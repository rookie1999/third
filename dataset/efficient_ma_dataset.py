import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from dataset.utils_norm import normalize_data


class EfficientEpisodicDataset(Dataset):
    def __init__(self, dataset_path_list, stats, camera_names=['cam_high'], chunk_size=100, use_cache=True,
                 n_obs_steps=1):
        """
        Args:
            dataset_path_list: 数据集文件路径列表
            stats: 归一化统计数据 (mean, std)
            camera_names: 使用的摄像头名称列表
            chunk_size: 预测动作的长度 (Action Chunk)
            use_cache: 是否使用缓存 (此版本代码主要依赖 h5py 动态读取，use_cache 预留接口)
            n_obs_steps: 观察的历史步数。
                         1 = 标准 ACT (输出单帧)
                         >=2 = MA-ACT/SpeedACT (输出时序帧 T, C, H, W)
        """
        super().__init__()
        self.stats = stats
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.use_cache = use_cache
        self.n_obs_steps = n_obs_steps

        self.dataset_path_list = dataset_path_list

        # 1. 建立索引 (轻量级，不加载具体数据)
        # 结构: self.indices[global_idx] = (file_path, episode_idx_in_file, start_ts, episode_len)
        self.indices = []

        for path in dataset_path_list:
            with h5py.File(path, 'r') as f:
                # 获取当前 episode 的总长度
                # 假设每个 hdf5 文件就是一个 episode
                # 如果你的 hdf5 结构不同，请在此调整 keys，例如 f['action'] -> f['data/action']
                action_len = f['action'].shape[0]

                # 每一个时间步都可以作为一个样本的起始点
                for i in range(action_len):
                    self.indices.append((path, 0, i, action_len))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        """
        返回:
            image_tensors: list of tensors.
                           如果 n_obs_steps=1: [(C, H, W), ...]
                           如果 n_obs_steps>1: [(T, C, H, W), ...]
            qpos_tensor: tensor.
                           如果 n_obs_steps=1: (D,)
                           如果 n_obs_steps>1: (T, D)
            action_tensor: (chunk_size, D)
            is_pad_tensor: (chunk_size,) bool
        """
        file_path, episode_idx_in_file, start_ts, episode_len = self.indices[index]

        with h5py.File(file_path, 'r') as f:
            # ======================================
            # 1. 动态读取图像 (支持多帧历史)
            # ======================================
            imgs_per_cam = []

            for cam in self.camera_names:
                frames = []
                # 循环读取历史帧：t, t-1, t-2 ...
                # 我们希望输出的时间顺序是 [t-(N-1), ..., t] (旧 -> 新)
                for i in range(self.n_obs_steps):
                    # 计算要读取的时间戳
                    # 例如 n_obs_steps=2, start_ts=10
                    # i=0 -> t_read = 10 - 1 + 0 = 9  (前一帧)
                    # i=1 -> t_read = 10 - 1 + 1 = 10 (当前帧)
                    t_read = start_ts - (self.n_obs_steps - 1) + i

                    # 边界处理：如果 t < 0，使用第0帧填充 (Padding with edge)
                    if t_read < 0:
                        t_read = 0

                    # 读取图像数据
                    # 假设 hdf5 结构是 observations/images/cam_name
                    img = f[f'observations/images/{cam}'][t_read]
                    frames.append(img)

                # 堆叠时间维度: (T, H, W, C)
                img_stack = np.stack(frames, axis=0)

                # 调整通道顺序: (T, H, W, C) -> (T, C, H, W) 符合 PyTorch 习惯
                img_stack = np.transpose(img_stack, (0, 3, 1, 2))

                # [关键兼容逻辑] ACT模式下去掉时间维度，变回 (C, H, W)
                if self.n_obs_steps == 1:
                    img_stack = img_stack[0]

                imgs_per_cam.append(img_stack)

            # ======================================
            # 2. 动态读取机械臂状态 (Qpos)
            # ======================================
            qpos_list = []
            for i in range(self.n_obs_steps):
                t_read = start_ts - (self.n_obs_steps - 1) + i
                if t_read < 0:
                    t_read = 0

                qpos_frame = f['observations/qpos'][t_read]
                qpos_list.append(qpos_frame)

            qpos_data = np.stack(qpos_list, axis=0)  # (T, D)

            # [关键兼容逻辑] ACT模式下变回 (D,)
            if self.n_obs_steps == 1:
                qpos_data = qpos_data[0]

            # 归一化 Qpos
            qpos_data = normalize_data(qpos_data, self.stats, 'qpos')
            qpos_tensor = torch.from_numpy(qpos_data).float()

            # 归一化 Images (0-255 -> 0.0-1.0)
            image_tensors = [torch.from_numpy(img / 255.0).float() for img in imgs_per_cam]

            # ======================================
            # 3. 读取未来动作块 (Action Chunk)
            #    注意：Action 不需要历史，只需要从当前 start_ts 开始的未来 chunk_size 步
            # ======================================
            end_ts = start_ts + self.chunk_size

            # 从文件元数据或 shape 获取 action 总长度
            total_action_len = f['action'].shape[0]

            if end_ts > total_action_len:
                # 情况 A: 剩余步数不足 chunk_size，需要填充
                curr_action = f['action'][start_ts:]
                pad_len = end_ts - total_action_len

                # 拿到最后一步动作，用于重复填充
                last_action = curr_action[-1] if len(curr_action) > 0 else np.zeros_like(f['action'][0])

                # 创建填充部分
                pad_action = np.repeat(last_action[np.newaxis, :], pad_len, axis=0)

                # 拼接
                action_chunk = np.concatenate([curr_action, pad_action], axis=0)

                # 创建 Mask (True 代表是填充的数据，loss 计算时会忽略)
                is_pad = np.zeros(self.chunk_size, dtype=bool)
                is_pad[-pad_len:] = True
            else:
                # 情况 B: 剩余步数充足，直接读取
                action_chunk = f['action'][start_ts:end_ts]
                is_pad = np.zeros(self.chunk_size, dtype=bool)

            # 归一化 Action
            action_chunk = normalize_data(action_chunk, self.stats, 'action')

            action_tensor = torch.from_numpy(action_chunk).float()
            is_pad_tensor = torch.from_numpy(is_pad).bool()

            return image_tensors, qpos_tensor, action_tensor, is_pad_tensor