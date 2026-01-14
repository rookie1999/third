import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from dataset.utils_norm import normalize_data
from tqdm import tqdm


class EfficientEpisodicDataset(Dataset):
    def __init__(self, dataset_path_list, stats, camera_names=['cam_high'], chunk_size=100,
                 n_obs_steps=1):
        """
        Args:
            dataset_path_list: 数据集文件路径列表
            stats: 归一化统计数据 (mean, std)
            camera_names: 使用的摄像头名称列表
            chunk_size: 预测动作的长度 (Action Chunk)
            use_cache: True=全量加载到内存(极快); False=动态读取(省内存)
            n_obs_steps: 观察的历史步数。
                         1 = 标准 ACT (输出单帧)
                         >=2 = MA-ACT/SpeedACT (输出时序帧 T, C, H, W)
        """
        super().__init__()
        self.stats = stats
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.n_obs_steps = n_obs_steps
        self.dataset_path_list = dataset_path_list

        self.episodes = []
        self.indices = []

        print(f"🚀 [MA-Dataset] Pre-loading {len(dataset_path_list)} episodes into RAM (UInt8 mode)...")
        for ep_idx, path in enumerate(tqdm(dataset_path_list)):
            with h5py.File(path, 'r') as f:
                # 1. 读取基础数据 (T, D)
                qpos = f['observations/qpos'][:]
                action = f['action'][:]

                # 2. 读取图像 (保持 uint8 以节省内存)
                image_dict = {}
                for cam in camera_names:
                    img_data = f[f'observations/images/{cam}'][:]
                    # 统一转换为 (T, C, H, W) 格式
                    # 假设原始是 (T, H, W, C) 或者 (T, C, H, W)，这里做个判断
                    if img_data.ndim == 4 and img_data.shape[-1] == 3:  # (T, H, W, C) -> (T, C, H, W)
                        img_data = img_data.transpose(0, 3, 1, 2)
                    image_dict[cam] = img_data

                episode_len = len(qpos)
                self.episodes.append({
                    'qpos': qpos,
                    'action': action,
                    'images': image_dict,
                    'len': episode_len
                })

                # 3. 建立索引 (ep_idx, current_ts)
                for t in range(episode_len):
                    self.indices.append((ep_idx, t))

        print(f"✅ Loaded {len(self.indices)} samples. RAM optimized.")


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self._getitem_cache(index)

    def _getitem_cache(self, index):
        """从内存中读取 (极快)"""
        ep_idx, start_ts = self.indices[index]
        episode = self.episodes[ep_idx]

        # -------------------------------------------
        # 1. 读取图像 (支持多帧历史)
        # -------------------------------------------
        imgs_per_cam = []
        for cam in self.camera_names:
            # 获取该相机的所有帧 (T_total, C, H, W)
            full_video = episode['images'][cam]

            # 收集历史帧索引
            indices_to_read = []
            for i in range(self.n_obs_steps):
                # t, t-1, t-2 ... (倒序逻辑需注意，这里通常是从旧到新)
                # 假设 n_obs=2, start=10. 我们需要 [9, 10]
                t_read = start_ts - (self.n_obs_steps - 1) + i
                if t_read < 0: t_read = 0  # Padding
                indices_to_read.append(t_read)

            # 利用 Numpy 高级索引一次性取出 (n_obs, C, H, W)
            img_stack = full_video[indices_to_read]

            # ACT 兼容性: 去掉时间维度
            if self.n_obs_steps == 1:
                img_stack = img_stack[0]

            imgs_per_cam.append(img_stack)

        # 归一化图像 (0-255 -> 0.0-1.0)
        image_tensors = [torch.from_numpy(img).float() / 255.0 for img in imgs_per_cam]

        # -------------------------------------------
        # 2. 读取 Qpos (支持多帧历史)
        # -------------------------------------------
        indices_to_read = []
        for i in range(self.n_obs_steps):
            t_read = start_ts - (self.n_obs_steps - 1) + i
            if t_read < 0: t_read = 0
            indices_to_read.append(t_read)

        qpos_data = episode['qpos'][indices_to_read]  # (n_obs, D)

        if self.n_obs_steps == 1:
            qpos_data = qpos_data[0]

        qpos_data = normalize_data(qpos_data, self.stats, 'qpos')
        qpos_tensor = torch.from_numpy(qpos_data).float()

        # -------------------------------------------
        # 3. 读取 Action Chunk (未来预测)
        # -------------------------------------------
        action_full = episode['action']
        total_len = episode['len']
        end_ts = start_ts + self.chunk_size

        if end_ts > total_len:
            # Padding
            curr_action = action_full[start_ts:]
            pad_len = end_ts - total_len
            last_action = curr_action[-1] if len(curr_action) > 0 else np.zeros_like(action_full[0])
            pad_action = np.repeat(last_action[np.newaxis, :], pad_len, axis=0)
            action_chunk = np.concatenate([curr_action, pad_action], axis=0)
            is_pad = np.zeros(self.chunk_size, dtype=bool)
            is_pad[-pad_len:] = True
        else:
            action_chunk = action_full[start_ts:end_ts]
            is_pad = np.zeros(self.chunk_size, dtype=bool)

        action_chunk = normalize_data(action_chunk, self.stats, 'action')
        return image_tensors, qpos_tensor, torch.from_numpy(action_chunk).float(), torch.from_numpy(is_pad).bool()