import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from dataset.utils_norm import normalize_data


class EfficientEpisodicDataset(Dataset):
    def __init__(self, dataset_path_list, stats, camera_names=['cam_high'], chunk_size=100, use_cache=True):
        super().__init__()
        self.stats = stats
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.use_cache = use_cache

        self.dataset_path_list = dataset_path_list

        # 1. 建立索引，但不加载数据 (轻量级)
        # 结构: self.indices[global_idx] = (file_path, episode_idx_in_file, start_ts, episode_len)
        self.indices = []

        # 我们需要知道每个文件的长度，这里只读元数据
        for path in dataset_path_list:
            with h5py.File(path, 'r') as f:
                # 假设每个文件包含一个 episode 或者多个，原代码逻辑看起来是一个文件可能包含一个episode
                # 原代码逻辑: path -> qpos (len)
                # 为了不破坏逻辑，我们假设每个 path 就是一个 episode
                qpos_len = f['observations/qpos'].shape[0]
                action_len = f['action'].shape[0]

                # 每一个时间步都可以作为一个样本的起始点
                for t in range(qpos_len):
                    self.indices.append({
                        'path': path,
                        'start_ts': t,
                        'total_len': qpos_len,
                        'action_len': action_len
                    })

        # 2. 文件句柄缓存 (Worker 级别)
        # PyTorch DataLoader 的每个 worker 进程是独立的，不能共享 h5py 句柄
        # 我们将在 __getitem__ 中懒加载句柄
        self._file_handles = {}

    def __len__(self):
        return len(self.indices)

    def _get_file_handle(self, path):
        if not self.use_cache:
            return h5py.File(path, 'r')

        if path not in self._file_handles:
            # swmr=True (Single Writer Multiple Reader) 有助于并发读取稳定性
            # libver='latest' 使用最新的格式，通常更快
            self._file_handles[path] = h5py.File(path, 'r', swmr=True, libver='latest')
        return self._file_handles[path]

    def __getitem__(self, index):
        meta = self.indices[index]
        path = meta['path']
        start_ts = meta['start_ts']
        total_len = meta['total_len']

        # 获取文件句柄
        f = self._get_file_handle(path)

        try:
            # 1. 获取当前观测 (Observation)
            # 动态读取: 只读 start_ts 这一帧
            qpos = f['observations/qpos'][start_ts]

            # 图像处理: 动态读取并转换
            imgs = []
            for cam in self.camera_names:
                # 假设原始存储格式是 (T, H, W, C) 或者 (T, C, H, W)
                # 你的原代码中 img = img.transpose(0, 3, 1, 2) 暗示原始是 (T, H, W, C)
                # 我们只读第 start_ts 帧，得到 (H, W, C)
                img_data = f[f'observations/images/{cam}'][start_ts]

                # 转换为 (C, H, W)
                # 如果原始是 (H, W, C) -> permute (2, 0, 1)
                # 如果原始是 (C, H, W) -> 不动
                # 根据原代码 `transpose(0, 3, 1, 2)` (N, H, W, C) -> (N, C, H, W) 推断，单帧是 (H, W, C)
                img_data = img_data.transpose(2, 0, 1)

                # 归一化 (放在这里做，随用随算，节省 RAM 存储)
                img_data = img_data / 255.0
                imgs.append(img_data)

            image = np.stack(imgs, axis=0)  # (Num_cams, C, H, W)

            # Qpos Normalize
            qpos = normalize_data(qpos, self.stats, 'qpos')
            qpos = torch.from_numpy(qpos).float()
            image = torch.from_numpy(image).float()

            # 2. 获取未来动作块 (Action Chunk)
            end_ts = start_ts + self.chunk_size
            action_len = meta['action_len']

            if end_ts > action_len:
                # 需要 Padding
                # 只读取需要的切片
                curr_action = f['action'][start_ts:]
                pad_len = end_ts - action_len
                last_action = curr_action[-1]
                pad_action = np.repeat(last_action[np.newaxis, :], pad_len, axis=0)
                action_chunk = np.concatenate([curr_action, pad_action], axis=0)
                is_pad = np.zeros(self.chunk_size, dtype=bool)
                is_pad[-pad_len:] = True
            else:
                # 直接读取切片
                action_chunk = f['action'][start_ts:end_ts]
                is_pad = np.zeros(self.chunk_size, dtype=bool)

            action_chunk = normalize_data(action_chunk, self.stats, 'action')

            # 如果没有使用缓存，记得关闭文件
            if not self.use_cache:
                f.close()

            return image, qpos, torch.from_numpy(action_chunk).float(), torch.from_numpy(is_pad).bool()

        except Exception as e:
            print(f"Error reading file {path} at index {start_ts}: {e}")
            raise e