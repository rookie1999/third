import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from dataset.utils_norm import normalize_data
from tqdm import tqdm


class EfficientEpisodicDataset(Dataset):
    def __init__(self, dataset_path_list, stats, camera_names=['cam_high'], chunk_size=100, use_cache=True):
        """
        Args:
            dataset_path_list: 数据集路径列表
            stats: 归一化统计数据
            camera_names: 摄像头名称列表
            chunk_size: 动作预测长度
            use_cache: (本版修改) True=全量加载到内存(极快); False=使用旧版磁盘读取(慢但省内存)
        """
        super().__init__()
        self.stats = stats
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.use_cache = use_cache
        self.dataset_path_list = dataset_path_list

        self.episodes = []
        self.indices = []

        if self.use_cache:
            print(f"🚀 Pre-loading {len(dataset_path_list)} episodes into RAM (UInt8 mode)...")
            # --- 模式 A: 内存全量加载 (极速模式) ---
            for ep_idx, path in enumerate(tqdm(dataset_path_list)):
                with h5py.File(path, 'r') as f:
                    # 1. 读取基础数据
                    qpos = f['observations/qpos'][:]
                    action = f['action'][:]

                    # 2. 读取图像 (保持 uint8)
                    image_dict = {}
                    for cam in camera_names:
                        img_data = f[f'observations/images/{cam}'][:]
                        # 统一转换为 (T, C, H, W) 格式
                        if img_data.shape[-1] == 3:  # 如果是 (T, H, W, C)
                            img_data = img_data.transpose(0, 3, 1, 2)
                        image_dict[cam] = img_data

                    episode_len = len(qpos)
                    self.episodes.append({
                        'qpos': qpos,
                        'action': action,
                        'images': image_dict,
                        'len': episode_len
                    })

                    # 建立索引
                    for t in range(episode_len):
                        self.indices.append((ep_idx, t))
            print(f"✅ Loaded {len(self.indices)} samples. RAM optimized.")

        else:
            # --- 模式 B: 磁盘动态读取 (省内存模式 - 旧版逻辑) ---
            print(f"🐢 Lazy-loading mode (Disk I/O heavy).")
            for path in dataset_path_list:
                with h5py.File(path, 'r') as f:
                    qpos_len = f['observations/qpos'].shape[0]
                    action_len = f['action'].shape[0]
                    for t in range(qpos_len):
                        self.indices.append({
                            'path': path,
                            'start_ts': t,
                            'total_len': qpos_len,
                            'action_len': action_len
                        })
            self._file_handles = {}

    def __len__(self):
        return len(self.indices)

    def _get_file_handle(self, path):
        if path not in self._file_handles:
            self._file_handles[path] = h5py.File(path, 'r', swmr=True, libver='latest')
        return self._file_handles[path]

    def __getitem__(self, index):
        if self.use_cache:
            # ==========================================
            # 模式 A: 从内存读取 (无 IO)
            # ==========================================
            ep_idx, start_ts = self.indices[index]
            episode = self.episodes[ep_idx]

            # 1. Qpos
            qpos = episode['qpos'][start_ts]
            qpos = normalize_data(qpos, self.stats, 'qpos')
            qpos_tensor = torch.from_numpy(qpos).float()

            # 2. Images (UInt8 -> Float / 255.0)
            imgs = []
            for cam in self.camera_names:
                img_uint8 = episode['images'][cam][start_ts]  # (C, H, W)
                # 实时归一化 0-1
                img_float = torch.from_numpy(img_uint8).float() / 255.0
                imgs.append(img_float)
            image_tensor = torch.stack(imgs, dim=0)

            # 3. Action Chunk
            action_full = episode['action']
            action_len = episode['len']

        else:
            # ==========================================
            # 模式 B: 从磁盘读取 (有 IO)
            # ==========================================
            meta = self.indices[index]
            path, start_ts = meta['path'], meta['start_ts']
            f = self._get_file_handle(path)

            # 1. Qpos
            qpos = f['observations/qpos'][start_ts]
            qpos = normalize_data(qpos, self.stats, 'qpos')
            qpos_tensor = torch.from_numpy(qpos).float()

            # 2. Images
            imgs = []
            for cam in self.camera_names:
                img_data = f[f'observations/images/{cam}'][start_ts]
                # 假设原始是 (H, W, C) -> permute (2, 0, 1)
                img_data = img_data.transpose(2, 0, 1)
                img_float = torch.from_numpy(img_data).float() / 255.0
                imgs.append(img_float)
            image_tensor = torch.stack(imgs, dim=0)

            # 3. Action Setup
            action_full = f['action']  # h5py dataset object
            action_len = meta['action_len']

        # --- 公共部分：Action Chunk 切片逻辑 ---
        end_ts = start_ts + self.chunk_size

        if end_ts > action_len:
            # 需要 Padding
            curr_action = action_full[start_ts:]
            pad_len = end_ts - action_len
            # 注意: 如果是 h5py 对象，curr_action已经是numpy array了
            last_action = curr_action[-1]
            pad_action = np.repeat(last_action[np.newaxis, :], pad_len, axis=0)
            action_chunk = np.concatenate([curr_action, pad_action], axis=0)
            is_pad = np.zeros(self.chunk_size, dtype=bool)
            is_pad[-pad_len:] = True
        else:
            action_chunk = action_full[start_ts:end_ts]
            is_pad = np.zeros(self.chunk_size, dtype=bool)

        # Action Normalize
        action_chunk = normalize_data(action_chunk, self.stats, 'action')

        return image_tensor, qpos_tensor, torch.from_numpy(action_chunk).float(), torch.from_numpy(is_pad).bool()