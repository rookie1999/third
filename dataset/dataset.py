import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.utils_norm import normalize_data


class EpisodicDataset(Dataset):
    def __init__(self, dataset_path_list, stats, camera_names=['cam_high'], chunk_size=100):
        super().__init__()
        self.stats = stats
        self.camera_names = camera_names
        self.chunk_size = chunk_size  # ACT 核心参数：预测未来多少步

        self.episodes = []
        # 预加载所有数据到内存 (如果内存不够，可以在 __getitem__ 里实时读取)
        for path in dataset_path_list:
            with h5py.File(path, 'r') as f:
                qpos = f['observations/qpos'][:]
                action = f['action'][:]
                images = {}
                for cam in camera_names:
                    # 读取图像并转为 (T, C, H, W) 且归一化到 0-1
                    img = f[f'observations/images/{cam}'][:]
                    img = img.transpose(0, 3, 1, 2) / 255.0
                    images[cam] = img

                self.episodes.append({
                    'qpos': qpos,
                    'action': action,
                    'images': images,
                    'len': len(qpos)
                })

        # 建立索引映射：(episode_index, start_ts)
        self.indices = []
        for i, ep in enumerate(self.episodes):
            # 每一个时间步都可以作为一个样本的起始点
            for t in range(ep['len']):
                self.indices.append((i, t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        ep_idx, start_ts = self.indices[index]
        episode = self.episodes[ep_idx]

        # 1. 获取当前观测 (Observation)
        # qpos 需要归一化
        qpos = episode['qpos'][start_ts]
        qpos = normalize_data(qpos, self.stats, 'qpos')
        qpos = torch.from_numpy(qpos).float()

        # 图像 (假设只用一个相机)
        imgs = []
        for cam in self.camera_names:
            imgs.append(episode['images'][cam][start_ts])
        image = np.stack(imgs, axis=0)  # (Num_cams, C, H, W)
        image = torch.from_numpy(image).float()

        # 2. 获取未来动作块 (Action Chunk)
        # 目标是预测从 start_ts 开始的 chunk_size 个动作
        action_len = len(episode['action'])
        end_ts = start_ts + self.chunk_size

        # 如果超出这一集的长度，需要 Padding (重复最后一帧)
        if end_ts > action_len:
            curr_action = episode['action'][start_ts:]
            pad_len = end_ts - action_len
            last_action = curr_action[-1]
            pad_action = np.repeat(last_action[np.newaxis, :], pad_len, axis=0)
            action_chunk = np.concatenate([curr_action, pad_action], axis=0)
            # is_pad 标记哪些是填充的（不计算Loss）
            is_pad = np.zeros(self.chunk_size, dtype=bool)
            is_pad[-pad_len:] = True
        else:
            action_chunk = episode['action'][start_ts:end_ts]
            is_pad = np.zeros(self.chunk_size, dtype=bool)

        # 动作也需要归一化
        action_chunk = normalize_data(action_chunk, self.stats, 'action')

        return image, qpos, torch.from_numpy(action_chunk).float(), torch.from_numpy(is_pad).bool()