import h5py
import numpy as np
import torch
import cv2
import os
from torch.utils.data import Dataset
from dataset.utils_norm import normalize_data
from tqdm import tqdm


class VideoBasedEfficientMADataset(Dataset):
    def __init__(self, dataset_path_list, stats, camera_names=['cam_high'], chunk_size=100,
                 n_obs_steps=1):
        """
        Args:
            dataset_path_list: HDF5 路径列表 (用于读取 qpos/action 和定位 video)
            stats: 归一化统计数据
            camera_names: 摄像头名称
            chunk_size: 动作预测长度
            n_obs_steps: 历史观测步数 (MA-ACT 核心参数)
        """
        super().__init__()
        self.stats = stats
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.n_obs_steps = n_obs_steps
        self.dataset_path_list = dataset_path_list

        self.episodes = []
        self.indices = []

        print(f"🚀 [MA-ACT Video-Mode] Pre-loading {len(dataset_path_list)} episodes into RAM...")

        for ep_idx, path in enumerate(tqdm(dataset_path_list)):
            # 1. 从 HDF5 读取非图像数据 (qpos, action)
            with h5py.File(path, 'r') as f:
                qpos = f['observations/qpos'][:]
                action = f['action'][:]
                episode_len = len(qpos)
                if 'speed_level' not in f.attrs:
                    raise ValueError(
                        f"❌ Critical Error: Episode {path} bas NO speed tag! Please run add_speed_tag.py first.")
                speed_level = f.attrs['speed_level']

            # 2. 从 MP4 读取图像序列
            image_dict = {}
            for cam in camera_names:
                # 路径映射逻辑： .../episode/episode_X.hdf5 -> .../video/episode_X.mp4
                video_path = path.replace('episode', 'video').replace('.hdf5', '.mp4')

                if not os.path.exists(video_path):
                    # 容错：如果在同级目录找不到，尝试在 dataset_path_list 所在目录找
                    # 这是一个常见的路径坑，根据你的实际目录结构调整
                    video_path = path.replace('.hdf5', '.mp4')
                    if not os.path.exists(video_path):
                        raise FileNotFoundError(f"Video file not found: {video_path}")

                # 读取视频所有帧 -> (T, H, W, C)
                frames = self._read_video_frames(video_path)

                # 帧数对齐检查
                if len(frames) != episode_len:
                    # print(f"⚠️ Alignment Warning: {video_path} has {len(frames)} frames, but qpos has {episode_len}.")
                    min_len = min(len(frames), episode_len)
                    frames = frames[:min_len]
                    qpos = qpos[:min_len]
                    action = action[:min_len]
                    episode_len = min_len

                # 格式转换: (T, H, W, C) -> (T, C, H, W)
                # 保持 uint8 以节省内存，这一点对于 MA-ACT 尤为重要，因为历史帧多，内存压力大
                frames = frames.transpose(0, 3, 1, 2)
                image_dict[cam] = frames

            self.episodes.append({
                'qpos': qpos,
                'action': action,
                'images': image_dict,
                'len': episode_len,
                'speed_level': speed_level
            })

            # 3. 建立索引
            for t in range(episode_len):
                self.indices.append((ep_idx, t))

        print(f"✅ Loaded {len(self.indices)} samples (Video Source). RAM optimized.")

    def _read_video_frames(self, video_path):
        """读取视频并转为 RGB"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR -> RGB 是必须的
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return np.array(frames, dtype=np.uint8)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        ep_idx, start_ts = self.indices[index]
        episode = self.episodes[ep_idx]

        imgs_per_cam = []
        for cam in self.camera_names:
            full_video = episode['images'][cam]  # (T_total, C, H, W)

            # 收集历史帧索引
            indices_to_read = []
            for i in range(self.n_obs_steps):
                # 如果 n_obs=2, start=10 -> 读取 [9, 10]
                t_read = start_ts - (self.n_obs_steps - 1) + i
                if t_read < 0: t_read = 0  # Padding for first few frames
                indices_to_read.append(t_read)

            # Numpy 高级索引切片
            img_stack = full_video[indices_to_read]  # (n_obs, C, H, W)

            # 如果 n_obs_steps == 1 (标准 ACT)，去掉时间维度兼容接口
            if self.n_obs_steps == 1:
                img_stack = img_stack[0]

            imgs_per_cam.append(img_stack)

        # 归一化 (0-255 -> 0.0-1.0)
        image_tensors = [torch.from_numpy(img).float() / 255.0 for img in imgs_per_cam]

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

        action_full = episode['action']
        total_len = episode['len']
        end_ts = start_ts + self.chunk_size

        speed_label = torch.tensor(episode['speed_level'], dtype=torch.long)

        if end_ts > total_len:
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

        return image_tensors, qpos_tensor, torch.from_numpy(action_chunk).float(), torch.from_numpy(is_pad).bool(), speed_label