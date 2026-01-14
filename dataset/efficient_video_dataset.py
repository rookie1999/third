import h5py
import numpy as np
import torch
import cv2
import os
from torch.utils.data import Dataset
from dataset.utils_norm import normalize_data
from tqdm import tqdm


class VideoBasedEfficientDataset(Dataset):
    def __init__(self, dataset_path_list, stats, camera_names=['cam_high'], chunk_size=100):
        super().__init__()
        self.stats = stats
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.dataset_path_list = dataset_path_list

        self.episodes = []
        self.indices = []

        print(f"🚀 [Video-Mode] Pre-loading {len(dataset_path_list)} episodes into RAM...")

        for ep_idx, path in enumerate(tqdm(dataset_path_list)):
            # 1. 从 HDF5 读取非图像数据 (qpos, action)
            with h5py.File(path, 'r') as f:
                qpos = f['observations/qpos'][:]
                action = f['action'][:]
                episode_len = len(qpos)

            # 2. 从视频文件读取图像
            image_dict = {}
            for cam in camera_names:
                # 构建视频路径：假设 hdf5 在 .../episode/xxx.hdf5，视频在 .../video/xxx.mp4
                # 根据你的实际路径结构进行调整
                video_path = path.replace('episode', 'video').replace('.hdf5', '.mp4')

                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video not found: {video_path}")

                frames = self._read_video_frames(video_path)

                # 检查帧数对齐
                if len(frames) != episode_len:
                    print(f"⚠️ Warning: Frame mismatch in {video_path}. "
                          f"Video: {len(frames)}, Qpos: {episode_len}. Truncating to shorter one.")
                    min_len = min(len(frames), episode_len)
                    frames = frames[:min_len]
                    qpos = qpos[:min_len]
                    action = action[:min_len]
                    episode_len = min_len

                # 转换为 (T, C, H, W) 格式，保持 uint8 节省内存
                # frames 是 (T, H, W, C)
                frames = frames.transpose(0, 3, 1, 2)
                image_dict[cam] = frames

            self.episodes.append({
                'qpos': qpos,
                'action': action,
                'images': image_dict,
                'len': episode_len
            })

            # 建立索引
            for t in range(episode_len):
                self.indices.append((ep_idx, t))

        print(f"✅ Loaded {len(self.indices)} samples from Video sources.")

    def _read_video_frames(self, video_path):
        """使用 OpenCV 读取所有视频帧"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV 默认读取为 BGR，需要转为 RGB 以匹配训练预期
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return np.array(frames, dtype=np.uint8)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
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
        end_ts = start_ts + self.chunk_size

        if end_ts > action_len:
            curr_action = action_full[start_ts:]
            pad_len = end_ts - action_len
            last_action = curr_action[-1]
            pad_action = np.repeat(last_action[np.newaxis, :], pad_len, axis=0)
            action_chunk = np.concatenate([curr_action, pad_action], axis=0)
            is_pad = np.zeros(self.chunk_size, dtype=bool)
            is_pad[-pad_len:] = True
        else:
            action_chunk = action_full[start_ts:end_ts]
            is_pad = np.zeros(self.chunk_size, dtype=bool)

        action_chunk = normalize_data(action_chunk, self.stats, 'action')

        return image_tensor, qpos_tensor, torch.from_numpy(action_chunk).float(), torch.from_numpy(is_pad).bool()