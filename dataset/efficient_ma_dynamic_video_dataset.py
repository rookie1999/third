import h5py
import numpy as np
import torch
import cv2
import os
import psutil  # 建议安装: pip install psutil，用于自动检测内存，或者手动指定数量
from torch.utils.data import Dataset
from dataset.utils_norm import normalize_data
from tqdm import tqdm


class VideoBasedEfficientMADataset(Dataset):
    def __init__(self, dataset_path_list, stats, camera_names=['cam_high'], chunk_size=100,
                 n_obs_steps=1, max_preload_episodes=400):
        """
        Args:
            max_preload_episodes (int): 核心参数。指定有多少集数据会"常驻内存"。
                                        - 如果内存大（64G+），可以设为 200甚至更多。
                                        - 如果内存小，设为 0 就变成了纯动态读取模式。
        """
        super().__init__()
        self.stats = stats
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.n_obs_steps = n_obs_steps
        self.dataset_path_list = dataset_path_list

        # 记录缓存策略
        self.max_preload_episodes = max_preload_episodes
        self.preloaded_count = 0

        self.episodes = []
        self.indices = []

        print(f"🚀 [Hybrid Dataset] Initializing...")
        print(f"   - Target Cache: {max_preload_episodes} episodes in RAM")
        print(f"   - The rest will be loaded from Disk on-the-fly.")

        for ep_idx, path in enumerate(tqdm(dataset_path_list)):
            # 1. 读取基础数据 (永远在内存，因为很小)
            with h5py.File(path, 'r') as f:
                qpos = f['observations/qpos'][:]
                action = f['action'][:]
                episode_len = len(qpos)
                if 'speed_level' not in f.attrs:
                    raise ValueError(
                        f"❌ Critical Error: Episode {path} bas NO speed tag! Please run add_speed_tag.py first.")
                speed_level = f.attrs['speed_level']

            # 2. 决定是否预加载图像
            image_dict = {}
            video_paths = {}
            is_cached = False

            # 寻找视频路径
            for cam in camera_names:
                video_path = path.replace('episode', 'video').replace('.hdf5', '.mp4')
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video not found: {video_path}")
                video_paths[cam] = video_path

            # >>> 核心逻辑：如果配额没用完，就加载进内存 <<<
            if self.preloaded_count < self.max_preload_episodes:
                try:
                    # 尝试加载
                    for cam, v_path in video_paths.items():
                        frames = self._read_all_video_frames(v_path)

                        # 对齐检查
                        if len(frames) != episode_len:
                            min_len = min(len(frames), episode_len)
                            frames = frames[:min_len]
                            if cam == camera_names[0]:  # 只在第一个相机修正一次长度
                                qpos = qpos[:min_len]
                                action = action[:min_len]
                                episode_len = min_len

                        # 转为 (T, C, H, W)
                        frames = frames.transpose(0, 3, 1, 2)
                        image_dict[cam] = frames

                    is_cached = True
                    self.preloaded_count += 1
                except Exception as e:
                    print(f"⚠️ Preload failed for {path}: {e}. Fallback to disk mode.")
                    is_cached = False
                    image_dict = {}  # 清空可能加载了一半的数据

            # 存入 episode 列表
            self.episodes.append({
                'qpos': qpos,
                'action': action,
                'images': image_dict if is_cached else None,  # 如果缓存了就是数据
                'video_paths': video_paths,  # 永远存路径作为后备
                'is_cached': is_cached,  # 标记这一集是否在内存
                'len': episode_len,
                'speed_level': speed_level
            })

            # 3. 建立索引
            for t in range(episode_len):
                self.indices.append((ep_idx, t))

        print(f"✅ Init Done. Memory Status: {self.preloaded_count}/{len(dataset_path_list)} episodes cached in RAM.")

    def _read_all_video_frames(self, video_path):
        """一次性读取完整视频 (用于预加载)"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return np.array(frames, dtype=np.uint8)

    def _read_partial_frames(self, video_path, frame_indices):
        """动态读取指定帧 (用于未缓存的集)"""
        sorted_indices = sorted(list(set(frame_indices)))
        cap = cv2.VideoCapture(video_path)
        frames_dict = {}
        for target_idx in sorted_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_dict[target_idx] = frame
            else:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frames_dict[target_idx] = np.zeros((h, w, 3), dtype=np.uint8)
        cap.release()

        output = []
        for idx in frame_indices:
            output.append(frames_dict.get(idx, frames_dict.get(sorted_indices[-1])))

        return np.array(output, dtype=np.uint8).transpose(0, 3, 1, 2)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        ep_idx, start_ts = self.indices[index]
        episode = self.episodes[ep_idx]

        # 1. 计算需要的历史帧索引
        indices_to_read = []
        for i in range(self.n_obs_steps):
            t_read = start_ts - (self.n_obs_steps - 1) + i
            if t_read < 0: t_read = 0
            if t_read >= episode['len']: t_read = episode['len'] - 1
            indices_to_read.append(t_read)

        imgs_per_cam = []
        for cam in self.camera_names:
            if episode['is_cached']:
                full_video = episode['images'][cam]
                img_stack = full_video[indices_to_read]
            else:
                video_path = episode['video_paths'][cam]
                img_stack = self._read_partial_frames(video_path, indices_to_read)

            if self.n_obs_steps == 1:
                img_stack = img_stack[0]
            imgs_per_cam.append(img_stack)

        image_tensors = [torch.from_numpy(img).float() / 255.0 for img in imgs_per_cam]

        # 读取 qpos
        qpos_data = episode['qpos'][indices_to_read]
        if self.n_obs_steps == 1: qpos_data = qpos_data[0]
        qpos_tensor = torch.from_numpy(normalize_data(qpos_data, self.stats, 'qpos')).float()

        # 读取 Action
        action_full = episode['action']
        total_len = episode['len']
        end_ts = start_ts + self.chunk_size
        speed_label = torch.tensor(episode['speed_level'], dtype=torch.long)  # 返回档位标签

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

        return image_tensors, qpos_tensor, torch.from_numpy(action_chunk).float(), torch.from_numpy(
            is_pad).bool(), speed_label