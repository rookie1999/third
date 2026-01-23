#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Universal Data Converter
1. Recursively finds 'session_*' folders in a Task root (supports multi-level nesting).
2. Auto-detects Dual-Arm vs Single-Arm layout per session.
3. Aligns data to target frequency (20Hz/30Hz/60Hz).

python raw2hdf5.py \
  --task_root /path/to/task \
  --output /path/to/hdf5_output \
  --frequency 20 \
  --traj_source merge

'''
import h5py
import pandas as pd
import os
import cv2
import glob
from tqdm import tqdm
import numpy as np
import subprocess
from concurrent.futures import ProcessPoolExecutor
import argparse

# 假设原始相机帧率为 60fps
SOURCE_CAMERA_FPS = 60

# ----------------- 1. 基础工具函数 -----------------

def clamp_txt_to_csv(txt_path, csv_path):
    try:
        df = pd.read_csv(txt_path, sep=r'\s+', header=None)
        df.columns = ['timestamp', 'clamp']
        df.to_csv(csv_path, index=False)
        return True
    except Exception:
        return False

def ensure_clamp_csv_for_path(data_path):
    """
    检查指定数据目录下的 Clamp_Data, 如果只有 txt 则转 csv
    """
    clamp_dir = os.path.join(data_path, "Clamp_Data")
    if not os.path.exists(clamp_dir):
        return
    
    clamp_txt_path = os.path.join(clamp_dir, "clamp_data_tum.txt")
    clamp_csv_path = os.path.join(clamp_dir, "clamp.csv")

    if os.path.exists(clamp_csv_path):
        return
    if os.path.exists(clamp_txt_path):
        clamp_txt_to_csv(clamp_txt_path, clamp_csv_path)

def detect_layout(session_path):
    """
    检测 session 文件夹是双臂还是单臂
    返回: mode ('dual', 'single', 'invalid'), paths (dict)
    """
    if not os.path.isdir(session_path):
        return 'invalid', {}

    subdirs = [d for d in os.listdir(session_path) if os.path.isdir(os.path.join(session_path, d))]
    
    # 查找 left_hand_* 和 right_hand_*
    left_dir_name = next((d for d in subdirs if d.startswith('left_hand')), None)
    right_dir_name = next((d for d in subdirs if d.startswith('right_hand')), None)

    # 判定双臂
    if left_dir_name and right_dir_name:
        return 'dual', {
            'left': os.path.join(session_path, left_dir_name),
            'right': os.path.join(session_path, right_dir_name)
        }
    
    # 判定单臂 (只要存在 RGB_Images 或 Merged_Trajectory 就算单臂)
    if os.path.exists(os.path.join(session_path, "RGB_Images")):
        return 'single', {
            'single': session_path
        }
    
    return 'invalid', {}

def get_video_state(video_path):
    if not os.path.isfile(video_path):
        return False
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.returncode == 0 and float(result.stdout.strip()) > 0
    except:
        return False

def read_trj_txt(txt_path):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"trajectory txt not found: {txt_path}")
    df = pd.read_csv(txt_path, sep=r'\s+', header=None)
    if df.shape[1] < 8:
        raise ValueError(f"trajectory txt expects >=8 columns")
    df = df.iloc[:, :8]
    df.columns = ['timestamp', 'Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W']
    return df

def load_trajectory(session_path, traj_source):
    if traj_source == 'merge':
        trj_dir = os.path.join(session_path, "Merged_Trajectory")
        trj_csv = os.path.join(trj_dir, "merged_trajectory.csv")
        trj_txt = os.path.join(trj_dir, "merged_trajectory.txt")
        if os.path.exists(trj_csv):
            df = pd.read_csv(trj_csv)
            expected_cols = ['timestamp', 'Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W']
            if all(c in df.columns for c in expected_cols):
                return df[expected_cols]
        if os.path.exists(trj_txt):
            return read_trj_txt(trj_txt)
        raise FileNotFoundError(f"No merged trajectory found in {session_path}")
    elif traj_source == 'slam':
        return read_trj_txt(os.path.join(session_path, "SLAM_Poses", "slam_processed.txt"))
    elif traj_source == 'vive':
        return read_trj_txt(os.path.join(session_path, "Vive_Poses", "vive_data_tum.txt"))
    else:
        raise ValueError(f"Unknown traj_source={traj_source}")


def load_relative_transform_txt(txt_path):
    """
    读取相对位姿 txt 文件:
      - 以 '# ' 开头的行视为注释跳过
      - 数据格式: timestamp tx ty tz qx qy qz qw
    只做四元数 -> 6D 旋转表示的转换的前置读取，不做坐标系变换计算。
    返回:
      None 或 dict(timestamp, translation, quat)
    """
    if not os.path.exists(txt_path):
        return None

    timestamps = []
    translations = []
    quats = []

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                t = float(parts[0])
                tx, ty, tz = map(float, parts[1:4])
                qx, qy, qz, qw = map(float, parts[4:8])
            except ValueError:
                continue
            timestamps.append(t)
            translations.append([tx, ty, tz])
            quats.append([qx, qy, qz, qw])

    if not timestamps:
        return None

    return {
        "timestamp": np.asarray(timestamps, dtype=np.float64),
        "translation": np.asarray(translations, dtype=np.float32),
        "quat": np.asarray(quats, dtype=np.float32),
    }


def quat_to_rot6d(quat_array):
    """
    将四元数 (qx, qy, qz, qw) 转换为 6D 旋转表示 (取旋转矩阵的前两列拼接)。
    不进行任何坐标变换运算, 仅做姿态表达形式的转换。

    参数:
        quat_array: (..., 4) 的 numpy 数组, 顺序为 [qx, qy, qz, qw]
    返回:
        (..., 6) 的 numpy 数组
    """
    q = np.asarray(quat_array, dtype=np.float32)
    if q.ndim == 1:
        q = q[None, :]
    if q.shape[-1] != 4:
        raise ValueError("quat_to_rot6d expects last dim = 4 (qx, qy, qz, qw)")

    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]

    # 四元数 -> 旋转矩阵 (3x3) 的显式公式
    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    r00 = ww + xx - yy - zz
    r01 = 2.0 * (xy - zw)
    r10 = 2.0 * (xy + zw)
    r11 = ww - xx + yy - zz
    r20 = 2.0 * (xz - yw)
    r21 = 2.0 * (yz + xw)

    # 取旋转矩阵的前两列，按列拼接成 6D
    rot6d = np.stack(
        [
            r00, r10, r20,  # 第 1 列
            r01, r11, r21,  # 第 2 列
        ],
        axis=1,
    )
    return rot6d

def load_arm_data(data_path, traj_source):
    """读取指定路径下的机械臂数据"""
    if not os.path.isdir(data_path):
        return None

    image_dir = os.path.join(data_path, "RGB_Images")
    clamp_dir = os.path.join(data_path, "Clamp_Data")
    video_path = os.path.join(image_dir, "video.mp4")
    timestamps_path = os.path.join(image_dir, "timestamps.csv")
    clamp_path = os.path.join(clamp_dir, "clamp.csv")

    ensure_clamp_csv_for_path(data_path)

    if not (os.path.exists(video_path) and os.path.exists(timestamps_path) and os.path.exists(clamp_path)):
        return None
    if not get_video_state(video_path):
        return None

    try:
        traj = load_trajectory(data_path, traj_source)
        clamp = pd.read_csv(clamp_path)
        timestamps = pd.read_csv(timestamps_path)
    except Exception as e:
        # print(f"[ERROR] Loading data failed for {os.path.basename(data_path)}: {e}")
        return None

    if 'aligned_stamp' in timestamps.columns:
        timestamps['timestamp'] = timestamps['aligned_stamp']
    elif 'timestamp' not in timestamps.columns:
        timestamps['timestamp'] = np.arange(len(timestamps), dtype=float)
    
    if 'frame_index' not in timestamps.columns:
        timestamps['frame_index'] = np.arange(len(timestamps), dtype=int)

    return {
        "traj": traj,
        "clamp": clamp,
        "timestamps": timestamps,
        "video_path": video_path
    }

# ----------------- 2. HDF5 写入逻辑 -----------------

def write_hdf5_dual(output_path, records, camera_names):
    """写入双臂格式 HDF5"""
    with h5py.File(output_path, 'w', rdcc_nbytes=2 * 1024 ** 2) as root:
        root.attrs['sim'] = False
        arms_config = [("robot_0", records["robot_0"]), ("robot_1", records["robot_1"])]

        for arm_name, arm_data in arms_config:
            arm_grp = root.create_group(arm_name)
            obs_grp = arm_grp.create_group("observations")
            obs_grp.create_dataset("qpos", data=np.array(arm_data["qpos"], dtype=np.float32))

            # 可选: 相对位姿低维表示 (平移 + 6D 旋转, 共 9 维)
            # 约定:
            #   - 若存在 T_left_to_right_9d, 则写入 observations/T_left_to_right_9d
            #   - 若存在 T_right_to_left_9d, 则写入 observations/T_right_to_left_9d
            if "T_left_to_right_9d" in arm_data:
                obs_grp.create_dataset(
                    "T_left_to_right_9d",
                    data=np.array(arm_data["T_left_to_right_9d"], dtype=np.float32),
                )
            if "T_right_to_left_9d" in arm_data:
                obs_grp.create_dataset(
                    "T_right_to_left_9d",
                    data=np.array(arm_data["T_right_to_left_9d"], dtype=np.float32),
                )
            
            img_grp = obs_grp.create_group("images")
            imgs_np = np.array(arm_data["images"], dtype=np.uint8)
            for cam_name in camera_names:
                img_grp.create_dataset(cam_name, data=imgs_np, compression='gzip', compression_opts=4)
            
            arm_grp.create_dataset("action", data=np.array(arm_data["action"], dtype=np.float32))

def write_hdf5_single(output_path, records, camera_names):
    """写入单臂格式 HDF5"""
    with h5py.File(output_path, 'w', rdcc_nbytes=2 * 1024 ** 2) as root:
        root.attrs['sim'] = False
        
        obs_grp = root.create_group("observations")
        obs_grp.create_dataset("qpos", data=np.array(records["single"]["qpos"], dtype=np.float32))
        
        img_grp = obs_grp.create_group("images")
        imgs_np = np.array(records["single"]["images"], dtype=np.uint8)
        for cam_name in camera_names:
            img_grp.create_dataset(cam_name, data=imgs_np, compression='gzip', compression_opts=4)
            
        root.create_dataset("action", data=np.array(records["single"]["action"], dtype=np.float32))

# ----------------- 3. 处理单个 Session -----------------

def process_session_auto(session_path, output_root, episode_idx, camera_names, traj_source, target_fps):
    session_name = os.path.basename(session_path)
    mode, paths = detect_layout(session_path)

    if mode == 'invalid':
        return

    # 计算 Step
    if target_fps > SOURCE_CAMERA_FPS:
        step = 1
    else:
        step = int(SOURCE_CAMERA_FPS / target_fps)
        if step < 1: step = 1

    # --- 双臂 ---
    if mode == 'dual':
        left_data = load_arm_data(paths['left'], traj_source)
        right_data = load_arm_data(paths['right'], traj_source)

        if left_data is None or right_data is None:
            print(f"[SKIP] Missing data in dual session: {session_name}")
            return

        master_timestamps = left_data["timestamps"].iloc[::step].reset_index(drop=True)
        if master_timestamps.empty:
            return

        records = {
            "robot_0": {"qpos": [], "action": [], "images": []},
            "robot_1": {"qpos": [], "action": [], "images": []},
        }

        # 读取 session 根目录下的相对位姿文件（若存在则使用，不存在则跳过）
        # 例如: relative_transforms_right_to_left.txt, relative_transforms_left_to_right.txt
        rel_r2l_path = os.path.join(session_path, "relative_transforms_right_to_left.txt")
        rel_l2r_path = os.path.join(session_path, "relative_transforms_left_to_right.txt")
        rel_r2l = load_relative_transform_txt(rel_r2l_path)
        rel_l2r = load_relative_transform_txt(rel_l2r_path)

        if rel_r2l is not None:
            rel_r2l_ts = rel_r2l["timestamp"]
            rel_r2l_trans = rel_r2l["translation"]
            rel_r2l_quat = rel_r2l["quat"]
        else:
            rel_r2l_ts = rel_r2l_trans = rel_r2l_quat = None

        if rel_l2r is not None:
            rel_l2r_ts = rel_l2r["timestamp"]
            rel_l2r_trans = rel_l2r["translation"]
            rel_l2r_quat = rel_l2r["quat"]
        else:
            rel_l2r_ts = rel_l2r_trans = rel_l2r_quat = None

        # 当前 episode 内与帧对齐的 9D 相对位姿 (平移 3D + 旋转 6D)
        episode_r2l_9d = []  # 写到 robot_1 (right) 的 observations/T_right_to_left_9d
        episode_l2r_9d = []  # 写到 robot_0 (left) 的 observations/T_left_to_right_9d

        cap_l = cv2.VideoCapture(left_data["video_path"])
        cap_r = cv2.VideoCapture(right_data["video_path"])

        l_traj_ts, l_clamp_ts = (
            left_data["traj"]["timestamp"].to_numpy(),
            left_data["clamp"]["timestamp"].to_numpy(),
        )
        r_traj_ts, r_clamp_ts = (
            right_data["traj"]["timestamp"].to_numpy(),
            right_data["clamp"]["timestamp"].to_numpy(),
        )
        r_cam_ts = right_data["timestamps"]["timestamp"].to_numpy()
        r_cam_fidx = right_data["timestamps"]["frame_index"].to_numpy()

        valid_frames = 0
        for _, row in master_timestamps.iterrows():
            t_master = row["timestamp"]

            # Robot 0 (left)
            l_idx_t = int(np.argmin(np.abs(l_traj_ts - t_master)))
            l_idx_c = int(np.argmin(np.abs(l_clamp_ts - t_master)))
            l_pos = [
                left_data["traj"].iloc[l_idx_t]["Pos X"],
                left_data["traj"].iloc[l_idx_t]["Pos Y"],
                left_data["traj"].iloc[l_idx_t]["Pos Z"],
                left_data["traj"].iloc[l_idx_t]["Q_X"],
                left_data["traj"].iloc[l_idx_t]["Q_Y"],
                left_data["traj"].iloc[l_idx_t]["Q_Z"],
                left_data["traj"].iloc[l_idx_t]["Q_W"],
                left_data["clamp"].iloc[l_idx_c]["clamp"],
            ]

            cap_l.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_index"]))
            ret_l, frame_l = cap_l.read()

            # Robot 1 (right)
            r_cam_idx = int(np.argmin(np.abs(r_cam_ts - t_master)))
            t_right = r_cam_ts[r_cam_idx]
            r_idx_t = int(np.argmin(np.abs(r_traj_ts - t_right)))
            r_idx_c = int(np.argmin(np.abs(r_clamp_ts - t_right)))
            r_pos = [
                right_data["traj"].iloc[r_idx_t]["Pos X"],
                right_data["traj"].iloc[r_idx_t]["Pos Y"],
                right_data["traj"].iloc[r_idx_t]["Pos Z"],
                right_data["traj"].iloc[r_idx_t]["Q_X"],
                right_data["traj"].iloc[r_idx_t]["Q_Y"],
                right_data["traj"].iloc[r_idx_t]["Q_Z"],
                right_data["traj"].iloc[r_idx_t]["Q_W"],
                right_data["clamp"].iloc[r_idx_c]["clamp"],
            ]

            cap_r.set(cv2.CAP_PROP_POS_FRAMES, int(r_cam_fidx[r_cam_idx]))
            ret_r, frame_r = cap_r.read()

            if not (ret_l and ret_r):
                continue

            # 基本观测
            records["robot_0"]["qpos"].append(l_pos)
            records["robot_0"]["action"].append(l_pos)
            records["robot_0"]["images"].append(frame_l)
            records["robot_1"]["qpos"].append(r_pos)
            records["robot_1"]["action"].append(r_pos)
            records["robot_1"]["images"].append(frame_r)
            valid_frames += 1

            # 对齐已有的相对位姿 (仅根据时间戳做最近邻, 不进行坐标变换)
            if rel_r2l_ts is not None:
                idx_r2l = int(np.argmin(np.abs(rel_r2l_ts - t_master)))
                trans = rel_r2l_trans[idx_r2l]  # (3,)
                quat = rel_r2l_quat[idx_r2l]    # (4,)
                rot6d = quat_to_rot6d(quat)[0]  # (6,)
                pose9 = np.concatenate([trans, rot6d], axis=0).astype(np.float32)
                episode_r2l_9d.append(pose9)

            if rel_l2r_ts is not None:
                idx_l2r = int(np.argmin(np.abs(rel_l2r_ts - t_master)))
                trans = rel_l2r_trans[idx_l2r]
                quat = rel_l2r_quat[idx_l2r]
                rot6d = quat_to_rot6d(quat)[0]
                pose9 = np.concatenate([trans, rot6d], axis=0).astype(np.float32)
                episode_l2r_9d.append(pose9)

        cap_l.release()
        cap_r.release()

        if valid_frames > 0:
            # 将对齐后的相对位姿放入对应的 robot 组中
            if episode_l2r_9d:
                records["robot_0"]["T_left_to_right_9d"] = np.stack(
                    episode_l2r_9d, axis=0
                )
            if episode_r2l_9d:
                records["robot_1"]["T_right_to_left_9d"] = np.stack(
                    episode_r2l_9d, axis=0
                )

            hdf5_path = os.path.join(output_root, f"episode_{episode_idx:06d}.hdf5")
            write_hdf5_dual(hdf5_path, records, camera_names)
            print(
                f"[OK] Dual: {os.path.basename(os.path.dirname(session_path))}/{session_name} -> {os.path.basename(hdf5_path)}"
            )

    # --- 单臂 ---
    elif mode == 'single':
        single_data = load_arm_data(paths['single'], traj_source)
        if single_data is None:
            return

        master_timestamps = single_data["timestamps"].iloc[::step].reset_index(drop=True)
        if master_timestamps.empty: return

        records = {"single": {"qpos": [], "action": [], "images": []}}
        cap = cv2.VideoCapture(single_data["video_path"])

        traj_ts = single_data["traj"]['timestamp'].to_numpy()
        clamp_ts = single_data["clamp"]['timestamp'].to_numpy()
        
        valid_frames = 0
        for _, row in master_timestamps.iterrows():
            t_master = row['timestamp']
            
            idx_t = int(np.argmin(np.abs(traj_ts - t_master)))
            idx_c = int(np.argmin(np.abs(clamp_ts - t_master)))
            
            pos_quat = [
                single_data["traj"].iloc[idx_t]['Pos X'], single_data["traj"].iloc[idx_t]['Pos Y'], single_data["traj"].iloc[idx_t]['Pos Z'],
                single_data["traj"].iloc[idx_t]['Q_X'], single_data["traj"].iloc[idx_t]['Q_Y'], single_data["traj"].iloc[idx_t]['Q_Z'], single_data["traj"].iloc[idx_t]['Q_W'],
                single_data["clamp"].iloc[idx_c]['clamp']
            ]
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(row['frame_index']))
            ret, frame = cap.read()
            
            if ret:
                records["single"]["qpos"].append(pos_quat)
                records["single"]["action"].append(pos_quat)
                records["single"]["images"].append(frame)
                valid_frames += 1
        
        cap.release()

        if valid_frames > 0:
            hdf5_path = os.path.join(output_root, f'episode_{episode_idx:06d}.hdf5')
            write_hdf5_single(hdf5_path, records, camera_names)
            print(f"[OK] Single: {os.path.basename(os.path.dirname(session_path))}/{session_name} -> {os.path.basename(hdf5_path)}")


def find_all_sessions(root_path):
    """
    递归查找 root_path 下所有以 'session' 开头的文件夹。
    返回按路径排序的列表。
    """
    found_sessions = []
    print(f"[INFO] Scanning '{root_path}' recursively...")
    
    for root, dirs, files in os.walk(root_path):
        # 筛选出以 "session" 开头的目录
        for d in dirs:
            if d.startswith("session"):
                full_path = os.path.join(root, d)
                found_sessions.append(full_path)
    
    # 排序保证处理顺序的一致性 (例如 multi_sessions_1 先于 multi_sessions_2)
    return sorted(found_sessions)

def convert_task(task_root, output_dir, camera_names, traj_source, target_fps, num_workers):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 递归扫描所有 session
    session_dirs = find_all_sessions(task_root)
    
    if not session_dirs:
        print(f"[ERROR] No 'session_*' directories found anywhere under {task_root}")
        return

    print(f"[INFO] Found {len(session_dirs)} total sessions.")
    print(f"[INFO] Output Directory: {output_dir}")
    print(f"[INFO] Target Frequency: {target_fps}Hz")
    
    # 2. 全局连续编号
    episode_indices = list(range(len(session_dirs)))

    # 3. 并行处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(
            executor.map(
                process_session_auto,
                session_dirs,
                [output_dir] * len(session_dirs),
                episode_indices,
                [camera_names] * len(session_dirs),
                [traj_source] * len(session_dirs),
                [target_fps] * len(session_dirs)
            ),
            total=len(session_dirs),
            desc="Processing"
        ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively process Single/Dual arm sessions under a Task root.")
    
    parser.add_argument("--task_root", type=str, required=True, 
                        help="Root directory of the Task (scans recursively for session_*)")
    
    parser.add_argument("--output", type=str, required=True, help="Output directory for HDF5 files")
    
    parser.add_argument("--frequency", type=int, choices=[20, 30, 60], default=20,
                        help="Target alignment frequency (20, 30, 60Hz)")
    
    parser.add_argument("--camera_names", type=str, default="front", help="Camera dataset names")
    parser.add_argument("--traj_source", type=str, choices=["merge", "slam", "vive"], default="merge")
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()
    
    camera_names = [s.strip() for s in args.camera_names.split(",") if s.strip()]

    convert_task(
        args.task_root,
        args.output,
        camera_names,
        args.traj_source,
        args.frequency,
        args.num_workers
    )