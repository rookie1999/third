import h5py
import numpy as np
import os
import glob

"""
计算整个数据集的Mean均值和Std标准差
"""

def get_norm_stats(dataset_dir):
    all_qpos_data = []
    all_action_data = []

    # 1. 遍历所有 hdf5 文件
    files = glob.glob(os.path.join(dataset_dir, '*.hdf5'))
    print(f"Found {len(files)} episodes in {dataset_dir}")

    for filename in files:
        with h5py.File(filename, 'r') as f:
            qpos = f['observations/qpos'][:]  # (T, 8)
            action = f['action'][:]  # (T, 8)
            all_qpos_data.append(qpos)
            all_action_data.append(action)

    # 2. 拼接所有数据
    all_qpos_data = np.concatenate(all_qpos_data, axis=0)
    all_action_data = np.concatenate(all_action_data, axis=0)

    # 3. 计算均值和方差
    stats = {
        'action_mean': np.mean(all_action_data, axis=0),
        'action_std': np.std(all_action_data, axis=0),
        'qpos_mean': np.mean(all_qpos_data, axis=0),
        'qpos_std': np.std(all_qpos_data, axis=0)
    }

    # 防止 std 为 0 (例如夹爪一直没动)，加一个极小值
    stats['action_std'] = np.clip(stats['action_std'], 1e-2, np.inf)
    stats['qpos_std'] = np.clip(stats['qpos_std'], 1e-2, np.inf)

    print("Stats calculated successfully!")
    return stats


# 简单的归一化和反归一化辅助函数
def normalize_data(data, stats, key_prefix):
    mean = stats[f'{key_prefix}_mean']
    std = stats[f'{key_prefix}_std']
    return (data - mean) / std


def unnormalize_data(data, stats, key_prefix):
    mean = stats[f'{key_prefix}_mean']
    std = stats[f'{key_prefix}_std']
    return data * std + mean