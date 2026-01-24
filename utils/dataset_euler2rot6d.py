import h5py
import numpy as np
import os
import argparse
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# ==========================================
# 1. 你的转换函数 (直接复用)
# ==========================================

def euler_to_rot6d(euler_angles, seq='xyz', degrees=False):
    """
    将欧拉角转换为 6D 旋转表示 (旋转矩阵的前两列)。
    支持单条数据 (3,) 和 批量数据 (N, 3)。
    """
    # 1. 维度处理：记录原始形状是否为 1D
    euler_angles = np.asarray(euler_angles)
    is_single = (euler_angles.ndim == 1)

    # 强制转为 2D (N, 3) 进行计算
    if is_single:
        euler_angles = euler_angles[None, :]

    # 2. 核心转换
    # seq='xyz' (外旋) 等价于 Intrinsic ZYX (内旋)，只要输入顺序是 [roll, pitch, yaw]
    r = R.from_euler(seq, euler_angles, degrees=degrees)
    matrices = r.as_matrix()  # (N, 3, 3)

    # 取前两列 (Rx, Ry) 并拼接
    # matrices[:, :, 0] -> (N, 3)
    # matrices[:, :, 1] -> (N, 3)
    # result -> (N, 6)
    rot6d = np.concatenate([matrices[:, :, 0], matrices[:, :, 1]], axis=1)

    # 3. 维度还原
    if is_single:
        return rot6d[0]  # 返回 (6,)
    return rot6d  # 返回 (N, 6)


# ==========================================
# 2. 数据处理逻辑
# ==========================================

def process_vector_data(data):
    """
    处理单个数据集 (qpos 或 action)
    输入形状: (N, 7) -> [x, y, z, roll, pitch, yaw, gripper]
    输出形状: (N, 10) -> [x, y, z, r1, r2, r3, r4, r5, r6, gripper]
    """
    # 确保数据是 2D 的 (N, D)
    if data.ndim == 1:
        data = data[None, :]
        is_single_frame = True
    else:
        is_single_frame = False

    # 维度检查
    if data.shape[1] != 7:
        print(f"⚠️ Warning: Data dim is {data.shape[1]}, expected 7. Skipping conversion for this dataset.")
        return data

    # 1. 切片提取
    pos = data[:, :3]  # [x, y, z]
    euler = data[:, 3:6]  # [roll, pitch, yaw]
    gripper = data[:, 6:]  # [gripper]

    # 2. 转换旋转 (Euler -> Rot6D)
    # 假设你的欧拉角是弧度制，且顺序为 xyz (RPY)
    rot6d = euler_to_rot6d(euler, seq='xyz', degrees=False)

    # 3. 拼接
    # (N, 3) + (N, 6) + (N, 1) = (N, 10)
    new_data = np.concatenate([pos, rot6d, gripper], axis=1)

    if is_single_frame:
        return new_data[0]
    return new_data


def copy_and_convert_group(source_group, dest_group):
    """
    递归复制 HDF5 组，如果是 qpos/action 则进行转换
    """
    for key, item in source_group.items():
        if isinstance(item, h5py.Group):
            # 如果是组，递归创建并处理
            new_group = dest_group.create_group(key)
            copy_and_convert_group(item, new_group)
        elif isinstance(item, h5py.Dataset):
            # 如果是数据集，检查名字
            # 根据你的描述，需要转换的是 'qpos' 和 'action'
            # 有些数据集可能嵌套在 observations/qpos 下，这里只匹配 key 的名字
            if key in ['qpos', 'action']:
                # 读取原始数据
                data = item[:]

                # 执行转换
                new_data = process_vector_data(data)

                # 创建新数据集
                dest_group.create_dataset(key, data=new_data)

                # 复制属性 (Attributes) - 这一步很重要，防止丢失元数据
                for attr_name, attr_value in item.attrs.items():
                    dest_group[key].attrs[attr_name] = attr_value
            else:
                # 其他数据 (如 images, timestamps) 直接复制
                source_group.copy(key, dest_group)


def process_folder(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    files = [f for f in os.listdir(input_dir) if f.endswith('.hdf5')]
    print(f"Found {len(files)} HDF5 files to process.")

    for filename in tqdm(files, desc="Converting HDF5"):
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)

        try:
            with h5py.File(src_path, 'r') as f_src:
                with h5py.File(dst_path, 'w') as f_dst:
                    # 递归复制并转换
                    copy_and_convert_group(f_src, f_dst)

                    # 复制文件根目录的属性 (如果有)
                    for attr_name, attr_value in f_src.attrs.items():
                        f_dst.attrs[attr_name] = attr_value

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print("\n✅ All conversions completed!")
    print(f"Input dir:  {input_dir}")
    print(f"Output dir: {output_dir}")


# ==========================================
# 3. 主程序入口
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert HDF5 qpos/action from Euler(7D) to Rot6D(10D).")
    parser.add_argument('--input', type=str, required=True, help='Path to input folder containing .hdf5 files')
    parser.add_argument('--output', type=str, required=True, help='Path to output folder for converted files')

    args = parser.parse_args()

    process_folder(args.input, args.output)
    """
    python utils/dataset_euler2rot6d.py --input "F:\projects\lumos\data\20260121_dp\20260121_all\episode" --output "F:\projects\lumos\data\20260121_dp\20260121_all_rot\episode"
    """