import h5py
import numpy as np
import os
import argparse
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# ==========================================
# 1. 核心转换逻辑 (Quaternion -> Euler)
# ==========================================

def quat_xyzw_to_euler(quat, seq='xyz'):
    """
    将四元数转换为欧拉角。
    输入: (N, 4) 或 (4,) -> [x, y, z, w]
    输出: (N, 3) 或 (3,) -> [roll, pitch, yaw] (弧度)
    """
    # 确保是 numpy 数组
    quat = np.asarray(quat)

    # 记录是否是单帧数据
    is_single = (quat.ndim == 1)
    if is_single:
        quat = quat[None, :]

    # 使用 Scipy 进行转换
    # Scipy 的 from_quat 默认接受 [x, y, z, w] 顺序，与你的数据一致
    r = R.from_quat(quat)

    # 转换为欧拉角 (seq='xyz' 对应 roll, pitch, yaw)
    euler = r.as_euler(seq, degrees=False)

    if is_single:
        return euler[0]
    return euler


# ==========================================
# 2. 数据处理逻辑 (8维 -> 7维)
# ==========================================

def process_vector_data(data):
    """
    处理单个数据集 (qpos 或 action)
    输入形状: (N, 8) -> [x, y, z, qx, qy, qz, qw, gripper]
    输出形状: (N, 7) -> [x, y, z, roll, pitch, yaw, gripper]
    """
    # 1. 确保数据是 2D 的 (N, D)
    if data.ndim == 1:
        data = data[None, :]
        is_single_frame = True
    else:
        is_single_frame = False

    # 2. 维度检查
    # 你的输入必须是 8 维: Pos(3) + Quat(4) + Gripper(1)
    if data.shape[1] != 8:
        print(f"⚠️ Warning: Data dim is {data.shape[1]}, expected 8 (Pos+Quat+Gripper). Skipping conversion.")
        return data

    # 3. 切片提取
    pos = data[:, :3]  # [x, y, z]
    quat = data[:, 3:7]  # [qx, qy, qz, qw] (注意: 这里的 3:7 对应第4到第7列)
    gripper = data[:, 7:]  # [gripper]

    # 4. 执行转换 (Quat -> Euler)
    # Scipy 内部实现了你提供的数学逻辑，并且带有数值稳定性优化
    euler = quat_xyzw_to_euler(quat, seq='xyz')

    # 5. 拼接
    # (N, 3) + (N, 3) + (N, 1) = (N, 7)
    new_data = np.concatenate([pos, euler, gripper], axis=1)

    # 还原维度 (如果是单帧)
    if is_single_frame:
        return new_data[0]
    return new_data


# ==========================================
# 3. HDF5 文件遍历与复制 (保持不变)
# ==========================================

def copy_and_convert_group(source_group, dest_group):
    """
    递归复制 HDF5 组，如果是 qpos/action 则进行转换
    """
    for key, item in source_group.items():
        if isinstance(item, h5py.Group):
            new_group = dest_group.create_group(key)
            copy_and_convert_group(item, new_group)
        elif isinstance(item, h5py.Dataset):
            # 这里匹配你需要转换的 dataset 名字
            if key in ['qpos', 'action']:
                data = item[:]

                # 调用新的处理函数
                new_data = process_vector_data(data)

                dest_group.create_dataset(key, data=new_data)

                # 复制属性
                for attr_name, attr_value in item.attrs.items():
                    dest_group[key].attrs[attr_name] = attr_value
            else:
                source_group.copy(key, dest_group)


def process_folder(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    files = [f for f in os.listdir(input_dir) if f.endswith('.hdf5')]
    print(f"Found {len(files)} HDF5 files to process.")

    for filename in tqdm(files, desc="Converting Quat->Euler"):
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)

        try:
            with h5py.File(src_path, 'r') as f_src:
                with h5py.File(dst_path, 'w') as f_dst:
                    copy_and_convert_group(f_src, f_dst)
                    for attr_name, attr_value in f_src.attrs.items():
                        f_dst.attrs[attr_name] = attr_value

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print("\n✅ All conversions completed!")
    print(f"Input dir:  {input_dir}")
    print(f"Output dir: {output_dir}")


# ==========================================
# 4. 主程序入口
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert HDF5 qpos/action from Quaternion(8D) to Euler(7D).")
    parser.add_argument('--input', type=str, help='Path to input folder containing .hdf5 files')
    parser.add_argument('--output', type=str, help='Path to output folder for converted files')

    args = parser.parse_args()
    args.input = r"F:\projects\lumos\data\fastumi_0123\episode2"
    args.output = r"F:\projects\lumos\data\fastumi_0123\episode"

    process_folder(args.input, args.output)