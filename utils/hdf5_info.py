import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 配置 =================
# 这里填入你刚才生成的 .hdf5 文件路径
FILE_PATH = r'F:\projects\lumos\data\20260109\episode_0.hdf5'


# =======================================

def print_structure(name, obj):
    """
    回调函数：用于递归打印 HDF5 的层级结构
    """
    # 计算层级缩进
    level = name.count('/')
    indent = '  ' * level

    if isinstance(obj, h5py.Group):
        print(f"{indent}📂 [Group]   {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}📄 [Dataset] {obj.name.split('/')[-1]} | Shape: {obj.shape} | Type: {obj.dtype}")


def analyze_numeric_data(dataset_name, data):
    """
    分析数值数据的统计信息 (Min, Max, Mean)
    """
    print(f"\n--- 分析数据: {dataset_name} ---")
    print(f"  Shape: {data.shape}")
    print(f"  Type:  {data.dtype}")

    # 如果是数值型数据，打印统计信息
    if np.issubdtype(data.dtype, np.number):
        print(f"  Min:   {np.min(data):.4f}")
        print(f"  Max:   {np.max(data):.4f}")
        print(f"  Mean:  {np.mean(data):.4f}")

        # 打印前 2 行数据示例
        if len(data) > 0:
            print(f"  Sample (First 2 rows):\n{data[:2]}")
    else:
        print("  (非数值数据，跳过统计)")


def show_image_sample(data, title="Image Sample"):
    """
    显示图像数据的第一帧
    """
    # ACT 数据集图像通常格式: (Time, Height, Width, Channel) 或 (Time, Channel, H, W)
    if data.ndim == 4:
        # 取第一帧
        img = data[0]

        # 检查是否需要转置: 如果 Channel 在前 (3, H, W) -> 转成 (H, W, 3)
        if img.shape[0] == 3 and img.shape[2] != 3:
            img = np.transpose(img, (1, 2, 0))

        plt.figure(figsize=(6, 4))
        plt.imshow(img.astype(np.uint8))  # 确保是 uint8 显示
        plt.title(f"{title} (Frame 0)")
        plt.axis('off')
        plt.show()
        print(f"  >>> 已显示图像预览: {title}")
    else:
        print(f"  (维度 {data.shape} 不像常规图像，跳过显示)")


def main():
    if not os.path.exists(FILE_PATH):
        print(f"错误: 找不到文件 {FILE_PATH}")
        return

    print(f"正在打开文件: {FILE_PATH} ...\n")

    with h5py.File(FILE_PATH, 'r') as f:
        # 1. 打印整体结构树
        print("=" * 40)
        print("Dataset Structure Tree:")
        print("=" * 40)
        f.visititems(print_structure)
        print("=" * 40 + "\n")

        # 2. 智能详细分析 (针对 ACT/Robotics 数据格式)
        # 分析 Action (动作指令)
        if 'action' in f:
            analyze_numeric_data('/action', f['action'][:])

        # 分析 Qpos (关节观测)
        if 'observations/qpos' in f:
            analyze_numeric_data('/observations/qpos', f['observations/qpos'][:])

        # 分析图像 (如果有)
        # 自动搜索 observations/images 下的所有数据集
        if 'observations' in f and 'images' in f['observations']:
            img_group = f['observations/images']
            for cam_name in img_group.keys():
                print(f"\n--- 检测到图像数据: {cam_name} ---")
                img_data = img_group[cam_name][:]
                print(f"  Shape: {img_data.shape}")

                # 尝试显示第一帧
                show_image_sample(img_data, title=cam_name)


if __name__ == "__main__":
    main()