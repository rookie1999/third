import os
import glob
import h5py
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import shutil


def extract_video(src_h5, video_output_dir, file_stem):
    """
    从 HDF5 中提取图片并保存为视频
    返回: 是否成功提取了至少一个视频
    """
    if 'observations/images' not in src_h5:
        return False

    image_group = src_h5['observations/images']
    success = False

    for cam_name in image_group.keys():
        dset = image_group[cam_name]
        num_frames = dset.shape[0]
        if num_frames == 0: continue

        # 获取图像尺寸
        # 假设格式是 (N, H, W, C) 或 (N, C, H, W) -> 需要根据实际情况调整
        # LeRobot 通常是 (N, C, H, W) 或 (N, H, W, C)，这里做个自动判断
        sample_frame = dset[0]
        if sample_frame.shape[0] == 3:  # (C, H, W)
            # 转置为 (H, W, C)
            height, width = sample_frame.shape[1], sample_frame.shape[2]
            is_channel_first = True
        else:  # (H, W, C)
            height, width = sample_frame.shape[0], sample_frame.shape[1]
            is_channel_first = False

        video_name = f"{file_stem}_{cam_name}.mp4"
        video_path = os.path.join(video_output_dir, video_name)

        # 使用 mp4v 编码
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

        for i in range(num_frames):
            img = dset[i]
            if is_channel_first:
                img = np.transpose(img, (1, 2, 0))  # CHW -> HWC

            # RGB -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out.write(img)

        out.release()
        success = True

    return success


def copy_group(src_group, dst_group, exclude_paths=[]):
    """
    递归复制 HDF5 组，跳过 exclude_paths 指定的路径
    """
    # 1. 复制属性 (Attributes)
    for name, value in src_group.attrs.items():
        dst_group.attrs[name] = value

    # 2. 递归复制数据集和子组
    for name in src_group.keys():
        path = src_group[name].name  # 获取绝对路径例如 /observations/images

        # 检查是否在排除列表中
        should_skip = False
        for exclude in exclude_paths:
            if exclude in path:
                should_skip = True
                break
        if should_skip:
            continue

        item = src_group[name]
        if isinstance(item, h5py.Dataset):
            # 复制数据集
            dst_group.create_dataset(name, data=item[...])
        elif isinstance(item, h5py.Group):
            # 创建新组并递归
            new_sub_group = dst_group.create_group(name)
            copy_group(item, new_sub_group, exclude_paths)


def process_single_file(file_path, video_dst_dir):
    filename = os.path.basename(file_path)
    file_stem = os.path.splitext(filename)[0]
    temp_file_path = file_path + ".temp"

    try:
        with h5py.File(file_path, 'r') as src_f:
            # 1. 先提取视频
            print(f"正在提取视频: {filename} ...")
            has_video = extract_video(src_f, video_dst_dir, file_stem)

            if not has_video:
                print(f"[警告] {filename} 中没有发现图片数据，将跳过瘦身操作。")
                return

            # 2. 创建临时文件进行瘦身复制
            print(f"正在生成瘦身文件: {filename} ...")
            with h5py.File(temp_file_path, 'w') as dst_f:
                # 排除 images 组
                copy_group(src_f, dst_f, exclude_paths=['/observations/images'])

        # 3. 覆盖原文件 (原子操作)
        # 在 Windows 上需要先删除原文件，Linux 可以直接 replace
        if os.path.exists(file_path):
            os.remove(file_path)
        os.rename(temp_file_path, file_path)
        print(f"[成功] 文件已瘦身并覆盖: {file_path}")

    except Exception as e:
        print(f"[错误] 处理 {filename} 失败: {e}")
        # 如果出错，尝试清理临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def main():
    parser = argparse.ArgumentParser(description="HDF5 数据集瘦身工具：提取视频并移除原文件中的图片")
    parser.add_argument("--src_dir", type=str, required=True, help="包含 .hdf5 文件的文件夹路径")
    parser.add_argument("--video_dir", type=str, required=True, help="视频输出保存路径")

    args = parser.parse_args()


    if not os.path.exists(args.video_dir):
        os.makedirs(args.video_dir)

    # 查找所有 hdf5 文件
    files = glob.glob(os.path.join(args.src_dir, "*.hdf5")) + glob.glob(os.path.join(args.src_dir, "*.h5"))
    files.sort()

    print(f"找到 {len(files)} 个文件，准备处理...")
    print(f"注意：这将修改原始文件，请确保重要数据已备份！\n")

    for f in tqdm(files):
        process_single_file(f, args.video_dir)

    print("\n所有任务完成！")


if __name__ == "__main__":
    main()