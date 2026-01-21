import os
import shutil
import re
from tqdm import tqdm  # 这是一个显示进度条的库，如果没有请运行 pip install tqdm，或者删除相关代码


def merge_datasets(source_folders, target_folder):
    """
    将多个源文件夹中的 episode 和 video 数据合并到一个目标文件夹，并重新从0开始编号。
    """

    # 1. 确保目标文件夹结构存在
    target_ep_dir = os.path.join(target_folder, 'episode')
    target_vid_dir = os.path.join(target_folder, 'video')

    os.makedirs(target_ep_dir, exist_ok=True)
    os.makedirs(target_vid_dir, exist_ok=True)

    global_index = 0  # 全局索引，从0开始，一直累加

    print(f"准备开始合并数据...")
    print(f"源文件夹: {source_folders}")
    print(f"目标文件夹: {target_folder}")
    print("-" * 30)

    # 2. 遍历每一个源文件夹
    for src_root in source_folders:
        src_ep_dir = os.path.join(src_root, 'episode')
        src_vid_dir = os.path.join(src_root, 'video')

        # 检查源文件夹是否存在
        if not os.path.exists(src_ep_dir) or not os.path.exists(src_vid_dir):
            print(f"警告: 跳过 {src_root}，因为找不到 episode 或 video 子文件夹。")
            continue

        # 获取该文件夹下所有的 .hdf5 文件
        files = os.listdir(src_ep_dir)
        # 过滤并提取出 index (确保按照数字顺序处理，而不是字符串顺序)
        # 例如: episode_2.hdf5, episode_10.hdf5
        valid_files = []
        for f in files:
            match = re.match(r'episode_(\d+)\.hdf5', f)
            if match:
                idx = int(match.group(1))
                valid_files.append((idx, f))

        # 按旧索引排序，保证处理顺序
        valid_files.sort(key=lambda x: x[0])

        print(f"正在处理文件夹: {src_root} (共 {len(valid_files)} 组数据)")

        # 3. 逐个文件复制并重命名
        for old_idx, ep_filename in tqdm(valid_files, desc="复制进度"):
            # 构建旧的文件路径
            old_ep_path = os.path.join(src_ep_dir, ep_filename)
            old_vid_path = os.path.join(src_vid_dir, f'video_{old_idx}.mp4')

            # 校验：必须成对出现。如果只有 episode 没有 video，则跳过，防止数据损坏
            if not os.path.exists(old_vid_path):
                print(f"\n错误: 找不到对应的视频文件 {old_vid_path}，跳过索引 {old_idx}")
                continue

            # 构建新的文件名 (使用 global_index)
            new_ep_name = f'episode_{global_index}.hdf5'
            new_vid_name = f'video_{global_index}.mp4'

            new_ep_path = os.path.join(target_ep_dir, new_ep_name)
            new_vid_path = os.path.join(target_vid_dir, new_vid_name)

            # 执行复制 (使用 copy2 保留文件元数据如时间戳)
            shutil.copy2(old_ep_path, new_ep_path)
            shutil.copy2(old_vid_path, new_vid_path)

            # 累加全局索引
            global_index += 1

    print("-" * 30)
    print(f"合并完成！总共处理了 {global_index} 组数据 (0 到 {global_index - 1})。")


# ================= 配置区域 =================

# 在这里填入你现在的三个文件夹路径
source_dirs = [
    r'F:\projects\lumos\data\20260120_dynamic_grasp\20260120_125824',  # 例如: D:/data/batch1
    r'F:\projects\lumos\data\20260120_dynamic_grasp\20260120_130325',
    r'F:\projects\lumos\data\20260120_dynamic_grasp\20260120_131342'
]

# 在这里填入你想保存的新路径
output_dir = r'F:\projects\lumos\data\20260120_dynamic_grasp'

# ================= 运行脚本 =================
if __name__ == '__main__':
    # 如果没有安装 tqdm，可以注释掉 import tqdm，并将 for 循环改为：
    # for old_idx, ep_filename in valid_files:
    merge_datasets(source_dirs, output_dir)