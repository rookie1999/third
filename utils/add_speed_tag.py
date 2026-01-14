import h5py
import os
import glob
from tqdm import tqdm
import argparse

# ==========================================
# 核心配置区域：在这里定义你的“上帝视角”
# ==========================================
# 格式：(Start_Index, End_Index, "Tag_Name")
# 规则：包含 Start，不包含 End (Python 切片风格 [start, end))
# 比如 (0, 50, "slow") 意味着 episode_0 到 episode_49 都是 slow
TAG_RULES = [
    (0, 50, "slow"),  # 0-49 集：慢速 (Speed 0)
    (50, 100, "fast"),  # 50-99 集：快速 (Speed 1)
    (100, 150, "normal"),  # 100-149 集：正常速度
    # 你可以继续添加更多规则...
]

# 默认标签 (如果某个 ID 不在上述任何范围内)
DEFAULT_TAG = "normal"


def get_tag_by_index(idx):
    for start, end, tag in TAG_RULES:
        if start <= idx < end:
            return tag
    return DEFAULT_TAG


def main():
    parser = argparse.ArgumentParser(description="Manually tag HDF5 files based on Index Ranges")
    # 请修改为你的实际 episode 路径
    parser.add_argument('--dataset_dir', type=str, default=r'F:\projects\lumos\data\20260109\episode')
    args = parser.parse_args()

    # 1. 获取所有 hdf5 文件
    files = glob.glob(os.path.join(args.dataset_dir, '*.hdf5'))
    if not files:
        print(f"❌ No .hdf5 files found in {args.dataset_dir}")
        return

    print(f"📂 Found {len(files)} episodes. Applying manual tags...")

    count_stats = {"slow": 0, "normal": 0, "fast": 0}

    # 2. 遍历并打标
    for file_path in tqdm(files):
        # 从文件名解析 ID: "episode_12.hdf5" -> 12
        try:
            filename = os.path.basename(file_path)
            idx_str = filename.split('_')[-1].split('.')[0]
            idx = int(idx_str)
        except ValueError:
            print(f"⚠️ Skipping weird filename: {filename}")
            continue

        # 获取目标标签
        tag = get_tag_by_index(idx)

        # 写入 HDF5 属性
        with h5py.File(file_path, 'r+') as f:
            # 如果已有标签，这一步会覆盖它
            f.attrs['speed_tag'] = tag

            # 可选：如果你想记录具体的 speed level (0, 1, 2)
            if tag == "slow":
                level = 0
            elif tag == "fast":
                level = 1
            else:
                level = 2
            f.attrs['speed_level'] = level

        # 统计
        if tag not in count_stats: count_stats[tag] = 0
        count_stats[tag] += 1

    print("\n✅ Manual Tagging Done!")
    print("📊 Tag Statistics:")
    for k, v in count_stats.items():
        print(f"  - {k}: {v} episodes")

    # 简单的验证检查
    print("\n🔍 Verification (First 3 files):")
    for file_path in files[:3]:
        with h5py.File(file_path, 'r') as f:
            print(f"  - {os.path.basename(file_path)}: {f.attrs.get('speed_tag', 'None')}")


if __name__ == "__main__":
    main()