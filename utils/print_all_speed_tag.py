import h5py
import os
import glob
import argparse


def main():
    parser = argparse.ArgumentParser(description="批量打印文件夹下所有 HDF5 文件的速度标签")
    parser.add_argument('--dataset_dir', type=str, required=True, help='包含 .hdf5 文件的文件夹路径')
    args = parser.parse_args()

    # 获取所有 hdf5 文件并按文件名排序
    # recursive=True 允许查找子文件夹（视需求而定，这里默认只找当前层级）
    files = glob.glob(os.path.join(args.dataset_dir, '*.hdf5'))

    # 尝试按文件名中的数字排序 (例如 episode_0, episode_1, episode_10)
    try:
        files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    except:
        files.sort()  # 如果文件名格式不统一，则使用默认字符串排序

    if not files:
        print(f"❌ 在 {args.dataset_dir} 下未找到 .hdf5 文件")
        return

    print(f"\n📂 正在检查文件夹: {args.dataset_dir}")
    print(f"共找到 {len(files)} 个文件\n")

    # 打印表头
    header = f"{'Filename':<35} | {'Speed Level (Int)':<18} | {'Speed Tag (Str)':<15}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # 遍历打印
    for file_path in files:
        filename = os.path.basename(file_path)
        try:
            with h5py.File(file_path, 'r') as f:
                # 获取属性，如果没有则显示 '-'
                # 注意：读取出来的可能是 numpy 类型，转为 str 显示更安全
                speed_level = f.attrs.get('speed_level', '-')
                speed_tag = f.attrs.get('speed_tag', '-')

                print(f"{filename:<35} | {str(speed_level):<18} | {str(speed_tag):<15}")
        except Exception as e:
            print(f"{filename:<35} | ❌ 读取错误: {e}")

    print("=" * len(header))


if __name__ == "__main__":
    main()