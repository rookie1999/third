import h5py
import os
import glob
import argparse
from tqdm import tqdm

"""
python utils/add_speed_tag.py --dataset_dir "F:\projects\lumos\data\20260121_dp\0\episode" --speed_value 0
"""
def main():
    parser = argparse.ArgumentParser(description="Force set a numeric speed tag for ALL files in a folder.")

    # 1. 文件夹路径
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='包含 .hdf5 文件的文件夹路径')

    # 2. 你要指定的数字 (直接是 int 类型)
    parser.add_argument('--speed_value', type=int, required=True,
                        help='你要打进去的数字标签 (例如: 0, 1, 2 ...)')

    args = parser.parse_args()

    # 搜索文件
    files = glob.glob(os.path.join(args.dataset_dir, '*.hdf5'))

    if not files:
        print(f"❌ 在 {args.dataset_dir} 下没找到 .hdf5 文件")
        return

    print(f"📂 找到 {len(files)} 个文件。正在统一写入标签: speed_level = {args.speed_value} ...")

    count = 0
    for file_path in tqdm(files):
        try:
            with h5py.File(file_path, 'r+') as f:
                # 直接写入你指定的整数
                f.attrs['speed_level'] = args.speed_value

                # 为了防止以前的代码报错，可选：顺便把 speed_tag 也写成这个数字的字符串形式 (如 "0")
                # 如果你的 Dataset 只读 speed_level，这行可以删掉
                f.attrs['speed_tag'] = str(args.speed_value)

            count += 1
        except Exception as e:
            print(f"⚠️ 处理出错: {os.path.basename(file_path)} - {e}")

    print(f"\n✅ 完成！已将 {count} 个文件的 speed_level 设为 {args.speed_value}。")

    # 验证第一个文件
    if files:
        with h5py.File(files[0], 'r') as f:
            print(f"\n🔍 验证检查 ({os.path.basename(files[0])}):")
            print(f"   speed_level: {f.attrs.get('speed_level', 'Not Found')}")


if __name__ == "__main__":
    main()