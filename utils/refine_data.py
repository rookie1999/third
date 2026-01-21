import cv2
import h5py
import os
import shutil
import re
import numpy as np
from tqdm import tqdm

"""
逐帧播放每一集数据进行清洗，
-- episode
    -- *.hdf5
-- video
    -- *.mp4
"""

def get_start_frame_gui(video_path, filename):
    """
    弹出窗口选择起始帧。
    快捷键:
    - D / Right : 下一帧
    - A / Left  : 上一帧
    - Enter     : 确认当前帧为裁剪点 (进行处理)
    - N         : 不需要改动 (直接原样复制，跳到下一个)
    - S         : 丢弃该数据 (不保存，跳到下一个) [可选]
    - Q         : 退出整个程序
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    window_name = f"Check: {filename}"
    cv2.namedWindow(window_name)

    def on_trackbar(val):
        nonlocal current_frame
        current_frame = val
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if ret:
            # 显示操作提示
            cv2.putText(frame, f"Frame: {current_frame}/{total_frames}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "[Enter] Confirm Trim", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "[N] No Change (Keep Original)", (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "[S] Skip (Discard File)", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow(window_name, frame)

    cv2.createTrackbar("Frame", window_name, 0, total_frames - 1, on_trackbar)
    on_trackbar(0)

    selected_frame = 0
    action_type = 'trim'  # 默认动作

    while True:
        key = cv2.waitKey(10) & 0xFF

        # 左右键微调
        if key == ord('d') or key == 83:
            current_frame = min(total_frames - 1, current_frame + 1)
            cv2.setTrackbarPos("Frame", window_name, current_frame)
        elif key == ord('a') or key == 81:
            current_frame = max(0, current_frame - 1)
            cv2.setTrackbarPos("Frame", window_name, current_frame)

        # [Enter] 确认裁剪
        elif key == 13 or key == 32:
            selected_frame = current_frame
            action_type = 'trim'
            print(f" -> 确认裁剪: 从第 {selected_frame} 帧开始")
            break

        # [N] 不需要改动 (Next / No Change)
        elif key == ord('n'):
            action_type = 'copy'
            print(" -> 不需要改动，原样复制。")
            break

        # [S] 丢弃 (Skip) - 如果你看到坏数据不想保留
        elif key == ord('s'):
            action_type = 'skip'
            print(" -> 丢弃该数据。")
            break

        # [Q] 退出程序
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cap.release()
    cv2.destroyAllWindows()
    return selected_frame, action_type


def trim_dataset(src_h5, dst_h5, src_vid, dst_vid, start_idx):
    """具体的裁剪保存逻辑"""
    # 1. HDF5
    try:
        with h5py.File(src_h5, 'r') as f_src, h5py.File(dst_h5, 'w') as f_dst:
            def visit_func(name, node):
                if isinstance(node, h5py.Dataset):
                    data = node[:]
                    if len(data) > start_idx:
                        f_dst.create_dataset(name, data=data[start_idx:])
                    else:
                        f_dst.create_dataset(name, data=data)
                else:
                    f_dst.create_group(name)

            f_src.visititems(visit_func)
            for key, val in f_src.attrs.items():
                f_dst.attrs[key] = val
    except Exception as e:
        print(f"HDF5 Error: {e}")

    # 2. Video (重新编码)
    cap = cv2.VideoCapture(src_vid)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(dst_vid, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    while True:
        ret, frame = cap.read()
        if not ret: break
        out.write(frame)
    cap.release()
    out.release()


# ================= 2. 主逻辑 =================

def clean_single_folder(data_root, output_root):
    src_ep_dir = os.path.join(data_root, 'episode')
    src_vid_dir = os.path.join(data_root, 'video')
    dst_ep_dir = os.path.join(output_root, 'episode')
    dst_vid_dir = os.path.join(output_root, 'video')

    os.makedirs(dst_ep_dir, exist_ok=True)
    os.makedirs(dst_vid_dir, exist_ok=True)

    files = [f for f in os.listdir(src_ep_dir) if f.endswith('.hdf5')]
    files.sort(key=lambda f: int(re.search(r'(\d+)', f).group(1)) if re.search(r'(\d+)', f) else 0)

    print(f"找到 {len(files)} 个文件。")
    print("快捷键: [Enter]裁剪, [N]原样保留, [S]丢弃, [Q]退出")

    # 用于新文件夹的计数，防止跳过文件后序号断裂
    new_index_counter = 0

    for f in tqdm(files, desc="Processing"):
        # 1. 解析路径
        match = re.match(r'episode_(\d+)\.hdf5', f)
        if not match: continue

        old_idx = match.group(1)
        video_name = f"video_{old_idx}.mp4"

        src_h5_path = os.path.join(src_ep_dir, f)
        src_vid_path = os.path.join(src_vid_dir, video_name)

        if not os.path.exists(src_vid_path): continue

        # 2. GUI 交互
        start_idx, action = get_start_frame_gui(src_vid_path, f)

        # 3. 准备新文件名 (保持序号连续)
        if action == 'skip':
            # 如果选择了跳过/丢弃，就不进行保存，也不增加计数器
            continue

        new_ep_name = f"episode_{new_index_counter}.hdf5"
        new_vid_name = f"video_{new_index_counter}.mp4"

        dst_h5_path = os.path.join(dst_ep_dir, new_ep_name)
        dst_vid_path = os.path.join(dst_vid_dir, new_vid_name)

        # 4. 根据动作执行处理
        if action == 'copy':
            # 直接文件复制 (速度极快)
            shutil.copy2(src_h5_path, dst_h5_path)
            shutil.copy2(src_vid_path, dst_vid_path)

        elif action == 'trim':
            # 如果用户选择了第0帧，本质上也是不改动，优化为直接复制
            if start_idx == 0:
                shutil.copy2(src_h5_path, dst_h5_path)
                shutil.copy2(src_vid_path, dst_vid_path)
            else:
                # 进行裁剪处理
                trim_dataset(src_h5_path, dst_h5_path, src_vid_path, dst_vid_path, start_idx)

        # 处理成功，计数器+1
        new_index_counter += 1

    print(f"\n全部完成！共保存 {new_index_counter} 组数据。")


# ================= 3. 运行 =================
input_folder = r'F:\projects\lumos\data\20260120_dynamic_grasp\new'
output_folder = r'F:\projects\lumos\data\20260120_dynamic_grasp\generated'

if __name__ == '__main__':
    clean_single_folder(input_folder, output_folder)