import cv2
import os


def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {frame_count}")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 保存图片，文件名建议带上序号，方便和 HDF5 索引对应
        # 例如: frame_0000.jpg, frame_0001.jpg
        cv2.imwrite(os.path.join(output_folder, f"frame_{idx:06d}.jpg"), frame)
        idx += 1

    cap.release()
    print(f"Extracted {idx} frames to {output_folder}")

# 使用方法
# extract_frames("data/episode_0.mp4", "data/episode_0_images")