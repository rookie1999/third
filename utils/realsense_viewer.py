import pyrealsense2 as rs
import numpy as np
import cv2


"""
用来测试intel realsense相机是否正常
"""
def main():
    # 1. 配置 RealSense 管道
    pipeline = rs.pipeline()
    config = rs.config()

    # 获取设备信息（可选，用于确认设备已连接）
    # pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    # pipeline_profile = config.resolve(pipeline_wrapper)
    # device = pipeline_profile.get_device()
    # print(f"Device found: {device.get_info(rs.camera_info.name)}")

    # 2. 配置流
    # 常见的配置：640x480, 30fps
    # format.z16 代表 16位深度数据 (单位通常是毫米)
    # format.bgr8 代表标准的 OpenCV 彩色格式
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 3. 开启管道
    print("正在启动 RealSense 相机...")
    profile = pipeline.start(config)

    # 获取深度传感器的深度标度 (Depth Scale)
    # 不同的相机型号深度标度可能不同，通常是 0.001米 (1毫米)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale is: {depth_scale}")

    # 创建对齐对象 (Alignment)
    # rs.stream.color 表示我们要将深度图对齐到彩色图的视角
    # 这在机器人抓取或视觉处理中非常重要，保证像素点对应
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        print("相机已启动，按 'q' 或 'ESC' 退出程序。")
        while True:
            # 4. 等待并获取一帧数据 (阻塞模式)
            frames = pipeline.wait_for_frames()

            # 5. 将深度帧对齐到彩色帧
            aligned_frames = align.process(frames)

            # 获取对齐后的帧
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # 验证帧是否有效
            if not aligned_depth_frame or not color_frame:
                continue

            # 6. 将图像转换为 Numpy 数组
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 7. 深度图可视化处理
            # 深度图是16位的，直接显示是一片黑。我们需要将其转换为8位彩色图以便人眼观察。
            # alpha=0.03 是缩放比例，根据距离调整，让图像看起来对比度更好
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # 8. 拼接图像 (左边是彩色，右边是深度)
            # 确保两个图像的高度和宽度一致
            images = np.hstack((color_image, depth_colormap))

            # 9. 使用 OpenCV 显示
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)

            # 按下 'q' 或 ESC (ASCII 27) 退出
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break

    finally:
        # 10. 停止管道释放资源
        pipeline.stop()
        cv2.destroyAllWindows()
        print("程序已结束，相机已关闭。")

if __name__ == "__main__":
    main()