import pyrealsense2 as rs
import numpy as np
import cv2


class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        # 如果需要深度图，取消下面注释
        # self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)  # 对齐深度到彩色

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    def stop(self):
        self.pipeline.stop()