#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import time

class FinalFisheyeViewer:
    def __init__(self):
        rospy.init_node('final_fisheye_viewer', anonymous=True)

        # --- 参数配置 ---
        # 这里的 topic 记得核对一下是否正确
        self.topic_name = rospy.get_param("~topic_name", "/xv_sdk/250801DR48FP25002689/color_camera/image")
        self.display_fps = rospy.get_param("~display_fps", 30.0) # 提高一点帧率体验更丝滑
        
        self.min_frame_interval = 1.0 / self.display_fps if self.display_fps > 0 else 0
        self.last_display_time = 0

        # --- 【核心参数：来自你的标定结果】 ---
        # --- 【第二次标定参数：中心点更准】 ---
        self.img_dim = (1280, 1280)
        
        # 内参矩阵 K (注意第一行第二个是 0.408685，虽然很小但我们照填)
        self.K = np.array([[395.262928, 0.408685, 641.067781],
                           [0.000000, 395.611045, 644.648616],
                           [0.000000, 0.000000, 1.000000]])
        
        # 畸变系数 D (直接填入你的4个新参数)
        self.D = np.array([0.046958, 0.130829, -0.259944, 0.163639])

        # --- 初始化工具 ---
        self.bridge = CvBridge()
        self.map1 = None
        self.map2 = None
        
        # 预先计算映射表
        self.init_rectification_map()

        self.image_sub = rospy.Subscriber(self.topic_name, Image, self.callback)
        rospy.loginfo(f"标定参数已加载。正在监听: {self.topic_name}")

    # def init_rectification_map(self):
    #     """
    #     生成去畸变映射表
    #     """
    #     # balance 参数说明：
    #     # 0.0 = 裁剪模式（尽量去黑边，但会损失视野）
    #     # 1.0 = 保留模式（保留所有像素，图像四周会有黑边，这是最稳妥的）
    #     # 你可以尝试修改成 0.5 或 0.8 看看效果
    #     balance = 1.0 
        
    #     try:
    #         # 1. 估算新的相机矩阵
    #         new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    #             self.K, self.D, self.img_dim, np.eye(3), balance=balance
    #         )
            
    #         # 2. 生成映射表
    #         self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
    #             self.K, self.D, np.eye(3), new_K, self.img_dim, cv2.CV_16SC2
    #         )
    #         rospy.loginfo("鱼眼映射表计算成功！")
            
    #     except cv2.error as e:
    #         rospy.logerr(f"OpenCV Error: {e}")
    #         rospy.logerr("请检查你的 D 数组是否为 4 个元素。")
    # ==================== 调试版函数 开始 ====================
    def init_rectification_map(self):
        """
        (调试版) 生成去畸变映射表，并打印详细日志
        """
        balance = 1
        rospy.loginfo("--------------------------------------------------")
        rospy.loginfo("【调试模式】开始计算鱼眼映射表...")
        
        # 1. 打印输入参数状态
        rospy.loginfo(f"输入 K 矩阵形状: {self.K.shape}, 数据类型: {self.K.dtype}")
        # rospy.loginfo(f"K 内容:\n{self.K}") # 如果需要可以取消注释看具体数值
        rospy.loginfo(f"输入 D 数组形状: {self.D.shape}, 数据类型: {self.D.dtype}")
        rospy.loginfo(f"D 内容: {self.D}")
        rospy.loginfo(f"图像尺寸 (W, H): {self.img_dim}")

        # 检查 D 的长度
        if self.D.size != 4:
             rospy.logerr(f"【严重错误】D 数组的长度是 {self.D.size}，但 cv2.fisheye 要求必须是 4！")
             rospy.logerr("请检查你是否正确复制了前4个系数。")
             return

        try:
            # 2. 估算新的相机矩阵
            rospy.loginfo("正在运行: cv2.fisheye.estimateNewCameraMatrixForUndistortRectify ...")
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K, self.D, self.img_dim, np.eye(3), balance=balance, fov_scale=0.5
            )
            rospy.loginfo("新相机矩阵 new_K 计算成功！")

            # 3. 生成映射表
            rospy.loginfo("正在运行: cv2.fisheye.initUndistortRectifyMap ...")
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), new_K, self.img_dim, cv2.CV_16SC2
            )
            rospy.loginfo(f"映射表计算成功！map1 形状: {self.map1.shape}")
            rospy.loginfo("--------------------------------------------------")

        except cv2.error as e:
            rospy.logerr("==================================================")
            rospy.logerr("【OpenCV 报错了！】初始化失败。")
            rospy.logerr(f"错误详情:\n{e}")
            rospy.logerr("==================================================")
            # 将 map 置空，确保主循环知道初始化失败了
            self.map1 = None 
            self.map2 = None

        except Exception as e:
             rospy.logerr(f"【发生了未知错误】: {e}")
    # ==================== 调试版函数 结束 ====================

    def callback(self, data):
        current_time = time.time()
        if (current_time - self.last_display_time) < self.min_frame_interval:
            return

        try:
            # 1. 转为 OpenCV 图像
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # 2. 查表去畸变 (Remap)
            if self.map1 is not None:
                rectified_image = cv2.remap(
                    cv_image, self.map1, self.map2, 
                    interpolation=cv2.INTER_LINEAR, 
                    borderMode=cv2.BORDER_CONSTANT
                )
                
                # 3. 显示效果
                # 为了能在屏幕上放下，我们把两个图拼在一起并缩小一点显示
                combined = np.hstack((cv_image, rectified_image))
                
                # 缩放 50% 显示 (1280太大了，并排就是2560宽)
                scale = 0.5
                display_img = cv2.resize(combined, (0,0), fx=scale, fy=scale)
                
                cv2.imshow("Left: Raw | Right: Undistorted", display_img)
            else:
                cv2.imshow("Raw Only", cv_image)
            
            self.last_display_time = current_time
            cv2.waitKey(1)

        except CvBridgeError as e:
            rospy.logerr(f"Bridge Error: {e}")

    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    node = FinalFisheyeViewer()
    node.run()