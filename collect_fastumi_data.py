#!/usr/bin/env python3
import datetime
import os
import subprocess
import threading
import time

import cv2
import h5py
import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from xv_sdk.msg import Clamp

from utils import transform_vive_to_gripper

RECORD_FPS = 30


def speak_feedback(text):
    """简单的语音反馈"""
    try:
        subprocess.Popen(['espeak', '-v', 'zh', '-s', '160', text],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    except Exception:
        pass


class FastumiDataCollector:
    def __init__(self,
                 pose_topic,
                 clamp_topic,
                 camera_topic,
                 save_root="./data_fastumi"):

        self.pose_topic = pose_topic
        self.clamp_topic = clamp_topic
        self.camera_topic = camera_topic

        # 状态变量
        self.current_pose = None  # [x, y, z, qx, qy, qz, qw] (经过 vive_to_gripper 修正)
        self.current_clamp = 0.0
        self.current_image = None

        # 线程锁
        self.pose_lock = threading.Lock()
        self.clamp_lock = threading.Lock()
        self.image_lock = threading.Lock()
        self.data_lock = threading.Lock()

        # 线程控制
        self.shutdown_flag = False
        self.pose_thread = None
        self.clamp_thread = None
        self.camera_thread = None
        self.record_thread = None

        # ROS 工具
        self.bridge = CvBridge()

        # 录制状态
        self.is_recording = False
        self.episode_idx = 0
        self.last_obs = None  # 用于存储上一帧 (image, state)，实现 action=next_state 逻辑

        self.recorded_data = {
            "qpos": [],
            "action": [],
            "images": []
        }

        # 文件保存路径初始化
        current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_root, current_time_str)
        self.episode_dir = os.path.join(self.save_dir, "episode")
        self.video_dir = os.path.join(self.save_dir, "video")

        for d in [self.episode_dir, self.video_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

        rospy.loginfo(f"Data Collector Initialized. Saving to: {self.save_dir}")

    # ================= 监听线程 (使用 wait_for_message) =================

    def _pose_listener(self):
        """循环等待 Pose 消息"""
        rospy.loginfo("Pose listener thread started.")
        while not self.shutdown_flag and not rospy.is_shutdown():
            try:
                # 阻塞等待消息，超时 0.1s 以便检查 shutdown 标志
                msg = rospy.wait_for_message(self.pose_topic, PoseStamped, timeout=0.1)

                # 提取原始数据
                raw_qpos = np.array([
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w
                ])

                # 只进行手柄自身的偏置修正 (vive tracker -> gripper center)
                # 不进行 transform_to_base_quat
                final_pose = transform_vive_to_gripper(raw_qpos)

                # 注意：transform_vive_to_gripper 返回的是 (x,y,z,qx,qy,qz,qw) 还是 (x,y,z,roll,pitch,yaw)?
                # 假设 utils 里的这个函数返回的是 7维 (pos+quat) 或者 6维。
                # 如果 utils 返回的是 pose+euler，这里可能需要转回 quat 以保持数据统一性。
                # 通常 vive_to_gripper 是刚体变换，这里假设它返回的是 np.array 7维。
                # 如果它是 6 维 (x,y,z,r,p,y)，请自行决定是否转回四元数。
                # 这里假设它返回的是符合您要求的格式。

                with self.pose_lock:
                    self.current_pose = final_pose

            except rospy.ROSException:
                # 超时，继续循环
                pass
            except Exception as e:
                print(f"Pose Error: {e}")

    def _clamp_listener(self):
        """循环等待 Clamp 消息"""
        rospy.loginfo("Clamp listener thread started.")
        while not self.shutdown_flag and not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message(self.clamp_topic, Clamp, timeout=0.1)
                with self.clamp_lock:
                    self.current_clamp = msg.data
            except rospy.ROSException:
                pass
            except Exception as e:
                print(f"Clamp Error: {e}")

    def _camera_listener(self):
        """循环等待 Image 消息 (ROS Topic)"""
        rospy.loginfo("Camera listener thread started.")
        while not self.shutdown_flag and not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message(self.camera_topic, Image, timeout=0.1)
                try:
                    # 转换图像
                    cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                    with self.image_lock:
                        self.current_image = cv_image
                except CvBridgeError as e:
                    print(f"CvBridge Error: {e}")
            except rospy.ROSException:
                pass
            except Exception as e:
                print(f"Camera Error: {e}")

    # ================= 录制循环 (30Hz) =================

    def _record_loop(self):
        """以固定频率采集数据"""
        rate = rospy.Rate(RECORD_FPS)
        rospy.loginfo(f"Recording loop started at {RECORD_FPS} Hz")

        while not self.shutdown_flag and not rospy.is_shutdown():
            if self.is_recording:
                # 1. 获取当前瞬时数据
                img_snap = None
                with self.image_lock:
                    if self.current_image is not None:
                        img_snap = self.current_image.copy()

                pose_snap = None
                with self.pose_lock:
                    if self.current_pose is not None:
                        pose_snap = self.current_pose.copy()  # 假设是 numpy array

                clamp_snap = None
                with self.clamp_lock:
                    clamp_snap = self.current_clamp

                # 2. 只有当数据都有效时才处理
                if img_snap is not None and pose_snap is not None:
                    # 拼接 State: [Pose(7) + Clamp(1)] = 8维
                    # 如果 pose_snap 是 (x,y,z,r,p,y)，则为 7维。请确保维度符合您的预期。
                    # 这里假设 pose_snap 是 7维 (x,y,z,qx,qy,qz,qw)
                    current_state = np.concatenate([pose_snap, [clamp_snap]])

                    # 3. 执行 "Next State as Action" 逻辑
                    if self.last_obs is not None:
                        last_img, last_state = self.last_obs
                        with self.data_lock:
                            self.recorded_data["images"].append(last_img)
                            self.recorded_data["qpos"].append(last_state)  # t 时刻的状态
                            self.recorded_data["action"].append(current_state)  # t+1 时刻的状态 (即当前的 state)

                    # 更新上一帧
                    self.last_obs = (img_snap, current_state)

            rate.sleep()

    # ================= 控制接口 =================

    def start(self):
        self.shutdown_flag = False

        # 启动所有监听线程
        self.pose_thread = threading.Thread(target=self._pose_listener, daemon=True)
        self.clamp_thread = threading.Thread(target=self._clamp_listener, daemon=True)
        self.camera_thread = threading.Thread(target=self._camera_listener, daemon=True)
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)

        self.pose_thread.start()
        self.clamp_thread.start()
        self.camera_thread.start()
        self.record_thread.start()

    def stop(self):
        self.shutdown_flag = True
        rospy.loginfo("Stopping threads...")

    def start_recording(self):
        rospy.loginfo(f"Episode {self.episode_idx}: Recording Started...")
        with self.data_lock:
            # 清空上一集数据
            self.recorded_data = {"qpos": [], "action": [], "images": []}
        self.last_obs = None
        self.is_recording = True

    def stop_recording_and_save(self):
        self.is_recording = False
        rospy.loginfo("Recording stopped. Saving...")
        time.sleep(0.5)  # 等待最后的写入
        self.save_episode()

    def abort_recording(self):
        self.is_recording = False
        with self.data_lock:
            self.recorded_data = {"qpos": [], "action": [], "images": []}
        self.last_obs = None
        rospy.logwarn("Recording Aborted.")

    def save_episode(self):
        with self.data_lock:
            qpos_data = np.array(self.recorded_data["qpos"])
            action_data = np.array(self.recorded_data["action"])
            image_data = self.recorded_data["images"]

        data_len = len(qpos_data)
        if data_len == 0:
            rospy.logwarn("No data recorded, skipping save.")
            return

        filename = os.path.join(self.episode_dir, f"episode_{self.episode_idx}.hdf5")
        rospy.loginfo(f"Saving {data_len} steps to {filename}...")

        # HDF5 保存
        with h5py.File(filename, 'w') as f:
            obs_grp = f.create_group('observations')
            obs_grp.create_dataset('qpos', data=qpos_data)
            # obs_grp.create_dataset('images', data=np.array(image_data)) # 如果需要存图进 h5，取消注释
            f.create_dataset('action', data=action_data)

            # 元数据
            f.attrs['robot'] = "fastumi_gripper_only"

        # MP4 保存 (用于预览)
        if len(image_data) > 0:
            video_path = os.path.join(self.video_dir, f"episode_{self.episode_idx}.mp4")
            height, width, _ = image_data[0].shape
            # MP4V 编码
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, float(RECORD_FPS), (width, height))
            for img in image_data:
                out.write(img)
            out.release()
            rospy.loginfo(f"Video saved to {video_path}")

        self.episode_idx += 1
        rospy.loginfo("Save Complete.")


# ================= 主函数 =================
if __name__ == "__main__":
    rospy.init_node('fastumi_recorder', anonymous=True)

    pose_topic = "/vive/LHR_3BEB0779/pose"
    clamp_topic = "/xv_sdk/250801DR48FP25002689/clamp/Data"
    camera_topic = "/camera/fisheye/image_raw"

    print("========================================")
    print(f"Pose Topic:   {pose_topic}")
    print(f"Clamp Topic:  {clamp_topic}")
    print(f"Camera Topic: {camera_topic}")
    print("========================================")

    collector = FastumiDataCollector(pose_topic, clamp_topic, camera_topic)

    # 启动线程
    collector.start()

    print("========================================")
    print("Ready to Record.")
    print(" [s] Start")
    print(" [e] End & Save")
    print(" [x] Abort")
    print(" [q] Quit")
    print("========================================")

    try:
        while not rospy.is_shutdown():
            cmd = input().strip().lower()

            if cmd == 's':
                if not collector.is_recording:
                    collector.start_recording()
                    speak_feedback("Start")
                else:
                    print("Already recording!")

            elif cmd == 'e':
                if collector.is_recording:
                    speak_feedback("Saving")
                    collector.stop_recording_and_save()
                    speak_feedback("Done")
                else:
                    print("Not recording.")

            elif cmd == 'x':
                if collector.is_recording:
                    collector.abort_recording()
                    speak_feedback("Aborted")
                else:
                    print("Not recording.")

            elif cmd == 'q':
                print("Quitting...")
                speak_feedback("Bye")
                break
    except KeyboardInterrupt:
        pass
    finally:
        collector.stop()
        print("Shutdown.")