import datetime
from collections import deque

import cv2
import rospy
from geometry_msgs.msg import PoseStamped
from xv_sdk.msg import Clamp
import os
import threading
from scipy.spatial.transform import Rotation as R
import numpy as np
import time
import sys
from utils import VIVE2VIVE_FLAT, transform_vive_to_gripper
import h5py
from datetime import datetime

TRANSFORMATION = np.eye(4)
TRANSFORMATION[:3, :3] = R.from_euler('xyz', [30, 0, 0], degrees=True).as_matrix()

# 定义数据保存路径
DATA_DIR = "../data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


class RemoteSubv3:
    def __init__(self,
                 robot_sdk,
                 T_base2local,
                 pose_topic="/vive/LHR_339B545E/pose",
                 clamp_topic="/xv_sdk/250801DR48FP25002267/clamp/Data",
                 arm_instance=None,
                 camera_interface=None,
                 save_jpgs=False,
                 save_video=True,
                 speed_tag=0,
                 future_steps=2):
        self._arm_instance = arm_instance
        self._camera = camera_interface
        self.save_jpgs = save_jpgs
        self.save_video = save_video

        self.target_pose = None
        self.target_clamp = None
        self.image = None
        self.T_base2local = T_base2local
        self._robot_sdk = robot_sdk
        self.pose_lock = threading.Lock()
        self.clamp_lock = threading.Lock()
        self.image_lock = threading.Lock()
        self.pose_thread = None
        self.clamp_thread = None
        self.control_thread = None
        self.camera_thread = None
        self.shutdown_flag = False
        self.pose_topic = pose_topic
        self.clamp_topic = clamp_topic

        # Recording Variables
        self.is_recording = False
        self.episode_idx = 0
        self.recorded_data = {
            "qpos": [],
            "action": [],
            "images": []
        }
        self.data_lock = threading.Lock()
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join("./data", current_time_str)
        self.episode_dir = os.path.join(self.save_dir, "episode")
        self.img_save_dir = os.path.join(self.save_dir, "img")
        self.video_save_dir = os.path.join(self.save_dir, "video")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.img_save_dir):
            os.makedirs(self.img_save_dir)
        if not os.path.exists(self.video_save_dir):
            os.makedirs(self.video_save_dir)

        self.speed_tag = speed_tag
        rospy.loginfo(f"Current Recording Speed Mode: [{self.speed_tag}]")

        rospy.loginfo(f"Data will be saved to: {self.save_dir}")
        rospy.loginfo(f"Images will be saved to: {self.img_save_dir}")

        self.future_steps = future_steps
        self.obs_buffer = deque(maxlen=self.future_steps + 1)

    def _pose_listener(self):
        """Thread to listen for Vive pose messages."""
        rospy.loginfo("Pose listener thread started.")
        while not self.shutdown_flag and not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message(
                    self.pose_topic, PoseStamped, timeout=0.1
                )
                # print(msg)
                # with self.pose_lock:
                cur_qpos = np.array([
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w
                ])
                tmp_pose = transform_vive_to_gripper(cur_qpos)
                self.target_pose = self.transform_to_base_quat(*tmp_pose, self.T_base2local)
            except rospy.ROSException:
                # Timeout or error — just retry
                print("pose loss")
                pass

    def _clamp_listener(self):
        rospy.logdebug("Clamp listener started.")
        while not self.shutdown_flag and not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message(self.clamp_topic, Clamp, timeout=0.1)
                with self.clamp_lock:
                    self.target_clamp = msg.data
                    # print(self.target_clamp)
            except rospy.ROSException:
                print("clamp loss")
                continue

    def transform_to_base_quat(self, x, y, z, qx, qy, qz, qw, T_base_to_local):
        '''transform the pose of fastumi to robot base'''
        rotation_local = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T_local = np.eye(4)
        T_local[:3, :3] = rotation_local
        T_local[:3, 3] = [x, y, z]

        T_base_r = np.matmul(T_local[:3, :3], T_base_to_local[:3, :3])

        x_base, y_base, z_base = T_base_to_local[:3, 3] + T_local[:3, 3]
        rotation_base = R.from_matrix(T_base_r)
        roll_base, pitch_base, yaw_base = rotation_base.as_euler('xyz', degrees=False)
        return x_base, y_base, z_base, roll_base, pitch_base, yaw_base

    def _camera_loop(self):
        """独立线程：全速获取图像，存入缓冲区"""
        rospy.loginfo("Camera loop started.")
        # 这里的频率取决于相机性能，不需要人为 sleep 太多，保证最新即可
        # 如果相机是 30fps，这里自然就是 30fps
        while not self.shutdown_flag and not rospy.is_shutdown():
            try:
                frame = self._camera.get_frame()
                if frame is not None:
                    with self.image_lock:
                        self.image = frame
            except Exception as e:
                print(f"Camera error: {e}")
                time.sleep(0.1)

    def _control_loop(self):
        """This replaces your original `run()` — runs at 100 Hz."""
        rospy.loginfo("Control loop started at 100 Hz.")
        rate = rospy.Rate(40)
        print(not self.shutdown_flag)
        print(not rospy.is_shutdown())
        while not self.shutdown_flag and not rospy.is_shutdown():
            if self.target_pose is not None and self.target_clamp is not None:
                self._robot_sdk(self.target_pose, self.target_clamp, speed=100)
            rate.sleep()
        rospy.loginfo("Control loop exited.")

    def _write_loop(self):
        """录制线程：同步抓取 Image 和 EEF Pose"""
        rospy.loginfo(f"Write loop started at 30Hz. Delay shift: {self.obs_delay} steps.")
        rate = rospy.Rate(30)  # 30Hz 采样率

        while not self.shutdown_flag and not rospy.is_shutdown():
            if self.is_recording:
                # A. 获取图像
                current_img = None
                with self.image_lock:
                    if self.image is not None:
                        current_img = self.image.copy()

                # B. [核心修改] 获取当前 EEF 位姿作为“状态” (Observation)
                # 输入变成了 EEF，所以这里不再读关节角，而是读末端
                current_eef_state = []
                try:
                    # 调用 startouchclass.py 的接口
                    # 返回: pos(3), quat(4)
                    pos, quat = self._arm_instance.get_ee_pose_quat()
                    gripper = self._arm_instance.get_gripper_position()

                    # 拼接成 8 维向量: [x, y, z, qx, qy, qz, qw, gripper]
                    current_eef_state = np.concatenate([pos, quat, [gripper]])
                except Exception as e:
                    print(f"Error getting EEF state: {e}")

                # -------------------------------------------------------
                # 2. 存入 Buffer 与 Time Shift 处理
                # -------------------------------------------------------

                # 只有数据有效才处理
                if current_img is not None and len(current_eef_state) > 0:

                    # === 存入 Buffer 的是当前的观测 (Input) ===
                    # 注意：虽然变量名还叫 qpos (为了兼容 HDF5 结构)，但内容已经是 EEF 了
                    self.obs_buffer.append({
                        "image": current_img,
                        "qpos": current_eef_state
                    })

                    # === 生成训练对 (Input_Old, Output_Now) ===
                    # 如果 Buffer 满了，说明有了足够的时间延迟
                    if len(self.obs_buffer) > self.obs_delay:
                        # 1. 取出 T-2 时刻的观测 (作为模型的 Input)
                        past_obs = self.obs_buffer.popleft()

                        # 2. 取出 T 时刻的 EEF 状态 (作为模型的 Target Action)
                        # 因为是模仿学习，机器人当前的真实状态就是它 T-2 时刻应该预测到的“未来”
                        action_to_save = current_eef_state

                        with self.data_lock:
                            self.recorded_data["images"].append(past_obs["image"])
                            self.recorded_data["qpos"].append(past_obs["qpos"])  # 存入的是过去的 EEF
                            self.recorded_data["action"].append(action_to_save)  # 存入的是当前的 EEF

            rate.sleep()


    def start_recording(self):
        rospy.loginfo("Starting Recording...")
        with self.data_lock:
            self.recorded_data = {"qpos": [], "action": [], "images": []}
        self.is_recording = True

    def stop_recording_and_save(self):
        rospy.loginfo("Stopping Recording...")
        self.is_recording = False
        time.sleep(0.1)  # 等待最后的写入完成
        self.save_episode()

    def save_episode(self):
        data_len = len(self.recorded_data["qpos"])
        if data_len == 0:
            rospy.logwarn("No data recorded, skipping save.")
            return

        filename = os.path.join(self.episode_dir, f"episode_{self.episode_idx}.hdf5")
        rospy.loginfo(f"Saving {data_len} steps to {filename}...")

        # ---------- HDF5 ----------
        with h5py.File(filename, 'w') as f:
            obs_grp = f.create_group('observations')
            obs_grp.create_dataset('qpos', data=np.array(self.recorded_data["qpos"]))
            f.create_dataset('action', data=np.array(self.recorded_data["action"]))

        images = self.recorded_data["images"]

        # ---------- JPG ----------
        if self.save_jpgs:
            rospy.loginfo("Saving JPG images...")
            if not os.path.exists(self.img_save_dir):
                os.makedirs(self.img_save_dir)
            for step_idx, img in enumerate(images):  # ← 循环开始
                img_name = f"episode_{self.episode_idx}_step_{step_idx}.jpg"
                img_path = os.path.join(self.img_save_dir, img_name)
                cv2.imwrite(img_path, img)  # ← 循环体内

        # ---------- MP4 ----------
        if self.save_video:
            rospy.loginfo("Saving MP4 video...")
            if not os.path.exists(self.video_save_dir):
                os.makedirs(self.video_save_dir)

            video_name = os.path.join(self.video_save_dir, f"episode_{self.episode_idx}.mp4")
            if images:
                height, width, _ = images[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_name, fourcc, 30.0, (width, height))
                for img in images:  # ← 循环写帧
                    out.write(img)
                out.release()

        rospy.loginfo(f"Episode {self.episode_idx} saved successfully.")
        self.episode_idx += 1


    def start(self):
        """Start all threads."""
        rospy.loginfo("Starting RobotController threads...")
        self.shutdown_flag = False
        # ROS1 call back
        self.pose_thread = threading.Thread(target=self._pose_listener, daemon=True)
        self.clamp_thread = threading.Thread(target=self._clamp_listener, daemon=True)
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.write_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)

        self.pose_thread.start()
        self.clamp_thread.start()
        self.control_thread.start()
        self.write_thread.start()
        self.camera_thread.start()


    def stop(self):
        """Graceful shutdown."""
        rospy.loginfo("Shutting down RobotController...")
        self.shutdown_flag = True
