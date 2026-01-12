import rospy
from geometry_msgs.msg import PoseStamped
from xv_sdk.msg import Clamp

import threading
from scipy.spatial.transform import Rotation as R
import numpy as np
import time
import sys
from utils import VIVE2VIVE_FLAT,transform_vive_to_gripper
TRANSFORMATION = np.eye(4)
TRANSFORMATION[:3, :3] = R.from_euler('xyz', [30, 0, 0], degrees=True).as_matrix()
class RemoteSub:
    def __init__(self,robot_sdk ,T_base2local,pose_topic="/vive/LHR_339B545E/pose",clamp_topic="/xv_sdk/250801DR48FP25002267/clamp/Data"):

        self.robot=robot_sdk
        self.target_pose = None
        self.target_clamp = None   
        self.T_base2local = T_base2local
        self._robot_sdk = robot_sdk
        self.pose_lock = threading.Lock()
        self.clamp_lock = threading.Lock()
        self.pose_thread = None
        self.clamp_thread = None
        self.control_thread = None
        self.shutdown_flag = False
        self.pose_topic = pose_topic
        self.clamp_topic = clamp_topic
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


        
    def transform_to_base_quat(self,x, y, z, qx, qy, qz, qw, T_base_to_local):
        '''transform the pose of fastumi to robot base'''
        rotation_local = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T_local = np.eye(4)
        T_local[:3, :3] = rotation_local
        T_local[:3, 3] = [x, y, z]
        
        T_base_r = np.matmul(T_local[:3, :3] , T_base_to_local[:3, :3] )
        1
        x_base, y_base, z_base = T_base_to_local[:3, 3] + T_local[:3, 3]
        rotation_base = R.from_matrix(T_base_r)
        roll_base, pitch_base, yaw_base = rotation_base.as_euler('xyz', degrees=False)
        return x_base, y_base, z_base,  roll_base, pitch_base, yaw_base

    def _control_loop(self):
        """This replaces your original `run()` — runs at 100 Hz."""
        rospy.loginfo("Control loop started at 100 Hz.")
        rate = rospy.Rate(40)
        print(not self.shutdown_flag)
        print(not rospy.is_shutdown())
        while not self.shutdown_flag and not rospy.is_shutdown(): 
            # print(self.target_pose)
            if self.target_pose is not None and self.target_clamp is not None:
                self._robot_sdk(self.target_pose, self.target_clamp, speed=100)

            rate.sleep()

        rospy.loginfo("Control loop exited.")

    def start(self):
        """Start all threads."""
        rospy.loginfo("Starting RobotController threads...")
        self.shutdown_flag = False
        # ROS1 call back
        self.pose_thread = threading.Thread(target=self._pose_listener, daemon=True)
        self.clamp_thread = threading.Thread(target=self._clamp_listener, daemon=True)
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)

        self.pose_thread.start()
        self.clamp_thread.start()
        self.control_thread.start()

    def stop(self):
        """Graceful shutdown."""
        rospy.loginfo("Shutting down RobotController...")
        self.shutdown_flag = True
