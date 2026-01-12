# from piper_replay_sdk.piper_robot import PiperRobot
# import time
# piper = PiperRobot()
# piper.enable()

# while(1):
#     pose = piper.get_end_pose()
#     print(pose)

#!/usr/bin/env python3from scipy.

#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from xv_sdk.msg import Clamp
import config as config
from piper_robot import PiperRobot
from scipy.spatial.transform import Rotation as R
import numpy as np
import time
import sys
sys.path.append("/home/lumos/data_collection_11.14/data_collector_opt")
from pose_merge import VIVE2VIVE_FLAT,transform_vive_to_gripper
TRANSFORMATION = np.eye(4)
TRANSFORMATION[:3, :3] = R.from_euler('xyz', [30, 0, 0], degrees=True).as_matrix()
class PiperSubscriber:
    def __init__(self,robot: PiperRobot ):
        # Store latest messages (optional, for async use)
        self.latest_pose = None
        self.latest_gripper_width = None
        self.robot=robot
        self.target_pose = None
        self.target_clamp = None   
        # Subscribe to pose topic
        self.pose_sub = rospy.Subscriber(
            "/vive/LHR_339B545E/pose",          # topic name
            PoseStamped,          # message type
            self.pose_callback    # callback
        )

        # Subscribe to gripper width topic
        self.gripper_sub = rospy.Subscriber(
            "/xv_sdk/250801DR48FP25002267/clamp/Data",          # topic name
            Clamp,                   # message type
            self.gripper_callback      # callback
        )


    def pose_callback(self, msg):
        self.latest_pose = msg
        
        np.set_printoptions(precision=2, suppress=True)
        print("raw Position: x={:.2f}, y={:.2f}, z={:.2f}".format(msg.pose.position.x,msg.pose.position.y,msg.pose.position.z))
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
        self.target_pose = self.transform_to_base_quat(*tmp_pose, config.T_base2local2)

        print("target Position: x={:.2f}, y={:.2f}, z={:.2f}".format(self.target_pose[0], self.target_pose[1], self.target_pose[2]))
        print("="*30)
        # print(self.target_pose)
        # rospy.loginfo(msg.pose)

    def gripper_callback(self, msg):
        self.target_clamp = np.clip(msg.data, 0, 88)
        # print(self.target_clamp)
        
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

    def run(self):
        """Keep node alive."""
        rate = rospy.Rate(500)  # 100 Hz → sleep 0.01 sec per loop

        rospy.loginfo("Starting 100 Hz timer loop...")
        count = 0
        while not rospy.is_shutdown():
            # 🔁 Your 100 Hz task goes here
            # print(self.target_pose)
           
            if self.target_clamp is not None:
                self.robot.move_gripper(self.target_clamp)
                # self.robot.move_gripper(0)
            if self.target_pose is not None:    
                # print("x={:.2f}, y={:.2f}, z={:.2f}, rx={:.2f}, ry={:.2f}, rz={:.2f}".format(self.target_pose[0], self.target_pose[1], self.target_pose[2], self.target_pose[3], self.target_pose[4], self.target_pose[5]))
                self.robot.movp(self.target_pose,unit="mrad", speed=100)
            
            rate.sleep()
    # rospy.spin()

def main():
    try:
        rospy.init_node('piper_subscriber', anonymous=True)
        piper = PiperRobot("can0")
        piper.enable()
        node = PiperSubscriber(piper)
        time.sleep(2)
        node.run()
    finally:
        pass
        piper.go_zero(speed=100)
        # piper.disable()
if __name__ == '__main__':
    main()
        