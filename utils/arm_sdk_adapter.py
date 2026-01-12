
import numpy as np
# Utility functions to create SDK adapters for different robotic arms
def make_xarm_sdk(arm_instance):
    def xarm_sdk(pose, width, speed=100):
        pose = list(pose)
        pose[:3] = [p*1000 for p in pose[:3]]
        # xArm unit:mm
        arm_instance.set_servo_cartesian(pose,is_radian=True)
        
        closest_width = int(((88 - width) / 88 * 255))
        closest_width = np.clip(closest_width, 0, 255)
        # arm_instance.robotiq_set_position(closest_width,force=10,speed=255,wait=False)
    return xarm_sdk

def make_piper_sdk(piper_instance):
    def piper_sdk(pose, gripper_cmd, speed=100):
        piper_instance.movp(pose, unit="mrad", speed=60)
        print(gripper_cmd)
        piper_instance.move_gripper(gripper_cmd)
  
    return piper_sdk

def make_startouch_sdk(arm_instance):
    def startouch_sdk(pose, width, speed=100):
        arm_instance.set_end_effector_pose_euler_raw(np.array(pose[:3]), np.array(pose[3:]))
        ratio = np.clip(width/80,0,1)
        # print(ratio)
        arm_instance.setGripperPosition_raw(ratio)
    return startouch_sdk



def make_xarm_reader(arm_instance):
    def reader_fn():
        code, angles = arm_instance.get_servo_angle(is_radian=True)
        if code != 0:
            return None
        code, gripper_pos = arm_instance.get_gripper_position()
        gripper_norm = gripper_pos  # norm if need 0 - 255

        return np.array(list(angles) + [gripper_norm])
    return reader_fn


def make_piper_reader(piper_instance):
    def reader_fn():
        # Piper 获取状态
        # 假设 piper_instance.GetArmStatus() 返回包含关节和夹爪的消息
        msg = piper_instance.GetArmStatus()
        joints = msg.joint_state.position  # 假设是列表
        gripper = msg.gripper_state.position
        return np.array(list(joints) + [gripper])
    return reader_fn


def make_startouch_reader(arm_instance):
    def reader_fn():
        # StarTouch 获取状态
        joints = arm_instance.get_joint()  # 如果模型是用Joint训练
        gripper = arm_instance.get_gripper_position()
        return np.array(list(joints) + [gripper])
    return reader_fn