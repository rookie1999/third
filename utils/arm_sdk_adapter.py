import numpy as np

def rot6d_to_euler(rot6d, seq='xyz', degrees=False):
    """
    将 6D 旋转表示还原为欧拉角。
    包含 Gram-Schmidt 正交化，确保旋转矩阵合法。

    Args:
        rot6d (np.ndarray): 6D 向量。形状可以是 (6,) 或 (N, 6)。
        seq (str): 输出的欧拉角顺序，默认为 'xyz'。
        degrees (bool): 输出是否为角度制。

    Returns:
        np.ndarray: 欧拉角 [roll, pitch, yaw]。
                    输入 (6,) -> 返回 (3,)
                    输入 (N, 6) -> 返回 (N, 3)
    """
    rot6d = np.asarray(rot6d)
    is_single = (rot6d.ndim == 1)

    if is_single:
        rot6d = rot6d[None, :]  # (1, 6)

    # 1. 提取向量
    x_raw = rot6d[:, 0:3]  # 第一列 [r11, r21, r31]
    y_raw = rot6d[:, 3:6]  # 第二列 [r12, r22, r32]

    # 2. Gram-Schmidt 正交化 (关键步骤)
    # 标准化 X
    x_norm = x_raw / (np.linalg.norm(x_raw, axis=1, keepdims=True) + 1e-8)

    # 让 Y 垂直于 X
    dot_prod = np.sum(x_norm * y_raw, axis=1, keepdims=True)
    y_ortho = y_raw - dot_prod * x_norm
    y_norm = y_ortho / (np.linalg.norm(y_ortho, axis=1, keepdims=True) + 1e-8)

    # 叉乘得到 Z
    z_norm = np.cross(x_norm, y_norm)

    # 3. 重组矩阵 (N, 3, 3)
    matrices = np.stack([x_norm, y_norm, z_norm], axis=-1)

    # 4. 转欧拉角
    r = R.from_matrix(matrices)
    eulers = r.as_euler(seq, degrees=degrees)

    if is_single:
        return eulers[0]
    return eulers

# Utility functions to create SDK adapters for different robotic arms
def make_xarm_sdk(arm_instance):
    def xarm_sdk(pose, width, speed=100):
        pose = list(pose)
        pose[:3] = [p * 1000 for p in pose[:3]]
        # xArm unit:mm
        arm_instance.set_servo_cartesian(pose, is_radian=True)

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

def make_startouch_eef_rpy_sdk(arm_instance):
    def startouch_sdk(pose, width, speed=100, is_record=False):
        arm_instance.set_end_effector_pose_euler_raw(np.array(pose[:3]), np.array(pose[3:]))
        width = width / 80 if is_record else width
        ratio = np.clip(width, 0, 1)
        # print(ratio)
        arm_instance.setGripperPosition_raw(ratio)

    return startouch_sdk

def make_startouch_eef_rot_sdk(arm_instance):
    def startouch_sdk(pose, width, speed=100, is_record=False):
        rpy = rot6d_to_euler(pose[3:])
        arm_instance.set_end_effector_pose_euler_raw(np.array(pose[:3]), np.array(rpy))
        width = width / 80 if is_record else width
        ratio = np.clip(width, 0, 1)
        # print(ratio)
        arm_instance.setGripperPosition_raw(ratio)

    return startouch_sdk


def make_startouch_joint_sdk(arm_instance):
    def startouch_sdk(pose, width, speed=100, is_record=False):
        # print(pose, width)
        arm_instance.set_joint_raw(np.array(pose), velocities=[0, 0, 0, 0, 0, 0])
        width = width / 80 if is_record else width
        ratio = np.clip(width, 0, 1)
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


def make_startouch_joint_reader(arm_instance):
    def reader_fn():
        joints = arm_instance.get_joint_positions()
        gripper = arm_instance.get_gripper_position()
        return np.array(list(joints) + [gripper])

    return reader_fn

def make_startouch_ee_rot_reader(arm_instance):
    def reader_fn():
        pos, rot = arm_instance.get_end_effector_pose_rot6d()
        gripper = arm_instance.get_gripper_position()
        return np.concatenate([pos, rot, [gripper]])

    return reader_fn

def make_startouch_ee_rpy_reader(arm_instance):
    def reader_fn():
        pos, rpy = arm_instance.get_ee_pose_euler()
        gripper = arm_instance.get_gripper_position()
        return np.concatenate([pos, rpy, [gripper]])

    return reader_fn