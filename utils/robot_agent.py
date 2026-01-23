import time


class UniversalRobotAgent:
    def __init__(self, name, read_fn, write_fn, arm, initial_joints):
        self.name = name
        self._read_fn = read_fn
        self._write_fn = write_fn
        self.arm = arm
        self.initial_joints = initial_joints

    def get_qpos(self):
        """通用获取状态接口"""
        return self._read_fn()

    def command_action(self, action_vector):
        """
        通用执行动作接口
        Args:
            action_vector: 模型预测出的原始向量 (例如 7关节 + 1夹爪)
        """
        robot_cmd = action_vector[:-1]
        gripper_cmd = action_vector[-1]
        self._write_fn(robot_cmd, gripper_cmd, speed=20)

    def go_home(self, blocking=True, duration=3.0):
        """
        让机械臂回到初始位置
        :param blocking: 是否阻塞等待
        :param duration: 预估的归位耗时（秒），用于阻塞
        """
        print("🤖 Robot going home...")
        if self.name == "startouch":
            self.arm.set_joint(self.initial_joints)

        if blocking:
            time.sleep(duration)
            print("✅ Robot is at home.")