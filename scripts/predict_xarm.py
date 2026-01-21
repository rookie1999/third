import pickle
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from xarm.wrapper import XArmAPI

from policy.act.policy import ACTPolicy

# 导入你的模型定义
# 注意：确保这一行能正确导入 policy.py，如果在 src 下运行可能需要 src.policy.policy

# ================= 配置部分 =================
# 硬件配置
MASTER_IP = '192.168.1.xxx'  # 你的主臂 IP (虽然推理时主臂不动，但为了安全可以连上或者只连从臂)
SLAVE_IP = '192.168.1.xxx'  # 你的从臂 IP
FREQUENCY = 50  # 控制频率
DT = 1.0 / FREQUENCY

# 模型配置
CKPT_PATH = './checkpoints/policy_best.ckpt'  # 训练好的权重文件
STATS_PATH = './checkpoints/dataset_stats.pkl'  # 统计量文件
CHUNK_SIZE = 50  # 模型预测步数 (K)
EXECUTION_HORIZON = 20  # 每次执行步数 (M), 必须 <= K
STATE_DIM = 8  # 7关节 + 1夹爪
CAMERA_NAMES = ['cam_high']

# Robotiq 参数
GRIPPER_OPEN = 255
GRIPPER_CLOSE = 0


# ================= 硬件类 (复用之前的) =================
class RealSensePlayer:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(self.config)
        self.color_image = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update_frames)
        self.thread.daemon = True
        self.thread.start()
        print("[Camera] Warming up...")
        time.sleep(2)

    def update_frames(self):
        while not self.stop_event.is_set():
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue
            img = np.asanyarray(color_frame.get_data())
            with self.lock:
                self.color_image = img

    def get_frame(self):
        with self.lock:
            if self.color_image is None: return None
            return self.color_image.copy()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.pipeline.stop()


def setup_slave(ip):
    print(f"[SLAVE] Connecting to {ip}...")
    arm = XArmAPI(ip)
    arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)

    print(f"[SLAVE] Configuring Robotiq...")
    arm.set_tgpio_modbus_baudrate(115200)
    time.sleep(0.2)
    arm.robotiq_reset()
    arm.robotiq_set_activate(True)
    arm.set_gripper_mode(0)
    arm.set_gripper_speed(5000)

    # 切换到伺服模式 (Mode 1)
    arm.set_mode(1)
    arm.set_state(0)
    time.sleep(1)
    return arm


# ================= 推理核心逻辑 =================
def load_model_and_stats():
    # 1. 加载统计量
    with open(STATS_PATH, 'rb') as f:
        stats = pickle.load(f)

    # 2. 加载模型
    policy_config = {
        'lr': 1e-5,  # 推理时不重要
        'num_queries': CHUNK_SIZE,
        'kl_weight': 10,
        'hidden_dim': 512,
        'dim_feedforward': 3200,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': CAMERA_NAMES,
        'state_dim': STATE_DIM
    }

    policy = ACTPolicy(policy_config)
    # 加载权重
    state_dict = torch.load(CKPT_PATH)
    policy.load_state_dict(state_dict)
    policy.cuda()
    policy.eval()  # 切换到评估模式

    print(f"Model loaded from {CKPT_PATH}")
    return policy, stats


def main():
    # 1. 准备模型
    policy, stats = load_model_and_stats()

    # 辅助函数：归一化与反归一化
    def pre_process(qpos_numpy):
        return (qpos_numpy - stats['qpos_mean']) / stats['qpos_std']

    def post_process(action_numpy):
        return action_numpy * stats['action_std'] + stats['action_mean']

    # 2. 准备硬件
    cam = RealSensePlayer()
    try:
        slave = setup_slave(SLAVE_IP)
    except Exception as e:
        print(f"Robot Error: {e}")
        cam.stop()
        return

    # 3. 推理循环变量
    # 上一次控制夹爪的状态，用于减少通信
    last_gripper_target = -1

    print("\n" + "=" * 50)
    print(" ACT Inference Started (Open Loop) ")
    print(f" Prediction Horizon (K): {CHUNK_SIZE}")
    print(f" Execution Horizon (M):  {EXECUTION_HORIZON}")
    print(" Press [Q] in the window to stop.")
    print("=" * 50 + "\n")

    try:
        while True:
            # === Step 1: 获取观测 (Observation) ===
            img = cam.get_frame()
            if img is None: continue

            # 显示当前画面
            cv2.imshow("ACT Inference", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 获取机械臂状态
            code, s_angles = slave.get_servo_angle(is_radian=True)
            if code != 0:
                print("Read Error!")
                break

            # 获取夹爪状态 (这里我们需要一个变量来维护，或者假设当前就在上次的目标位置)
            # 因为Robotiq读不到，我们假设它处于上一次指令的位置，初始为Open
            current_gripper = last_gripper_target if last_gripper_target != -1 else GRIPPER_OPEN

            # 组装 Observation (8维)
            qpos_numpy = np.array(list(s_angles) + [current_gripper])

            # === Step 2: 预处理 (归一化 + 转 Tensor) ===
            qpos = pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)  # (1, 8)

            # 处理图像: (H, W, C) -> (1, 1, C, H, W)
            # 注意: 必须除以 255.0，因为 transforms.Normalize 期望 0-1
            img_tensor = torch.from_numpy(img).float().cuda() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 480, 640)

            # === Step 3: 模型推理 (Inference) ===
            with torch.inference_mode():
                # forward(qpos, image, actions=None, is_pad=None)
                all_actions = policy(qpos, img_tensor)  # 输出 (1, K, 8)

            # 反归一化
            all_actions = all_actions.squeeze(0).cpu().numpy()
            all_actions = post_process(all_actions)  # (K, 8)

            # === Step 4: 执行动作 (Execution Loop) ===
            # 我们只执行前 M 步 (Open Loop)
            for t in range(EXECUTION_HORIZON):
                loop_start = time.time()

                # 取出第 t 步的动作
                action = all_actions[t]  # 8维
                target_angles = action[:7]
                target_gripper = action[7]  # 0~255 的数值

                # A. 控制关节
                slave.set_servo_angle_j(angles=target_angles, is_radian=True)

                # B. 控制夹爪 (二值化处理)
                # 因为模型输出的是连续值，我们设一个阈值来决定开合
                # 例如：< 128 认为是闭合，> 128 认为是张开
                # 或者更简单的：离0近就是0，离255近就是255
                gripper_cmd = GRIPPER_CLOSE if target_gripper < 128 else GRIPPER_OPEN

                if gripper_cmd != last_gripper_target:
                    slave.robotiq_set_position(int(gripper_cmd), wait=False)
                    last_gripper_target = gripper_cmd

                # C. 频率控制
                elapsed = time.time() - loop_start
                if elapsed < DT:
                    time.sleep(DT - elapsed)

                # 可选：如果在执行过程中想强行打断，可以检查按键
                # if cv2.waitKey(1) & 0xFF == ord('q'): return

    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        slave.set_mode(0)
        slave.disconnect()
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()