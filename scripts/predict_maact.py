import pickle
import sys
import time
import collections  # [新增] 用于管理历史帧
import numpy as np
import cv2
import torch
import yaml
import os

# ==========================================
# 导入 SDK Wrapper (保持原样)
# ==========================================
from xarm.wrapper import XArmAPI
from utils import make_xarm_sdk, make_xarm_reader
from utils.camera import RealSenseCamera
from utils.robot_agent import UniversalRobotAgent

# ==========================================
# [新增] 导入两种策略
# ==========================================
# 假设你的文件结构如下，请根据实际情况调整路径
from policy.act.policy import ACTPolicy  # 旧版 Standard ACT
from common.model.speed_act_modulate_full_model import SpeedACT  # 新版 MA-ACT
from common.configs.configuration_act import SpeedACTConfig  # 新版配置

# ==========================================
# 配置区域
# ==========================================
# [关键开关] True = 推理 MA-ACT; False = 推理 ACT
USE_SPEED_ACT = True

# 机器人配置
CURRENT_ROBOT = 'xarm'
CONFIG_FILE = 'config.yaml'

# 模型与数据路径
# 请确保这两个路径分别对应你训练好的 MA-ACT 和 ACT 权重
CKPT_PATH = './checkpoints/policy_best_ma_act.ckpt' if USE_SPEED_ACT else './checkpoints/policy_best_act.ckpt'
STATS_PATH = './checkpoints/dataset_stats.pkl'
YOLO_CKPT = r"F:\projects\lumos\ma_act\src\object_detection\object_detection_ckpt\yolov8n.pt"  # MA-ACT 必须

# 推理参数
CHUNK_SIZE = 50  # 动作块大小
EXECUTION_HORIZON = 15  # 开环执行步数 (比 Chunk 小)
FREQUENCY = 30  # 控制频率 Hz
DT = 1.0 / FREQUENCY

# MA-ACT 需要至少2帧历史，ACT 只需要1帧
N_OBS_STEPS = 2 if USE_SPEED_ACT else 1
MAIN_CAMERA_NAME = 'cam_high'  # 必须与训练时的名称一致


# ==========================================
# 辅助函数
# ==========================================
def setup_robot(robot_type, config_path):
    """初始化机器人硬件"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if robot_type == 'xarm':
        ip = cfg['Xarm7']['ip']
        arm = make_xarm_sdk(ip)
        reader = make_xarm_reader(ip)
        robot = UniversalRobotAgent(arm, reader)
        return robot
    else:
        raise NotImplementedError(f"Robot {robot_type} not supported yet.")


def main():
    # 1. 加载统计数据
    print(f"Loading stats from {STATS_PATH}...")
    with open(STATS_PATH, 'rb') as f:
        stats = pickle.load(f)

    # 定义预处理和后处理
    def pre_process(qpos):
        return (qpos - stats['qpos_mean']) / stats['qpos_std']

    def post_process(action):
        return action * stats['action_std'] + stats['action_mean']

    # 2. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model (Mode: {'MA-ACT' if USE_SPEED_ACT else 'Standard ACT'})...")

    if USE_SPEED_ACT:
        # --- 初始化 SpeedACT ---
        config = SpeedACTConfig(
            dim_model=512,
            chunk_size=CHUNK_SIZE,
            n_obs_steps=N_OBS_STEPS,
            # 注意：这里的图像尺寸 (480, 640) 必须与 RealSenseCamera 设置一致
            image_features={MAIN_CAMERA_NAME: (3, 480, 640)},
            main_camera=MAIN_CAMERA_NAME,
            robot_state_feature=(14,),
            action_feature=(14,),
            use_optical_flow=True,
            object_detection_ckpt_path=YOLO_CKPT,
            cropped_flow_h=64,
            cropped_flow_w=64,
            feedforward_activation="relu",
            pre_norm=False
        )
        policy = SpeedACT(config)
    else:
        # --- 初始化 Standard ACT ---
        policy = ACTPolicy(
            action_dim=14,
            state_dim=14,
            hidden_dim=512,
            chunk_size=CHUNK_SIZE
        )

    # 加载权重
    if not os.path.exists(CKPT_PATH):
        print(f"Error: Checkpoint not found at {CKPT_PATH}")
        return

    state_dict = torch.load(CKPT_PATH, map_location=device)
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    print("Model loaded successfully.")

    # 3. 初始化硬件
    print("Initializing robot and camera...")
    robot = setup_robot(CURRENT_ROBOT, CONFIG_FILE)
    # 确保分辨率与 Config 一致
    camera = RealSenseCamera(width=640, height=480, fps=30)
    camera.start()

    # 预热相机
    for _ in range(10):
        camera.get_image()
        time.sleep(0.1)

    print("Hardware ready. Starting inference loop...")
    print("Press 'q' in the OpenCV window to quit.")

    # ==========================================
    # 4. 推理主循环
    # ==========================================
    # [关键] 历史观测缓冲区: 自动保持最近 N_OBS_STEPS 帧
    obs_history = collections.deque(maxlen=N_OBS_STEPS)

    try:
        while True:
            # --- STEP 1: 获取当前观测 (t) ---
            step_start_total = time.time()

            # 读取图像 (H, W, C) RGB
            img = camera.get_image()
            if img is None: continue

            # 显示
            cv2.imshow("Camera View", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # 读取机械臂状态
            qpos = robot.get_qpos()  # numpy array (14,)

            # --- STEP 2: 数据处理与缓冲区管理 ---
            # 归一化当前帧
            qpos_norm = pre_process(qpos)

            # 存入缓冲区
            obs_history.append({'image': img, 'qpos': qpos_norm})

            # [冷启动] 如果缓冲区没满 (第一帧)，复制填充
            # MA-ACT 第一帧时，让 t-1 = t
            while len(obs_history) < N_OBS_STEPS:
                obs_history.append({'image': img, 'qpos': qpos_norm})

            # --- STEP 3: 构造模型输入 ---

            # 1. 堆叠图像: List[(H,W,C)] -> (T, H, W, C)
            img_seq = np.stack([x['image'] for x in obs_history])
            # 转置: (T, H, W, C) -> (T, C, H, W)
            img_seq = np.transpose(img_seq, (0, 3, 1, 2))
            # 转 Tensor & 归一化 & 增加 Batch 维度: (1, T, C, H, W)
            img_tensor = torch.from_numpy(img_seq).float().cuda() / 255.0
            img_tensor = img_tensor.unsqueeze(0)

            # 2. 堆叠状态: (T, D) -> (1, T, D)
            qpos_seq = np.stack([x['qpos'] for x in obs_history])
            qpos_tensor = torch.from_numpy(qpos_seq).float().cuda().unsqueeze(0)

            # --- STEP 4: 模型前向推理 ---
            with torch.inference_mode():
                if USE_SPEED_ACT:
                    # [MA-ACT 模式] 构造字典
                    batch = {
                        "observation.state": qpos_tensor,  # (1, T, D)
                        "observation.images": [img_tensor],  # List [(1, T, C, H, W)]
                        # 主相机数据用于光流
                        MAIN_CAMERA_NAME: img_tensor,
                        # 占位 Mask
                        "action_is_pad": torch.zeros(1, CHUNK_SIZE, dtype=torch.bool, device=device)
                    }

                    # SpeedACT 返回 (actions, stats)
                    all_actions, _ = policy(batch)

                else:
                    # [Standard ACT 模式]
                    # ACT 通常只需要当前帧 (Buffer 的最后一帧)
                    # (1, T, C, H, W) -> (1, C, H, W)
                    curr_img = img_tensor[:, -1]
                    # (1, T, D) -> (1, D)
                    curr_qpos = qpos_tensor[:, -1]

                    all_actions = policy(curr_qpos, curr_img)

            # 反归一化
            all_actions = all_actions.squeeze(0).cpu().numpy()  # (Chunk, D)
            all_actions = post_process(all_actions)

            # --- STEP 5: 执行动作 (Open Loop) ---
            # 简单执行前 EXECUTION_HORIZON 步
            for t in range(EXECUTION_HORIZON):
                loop_start = time.time()

                # 发送指令
                target_action = all_actions[t]
                robot.act(target_action)

                # 频率控制
                loop_elapsed = time.time() - loop_start
                if loop_elapsed < DT:
                    time.sleep(DT - loop_elapsed)

                # 响应退出 (在执行子循环中也要检查)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Safety exit.")


if __name__ == '__main__':
    main()