import pickle
import sys
import time
import collections
import numpy as np
import cv2
import torch
import yaml
import os

# ==========================================
# 导入 SDK Wrapper
# ==========================================
from xarm.wrapper import XArmAPI
from utils import make_xarm_sdk, make_xarm_reader
from utils.camera import RealSenseCamera
from utils.robot_agent import UniversalRobotAgent

# ==========================================
# 导入 MA-ACT 模型组件
# ==========================================
# 请根据实际路径调整 import
from policy.maact.common.model.speed_act_modulate_full_model import SpeedACT
from policy.maact.common.configs.configuration_act import SpeedACTConfig

# ==========================================
# 配置区域
# ==========================================
# 机器人配置
CURRENT_ROBOT = 'xarm'
CONFIG_FILE = 'config.yaml'

# 模型路径 (请确保指向 MA-ACT 训练好的权重)
CKPT_PATH = './checkpoints_maact/policy_best.ckpt'
STATS_PATH = './checkpoints_maact/dataset_stats.pkl'
# YOLO 权重路径 (必须存在)
YOLO_CKPT = r"F:\projects\lumos\ma_act\src\object_detection\object_detection_ckpt\yolov8n.pt"

# 推理参数
CHUNK_SIZE = 100  # 动作块大小 (与训练保持一致)
EXECUTION_HORIZON = 20  # 开环执行步数 (小于 Chunk Size)
FREQUENCY = 30  # 控制频率 Hz
DT = 1.0 / FREQUENCY

# MA-ACT 必须至少2帧历史
N_OBS_STEPS = 2
MAIN_CAMERA_NAME = 'cam_high'  # 必须与训练时的名称一致

# 维度配置 (必须与 train_maact.py 保持一致)
STATE_DIM = 7
ACTION_DIM = 7


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
    # -------------------------------------------------------------------------
    # 1. 准备归一化参数 (ImageNet Stats)
    # -------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 形状: (1, 1, 3, 1, 1) 用于广播匹配 (Batch, Time, Channel, Height, Width)
    NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    NORM_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    # -------------------------------------------------------------------------
    # 2. 加载统计数据
    # -------------------------------------------------------------------------
    print(f"Loading stats from {STATS_PATH}...")
    with open(STATS_PATH, 'rb') as f:
        stats = pickle.load(f)

    # 定义预处理和后处理
    def pre_process(qpos):
        # 确保输入维度正确，如果机器人返回多于7维，进行切片
        qpos = qpos[:STATE_DIM]
        return (qpos - stats['qpos_mean']) / stats['qpos_std']

    def post_process(action):
        return action * stats['action_std'] + stats['action_mean']

    # -------------------------------------------------------------------------
    # 3. 初始化 SpeedACT 模型
    # -------------------------------------------------------------------------
    print(f"Loading MA-ACT (SpeedACT) model...")

    config = SpeedACTConfig(
        dim_model=512,
        chunk_size=CHUNK_SIZE,
        n_obs_steps=N_OBS_STEPS,
        # 注意：图像尺寸 (480, 640) 必须与 RealSenseCamera 设置一致
        image_features={MAIN_CAMERA_NAME: (3, 480, 640)},
        main_camera=MAIN_CAMERA_NAME,

        # [关键修正] 维度需匹配 train_maact.py
        robot_state_feature=(STATE_DIM,),
        action_feature=(ACTION_DIM,),

        use_optical_flow=True,
        object_detection_ckpt_path=YOLO_CKPT,
        cropped_flow_h=64,
        cropped_flow_w=64,
        feedforward_activation="relu",
        pre_norm=False
    )

    policy = SpeedACT(config)

    # 加载权重
    if not os.path.exists(CKPT_PATH):
        print(f"Error: Checkpoint not found at {CKPT_PATH}")
        return

    state_dict = torch.load(CKPT_PATH, map_location=device)
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    print("Model loaded successfully.")

    # -------------------------------------------------------------------------
    # 4. 初始化硬件
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 5. 推理主循环
    # -------------------------------------------------------------------------
    # 历史观测缓冲区: 自动保持最近 N_OBS_STEPS 帧
    obs_history = collections.deque(maxlen=N_OBS_STEPS)

    try:
        while True:
            # --- STEP 1: 获取当前观测 (t) ---
            step_start_total = time.time()

            # 读取图像 (H, W, C) RGB
            img = camera.get_image()
            if img is None: continue

            # 显示 (转BGR显示)
            cv2.imshow("Camera View", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # 读取机械臂状态
            qpos = robot.get_qpos()  # numpy array

            # --- STEP 2: 数据处理与缓冲区管理 ---
            # 归一化当前帧状态
            qpos_norm = pre_process(qpos)

            # 存入缓冲区
            obs_history.append({'image': img, 'qpos': qpos_norm})

            # [冷启动] 如果缓冲区没满 (例如第一帧)，复制填充直到填满
            while len(obs_history) < N_OBS_STEPS:
                obs_history.append({'image': img, 'qpos': qpos_norm})

            # --- STEP 3: 构造模型输入 ---

            # 1. 堆叠图像: List[(H,W,C)] -> (T, H, W, C)
            img_seq = np.stack([x['image'] for x in obs_history])
            # 转置: (T, H, W, C) -> (T, C, H, W)
            img_seq = np.transpose(img_seq, (0, 3, 1, 2))

            # 转 Tensor (0-255 -> 0-1)
            img_tensor = torch.from_numpy(img_seq).float().cuda() / 255.0

            # 增加 Batch 维度 -> (1, T, C, H, W)
            img_tensor = img_tensor.unsqueeze(0)

            # [关键修正] 执行 ImageNet 归一化 (SpeedACT 必需)
            # 此时 img_tensor 是 0-1 范围，NORM_MEAN/STD 也是针对 0-1 的
            img_tensor = (img_tensor - NORM_MEAN) / NORM_STD

            # 2. 堆叠状态: (T, D) -> (1, T, D)
            qpos_seq = np.stack([x['qpos'] for x in obs_history])
            qpos_tensor = torch.from_numpy(qpos_seq).float().cuda().unsqueeze(0)

            # --- STEP 4: 模型前向推理 ---
            with torch.inference_mode():
                # 构造 MA-ACT 专用输入字典
                batch = {
                    "observation.state": qpos_tensor,  # (1, T, D)
                    "observation.images": [img_tensor],  # List [(1, T, C, H, W)]
                    # 主相机数据用于光流计算
                    MAIN_CAMERA_NAME: img_tensor,
                    # 占位 Mask (全 False 表示没有 Padding)
                    "action_is_pad": torch.zeros(1, CHUNK_SIZE, dtype=torch.bool, device=device)
                }

                # SpeedACT 返回 (actions, stats)
                # actions: (1, Chunk_Size, Action_Dim)
                all_actions, _ = policy(batch)

            # 反归一化
            all_actions = all_actions.squeeze(0).cpu().numpy()  # (Chunk, D)
            all_actions = post_process(all_actions)

            # --- STEP 5: 执行动作 (Open Loop) ---
            # 简单执行前 EXECUTION_HORIZON 步
            for t in range(EXECUTION_HORIZON):
                loop_start = time.time()

                # 发送指令
                target_action = all_actions[t]

                # 安全检查：如果机器人是7自由度，确保 action 也是7维
                if len(target_action) != 7:
                    print(f"Warning: Action dim {len(target_action)} != 7")

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
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Safety exit.")


if __name__ == '__main__':
    main()