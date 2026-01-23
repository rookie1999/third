import argparse
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

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

sys.path.append('/home/lumos/act_move/replay_remote_ctrl')

from scripts.predict import setup_robot
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
CURRENT_ROBOT = 'startouch'
CONFIG_FILE = 'config.yaml'

# 模型路径 (请确保指向 MA-ACT 训练好的权重)
CKPT_PATH = '/home/lumos/act_move/checkpoints/maact/policy_epoch_699.ckpt'
STATS_PATH = '/home/lumos/act_move/checkpoints/maact/dataset_stats.pkl'
# YOLO 权重路径 (必须存在)
# YOLO_CKPT = r"F:\projects\lumos\ma_act\src\object_detection\object_detection_ckpt\yolov8n.pt"

# 推理参数
CHUNK_SIZE = 50  # 动作块大小 (与训练保持一致)
EXECUTION_HORIZON = 20  # 开环执行步数 (小于 Chunk Size)
FREQUENCY = 30  # 控制频率 Hz
DT = 1.0 / FREQUENCY

# MA-ACT 必须至少2帧历史
N_OBS_STEPS = 2
MAIN_CAMERA_NAME = 'cam_high'  # 必须与训练时的名称一致
CAMERA_NAMES = ['cam_high']



def load_checkpoint_compatible(model, checkpoint_path, device):
    """
    自动处理 DDP 训练出来的 'module.' 前缀，使其能加载到单卡模型中
    """
    print(f"🔄 Loading checkpoint from: {checkpoint_path}")
    
    # 1. 加载文件
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 2. 兼容性处理：有时候 checkpoint 是字典，权重在 'state_dict' 键里
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 3. 关键步骤：去除 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        # 如果 key 以 'module.' 开头，去掉前 7 个字符
        if k.startswith('module.'):
            name = k[7:] 
        else:
            name = k
        new_state_dict[name] = v
        
    # 4. 加载处理后的权重
    msg = model.load_state_dict(new_state_dict, strict=False) # 建议先开 False 测试，没问题再 True
    print(f"✅ Loaded successfully! Missing keys: {msg.missing_keys}")
    return model




def main():
    parser = argparse.ArgumentParser(description="ACT Training Script")
    parser.add_argument('--joint_i', action='store_true', help='joint input')
    parser.add_argument('--joint_o', action='store_true', help='joint output')
    args = parser.parse_args()

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
    STATE_DIM = stats["qpos_mean"].shape[0]
    ACTION_DIM = stats["action_mean"].shape[0]
    # 定义预处理和后处理
    def pre_process(qpos):
        qpos = qpos[:STATE_DIM]
        return (qpos - stats['qpos_mean']) / stats['qpos_std']

    def post_process(action):
        return action * stats['action_std'] + stats['action_mean']

    # -------------------------------------------------------------------------
    # 3. 初始化 SpeedACT 模型
    # -------------------------------------------------------------------------
    print(f"Loading MA-ACT (SpeedACT) model...")

    # config = SpeedACTConfig(
    #     dim_model=512,
    #     chunk_size=CHUNK_SIZE,
    #     n_obs_steps=N_OBS_STEPS,
    #     # 注意：图像尺寸 (480, 640) 必须与 RealSenseCamera 设置一致
    #     image_features={MAIN_CAMERA_NAME: (3, 480, 640)},
    #     main_camera=MAIN_CAMERA_NAME,

    #     # [关键修正] 维度需匹配 train_maact.py
    #     robot_state_feature=(STATE_DIM,),
    #     action_feature=(ACTION_DIM,),

    #     use_optical_flow=True,
    #     # object_detection_ckpt_path=YOLO_CKPT,
    #     # cropped_flow_h=64,
    #     # cropped_flow_w=64,
    #     feedforward_activation="relu",
    #     pre_norm=False
    # )
    config = SpeedACTConfig(
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        n_encoder_layers=4,
        n_decoder_layers=1,
        chunk_size=CHUNK_SIZE,
        n_obs_steps=N_OBS_STEPS,
        image_features={cam: (3, 480, 640) for cam in CAMERA_NAMES},
        main_camera=MAIN_CAMERA_NAME,
        robot_state_feature=(STATE_DIM,),
        action_feature=(ACTION_DIM,),
        use_optical_flow=True,
        feedforward_activation="relu",
        pre_norm=False,
        global_flow_size=128,
        optical_flow_map_height=256,
        optical_flow_map_width=320,
    )

    policy = SpeedACT(config)

    # 加载权重
    if not os.path.exists(CKPT_PATH):
        print(f"Error: Checkpoint not found at {CKPT_PATH}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = SpeedACT(config).to(device)
    load_checkpoint_compatible(policy, CKPT_PATH, device)
    policy.eval()
    print("Model loaded successfully.")

    # -------------------------------------------------------------------------
    # 4. 初始化硬件
    # -------------------------------------------------------------------------
    print("Initializing robot and camera...")
    robot = setup_robot(CURRENT_ROBOT, CONFIG_FILE, args.joint_i, args.joint_o)
    # 确保分辨率与 Config 一致
    camera = RealSenseCamera(width=640, height=480, fps=30)
    # camera.start()

    # 预热相机
    for _ in range(10):
        camera.get_frame()
        time.sleep(0.1)

    print("Hardware ready. Starting inference loop...")
    print("Press 'q' in the OpenCV window to quit.")

    # -------------------------------------------------------------------------
    # 5. 推理主循环
    # -------------------------------------------------------------------------
    # 历史观测缓冲区: 自动保持最近 N_OBS_STEPS 帧
    obs_history = collections.deque(maxlen=N_OBS_STEPS)

    print("Warming up observation buffer...")
    for _ in range(N_OBS_STEPS):
        t0 = time.time()  # 记录开始时间
        img = camera.get_frame()
        qpos = robot.get_qpos()
        if img is not None and qpos is not None:
            obs_history.append({'image': img, 'qpos': pre_process(qpos)})

        # 扣除执行时间，精确等待
        elapsed = time.time() - t0
        if elapsed < DT:
            time.sleep(DT - elapsed)

    try:
        while True:
            # 1. 堆叠图像: (T, H, W, C) -> (T, C, H, W)
            img_seq = np.stack([x['image'] for x in obs_history])
            img_seq = np.transpose(img_seq, (0, 3, 1, 2))
            img_tensor = torch.from_numpy(img_seq).float().to(device) / 255.0
            img_tensor = img_tensor.unsqueeze(0)  # (1, T, C, H, W)

            # ImageNet 归一化
            img_tensor = (img_tensor - NORM_MEAN) / NORM_STD

            # 2. 堆叠状态
            qpos_seq = np.stack([x['qpos'] for x in obs_history])
            qpos_tensor = torch.from_numpy(qpos_seq).float().to(device).unsqueeze(0)

            # 3. 模型前向推理
            with torch.inference_mode():
                batch = {
                    "observation.state": qpos_tensor,
                    "observation.images": [img_tensor],
                    MAIN_CAMERA_NAME: img_tensor,
                    "action_is_pad": torch.zeros(1, CHUNK_SIZE, dtype=torch.bool, device=device)
                }
                # SpeedACT 返回4个值，只取第一个
                s = time.time()
                all_actions = policy(batch)[0]
                e = time.time()
                print(e - s)

            # 反归一化
            all_actions = all_actions.squeeze(0).cpu().numpy()
            all_actions = post_process(all_actions)

            for t in range(EXECUTION_HORIZON):
                loop_start = time.time()

                # A. 发送指令
                target_action = all_actions[t]
                robot.command_action(target_action)

                curr_img = camera.get_frame()
                curr_qpos = robot.get_qpos()

                if curr_img is not None and curr_qpos is not None:
                    # cv2.imshow("Camera View", curr_img)
                    # cv2.imshow("Camera View", cv2.cvtColor(curr_img, cv2.COLOR_RGB2BGR))
                    obs_history.append({'image': curr_img, 'qpos': pre_process(curr_qpos)})

                # C. 频率控制
                loop_elapsed = time.time() - loop_start
                if loop_elapsed < DT:
                    time.sleep(DT - loop_elapsed)

                # D. 响应退出
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