import argparse
import collections
import os
import pickle
import time

import cv2
import numpy as np
import torch

from policy.maact.common.configs.configuration_act import SpeedACTConfig
from policy.maact.common.model.speed_act_modulate_full_model import SpeedACT
from scripts.predict import setup_robot
from utils.camera import RealSenseCamera

# 机器人配置
CURRENT_ROBOT = 'startouch'
CONFIG_FILE = 'config.yaml'

# 模型路径 (请确保指向 MA-ACT 训练好的权重)
CKPT_PATH = './checkpoints_maact/policy_best.ckpt'
STATS_PATH = './checkpoints_maact/dataset_stats.pkl'

# 推理参数
CHUNK_SIZE = 100  # 动作块大小 (与训练保持一致)
EXECUTION_HORIZON = 20  # 开环执行步数 (小于 Chunk Size)
FREQUENCY = 30  # 控制频率 Hz
DT = 1.0 / FREQUENCY

# MA-ACT 必须至少2帧历史
N_OBS_STEPS = 2
MAIN_CAMERA_NAME = 'cam_high'  # 必须与训练时的名称一致


def main():
    parser = argparse.ArgumentParser(description="ACT Training Script")
    parser.add_argument('--joint_i', action='store_true', help='joint input')
    parser.add_argument('--joint_o', action='store_true', help='joint output')
    args = parser.parse_args()

    STATE_DIM = 7 if args.joint_i else 8
    ACTION_DIM = 7 if args.joint_o else 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 形状: (1, 1, 3, 1, 1) 用于广播匹配 (Batch, Time, Channel, Height, Width)
    NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    NORM_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    print(f"Loading stats from {STATS_PATH}...")
    with open(STATS_PATH, 'rb') as f:
        stats = pickle.load(f)

    # 定义预处理和后处理
    def pre_process(qpos):
        qpos = qpos[:STATE_DIM]
        return (qpos - stats['qpos_mean']) / stats['qpos_std']

    def post_process(action):
        return action * stats['action_std'] + stats['action_mean']

    print(f"Loading MA-ACT (SpeedACT) model...")

    config = SpeedACTConfig(
        dim_model=512,
        chunk_size=CHUNK_SIZE,
        n_obs_steps=N_OBS_STEPS,
        image_features={MAIN_CAMERA_NAME: (3, 480, 640)},
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

    state_dict = torch.load(CKPT_PATH, map_location=device)
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    print("Model loaded successfully.")

    print("Initializing robot and camera...")
    robot = setup_robot(CURRENT_ROBOT, CONFIG_FILE, args.joint_i, args.joint_o)
    camera = RealSenseCamera()

    print("Hardware ready. Starting inference loop...")
    print("Press 'q' in the OpenCV window to quit.")

    obs_history = collections.deque(maxlen=N_OBS_STEPS)

    print("Warming up observation buffer...")
    for _ in range(N_OBS_STEPS):
        t0 = time.time()
        img = camera.get_frame()
        qpos = robot.get_qpos()
        if img is not None and qpos is not None:
            obs_history.append({'image': img, 'qpos': pre_process(qpos)})

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
                all_actions = policy(batch)[0]

            # 反归一化
            all_actions = all_actions.squeeze(0).cpu().numpy()
            all_actions = post_process(all_actions)

            for t in range(EXECUTION_HORIZON):
                loop_start = time.time()

                target_action = all_actions[t]
                robot.command_action(target_action)

                curr_img = camera.get_frame()
                curr_qpos = robot.get_qpos()

                if curr_img is not None and curr_qpos is not None:
                    # cv2.imshow("Camera View", cv2.cvtColor(curr_img, cv2.COLOR_RGB2BGR))
                    obs_history.append({'image': curr_img, 'qpos': pre_process(curr_qpos)})

                loop_elapsed = time.time() - loop_start
                if loop_elapsed < DT:
                    time.sleep(DT - loop_elapsed)

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