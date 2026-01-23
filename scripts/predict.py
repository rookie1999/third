import argparse
import pickle
import sys
import time

import cv2
import torch
import yaml
# 导入 SDK Wrapper
# from xarm.wrapper import XArmAPI

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 导入适配器和代理
from policy.act.policy import ACTPolicy
from utils import make_xarm_sdk, make_xarm_reader, make_startouch_eef_rot_sdk, make_startouch_joint_reader, \
    make_startouch_ee_rot_reader, make_startouch_joint_sdk, make_startouch_ee_rot_reader, make_startouch_eef_rpy_sdk, make_startouch_ee_rpy_reader
from utils.camera import RealSenseCamera
from utils.robot_agent import UniversalRobotAgent
import numpy as np

startouch_path = os.path.join(root_dir, 'startouch-v1', 'interface_py')
if startouch_path not in sys.path:
    sys.path.append(startouch_path)
from startouchclass import SingleArm

# ==================== 配置 ====================
# 选择你的机器人类型: 'xarm' 或 'startouch'
CURRENT_ROBOT = 'startouch'
CONFIG_FILE = 'config.yaml'

# 模型参数
CKPT_PATH = '/home/lumos/act_move/checkpoints/act/policy_epoch_599.ckpt'
STATS_PATH = '/home/lumos/act_move/checkpoints/act/dataset_stats.pkl'
CHUNK_SIZE = 50  # 预测步长
EXECUTION_HORIZON = 20  # 执行步长 (Open Loop)
FREQUENCY = 30  # 控制频率 Hz
DT = 1.0 / FREQUENCY


# ==================== 硬件初始化工厂 ====================
def setup_robot(robot_type, config_path, joint_i, joint_o):
    """根据配置初始化具体的机器人，并返回通用代理"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if robot_type == 'xarm':
        ip = cfg['Xarm7']['robot_ip']
        print(f"[Init] Connecting to XArm at {ip}...")
        arm = XArmAPI(ip)
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(1)
        # 切换到伺服模式 (Cartesian Servo)
        arm.set_mode(1)
        arm.set_state(0)

        # 创建读写闭包
        write_fn = make_xarm_sdk(arm)
        read_fn = make_xarm_reader(arm)
        return UniversalRobotAgent('xarm', read_fn, write_fn, arm, None)

    elif robot_type == 'startouch':
        arm = SingleArm(can_interface_=cfg["StarTouch"]["can_port"], gripper=True)
        arm.setGripperPosition(1)
        time.sleep(1)
        startouch_write_fn = make_startouch_joint_sdk() if joint_o else make_startouch_eef_rpy_sdk(arm)
        startouch_read_fn = make_startouch_joint_reader() if joint_i else make_startouch_ee_rpy_reader(arm)
        return UniversalRobotAgent('startouch',
                                   startouch_read_fn,
                                   startouch_write_fn,
                                   arm,
                                   cfg["StarTouch"]["initial_joints"])

    else:
        raise ValueError(f"Unknown robot type: {robot_type}")

# ==================== 模型加载 ====================
def load_model(is_joint_input, is_joint_output):
    print(f"[Model] Loading stats from {STATS_PATH}...")
    with open(STATS_PATH, 'rb') as f:
        stats = pickle.load(f)

    print(f"[Model] Loading weights from {CKPT_PATH}...")

    # 这里的参数必须和训练时一致
    CAMERA_NAMES = ['cam_high']
    STATE_DIM = 7 if is_joint_input else 7
    ACTION_DIM = 7 if is_joint_output else 7
    LR = 1e-4
    CHUNK_SIZE = 50
    KL_WEIGHT = 10.0
    args_override = {
        'kl_weight': KL_WEIGHT,
        'chunk_size': CHUNK_SIZE,
        'hidden_dim': 512,
        'dim_feedforward': 3200,
        'nheads': 8,
        'enc_layers': 4,
        'dec_layers': 1,
        'n_decoder_layers': 1,  # 兼容不同命名习惯
        'camera_names': CAMERA_NAMES,
        'state_dim': STATE_DIM,
        'action_dim': ACTION_DIM,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'masks': False,
        'dilation': False,
        'dropout': 0.1,
        'pre_norm': False,
        'num_queries': CHUNK_SIZE,
    }
    policy = ACTPolicy(args_override)

    state_dict = torch.load(CKPT_PATH)
    policy.load_state_dict(state_dict)
    policy.cuda()
    policy.eval()
    return policy, stats


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="ACT Training Script")
    parser.add_argument('--joint_i', action='store_true', help='joint input')
    parser.add_argument('--joint_o', action='store_true', help='joint output')
    args = parser.parse_args()

    NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 1, 3, 1, 1)
    NORM_STD = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 1, 3, 1, 1)

    # 1. 准备硬件
    try:
        robot = setup_robot(CURRENT_ROBOT, CONFIG_FILE, args.joint_i, args.joint_o)
        camera = RealSenseCamera()
    except Exception as e:
        print(f"Hardware initialization failed: {e}")
        return

    # 2. 准备模型
    try:
        policy, stats = load_model(args.joint_i, args.joint_o)
    except Exception as e:
        print(f"Model loading failed: {e}")
        camera.stop()
        return

    # 数据归一化工具 (Lambda)
    pre_process = lambda qpos: (qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda action: action * stats['action_std'] + stats['action_mean']

    print("=" * 50)
    print(f" Start Inference Loop ({CURRENT_ROBOT})")
    print(" Press 'q' in window to Stop.")
    print("=" * 50)
    # robot.command_action(np.array([0.30240757, 0.01034589,0.18278981, 0.03465579, 0.00971405, 0.0560794, 0.61916577]))


    try:
        while True:
            # --- STEP 1: 获取观测 (Observation) ---
            t0 = time.time()
            img = camera.get_frame()
            qpos_numpy = robot.get_qpos()

            if img is None or qpos_numpy is None:
                print("Sensor read error, retrying...")
                time.sleep(0.1)
                continue

            # 显示图像
            cv2.imshow(f"Robot View ({CURRENT_ROBOT})", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # --- STEP 2: 数据预处理 ---
            # 构造 Torch Tensor
            qpos = pre_process(qpos_numpy)
            qpos_tensor = torch.from_numpy(qpos).float().cuda().unsqueeze(0)  # (1, state_dim)

            img_tensor = torch.from_numpy(img).float().cuda() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, H, W)

            # img_tensor = (img_tensor - NORM_MEAN) / NORM_STD

            # --- STEP 3: 模型推理 ---
            with torch.inference_mode():
                # ACT Forward
                all_actions = policy(qpos_tensor, img_tensor)
            print(all_actions)
            # 反归一化
            all_actions = all_actions.squeeze(0).cpu().numpy()
            all_actions = post_process(all_actions)  # (CHUNK_SIZE, action_dim)

            for t in range(EXECUTION_HORIZON):
                step_start = time.time()
                # 获取当前时刻的动作
                action = all_actions[t]
                print(action)
                robot.command_action(action)
                # 频率控制
                elapsed = time.time() - step_start
                if elapsed < DT:
                    time.sleep(DT - elapsed)

                # 在执行子循环中也要响应退出
                # cv2.imshow(f"Robot View ({CURRENT_ROBOT})", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        print(f"Runtime Error: {e}")
        raise e
    finally:
        # 安全退出
        robot.go_home()
        print("Cleaning up...")
        camera.stop()
        cv2.destroyAllWindows()
        # 注意：xarm api 在 python 脚本结束时通常会自动断开，或者可以在 RobotAgent 加 close 方法
        sys.exit(0)


if __name__ == "__main__":
    main()