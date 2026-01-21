import argparse
import pickle
import sys
import time
import numpy as np  # 确保导入 numpy

import cv2
import torch
import yaml
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 导入适配器和代理
from policy.act.policy import ACTPolicy
from utils import make_xarm_sdk, make_xarm_reader, make_startouch_eef_sdk, make_startouch_joint_reader, \
    make_startouch_ee_reader
from utils.camera import RealSenseCamera
from utils.robot_agent import UniversalRobotAgent

startouch_path = os.path.join(root_dir, 'startouch-v1', 'interface_py')
if startouch_path not in sys.path:
    sys.path.append(startouch_path)
from startouchclass import SingleArm

# ==================== 配置 ====================
# 选择你的机器人类型: 'xarm' 或 'startouch'
CURRENT_ROBOT = 'startouch'
CONFIG_FILE = 'config.yaml'

# 模型参数
CKPT_PATH = '/home/benson/projects/lumos/run_20260113_093204/checkpoints/policy_best.ckpt'
STATS_PATH = '/home/benson/projects/lumos/run_20260113_093204/checkpoints/dataset_stats.pkl'
CHUNK_SIZE = 50  # 预测步长
FREQUENCY = 30  # 控制频率 Hz
DT = 1.0 / FREQUENCY

# 时序集成参数
MAX_TIMESTEPS = 3000  # 最大运行步数（约100秒），可根据需要调大


# ==================== 硬件初始化工厂 ====================
def setup_robot(robot_type, config_path, joint_i, joint_o):
    """根据配置初始化具体的机器人，并返回通用代理"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if robot_type == 'xarm':
        # ... (xarm 代码保持不变)
        # 你的原始代码中没有 import XArmAPI，这里假设环境里有
        from xarm.wrapper import XArmAPI
        ip = cfg['Xarm7']['robot_ip']
        print(f"[Init] Connecting to XArm at {ip}...")
        arm = XArmAPI(ip)
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(1)
        arm.set_mode(1)
        arm.set_state(0)
        write_fn = make_xarm_sdk(arm)
        read_fn = make_xarm_reader(arm)
        return UniversalRobotAgent('xarm', read_fn, write_fn, arm, None)

    elif robot_type == 'startouch':
        arm = SingleArm(can_interface_=cfg["StarTouch"]["can_port"], gripper=True)
        # 注意：这里调用 utils 里的函数，需要确保 utils 被正确导入
        from utils import make_startouch_joint_sdk, make_startouch_eef_sdk, make_startouch_joint_reader, \
            make_startouch_ee_reader

        startouch_write_fn = make_startouch_joint_sdk(arm) if joint_o else make_startouch_eef_sdk(arm)
        startouch_read_fn = make_startouch_joint_reader(arm) if joint_i else make_startouch_ee_reader(arm)
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

    CAMERA_NAMES = ['cam_high']
    STATE_DIM = 7 if is_joint_input else 8
    ACTION_DIM = 7 if is_joint_output else 8
    KL_WEIGHT = 10.0
    args_override = {
        'kl_weight': KL_WEIGHT,
        'chunk_size': CHUNK_SIZE,
        'hidden_dim': 512,
        'dim_feedforward': 3200,
        'nheads': 8,
        'enc_layers': 4,
        'dec_layers': 1,
        'n_decoder_layers': 1,
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
    parser = argparse.ArgumentParser(description="ACT Inference Script with Temporal Aggregation")
    parser.add_argument('--joint_i', action='store_true', help='joint input')
    parser.add_argument('--joint_o', action='store_true', help='joint output')
    args = parser.parse_args()

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
        if 'camera' in locals(): camera.stop()
        return

    # 数据归一化工具
    pre_process = lambda qpos: (qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda action: action * stats['action_std'] + stats['action_mean']

    print("=" * 50)
    print(f" Start Inference Loop ({CURRENT_ROBOT}) with Temporal Aggregation")
    print(" Press 'q' in window to Stop.")
    print("=" * 50)

    # === 时序集成初始化 ===
    # 预先分配一个大数组用于存储所有时刻的预测
    # 形状: [max_steps, max_steps + chunk_size, action_dim]
    # 我们先设为 None，等第一次推理拿到 action_dim 后再初始化
    all_time_actions = None

    t = 0  # 全局时间步计数器

    try:
        # 重置到初始姿态 (可选)
        # robot.go_home()

        while True:
            step_start = time.time()

            # --- STEP 1: 获取观测 ---
            img = camera.get_frame()
            qpos_numpy = robot.get_qpos()

            if img is None or qpos_numpy is None:
                print("Sensor read error, retrying...")
                time.sleep(0.01)
                continue

            # --- STEP 2: 数据预处理 ---
            qpos = pre_process(qpos_numpy)
            qpos_tensor = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

            img_tensor = torch.from_numpy(img).float().cuda() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

            # 如果训练时使用了 ImageNet 归一化，这里也需要开启
            # NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 1, 3, 1, 1)
            # NORM_STD = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 1, 3, 1, 1)
            # img_tensor = (img_tensor - NORM_MEAN) / NORM_STD

            # --- STEP 3: 模型推理 ---
            with torch.inference_mode():
                # 预测未来 CHUNK_SIZE 步
                all_actions = policy(qpos_tensor, img_tensor)

            # 反归一化
            all_actions = all_actions.squeeze(0).cpu().numpy()
            all_actions = post_process(all_actions)  # shape: (CHUNK_SIZE, action_dim)

            # --- STEP 4: 时序集成 (Temporal Aggregation) ---

            # 首次运行时初始化缓冲区
            if all_time_actions is None:
                action_dim = all_actions.shape[-1]
                all_time_actions = np.zeros([MAX_TIMESTEPS, MAX_TIMESTEPS + CHUNK_SIZE, action_dim])

            # 检查是否超出最大步数
            if t >= MAX_TIMESTEPS:
                print("Reach Max Timesteps, Stopping...")
                break

            # 将本次预测的 50 步动作，叠加到全局时间表中
            # all_actions[i] 代表预测的第 i 步，对应真实时间 t + i
            all_time_actions[[t], t: t + CHUNK_SIZE] = all_actions

            # 计算当前时刻 t 的动作
            # 策略：取过去所有覆盖到当前时刻 t 的预测，求平均
            # 有效的起始预测帧索引：max(0, t - CHUNK_SIZE + 1) 到 t
            start_index = max(0, t - CHUNK_SIZE + 1)
            end_index = t + 1

            # 取出这些预测在 t 时刻的值
            # shape: [num_valid_predictions, action_dim]
            valid_actions = all_time_actions[start_index:end_index, t]

            # 求平均，得到平滑后的动作
            action = np.mean(valid_actions, axis=0)

            # --- STEP 5: 执行动作 ---
            robot.command_action(action)

            # --- STEP 6: 频率控制 ---
            t += 1
            elapsed = time.time() - step_start
            if elapsed < DT:
                time.sleep(DT - elapsed)

            # 响应退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        print(f"Runtime Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        robot.go_home()
        print("Cleaning up...")
        camera.stop()
        cv2.destroyAllWindows()
        sys.exit(0)


if __name__ == "__main__":
    main()