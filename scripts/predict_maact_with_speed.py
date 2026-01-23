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

CKPT_PATH = '/home/lumos/act_move/checkpoints/maact/policy_epoch_699.ckpt'
STATS_PATH = '/home/lumos/act_move/checkpoints/maact/dataset_stats.pkl'

# 推理参数
CHUNK_SIZE = 50  # 动作块大小 (与训练保持一致)
EXECUTION_HORIZON = 20  # 开环执行步数 (小于 Chunk Size)
FREQUENCY = 30  # 控制频率 Hz
DT = 1.0 / FREQUENCY

N_OBS_STEPS = 1
MAIN_CAMERA_NAME = 'cam_high'  # 必须与训练时的名称一致
CAMERA_NAMES = ['cam_high']
NUM_SPEED_CATEGORIES = 3

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


def get_user_speed_input(num_categories):
    """
    阻塞式询问用户当前的速度设置，根据 num_categories 动态生成验证范围
    """
    # 动态生成合法输入列表，例如 [0, 1, 2, 3, 4]
    valid_range = list(range(num_categories))

    if 1 in valid_range:
        default_val = 1
    elif len(valid_range) > 0:
        default_val = valid_range[len(valid_range) // 2]
    else:
        default_val = 0

    while True:
        print("\n" + "=" * 40)
        print("🚦 等待速度设置 (Wait for Speed Input) 🚦")
        print(f"请输入当前传送带速度等级 (范围: 0 ~ {num_categories - 1}):")
        print("=" * 40)
        try:
            # 动态生成提示字符串
            options_str = "/".join(map(str, valid_range))
            prompt = f"👉 请输入 ({options_str}) [按回车默认 {default_val}]: "

            user_input = input(prompt).strip()

            if user_input == "":
                print(f"未输入，使用默认值: [{default_val}]")
                return default_val

            val = int(user_input)
            if val in valid_range:
                print(f"✅ 已确认速度等级: [{val}]")
                return val
            else:
                print(f"❌ 输入无效，请输入 {valid_range} 中的一个数字")
        except ValueError:
            print("❌ 输入格式错误，请输入数字")

def main():
    parser = argparse.ArgumentParser(description="ACT Training Script")
    parser.add_argument('--joint_i', action='store_true', help='joint input')
    parser.add_argument('--joint_o', action='store_true', help='joint output')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 形状: (1, 1, 3, 1, 1) 用于广播匹配 (Batch, Time, Channel, Height, Width)
    NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    NORM_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    print(f"Loading stats from {STATS_PATH}...")
    with open(STATS_PATH, 'rb') as f:
        stats = pickle.load(f)
    STATE_DIM = stats["qpos_mean"].shape[0]
    ACTION_DIM = stats["action_mean"].shape[0]
    def pre_process(qpos):
        qpos = qpos[:STATE_DIM]
        return (qpos - stats['qpos_mean']) / stats['qpos_std']
    def post_process(action):
        return action * stats['action_std'] + stats['action_mean']

    print(f"Loading MA-ACT (SpeedACT) model...")

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
        num_speed_categories=NUM_SPEED_CATEGORIES,
        feedforward_activation="relu",
        pre_norm=False,
        global_flow_size=128,
        optical_flow_map_height=256,
        optical_flow_map_width=320,
    )

    if not os.path.exists(CKPT_PATH):
        print(f"Error: Checkpoint not found at {CKPT_PATH}")
        return

    policy = SpeedACT(config).to(device)
    load_checkpoint_compatible(policy, CKPT_PATH, device)
    policy.eval()
    print("Model loaded successfully.")

    print("Initializing robot and camera...")
    robot = setup_robot(CURRENT_ROBOT, CONFIG_FILE, args.joint_i, args.joint_o, STATE_DIM, ACTION_DIM)
    camera = RealSenseCamera(width=640, height=480, fps=30)

    print("Hardware ready. Starting inference loop...")
    print("Press 'q' in the OpenCV window to quit.")

    try:
        while True:
            current_speed = get_user_speed_input(NUM_SPEED_CATEGORIES)

            speed_tensor = torch.tensor([current_speed], dtype=torch.long, device=device)

            print("🤖 Robot going home...")
            robot.go_home(blocking=True, duration=3.0)

            print(f"🟢 Start Inference Loop (Speed: {current_speed})... Press [Enter] to Reset.")
            reset_triggered = False
            while not reset_triggered:
                img = camera.get_frame()
                qpos = robot.get_qpos()

                if img is None or qpos is None:
                    time.sleep(0.01)
                    continue

                # --- 图像处理: (H, W, C) -> (1, 1, C, H, W) ---
                # permute: (H, W, C) -> (C, H, W)
                img_tensor = torch.from_numpy(img).float().to(device)
                img_tensor = img_tensor.permute(2, 0, 1)
                # 增加 Batch 和 Time 维度
                img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
                img_tensor = img_tensor / 255.0
                img_tensor = (img_tensor - NORM_MEAN) / NORM_STD

                # --- 状态处理: (D,) -> (1, 1, D) ---
                qpos_norm = pre_process(qpos)
                qpos_tensor = torch.from_numpy(qpos_norm).float().to(device)
                qpos_tensor = qpos_tensor.unsqueeze(0).unsqueeze(0)

                # 2. 模型推理
                with torch.inference_mode():
                    batch = {
                        "observation.state": qpos_tensor,
                        "observation.images": [img_tensor],
                        "action_is_pad": torch.zeros(1, CHUNK_SIZE, dtype=torch.bool, device=device),
                        "speed_label": speed_tensor
                    }
                    all_actions, _ = policy(batch)

                all_actions = all_actions.squeeze(0).cpu().numpy()
                all_actions = post_process(all_actions)

                # 3. 动作执行循环 (Open-Loop Execution)
                for t in range(EXECUTION_HORIZON):
                    t_exec_start = time.time()

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Quitting...")
                        raise KeyboardInterrupt
                    elif key == 13:  # Enter 键 (ASCII 13)
                        print("\n🔄 Reset triggered! Restarting session...")
                        reset_triggered = True
                        break

                    target_action = all_actions[t]
                    robot.command_action(target_action)

                    # --- 更新观测 (仅用于显示) ---
                    # 因为下一轮推理不需要这里的历史数据，所以只做显示
                    curr_img = camera.get_frame()
                    if curr_img is not None:
                        bgr_img = cv2.cvtColor(curr_img, cv2.COLOR_RGB2BGR)
                        # 在左上角显示当前速度模式
                        cv2.putText(bgr_img, f"Speed Mode: {current_speed}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow("Camera View", bgr_img)

                    # --- 频率控制 ---
                    elapsed = time.time() - t_exec_start
                    if elapsed < DT:
                        time.sleep(DT - elapsed)

                # 如果触发了 Reset，break 跳出内层循环，回到外层 (Input -> Go Home)
                if reset_triggered:
                    break

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