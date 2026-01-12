import pickle
import sys
import time

import cv2
import torch
import yaml
# 导入 SDK Wrapper
from xarm.wrapper import XArmAPI

# 导入适配器和代理
from policy.act.policy import ACTPolicy
from utils import make_xarm_sdk, make_xarm_reader
from utils.camera import RealSenseCamera
from utils.robot_agent import UniversalRobotAgent

# 如果有 startouch 库，请取消注释
# from startouch_sdk import SingleArm

# ==================== 配置 ====================
# 选择你的机器人类型: 'xarm' 或 'startouch'
CURRENT_ROBOT = 'xarm'
CONFIG_FILE = 'config.yaml'

# 模型参数
CKPT_PATH = './checkpoints/policy_best.ckpt'
STATS_PATH = './checkpoints/dataset_stats.pkl'
CHUNK_SIZE = 50  # 预测步长
EXECUTION_HORIZON = 20  # 执行步长 (Open Loop)
FREQUENCY = 40  # 控制频率 Hz
DT = 1.0 / FREQUENCY


# ==================== 硬件初始化工厂 ====================
def setup_robot(robot_type, config_path):
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
        return UniversalRobotAgent('xarm_agent', read_fn, write_fn)

    elif robot_type == 'startouch':
        can_port = cfg['StarTouch']['can_port']
        print(f"[Init] Connecting to StarTouch at {can_port}...")
        # 伪代码：初始化 StarTouch
        # arm = SingleArm(can_port)
        # arm.enable()

        # write_fn = make_startouch_sdk(arm)
        # read_fn = make_startouch_reader(arm)
        # return UniversalRobotAgent('startouch_agent', read_fn, write_fn)
        raise NotImplementedError("StarTouch init needs actual library")

    else:
        raise ValueError(f"Unknown robot type: {robot_type}")

# ==================== 模型加载 ====================
def load_model():
    print(f"[Model] Loading stats from {STATS_PATH}...")
    with open(STATS_PATH, 'rb') as f:
        stats = pickle.load(f)

    print(f"[Model] Loading weights from {CKPT_PATH}...")
    # 这里的参数必须和训练时一致
    policy = ACTPolicy({
        'num_queries': CHUNK_SIZE,
        'camera_names': ['cam_high'],
        'state_dim': 7,  # 注意：这里要看你训练时是 7 (6pose+1gripper) 还是 14 (双臂)
        # ... 其他必要参数 (lr, hidden_dim 等)
        'lr': 1e-5, 'hidden_dim': 512, 'dim_feedforward': 3200,
        'backbone': 'resnet18', 'enc_layers': 4, 'dec_layers': 7, 'nheads': 8
    })

    state_dict = torch.load(CKPT_PATH)
    policy.load_state_dict(state_dict)
    policy.cuda()
    policy.eval()
    return policy, stats


# ==================== 主函数 ====================
def main():
    # 1. 准备硬件
    try:
        robot = setup_robot(CURRENT_ROBOT, CONFIG_FILE)
        camera = RealSenseCamera()
    except Exception as e:
        print(f"Hardware initialization failed: {e}")
        return

    # 2. 准备模型
    try:
        policy, stats = load_model()
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

            # --- STEP 3: 模型推理 ---
            with torch.inference_mode():
                # ACT Forward
                all_actions = policy(qpos_tensor, img_tensor)

            # 反归一化
            all_actions = all_actions.squeeze(0).cpu().numpy()
            all_actions = post_process(all_actions)  # (CHUNK_SIZE, action_dim)

            for t in range(EXECUTION_HORIZON):
                step_start = time.time()
                # 获取当前时刻的动作
                action = all_actions[t]
                robot.act(action)
                # 频率控制
                elapsed = time.time() - step_start
                if elapsed < DT:
                    time.sleep(DT - elapsed)

                # 在执行子循环中也要响应退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        print(f"Runtime Error: {e}")
    finally:
        # 安全退出
        print("Cleaning up...")
        camera.stop()
        cv2.destroyAllWindows()
        # 注意：xarm api 在 python 脚本结束时通常会自动断开，或者可以在 RobotAgent 加 close 方法
        sys.exit(0)


if __name__ == "__main__":
    main()