import h5py
import numpy as np
import time
import sys
import os
import argparse

# === 路径设置 (确保能导入 startouchclass) ===
current_dir = os.path.dirname(os.path.abspath(__file__))
startouch_path = os.path.join(current_dir, 'startouch-v1', 'interface_py')
if startouch_path not in sys.path:
    sys.path.append(startouch_path)

try:
    from startouchclass import SingleArm
except ImportError:
    print("❌ Error: 无法导入 SingleArm，请检查 startouch-v1 路径是否正确。")
    sys.exit(1)


def replay_episode(dataset_path, robot_interface="can0"):
    # 1. 读取 HDF5 数据
    print(f"📂 Loading dataset: {dataset_path}")
    try:
        with h5py.File(dataset_path, 'r') as f:
            # 读取 qpos (通常是 [N, 7]，前6个是关节角，第7个是夹爪)
            qpos_data = f['observations/qpos'][:]

            # 如果想回放 action (通常更平滑，是主手的命令值)，可以解开下面这行
            # qpos_data = f['action'][:]

            print(f"✅ Loaded {len(qpos_data)} frames. Shape: {qpos_data.shape}")
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    # 2. 初始化机械臂
    print("🤖 Initializing Robot...")
    try:
        # enable_fd_=False 关闭力反馈以确保位置控制更稳
        robot = SingleArm(can_interface_=robot_interface, gripper=True, enable_fd_=False)
    except Exception as e:
        print(f"Hardware initialization failed: {e}")
        return

    try:
        # 3. 移动到起始点 (使用带规划的 set_joint)
        # 假设 qpos 格式为: [j1, j2, j3, j4, j5, j6, gripper]
        start_joints = qpos_data[0][:6]
        start_gripper = qpos_data[0][-1]

        print(f"🚀 Moving to start position: {start_joints}")
        robot.set_joint(start_joints, tf=3.0)  # 3秒到达起始点

        # 同步夹爪状态
        robot.setGripperPosition(start_gripper)
        time.sleep(3.5)  # 等待运动完成

        # 4. 开始循环回放 (使用 set_joint_raw 透传)
        input("按 Enter 键开始回放 (Ctrl+C 停止)...")
        print("▶️ Replaying...")

        # 假设录制频率是 30Hz，这里设置间隔
        dt = 1.0 / 30.0

        for i, frame in enumerate(qpos_data):
            loop_start = time.time()

            # 解析数据
            target_joints = frame[:6]
            target_gripper = frame[6]  # 假设最后一位是夹爪 (0~1)

            # 发送关节指令 (velocities 设为0或根据差分计算，这里透传位置即可)
            robot.set_joint_raw(target_joints, velocities=[0.0] * 6)

            # 发送夹爪指令
            robot.setGripperPosition_raw(target_gripper)

            # 频率控制
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

            if i % 30 == 0:
                print(f"Step {i}/{len(qpos_data)}", end='\r')

        print("\n✅ Replay finished.")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Runtime Error: {e}")
    finally:
        # 安全退出：回到零位或保持当前位置
        # robot.go_home()
        print("Cleaning up...")
        # robot.cleanup() # 如果有 cleanup 方法
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay qpos from HDF5 dataset")
    parser.add_argument('--file', type=str, required=True, help='Path to the .hdf5 file')
    parser.add_argument('--can', type=str, default='can0', help='CAN interface name')

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
    else:
        replay_episode(args.file, args.can)