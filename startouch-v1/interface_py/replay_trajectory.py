#!/usr/bin/env python3
"""
轨迹复现脚本
根据 startouchclass.py 的控制逻辑，复现保存的轨迹数据
"""

import time
import numpy as np
import sys
from startouchclass import SingleArm
import threading


class TrajectoryPlayer:
    def __init__(self, trajectory_file: str, can_interface: str = "can0", enable_gravity: bool = True,
                 position_filter_alpha: float = 0.3, orientation_filter_alpha: float = 0.3):
        """
        初始化轨迹复现器
        
        Args:
            trajectory_file: 轨迹文件路径
            can_interface: CAN接口名称 (默认 "can0")
            enable_gravity: 是否启用重力补偿 (默认 True)
            position_filter_alpha: 位置低通滤波系数 (0-1，越小越平滑，默认0.3)
            orientation_filter_alpha: 姿态低通滤波系数 (0-1，越小越平滑，默认0.3)
        """
        self.trajectory_file = trajectory_file
        self.enable_gravity = enable_gravity
        self.position_filter_alpha = position_filter_alpha
        self.orientation_filter_alpha = orientation_filter_alpha
        
        # 滤波状态
        self.filtered_pos = None
        self.filtered_quat = None
        
        # 初始化机械臂控制器
        print("初始化机械臂控制器...")
        self.arm = SingleArm(can_interface_=can_interface, enable_fd_=False)
        print("机械臂控制器初始化完成")
        
        # 加载轨迹数据
        self.load_trajectory()
        
        # 启动重力补偿线程
        if self.enable_gravity:
            self.gravity_running = True
            self.gravity_thread = threading.Thread(target=self.gravity_compensation_loop, daemon=True)
            self.gravity_thread.start()
            print("重力补偿线程已启动")
    
    def load_trajectory(self):
        """加载轨迹文件"""
        print(f"加载轨迹文件: {self.trajectory_file}")
        
        try:
            # 读取轨迹数据
            # 格式: timestamp x y z qx qy qz qw
            data = np.loadtxt(self.trajectory_file)
            
            self.timestamps = data[:, 0]
            self.positions = data[:, 1:4]  # x, y, z
            self.quaternions = data[:, 4:8]  # qx, qy, qz, qw
            
            print(f"成功加载 {len(self.timestamps)} 个轨迹点")
            print(f"轨迹时长: {self.timestamps[-1] - self.timestamps[0]:.2f} 秒")
            
        except Exception as e:
            print(f"加载轨迹文件失败: {e}")
            sys.exit(1)
    
    def gravity_compensation_loop(self):
        """重力补偿线程"""
        while self.gravity_running:
            try:
                self.arm.gravity_compensation()
                time.sleep(0.01)  # 100Hz
            except Exception as e:
                print(f"重力补偿线程异常: {e}")
                time.sleep(0.1)
    
    def play_trajectory_position_control(self, speed_factor: float = 1.0):
        """
        使用位置控制模式复现轨迹
        
        Args:
            speed_factor: 速度因子，1.0为原速，>1.0为加速，<1.0为减速
        """
        print("\n开始复现轨迹（位置控制模式）...")
        print(f"速度因子: {speed_factor}x")
        
        # 获取当前位置
        current_pos, current_quat = self.arm.get_ee_pose_quat()
        print(f"当前位置: {current_pos}")
        print(f"起始位置: {self.positions[0]}")
        
        # 询问是否先移动到起始位置
        response = input("是否先移动到起始位置? (y/n): ")
        if response.lower() == 'y':
            print("移动到起始位置...")
            self.arm.set_end_effector_pose_quat(
                pos=self.positions[0],
                quat=self.quaternions[0],
                tf=3.0
            )
            time.sleep(3.5)
            print("已到达起始位置")
        
        input("按回车键开始复现轨迹...")
        
        # 记录开始时间
        start_time = time.time()
        trajectory_start_time = self.timestamps[0]
        
        for i in range(len(self.timestamps)):
            # 计算期望的执行时间
            expected_time = (self.timestamps[i] - trajectory_start_time) / speed_factor
            current_elapsed = time.time() - start_time
            
            # 等待到达期望时间
            if current_elapsed < expected_time:
                time.sleep(expected_time - current_elapsed)
            
            # 发送目标位置（使用四元数）
            # 注意：轨迹文件中的四元数格式是 [qx, qy, qz, qw]
            # 需要转换为 [qw, qx, qy, qz] 给 startouchclass
            quat_xyzw = self.quaternions[i]  # [qx, qy, qz, qw]
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # [qw, qx, qy, qz]
            
            self.arm.set_end_effector_pose_quat_raw(
                pos=self.positions[i],
                quat=quat_wxyz
            )
            
            # 打印进度
            if i % 50 == 0:
                progress = (i + 1) / len(self.timestamps) * 100
                print(f"进度: {progress:.1f}% ({i+1}/{len(self.timestamps)})")
        
        print("\n轨迹复现完成！")
    
    def play_trajectory_servo_control(self, speed_factor: float = 1.0, target_freq: float = 50.0, 
                                       enable_filter: bool = True, downsample: int = 1):
        """
        使用伺服控制模式复现轨迹（更平滑）
        
        Args:
            speed_factor: 速度因子，1.0为原速
            target_freq: 目标控制频率 (Hz)，默认50Hz更稳定
            enable_filter: 是否启用低通滤波
            downsample: 降采样因子，1=使用所有点，2=每2个点取1个
        """
        print("\n开始复现轨迹（伺服控制模式）...")
        print(f"速度因子: {speed_factor}x")
        print(f"目标控制频率: {target_freq} Hz")
        print(f"低通滤波: {'启用' if enable_filter else '禁用'}")
        print(f"降采样: {downsample}x")
        
        # 获取当前位置
        current_pos, current_quat = self.arm.get_ee_pose_quat()
        print(f"当前位置: {current_pos}")
        print(f"起始位置: {self.positions[0]}")
        
        # 初始化滤波器状态
        self.filtered_pos = self.positions[0].copy()
        quat_xyzw = self.quaternions[0]
        self.filtered_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        # 询问是否先移动到起始位置
        response = input("是否先移动到起始位置? (y/n): ")
        if response.lower() == 'y':
            print("移动到起始位置...")
            self.arm.set_end_effector_pose_quat(
                pos=self.positions[0],
                quat=self.filtered_quat,
                tf=3.0
            )
            time.sleep(3.5)
            print("已到达起始位置")
        
        input("按回车键开始复现轨迹...")
        
        # 降采样轨迹
        if downsample > 1:
            indices = np.arange(0, len(self.timestamps), downsample)
            timestamps = self.timestamps[indices]
            positions = self.positions[indices]
            quaternions = self.quaternions[indices]
            print(f"降采样后轨迹点数: {len(timestamps)}")
        else:
            timestamps = self.timestamps
            positions = self.positions
            quaternions = self.quaternions
        
        # 记录开始时间
        start_time = time.time()
        trajectory_start_time = timestamps[0]
        
        # 计算目标控制周期
        target_dt = 1.0 / target_freq
        
        # 计算原始轨迹的采样频率
        dt_list = np.diff(timestamps)
        avg_dt = np.mean(dt_list)
        original_freq = 1.0 / avg_dt if avg_dt > 0 else 100.0
        print(f"原始轨迹采样频率: {original_freq:.1f} Hz")
        print(f"控制周期: {target_dt*1000:.1f} ms\n")
        
        next_idx = 0
        last_time = time.time()
        
        try:
            while next_idx < len(timestamps):
                loop_start = time.time()
                
                # 当前应该执行到哪个轨迹点
                current_elapsed = (time.time() - start_time) * speed_factor
                target_time = trajectory_start_time + current_elapsed
                
                # 找到对应的轨迹索引
                while next_idx < len(timestamps) - 1 and timestamps[next_idx] < target_time:
                    next_idx += 1
                
                if next_idx >= len(timestamps):
                    break
                
                # 获取目标位姿
                target_pos = positions[next_idx]
                quat_xyzw = quaternions[next_idx]
                target_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
                
                # 应用低通滤波
                if enable_filter:
                    self.filtered_pos = (
                        self.position_filter_alpha * target_pos + 
                        (1 - self.position_filter_alpha) * self.filtered_pos
                    )
                    # 四元数球面插值（SLERP的简化版本）
                    self.filtered_quat = (
                        self.orientation_filter_alpha * target_quat + 
                        (1 - self.orientation_filter_alpha) * self.filtered_quat
                    )
                    # 重新归一化四元数
                    self.filtered_quat = self.filtered_quat / np.linalg.norm(self.filtered_quat)
                else:
                    self.filtered_pos = target_pos
                    self.filtered_quat = target_quat
                
                # 发送控制命令
                self.arm.set_end_effector_pose_quat_raw(
                    pos=self.filtered_pos,
                    quat=self.filtered_quat
                )
                
                # 打印进度
                if next_idx % 50 == 0:
                    progress = (next_idx + 1) / len(timestamps) * 100
                    actual_freq = 1.0 / (time.time() - last_time) if time.time() - last_time > 0 else 0
                    print(f"进度: {progress:.1f}% ({next_idx+1}/{len(timestamps)}) | "
                          f"实际频率: {actual_freq:.1f} Hz")
                    last_time = time.time()
                
                # 控制循环频率
                elapsed = time.time() - loop_start
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)
                
        except KeyboardInterrupt:
            print("\n用户中断复现")
            raise
        
        print("\n轨迹复现完成！")
    
    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")
        if self.enable_gravity:
            self.gravity_running = False
            if self.gravity_thread.is_alive():
                self.gravity_thread.join(timeout=1.0)
        self.arm.cleanup()
        print("资源清理完成")


def main():
    # 轨迹文件路径
    trajectory_file = "/home/ubuntu/FastUMI/startouch-v1/multi_sessions_20251223_100520/session_001/Merged_Trajectory/merged_trajectory.txt"
    
    print("=== 轨迹复现参数配置 ===")
    print("\n滤波强度 (0.1-0.9，越小越平滑但响应慢，推荐0.2-0.4):")
    filter_str = input("  位置滤波系数 (默认0.3): ")
    try:
        pos_filter = float(filter_str) if filter_str else 0.3
        pos_filter = max(0.1, min(0.9, pos_filter))
    except:
        pos_filter = 0.3
    
    ori_filter_str = input("  姿态滤波系数 (默认0.3): ")
    try:
        ori_filter = float(ori_filter_str) if ori_filter_str else 0.3
        ori_filter = max(0.1, min(0.9, ori_filter))
    except:
        ori_filter = 0.3
    
    # 创建轨迹播放器
    player = TrajectoryPlayer(
        trajectory_file=trajectory_file,
        can_interface="can0",
        enable_gravity=True,
        position_filter_alpha=pos_filter,
        orientation_filter_alpha=ori_filter
    )
    
    try:
        # 选择控制模式
        print("\n选择控制模式:")
        print("1. 位置控制模式（带规划）")
        print("2. 伺服控制模式（透传，推荐）")
        mode = input("请选择 (1/2): ")
        
        # 选择速度因子
        speed_str = input("\n请输入速度因子 (默认0.5，范围0.1-5.0，建议从慢速开始): ")
        try:
            speed_factor = float(speed_str) if speed_str else 0.5
            speed_factor = max(0.1, min(5.0, speed_factor))
        except:
            speed_factor = 0.5
        
        # 执行轨迹复现
        if mode == "1":
            player.play_trajectory_position_control(speed_factor=speed_factor)
        else:
            # 伺服控制模式的额外参数
            freq_str = input("\n控制频率 (Hz, 默认50，范围10-200): ")
            try:
                target_freq = float(freq_str) if freq_str else 50.0
                target_freq = max(10.0, min(200.0, target_freq))
            except:
                target_freq = 50.0
            
            enable_filter = input("启用低通滤波? (y/n，默认y): ").lower() != 'n'
            
            downsample_str = input("降采样因子 (1=全部点，2=每2个点取1个，默认2): ")
            try:
                downsample = int(downsample_str) if downsample_str else 2
                downsample = max(1, min(10, downsample))
            except:
                downsample = 2
            
            player.play_trajectory_servo_control(
                speed_factor=speed_factor,
                target_freq=target_freq,
                enable_filter=enable_filter,
                downsample=downsample
            )
            
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        player.cleanup()


if __name__ == "__main__":
    main()
