import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation as R
import time
import threading
import utils.utils as utils
from collections import defaultdict

class TrajReplayer:
    def __init__(self,
                 robot_sdk=None,#move and gripper function caller
                 ):
        self._robot_sdk = robot_sdk

    # def load_trajectory(
    #     self,
    #     traj_path: str,
    #     clamp_path: str
    # ) :
    #     '''read traj(x,y,z,qx,qy,qz,qw) and clamp width'''
    #     try:
    #         self.raw_clamp = np.loadtxt(clamp_path)
    #         self.raw_pose = np.loadtxt(traj_path)
    #         self.pose_timestamps = self.raw_pose[:,0]
    #     except Exception as e:
    #         print(e)
    
          
    # def transform_to_base_quat(self,x, y, z, qx, qy, qz, qw, T_base_to_local):
    #     '''transform the pose of fastumi to robot base'''
    #     rotation_local = R.from_quat([qx, qy, qz, qw]).as_matrix()
    #     T_local = np.eye(4)
    #     T_local[:3, :3] = rotation_local
    #     T_local[:3, 3] = [x, y, z]
        
    #     T_base_r = np.matmul(T_local[:3, :3] , T_base_to_local[:3, :3] )
        
    #     x_base, y_base, z_base = T_base_to_local[:3, 3] + T_local[:3, 3]
    #     rotation_base = R.from_matrix(T_base_r)
    #     roll_base, pitch_base, yaw_base = rotation_base.as_euler('xyz', degrees=False)
    #     return x_base, y_base, z_base,  roll_base, pitch_base, yaw_base

    def replay(self,speed_rate,target_pose,target_clamp_width,pose_timestamps):
        if self._robot_sdk is None:
            return 
        print(f"开始轨迹复现: {len(target_pose)} 个点, 预计时长: {pose_timestamps[-1] - pose_timestamps[0]:.2f} 秒")
        timestamps = pose_timestamps - pose_timestamps[0]
        # 对时间戳下采样
        ts_list = (pose_timestamps - pose_timestamps[0]) / speed_rate
        start_time = time.time()
        for i, target_ts in enumerate(ts_list):
            now = time.time() - start_time
            sleep_time = target_ts - now
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"[WARN] Step {i} is behind schedule by {-sleep_time:.3f}s")

            # 执行动作
            pose = target_pose[i]
            width = target_clamp_width[i]
            self._robot_sdk(pose, width, speed=100)
            # robot_sdk.movp(pose, unit="mrad",speed=100)
            # robot_sdk.move_gripper(width)
            
    def replay2(self, speed_rate, target_pose, target_clamp_width, pose_timestamps, duration=None):
        if self._robot_sdk is None:
            raise ValueError("robot_sdk not set!")

        # 原始时间范围
        t0 = pose_timestamps[0]
        t1 = pose_timestamps[-1]
        total_duration = (t1 - t0) / speed_rate

        # 转为相对时间（从0开始）
        rel_timestamps = pose_timestamps - t0  # [0, ..., T]

        # 转为 NumPy 数组便于插值
        poses = np.array(target_pose)          # shape: (N, 6)
        widths = np.array(target_clamp_width)  # shape: (N,)

        print(f"开始时间对齐轨迹复现: {len(target_pose)} 个点, 总时长: {total_duration:.2f} 秒")
        self._robot_sdk(poses[0], widths[0], speed=100)
        
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            current_playback_time = elapsed * speed_rate  # 在原始时间轴上的位置

            # 超出轨迹范围则结束
            if current_playback_time > (t1 - t0):
                print("轨迹回放完成")
                break
            if duration and elapsed > duration:
                print("达到指定回放时长，退出")
                break

            # 找到当前 playback_time 对应的 pose（线性插值）
            idx = np.searchsorted(rel_timestamps, current_playback_time)
            if idx == 0:
                pose_interp = poses[0]
                width_interp = widths[0]
            elif idx >= len(rel_timestamps):
                pose_interp = poses[-1]
                width_interp = widths[-1]
            else:
                t_left = rel_timestamps[idx - 1]
                t_right = rel_timestamps[idx]
                ratio = (current_playback_time - t_left) / (t_right - t_left)

                pose_interp = poses[idx - 1] + ratio * (poses[idx] - poses[idx - 1])
                width_interp = int(widths[idx - 1] + ratio * (widths[idx] - widths[idx - 1]))

            # 发送目标给机器人（非阻塞）
            try:
                self.threading._robot_sdk(pose_interp, width_interp, speed=100)
            except Exception as e:
                print(f"[ERROR] SDK error at t={current_playback_time:.3f}s: {e}")

            # 控制循环频率（避免CPU占满），例如 100 Hz
            time.sleep(0.01)  # 10ms ≈ 100Hz
    
    def replay3(self,speed_rate,target_pose,target_clamp_width,pose_timestamps,interval=1):
            # 下采样：每隔 `interval` 个点取一个
        sampled_indices = slice(0, None, interval)
        sampled_timestamps = pose_timestamps[sampled_indices]
        sampled_pose = target_pose[sampled_indices]
        sampled_clamp = target_clamp_width[sampled_indices]

        # 转为相对时间（从0开始）
        timestamps = sampled_timestamps - sampled_timestamps[0]
        start_time = time.time()
        
        
        print(f"开始轨迹复现 (interval={interval}): {len(sampled_pose)} 个点, "
            f"预计时长: {timestamps[-1] / speed_rate:.2f} 秒")
        inter_w = sampled_clamp[0]
        for i, ts in enumerate(timestamps / speed_rate):
            now = time.time() - start_time
            delay = ts - now
            if delay < 0.01:
                print(f"[WARN]: timestamp[{i}] delayed (delay={delay:.3f}s)")   
                delay = 0
            if i % 20 == 0 :
                inter_w = sampled_clamp[i]
            timer_thread = threading.Timer(
                delay,
                lambda p=sampled_pose[i], w=inter_w: self._robot_sdk(p, w)
            )
            timer_thread.daemon = True
            timer_thread.start()

    def move_robot(self,robot_sdk,pose,gripper_cmd,speed=100):
        if robot_sdk is None:
            raise ValueError("robot sdk call function not set")
        else:
            robot_sdk(pose, gripper_cmd, speed)
        # self.robot.movp(pose,unit="mrad",speed=speed)
        # self.robot.move_gripper(gripper_cmd)

    def smooth_trajectory(self, dt_est: float = 0.01, pos_std_meas: float = 0.5, pos_std_acc: float = 2.0,
                          ori_alpha: float = 0.2):
        """
        Apply Kalman filter to position (x,y,z) and EMA smoothing to orientation.
        Assumes self.raw_pose = [ts, x, y, z, qx, qy, qz, qw]

        Parameters:
            dt_est: estimated time step (used for Kalman F matrix)
            pos_std_meas: measurement noise std for position (in mm)
            pos_std_acc: process noise std for acceleration (in mm/s^2)
            ori_alpha: EMA factor for orientation (0 < alpha <= 1, smaller = smoother)
        """
        if self.raw_pose is None:
            raise ValueError("Call load_trajectory() first!")

        N = len(self.raw_pose)
        smoothed = np.copy(self.raw_pose)

        # === Filter position with Kalman ===
        kf = utils.create_kalman_3d(dt=dt_est, std_acc=pos_std_acc, std_meas=pos_std_meas)
        # Initialize with first measurement
        kf.x = np.zeros(6)
        kf.x[:3] = self.raw_pose[0, 1:4]
        smoothed[0, 1:4] = kf.x[:3]

        for i in range(1, N):
            dt = self.raw_pose[i, 0] - self.raw_pose[i-1, 0]
            if dt <= 0:
                dt = dt_est
            # Update F matrix with actual dt
            kf.F[0, 3] = dt
            kf.F[1, 4] = dt
            kf.F[2, 5] = dt
            kf.predict()
            kf.update(self.raw_pose[i, 1:4])
            smoothed[i, 1:4] = kf.x[:3]

        # === Smooth orientation with EMA (quaternion-aware) ===
        q_prev = smoothed[0, 4:8]
        q_prev /= np.linalg.norm(q_prev)
        smoothed[0, 4:8] = q_prev

        for i in range(1, N):
            q_curr = self.raw_pose[i, 4:8]
            q_curr /= np.linalg.norm(q_curr)
            # Ensure shortest path (quaternion double cover)
            if np.dot(q_prev, q_curr) < 0:
                q_curr = -q_curr
            # Spherical linear interpolation (approx. EMA on sphere)
            q_smooth = (1 - ori_alpha) * q_prev + ori_alpha * q_curr
            q_smooth /= np.linalg.norm(q_smooth)
            smoothed[i, 4:8] = q_smooth
            q_prev = q_smooth
        q_prev /= np.linalg.norm(q_prev)
        smoothed[0, 4:8] = q_prev

        for i in range(1, N):
                q_curr = self.raw_pose[i, 4:8]
                q_curr /= np.linalg.norm(q_curr)
                # Ensure shortest path (quaternion double cover)
                if np.dot(q_prev, q_curr) < 0:
                    q_curr = -q_curr
                # Spherical linear interpolation (approx. EMA on sphere)
                q_smooth = (1 - ori_alpha) * q_prev + ori_alpha * q_curr
                q_smooth /= np.linalg.norm(q_smooth)
                smoothed[i, 4:8] = q_smooth
                q_prev = q_smooth

        self.raw_pose = smoothed
        print(f"[INFO] Trajectory smoothed with Kalman (pos) and EMA (ori).")



