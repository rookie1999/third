import numpy as np
from typing import List, Tuple, Optional
from .piper_robot import PiperRobot
from scipy.spatial.transform import Rotation as R
import time
from .config import *
import threading
import utils
class PiperReplayer:
    def __init__(self,
                 robot,
                 T_base2local
                 ):
        self.robot = robot
        self.T_base2local=T_base2local
        self.target_pose=None

    def load_trajectory(
        self,
        traj_path: str,
        clamp_path: str
    ) :
        '''read traj(x,y,z,qx,qy,qz,qw) and clamp width'''
        try:
            self.raw_clamp = np.loadtxt(clamp_path)
            self.raw_pose = np.loadtxt(traj_path)
            self.pose_timestamps = self.raw_pose[:,0]
        except Exception as e:
            print(e)
    
    def transform_traj(self,T_base2local=None,interval=1)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''unit:mm'''
        self.target_pose = []    # argparser = argparse.ArgumentParser(description="Xarm6.")
    # # Required arguments
    # argparser.add_argument("robot_ip", default="192.168.1.208", help="IP address of the robot server")
    # # argparser.add_argument("local_ip", help="IP address of this PC")
    # # argparser.add_argument("frequency", type=int, help="Command frequency, 1 to 200 [Hz]")
    # # Optional arguments
    # # argparser.add_argument("--hold", action="store_true", help="Robot holds current joint positions, otherwise do a sine-sweep")
    # args = argparser.parse_args()
        self.target_clamp_width=[]
        self.interval = interval
        
        clamp_timestamps = self.raw_clamp[:,0]
        umi_clamp_widths = self.raw_clamp[:,-1]
        
        if T_base2local is None:
            T_base2local=self.T_base2local
        for i,(p, pose_ts) in enumerate(zip(self.raw_pose[:,1:],self.pose_timestamps)):
            if i%interval==0:
                idx = np.abs(clamp_timestamps - pose_ts).argmin()
                real_width = np.clip(umi_clamp_widths[idx].astype(int), 0, 88)
                self.target_clamp_width.append(real_width)
                self.target_pose.append(self.transform_to_base_quat(*p, T_base2local))
          
    def transform_to_base_quat(self,x, y, z, qx, qy, qz, qw, T_base_to_local):
        '''transform the pose of fastumi to robot base'''
        rotation_local = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T_local = np.eye(4)
        T_local[:3, :3] = rotation_local
        T_local[:3, 3] = [x, y, z]
        
        T_base_r = np.matmul(T_local[:3, :3] , T_base_to_local[:3, :3] )
        
        x_base, y_base, z_base = T_base_to_local[:3, 3] + T_local[:3, 3]
        rotation_base = R.from_matrix(T_base_r)
        roll_base, pitch_base, yaw_base = rotation_base.as_euler('xyz', degrees=False)
        return x_base, y_base, z_base,  roll_base, pitch_base, yaw_base

    def replay(self,speed_rate):
        print(f"开始轨迹复现: {len(self.raw_pose)} 个点, 预计时长: {self.pose_timestamps[-1] - self.pose_timestamps[0]:.2f} 秒")
        timestamps = self.pose_timestamps - self.pose_timestamps[0]
        start_time = time.time()
        timestamps = timestamps[0::self.interval]
        for i,ts in enumerate(timestamps/speed_rate) :
            now = time.time() - start_time
            delay = ts - now
            if delay < 0.01:
                print(f"[WARN]: timestamp[{i}] delayed (delay={delay:.3f}s)")
                delay=0
            timer_thread = threading.Timer(delay,
                                           lambda idx=i, pose=self.target_pose[i],gripper_cmd=self.target_clamp_width[i]:
                                                self.move_robot(pose,gripper_cmd)
                                                #print(f"STEP[{idx}]:move to {pose}")#self.robot.movp() 
                                           )
            timer_thread.daemon = True
            timer_thread.start()
    
    def move_robot(self,pose,gripper_cmd,speed=100):
        self.robot.movp(pose,unit="mrad",speed=speed)
        self.robot.move_gripper(gripper_cmd)

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



