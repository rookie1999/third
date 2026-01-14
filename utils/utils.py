from pathlib import Path
import re
from scipy.spatial.transform import Rotation as R
from datetime import datetime

def parse_timestamp_from_name(name: str):
    match = re.search(r"multi_sessions_(\d{8})_(\d{6})", name)
    if match:
        date_str, time_str = match.groups()
        try:
            return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return None

def select_multi_sessions_dir(base_path="~/data_collection_11.14/data_collector_opt"):
    base = Path(base_path).expanduser()
    dirs = [p for p in base.glob("multi_sessions*") if p.is_dir()]
    if not dirs:
        raise FileNotFoundError("未找到任何 'multi_sessions*' 目录。")

    def sort_key(p):
        dt = parse_timestamp_from_name(p.name)
        return (dt or datetime.min, p.name)
    
    sorted_dirs = sorted(dirs, key=sort_key, reverse=True)

    print("\n📁 可用的多会话记录（最新在前）：")
    print("─" * 50)
    for i, d in enumerate(sorted_dirs, 1):
        dt = parse_timestamp_from_name(d.name)
        time_label = dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "???"
        print(f"[{i:2}] {time_label}  →  {d.name}")
    print("─" * 50)

    while True:
        try:
            choice = input("\n➤ 请输入多会话编号（直接回车 = 使用最新会话）：").strip()
            if not choice:
                selected_root = sorted_dirs[0]
                print(f"→ 使用最新会话根目录：{selected_root.name}")
                return selected_root
            idx = int(choice)
            if 1 <= idx <= len(sorted_dirs):
                selected_root = sorted_dirs[idx - 1]
                print(f"→ 已选择：{selected_root.name}")
                return selected_root
            else:
                print(f"⚠️  编号无效，请输入 1–{len(sorted_dirs)} 之间的数字。")
        except ValueError:
            print("⚠️  请输入有效数字，或直接回车选择最新会话。")

def select_session_subdir(multi_session_root: Path):
    """在 multi_sessions_*/ 下选择 session_XXX 子目录"""
    session_dirs = sorted(
        [p for p in multi_session_root.glob("session_*") if p.is_dir()],
        key=lambda p: p.name  # session_001 < session_002 < ... < session_010
    )
    
    if not session_dirs:
        raise FileNotFoundError(f"在 '{multi_session_root}' 下未找到任何 'session_*' 子目录。")

    print(f"\n📂 当前多会话：{multi_session_root.name}")
    print("📁 可用的子会话（按编号升序排列）：")
    print("─" * 45)
    for i, d in enumerate(session_dirs, 1):
        print(f"[{i:2}] {d.name}")
    print("─" * 45)

    while True:
        try:
            choice = input("\n➤ 请输入子会话编号（直接回车 = 选择【最新】会话）：").strip()
            if not choice:
                # ✅ NOW DEFAULTS TO LATEST (last in sorted list)
                selected = session_dirs[-1]
                print(f"→ 使用最新子会话：{selected.name}")
                return selected
            idx = int(choice)
            if 1 <= idx <= len(session_dirs):
                selected = session_dirs[idx - 1]
                print(f"→ 已选择子会话：{selected.name}")
                return selected
            else:
                print(f"⚠️  编号无效，请输入 1–{len(session_dirs)} 之间的数字。")
        except ValueError:
            print("⚠️  请输入有效数字，或直接回车选择最新会话。")

def transform_to_base_quat(x, y, z, qx, qy, qz, qw, T_base_to_local,dgrees=False):
    '''transform the pose of fastumi to robot base
        unit:m
    '''
    rotation_local = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T_local = np.eye(4)
    T_local[:3, :3] = rotation_local
    T_local[:3, 3] = [x, y, z]
    
    T_base_r = np.matmul(T_local[:3, :3] , T_base_to_local[:3, :3] )
    
    x_base, y_base, z_base = T_base_to_local[:3, 3] + T_local[:3, 3]
    rotation_base = R.from_matrix(T_base_r)
    roll_base, pitch_base, yaw_base = rotation_base.as_euler('xyz', degrees=dgrees)
    return x_base, y_base, z_base,  roll_base, pitch_base, yaw_base





import numpy as np
from scipy.spatial.transform import Rotation as R
def load_trajectory(
    traj_path: str,
    clamp_path: str
) :
    '''read traj(x,y,z,qx,qy,qz,qw) and clamp width'''
    try:
        raw_clamp = np.loadtxt(clamp_path)
        raw_pose = np.loadtxt(traj_path)
        pose_timestamps = raw_pose[:,0]
        raw_pose = raw_pose[:,1:]
    except Exception as e:
        print(e)
    
    return raw_pose, raw_clamp, pose_timestamps

def transform_traj(raw_pose, raw_clamp, pose_timestamps, T_base2local):
    '''unit:mm'''
    target_pose = []   
    target_clamp_width=[]
    
    clamp_timestamps = raw_clamp[:,0]
    umi_clamp_widths = raw_clamp[:,-1]

    for i,(p, pose_ts) in enumerate(zip(raw_pose,pose_timestamps)):

        idx = np.abs(clamp_timestamps - pose_ts).argmin()
        real_width = np.clip(umi_clamp_widths[idx], 0, 88)
        target_clamp_width.append(real_width)
        target_pose.append(transform_to_base_quat(*p, T_base2local))
    return target_pose, target_clamp_width

def transform_to_base_quat(x, y, z, qx, qy, qz, qw, T_base_to_local,dgrees=False):
    '''transform the pose of fastumi to robot base
        unit:m
    '''
    rotation_local = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T_local = np.eye(4)
    T_local[:3, :3] = rotation_local
    T_local[:3, 3] = [x, y, z]
    
    T_base_r = np.matmul(T_local[:3, :3] , T_base_to_local[:3, :3] )
    
    x_base, y_base, z_base = T_base_to_local[:3, 3] + T_local[:3, 3]
    rotation_base = R.from_matrix(T_base_r)
    roll_base, pitch_base, yaw_base = rotation_base.as_euler('xyz', degrees=dgrees)
    return x_base, y_base, z_base,  roll_base, pitch_base, yaw_base


# ===== Usage Example =====
try:
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise
except Exception as e:
    pass
import numpy as np

def create_kalman_3d(dt=0.01, std_acc=1.0, std_meas=0.1):
    """
    Create a 3D Kalman filter with constant velocity model.
    State: [x, y, z, vx, vy, vz]
    """
    dt = float(dt)
    kf = KalmanFilter(dim_x=6, dim_z=3)
    kf.F = np.array([[1, 0, 0, dt, 0, 0],
                     [0, 1, 0, 0, dt, 0],
                     [0, 0, 1, 0, 0, dt],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]], dtype=float)
    
    kf.H = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0]], dtype=float)

    # Process noise (acceleration assumed)
    q = Q_discrete_white_noise(dim=3, dt=dt, var=std_acc**2)
    kf.Q[0:3, 0:3] = q
    kf.Q[3:6, 3:6] = q / 10  # lower noise on velocity

    # Measurement noise
    kf.R = np.eye(3) * (std_meas**2)

    # Initial state uncertainty
    kf.P *= 1000.

    return kf

def transform_vive_to_gripper(qpos):
    """
    完整转换链：VIVE → VIVE_FLAT → XV → Gripper
    
    参数：
        qpos: list or array, [x, y, z, qx, qy, qz, qw]
    
    返回：
        list, [x, y, z, qx, qy, qz, qw] in Gripper coordinate
    """
    # 定义 VIVE 到 VIVE_FLAT 的变换
    TRANSFORMATION = np.eye(4)
    TRANSFORMATION[:3, :3] = R.from_euler('xyz', [30, 0, 0], degrees=True).as_matrix()
    
    # VIVE → VIVE_FLAT
    qpos = VIVE2VIVE_FLAT(qpos, TRANSFORMATION)
    
    # VIVE_FLAT → XV
    qpos = VIVEFLAT2XV(qpos)
    
    # XV → Gripper
    qpos = XV2Gripper(qpos)
    
    return qpos
def VIVE2VIVE_FLAT(qpos, TRANSFORMATION):
    """
    qpos : x y z qx qy qz qw
    TRANSFORMATION: VIVE iv VIVE flat
    """
    M = qpos2mat(qpos)
    M_flat = np.dot(np.dot(TRANSFORMATION, M), np.linalg.inv(TRANSFORMATION))
    return mat2qpos(M_flat)
def XV2Gripper(qpos):
    TRANSFORMATION_MAT = np.array([
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    
    cur_qpos = np.array(qpos)
    cur_mat = qpos2mat(cur_qpos)
    
    #####################################
    left_right_offset = 0.02268
    front_back_offset = 0.08745
    up_down_offset = 0.09240
    #####################################
    cur_qpos[0] -= left_right_offset
    cur_qpos[1] -= up_down_offset
    cur_qpos[2] -= front_back_offset
    
    ori = cur_mat[:3, :3]
    cur_qpos[:3] += ori[:, 0] * left_right_offset
    cur_qpos[:3] += ori[:, 1] * up_down_offset
    cur_qpos[:3] += ori[:, 2] * front_back_offset
    
    cur_mat = qpos2mat(cur_qpos)
    xv_mat_in_gripper =  np.dot(np.dot(TRANSFORMATION_MAT, cur_mat), np.linalg.inv(TRANSFORMATION_MAT))
    return mat2qpos(xv_mat_in_gripper)

def qpos2mat(qpos):
    x, y, z, qx, qy, qz, qw = qpos
    R_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
    t = np.array([x, y, z])
    M = np.eye(4)
    M[:3, :3] = R_mat
    M[:3, 3] = t
    return M

def mat2qpos(M):
    t = M[:3, 3]
    R_mat = M[:3, :3]
    qx, qy, qz, qw = R.from_matrix(R_mat).as_quat()
    return [t[0], t[1], t[2], qx, qy, qz, qw]

def VIVEFLAT2XV(qpos):
    
    TRANSFORMATION_MAT = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    
    cur_qpos = np.array(qpos)
    cur_mat = qpos2mat(cur_qpos)
    #############################
    left_right_offset = 0.02220 # original 0.02220
    front_back_offset = -0.04020 # original -0.03820
    up_down_offset = -0.01003 # original -0.01003
    #############################
    cur_qpos[0] -= left_right_offset
    cur_qpos[1] -= front_back_offset
    cur_qpos[2] += up_down_offset
    
    ori = cur_mat[:3, :3]
    cur_qpos[:3] += ori[:, 0] * left_right_offset
    cur_qpos[:3] += ori[:, 1] * front_back_offset
    cur_qpos[:3] -= ori[:, 2] * up_down_offset
    
    cur_mat = qpos2mat(cur_qpos)
    
    vive_mat_in_xv =  np.dot(np.dot(TRANSFORMATION_MAT, cur_mat), np.linalg.inv(TRANSFORMATION_MAT)) 
    return mat2qpos(vive_mat_in_xv)

# ===== Example Usage of Session Selection =====
if __name__ == "__main__":
    try:
        # Step 1: Choose multi_sessions_* root
        multi_root = select_multi_sessions_dir()
        
        # Step 2: Choose session_XXX under it
        selected_session = select_session_subdir(multi_root)
        
        print(f"\n✅ 最终选择路径：\n{selected_session}")
        # → You can now load data from selected_session (e.g., poses, timestamps)
        
    except FileNotFoundError as e:
        print(f"❌ 错误：{e}")