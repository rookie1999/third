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