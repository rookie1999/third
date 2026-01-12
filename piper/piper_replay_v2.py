from piper_robot import PiperRobot
from piper_replayer import PiperReplayer
from config import *
import utils
import os
import time
if __name__=="__main__":
    # init robot
    try:
        robot = PiperRobot(can_port="can0")
        robot.enable()
    except Exception as e:
        print(e)
    # init replayer
    piper_replayer = PiperReplayer(robot=robot, T_base2local = T_base2local)

    multi_session_dir = utils.select_multi_sessions_dir(base_path=DATA_ROOT)
    

    selected_session = utils.select_session_subdir(multi_session_dir)
    
    clamp_path = os.path.join(selected_session,"Clamp_Data","clamp_data_tum.txt")    #"./session_001/Clamp_Data/clamp_data_tum.txt"
    traj_path = os.path.join(selected_session,"Merged_Trajectory","merged_trajectory.txt")#"./session_001/Merged_Trajectory/merged_trajectory.txt"\

    print("clamp path:",clamp_path)
    print("traj path:",traj_path)

    piper_replayer.load_trajectory(traj_path,clamp_path)
    
    # piper_replayer.smooth_trajectory(
    #     dt_est=0.01,          # ~60 Hz
    #     pos_std_meas=2.0,      # measurement noise: 1 mm  
    #     pos_std_acc=5.0,      # allow some acceleration
    #     ori_alpha=0.3          # moderate smoothing
    # )
    piper_replayer.transform_traj(T_base2local,interval=1)

    piper_replayer.replay(speed_rate=speed_rate)

    time.sleep((piper_replayer.pose_timestamps[-1] - piper_replayer.pose_timestamps[0])/speed_rate+1)
    # robot.movp([56.127, 0.0, 213.266, 0.0, 84.999, 0.0],unit="mrad",speed=30)
    robot.go_zero(speed=10)


