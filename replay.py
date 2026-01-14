import yaml
from piper import PiperRobot
from RoboticsToolBox import Bestman_Real_Xarm6
from utils import TrajReplayer
from utils import select_multi_sessions_dir, select_session_subdir,load_trajectory,transform_traj
from utils import make_piper_sdk, make_xarm_sdk,make_startouch_sdk
import os
import numpy as np
import time
from xarm.wrapper import XArmAPI

import sys
sys.path.append('/home/lumos/sync_replay_remote_ctrl/startouch-v1/interface_py')
try:
    from startouchclass import SingleArm
except Exception as e:
    pass
if __name__ == "__main__":  
    # Load configuration
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    #initialize Robots
    piper_sdk = None
    xarm_sdk = None
    startouch_sdk = None
    try:
        #piper
        if config["Piper"]["enable"]:
            piper = PiperRobot(can_port=config["Piper"]["can_port"])
            piper.enable()
            piper.go_zero(speed=10)
            piper.movj(config["Piper"]["initial_joints"],unit='deg')
            piper_sdk = make_piper_sdk(piper)
        # xarm7
        if config["Xarm7"]["enable"]:
            xarm7 = XArmAPI(config["Xarm7"]["robot_ip"])
            xarm7.set_mode(0)
            xarm7.set_state(0)
            xarm7.set_servo_angle(angle=config["Xarm7"]["initial_joints"], is_radian=False ,wait=True)   
            xarm7.set_mode(1)#servo mode
            xarm7.set_state(0)
            xarm_sdk = make_xarm_sdk(xarm7)
            time.sleep(0.5)
            xarm7.set_tgpio_modbus_baudrate(115200)
            code = xarm7.robotiq_reset()
            code = xarm7.robotiq_set_activate(True)
            time.sleep(2.0)  
        # StarTouch
        if config["StarTouch"]["enable"]:
            startouch = SingleArm(can_interface_=config["StarTouch"]["can_port"], gripper=True, enable_fd_=False)
            startouch_sdk = make_startouch_sdk(startouch)
            startouch.set_joint(config["StarTouch"]["initial_joints"], tf=3)
            time.sleep(1.0)
    except Exception as e:
        print(e)

    
    
    # replayer initialization
    piper_replayer = TrajReplayer(piper_sdk)
    xarm7_replayer = TrajReplayer(xarm_sdk)
    startouch_replayer = TrajReplayer(startouch_sdk)

    # Select session
    multi_session_dir = select_multi_sessions_dir(base_path=config["DATA_ROOT"])
    selected_session = select_session_subdir(multi_session_dir)
    clamp_path = os.path.join(selected_session,"Clamp_Data","clamp_data_tum.txt")    #"./session_001/Clamp_Data/clamp_data_tum.txt"
    traj_path = os.path.join(selected_session,"Merged_Trajectory","merged_trajectory.txt")#"./session_001/Merged_Trajectory/merged_trajectory.txt"\

    raw_pose, raw_clamp, pose_timestamps = load_trajectory(traj_path, clamp_path)

    # piper_pose, piper_clamp_width = transform_traj(raw_pose, raw_clamp, pose_timestamps, np.array(config["Piper"]["T_base2local"]))
    xarm_pose, xarm_clamp_width = transform_traj(raw_pose, raw_clamp, pose_timestamps, np.array(config["Xarm7"]["T_base2local"]))
    # startouch_pose, startouch_clamp_width = transform_traj(raw_pose, raw_clamp, pose_timestamps, np.array(config["StarTouch"]["T_base2local"]))
    # replay trajectory get_joint_positions

    speed_rate = config["speed_rate"]
    time.sleep(1)
    try:
        if config["Piper"]["enable"]:
            piper_replayer.replay3(speed_rate, piper_pose, piper_clamp_width, pose_timestamps,interval=1)
        if config["Xarm7"]["enable"]:
            xarm7_replayer.replay3(speed_rate, xarm_pose, xarm_clamp_width, pose_timestamps,interval=2)
        if config["StarTouch"]["enable"]:
            startouch_replayer.replay3(speed_rate, startouch_pose, startouch_clamp_width, pose_timestamps,interval=2)
        
        while(1):
            time.sleep(0.5)
    except Exception as e:
        print(e)
    finally:
        if config["StarTouch"]["enable"]:
            startouch.go_home()
        if config["Piper"]["enable"]:
            piper.go_zero(speed=10)


