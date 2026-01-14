from utils import RemoteSub
import rospy
import time
from piper import PiperRobot
from utils import make_piper_sdk,make_xarm_sdk, make_startouch_sdk
import yaml
from xarm.wrapper import XArmAPI
import numpy as np
import threading
import sys
sys.path.append('/home/lumos/single_replay/startouch-v1/interface_py')
from startouchclass import SingleArm
if __name__ == "__main__":
    
    #load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    rospy.init_node('piper_subscriber', anonymous=True)
    #initialize Robots
    piper_sdk = None
    xarm_sdk = None
    startouch_sdk = None 
    try:
    # piper
        if config["Piper"]["enable"]:
            piper = PiperRobot(can_port=config["Piper"]["can_port"])
            piper.enable()
            piper.go_zero(speed=10)
            piper_sdk = make_piper_sdk(piper)
    # xarm7
        if config["Xarm7"]["enable"]:
            xarm7 = XArmAPI(config["Xarm7"]["robot_ip"])
            xarm7.motion_enable(enable=True)
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
        if config["StarTouch"]["enable"]:
            startouch = SingleArm(can_interface_=config["StarTouch"]["can_port"], gripper=True, enable_fd_=False)
            startouch_sdk = make_startouch_sdk(startouch)
            startouch.set_joint(config["StarTouch"]["initial_joints"], tf=3)
            time.sleep(1.0)
    except Exception as e:
        print(e)

    #init and start remote control nodes
    # piper_node = RemoteSub(piper_sdk,T_base2local=np.array(config["Piper"]["T_base2local"]),pose_topic="/vive/LHR_339B545E/pose",clamp_topic="/xv_sdk/250801DR48FP25002267/clamp/Data")
    # xarm_node = RemoteSub(xarm_sdk,T_base2local=np.array(config["Xarm7"]["T_base2local"]),pose_topic="/vive/LHR_339B545E/pose",clamp_topic="/xv_sdk/250801DR48FP25002267/clamp/Data")
    startouch_node = RemoteSub(startouch_sdk,T_base2local=np.array(config["StarTouch"]["T_base2local"]),pose_topic="/vive/LHR_5A5B370E/pose",clamp_topic="/xv_sdk/250801DR48FP25002658/clamp/Data")
    # piper_node.start()
    # xarm_node.start()
    startouch_node.start()


    try:
       
        j_data = []
        w_data =[]
        v_data = []
        rate = rospy.Rate(30)
        while(not rospy.is_shutdown()):
            start = time.time()
            j = startouch.get_joint_positions()
            w = startouch.get_gripper_position()
            v = startouch.get_joint_velocities()
            # print(j,w)
            j_data.append(j)
            w_data.append(w)
            v_data.append(v)
            rate.sleep()
        # rospy.spin()  
            
    except KeyboardInterrupt:
        pass
    finally:
        # piper_node.stop()
        # xarm_node.stop()
        startouch_node.stop()
    
        print("write")
        j_data = np.array(j_data)
        w_data = np.array(w_data)
        v_data = np.array(v_data)
        np.savetxt("joint.csv",j_data)
        np.savetxt("gripper.csv",w_data)
        np.savetxt("vel.csv",v_data)