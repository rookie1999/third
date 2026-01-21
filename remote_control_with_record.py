import subprocess
import time

import numpy as np
import rospy
import yaml
from xarm.wrapper import XArmAPI

from piper import PiperRobot
from utils import make_piper_sdk, make_xarm_sdk
from utils import make_startouch_eef_rpy_sdk
from utils.camera import RealSenseCamera


def speak_feedback(text):
    """
    非阻塞语音播报函数
    使用系统底层的 espeak 命令，不会卡住主线程
    """
    try:
        # 使用 Popen 启动子进程，它是非阻塞的（立即返回）
        # '-v', 'zh' 尝试使用中文发音（如果系统支持），如果听不懂中文可以去掉这部分或改为 'en'
        # 你可以调整参数，比如 '-s', '160' 来调整语速
        subprocess.Popen(['espeak', '-v', 'zh', '-s', '160', text],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[TTS Error] 语音调用失败: {e}")

if __name__ == "__main__":
    
    #load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    rospy.init_node('piper_subscriber', anonymous=True)

    # initialize camera
    print("Initializing Camera...")
    camera = RealSenseCamera(width=640, height=480, fps=30)
    time.sleep(1)

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
    # xarm7/xv_sdk/250801DR48FP25002658/clamp/Data
        if config["Xarm7"]["enable"]:
            xarm7 = XArmAPI(config["Xarm7"]["robot_ip"])
            xarm7.motion_enable(enable=True)
            xarm7.set_mode(0)
            xarm7.set_state(0)
            xarm7.set_servo_angle(angle=config["Xarm7"]["initial_joints"], is_radian=False ,wait=True)   
            xarm7.set_mode(1)#servo mode
            xarm7.set_state(0)
            xarm_sdk = make_xarm_sdk(xarm7)
            # time.sleep(0.5)
            # xarm7.set_tgpio_modbus_baudrate(115200)
            # code = xarm7.robotiq_reset()
            # code = xarm7.robotiq_set_activate(True)
            time.sleep(2.0)  
        if config["StarTouch"]["enable"]:
            import sys
            sys.path.append('/home/lumos/replay_remote_ctrl/startouch-v1/interface_py')
            from startouchclass import SingleArm
            startouch = SingleArm(can_interface_=config["StarTouch"]["can_port"], gripper=True, enable_fd_=False)
            startouch_sdk = make_startouch_eef_rpy_sdk(startouch)
            startouch.set_joint(config["StarTouch"]["initial_joints"], tf=3)
            # print('*****************', startouch.get_joint_positions())
            time.sleep(1.0)
    except Exception as e:
        print(e)
    

    #init and start remote control nodes
    # piper_node = RemoteSub(piper_sdk,T_base2local=np.array(config["Piper"]["T_base2local"]),pose_topic=config["pose_topic"],clamp_topic=config["clamp_topic"])
    # xarm_node = RemoteSub(xarm_sdk,T_base2local=np.array(config["Xarm7"]["T_base2local"]),pose_topic=config["pose_topic"],clamp_topic=config["clamp_topic"])
    remote_node = RemoteSubv3(startouch_sdk,
                               T_base2local=np.array(config["StarTouch"]["T_base2local"]),
                               pose_topic=config["pose_topic"],
                               clamp_topic=config["clamp_topic"],
                               arm_instance=startouch,
                               camera_interface=camera
                               )
    # piper_node.start()
    # xarm_node.start()
    remote_node.start()
    # try:
    #     rospy.spin()
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     # piper_node.stop()
    #     # xarm_node.stop()
    #     remote_node.stop()

    # 4. 键盘控制循环 (替代 rospy.spin)
    print("========================================")
    print("Remote Teleop Ready.")
    print(" [s] Start Recording Episode")
    print(" [e] End Recording & Save Episode")
    print(" [x] Abort/Discard Current Recording")
    print(" [q] Quit")
    print("========================================")
    try:
        while not rospy.is_shutdown():
            cmd = input("Enter command: ").strip().lower()

            if cmd == 's':
                if not remote_node.is_recording:
                    print(f"Episode {remote_node.episode_idx} Recording Started...")
                    speak_feedback("Starting recording")
                    remote_node.start_recording()
                else:
                    print("Already recording!")
            elif cmd == 'e':
                if remote_node.is_recording:
                    remote_node.stop_recording_and_save()
                    print(f"Episode Saved. Next index: {remote_node.episode_idx}")
                    speak_feedback("Finish recording and save.")
                else:
                    print("Not currently recording.")
            elif cmd == 'x':
                if remote_node.is_recording:
                    remote_node.abort_recording()
                    print("Current recording discarded (NOT saved).")
                    speak_feedback("Recording aborted")
                else:
                    print("Nothing is being recorded.")
            elif cmd == 'q':
                print("Quitting...")
                speak_feedback("Stop Recording. Exiting.")
                break
            else:
                print("Unknown command.")
    except KeyboardInterrupt:
        pass
    finally:
        remote_node.stop()
        camera.stop()
        # xarm7.disconnect() # 如果需要
        print("Shutdown complete.")