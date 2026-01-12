import sys
sys.path.append('./startouch-v1/interface_py')
import time
from startouchclass import SingleArm
import numpy as np
startouch = SingleArm(can_interface_="can0", gripper=True, enable_fd_=False)

joints = np.loadtxt("joint.csv")
gripper = np.loadtxt("gripper.csv")
vel = np.loadtxt("vel.csv")
# j = joints[550]
# v = vel[550]
startouch.set_joint(joints[0], tf=3.0)

# time.sleep(2.0)
# startouch.set_joint_raw(j,v)
# quit()
# time.sleep(3.0)
for i in range(joints.shape[0]):
    j = startouch.get_joint_positions()
    print(j)
    j = joints[i]
    g = gripper[i]
    v = vel[i]
    # print(j,type)
    # break
    startouch.set_joint_raw(j,v)
    startouch.setGripperPosition_raw(g/2)
    # if(i==550):
    #     a=input("pause")
    time.sleep(0.033)

time.sleep(1.0)
startouch.go_home() 