import sys
sys.path.append('./startouch-v1/interface_py')
import time
from startouchclass import SingleArm
import numpy as np
startouch = SingleArm(can_interface_="can0", gripper=True, enable_fd_=False)
joint_path = "./joint.txt"
width_path = "./gripper.txt"
j_data = []
w_data =[]
v_data = []
try:
    while(1):
        start = time.time()
        j = startouch.get_joint_positions()
        w = startouch.get_gripper_position()
        v = startouch.get_joint_velocities()
        # print(j,w)
        j_data.append(j)
        w_data.append(w)
        v_data.append(v)
        time.sleep(0.01)
except Exception as e:
    pass
finally:
    print("write")
    j_data = np.array(j_data)
    w_data = np.array(w_data)
    v_data = np.array(v_data)
    np.savetxt("joint.csv",j_data)
    np.savetxt("gripper.csv",w_data)
    np.savetxt("velocities.csv",v_data)
    
    
       