import sys
import numpy as np
root_dir = "../"
sys.path.append(root_dir)
from arm import Arm
from dmp import DMP
img_num = 5
image_dir = f'{root_dir}/images/image{img_num}/'
refer_tjt_path = f'{image_dir}/dmp/refer_tjt.csv'

cfg_dir = f'{root_dir}/cfg/'
cam2base_H_path = f'{cfg_dir}/cam2base_H_ANDREFF_2.csv'
target2base_xyzrxryrz = np.array([0.74481064,-0.24952034,-0.15857617,-1.52667011,-1.55602563,-1.78504951])
#target2base_xyzrxryrz[0]-= 0.05 # x axis

## init arm
arm = Arm('192.168.10.19',8080,cam2base_H_path=cam2base_H_path,if_gripper=False,if_monitor=False)# 18 for left 19 for right
# print(arm)
arm.go_home()
arm.change_tool_frame(tool_name='dh3')
# arm.control_gripper(open_value=1000)
arm.move_p(pos=target2base_xyzrxryrz,vel=10)

## init dmp
dmp = DMP(refer_tjt_path)

## dmp
new_tjt = dmp.gen_new_tjt(initial_pos=arm.get_p(),goal_pos=target2base_xyzrxryrz.tolist(),show=False)
print(f'==========\nDMP ...')
poses = dmp.get_poses(tjt=new_tjt,step=50)
print(f'DMP Done\n==========')

## move to handle
print(f'==========\nMoving to Handle Through DMP ...')
# arm.move_poses(poses,vel=10,trajectory_connect=1,frame_name='dh3')
print(f'Motion Done\n==========')
