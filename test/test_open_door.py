'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-06-10 16:38:37
Version: v1
File: 
Brief: 
'''
import sys
import time
import os
import json
import numpy as np

root_dir = './open_door'
sys.path.append(root_dir)
cfg_dir = f'{root_dir}/cfg/'
image_dir = f'{root_dir}/images/image1/'
rgb_img_path = f'{image_dir}/rgb.png'
d_img_path = f'{image_dir}/d.png'
python_path = f'/media/datadisk10tb/leo/anaconda3/envs/rm/bin/python'
remote_root_dir = f'/media/datadisk10tb/leo/projects/realman-robot/open_door/'
remote_img_dir = f'{remote_root_dir}/images/image1/'



## camera capture
# from camera import Camera
# import time
# camera = Camera(width=1280,height=720,fps=30,intrinsic_path=f'{cfg_dir}/intrinsic.txt',depth_scale_path=f'{cfg_dir}/depth_scale.txt')
# print(camera)
# print('image capturing ...')
# time.sleep(3)
# rgb_img,d_img = camera.capture_rgbd(rgb_save_path=rgb_img_path,d_save_path=d_img_path)
# print('image captured')

## sever
# from server import Server
# server = Server(hostname='130.126.136.95',username='zhi',password='yourpassword',if_stfp=True)

# ## init dtsam
# from dtsam import DTSAM
# dtsam = DTSAM(img_path=rgb_img_path,classes='handle',device='cuda:1',threshold=0.3)

# # ## dtsam
# remote_dtsam_script_dir = f'{remote_root_dir}/dtsam_package/'
# remote_dtsam_script_path = f'detic_sam.py'
# local_img_path = dtsam.img_path
# remote_img_path = f'{remote_img_dir}/{os.path.basename(local_img_path)}'

# ## transfer the input image file to the server
# server.exec_cmd(f'mkdir -p {remote_img_dir}/dtsam/')
# server.transfer_file_local2remote(local_img_path,remote_img_path)

# # # dtsam
# dtsam_cmd = f'cd {remote_dtsam_script_dir}; {python_path} {remote_dtsam_script_path} -i {remote_img_path} -c {dtsam.classes} -d {dtsam.device} -t {dtsam.threshold}'
# server.exec_cmd(dtsam_cmd,if_p=True)

# # # transfer the output dir to the server
# server.transfer_folder_remote2local(f'{remote_img_dir}/dtsam/', f'{os.path.dirname(local_img_path)}/dtsam/')

# # # open dtsam_result.json to get x and y
# with open(f'{os.path.dirname(local_img_path)}/dtsam/dtsam_result.json','r') as f:
#     data = json.load(f)
#     x = data['Cx']
#     y = data['Cy']
#     w = data['w']
#     h = data['h']
#     box = data['box']
#     orientation = data['orientation']
# print(f'Cx: {x}, Cy: {y}')
# print(f'w: {w}, h: {h} orientation: {orientation}')
# print(f'box:{box}')


# from ransac import RANSAC
# ransac = RANSAC(rgb_img_path=rgb_img_path,d_img_path=d_img_path,config_file_path=f'{cfg_dir}/ransac_cfg.yaml',camera_info_file_path=f'{cfg_dir}/cam_params.json',vis=False)

# orientation = 'horizontal'
# local_rgb_img_path = ransac.rgb_img_path
# remote_rgb_img_path = f'{remote_img_dir}/{os.path.basename(local_rgb_img_path)}'
# local_d_img_path = ransac.d_img_path
# remote_d_img_path = f'{remote_img_dir}/{os.path.basename(local_d_img_path)}'
# local_config_file_path = ransac.config_file_path
# remote_config_file_path = f'{remote_img_dir}/ransac/{os.path.basename(local_config_file_path)}'
# local_camera_info_file_path = ransac.camera_info_file_path
# remote_camera_info_file_path = f'{remote_img_dir}/ransac/{os.path.basename(local_camera_info_file_path)}'

# # transfer the input files to the server
# server.exec_cmd(f'mkdir -p {remote_img_dir}/ransac/')
# server.transfer_file_local2remote(local_rgb_img_path,remote_rgb_img_path)
# server.transfer_file_local2remote(local_d_img_path,remote_d_img_path)
# server.transfer_file_local2remote(local_config_file_path,remote_config_file_path)
# server.transfer_file_local2remote(local_camera_info_file_path,remote_camera_info_file_path)

# # ransac
# remote_ransac_script_dir = f'{remote_root_dir}/ransac_package/'
# remote_ransac_script_path = f'plane_detector.py'
# ransac_cmd = f'cd {remote_ransac_script_dir}; {python_path} {remote_ransac_script_path} -rgb {remote_rgb_img_path} -d {remote_d_img_path} -cfg {remote_config_file_path} -camera {remote_camera_info_file_path} -o {orientation}'
# server.exec_cmd(ransac_cmd,if_p=True)

# # transfer the output dir to the server
# server.transfer_folder_remote2local(f'{remote_img_dir}/ransac/', f'{os.path.dirname(local_rgb_img_path)}/ransac/')

# # open ransac_result.json to get rx and ry and rz
# with open(f'{os.path.dirname(local_rgb_img_path)}/ransac/ransac_result.json','r') as f:
#     data = json.load(f)
#     rxryrz = data['rxryrz']
# print(f'rx:{rxryrz[0]},ry:{rxryrz[1]},rz:{rxryrz[2]}')

## 2d to 3d
from camera import Camera
camera = Camera(width=1280,height=720,fps=30,intrinsic_path=f'{cfg_dir}/intrinsic.txt',depth_scale_path=f'{cfg_dir}/depth_scale.txt')
print(camera)
x,y,z = camera.xy2xyz(897.6586715867159,470.39986581683996,d_img=d_img_path)
print(f'x:{x},y:{y},z:{z}')

## Arm
# init arm
from arm import Arm
arm = Arm('192.168.10.19',8080,cam2base_H_path=f'{root_dir}/cfg/cam2base_H.csv',if_gripper=True,if_monitor=False)# 18 for left 19 for right
print(arm)
arm.go_home()
arm.change_tool_frame(tool_name='dh3')
arm.control_gripper(open_value=1000)

## target2cam_xyzrxryrz 2 target2base_xyzrxryrz
# target2cam_xyzrxryrz = [x,y,z,-0.0689813069469519,0.3373031584204652,0.9388653570195614] # m
# target2cam_xyzrxryrz = [x,y,z,-0.34325921,-0.06448846,-3.118564] # m
# target2cam_xyzrxryrz = [x,y,z, 2.79833344,-0.06448846,-3.118564] # m
target2cam_xyzrxryrz = [x,y,z, 0.34325921,0.06448846,0.02302866] # m

target2base_xyzrxryrz = arm.target2cam_xyzrpy_to_target2base_xyzrpy(target2cam_xyzrxryrz,if_gripper=False)
# target2base_xyzrxryrz = [0.52898,-0.493726,-0.152548,-1.538,-1.093,-2.055]
target2base_xyzrxryrz[4] -= np.pi/2
print(f'target2base_xyzrxryrz:{target2base_xyzrxryrz}')

# ## dmp
from dmp import DMP
image_dir = f'{root_dir}/images/image1/'
refer_tjt_path = f'{cfg_dir}/refer_tjt.csv'
dmp = DMP(refer_tjt_path)
new_tjt = dmp.gen_new_tjt(initial_pos=arm.get_p(),goal_pos=target2base_xyzrxryrz.tolist(),show=True)
poses = dmp.get_poses(tjt=new_tjt,step=50,if_p=True)
arm.move_poses(poses,vel=10,trajectory_connect=1,frame_name='dh3')

# ## grasp
# arm.control_gripper(open_value=210)

# ## test rotate
# i=0
# while True:
#     arm.arm.Joint_Teach_Cmd(num=7, direction=0, v=30)
# while i<3:
#     arm.arm.Pos_Teach_Cmd(type=0,direction=1,v=10)
#     time.sleep(0.1)
#     i+=0.1

# while i<5:
#     arm.arm.Ort_Teach_Cmd(type=0,direction=0,v=10)
#     time.sleep(0.1)
#     i+=0.1

# arm.arm.Teach_Stop_Cmd()

## disconnect
arm.disconnect()