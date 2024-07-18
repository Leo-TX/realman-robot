'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-07-13 16:26:55
Version: v1
File: 
Brief: 
'''
from arm import Arm
import numpy as np
from utils.lib_math import *

arm = Arm.init_from_yaml(cfg_path='cfg/cfg_arm_left.yaml')

x1_3d,y1_3d,z1_3d = [-0.23367099323393, 0.11472560428474364, 0.5024066172696261]
normal = [0.13309615615026937, 0.3632092761658243, -0.922152067137043]

# rx,ry,rz = normal2rxryrz_left(normal)
rx,ry,rz = normal2rxryrz(normal)
print(f'[normal2rxryrz Result] rx: {rx} ry: {ry} rz: {rz}')

p1_3d_cam_xyzrxryrz = [x1_3d,y1_3d,z1_3d,ry,-rx,rz]
p1_3d_base_xyzrxryrz = arm.target2cam_xyzrpy_to_target2base_xyzrpy(p1_3d_cam_xyzrxryrz)

print(f'[p1_3d_cam_xyzrxryrz] {p1_3d_cam_xyzrxryrz}')
print(f'[p1_3d_base_xyzrxryrz] {p1_3d_base_xyzrxryrz}')

_rx,_ry,_rz = p1_3d_base_xyzrxryrz[3:6]
print(f'_rx: {_rx} _ry: {_ry} _rz: {_rz}')

rx = -1*_ry
ry = _rz+np.pi*2/3+np.pi/18
rz = -1*_rx-np.pi

print(f'rx: {rx} ry: {ry} rz: {rz}')
