'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-06-20 15:34:06
Version: v1
File: 
Brief: 
'''
from scipy.spatial.transform import Rotation as R
import numpy as np

def normal2rxryrz(normal,if_p=False):
    original_normal = np.array(normal)
    normal = original_normal * -1
    z_axis = normal / np.linalg.norm(normal)
    initial_x_axis = np.array([1, 0, 0])
    x_axis = initial_x_axis - np.dot(initial_x_axis, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz')
    rx,ry,rz = euler_angles
    if if_p:
        print(f'original_normal:\n{original_normal}')
        print(f'normal:\n{normal}')
        print(f'z_axis:\n{z_axis}')
        print(f'initial_x_axis:\n{initial_x_axis}')
        print(f'x_axis:\n{x_axis}')
        print(f'y_axis:\n{y_axis}')
        print(f'rotation_matrix:\n{rotation_matrix}')
        print(f'euler_angles:\n{euler_angles}')
        print(f'rx:{rx} ry:{ry} rz:{rz}')
    return rx,ry,rz

if __name__ == '__main__':
    normal = [-0.04086078106574935, 0.3599500732902602, -0.9320763602350576]
    normal2rxryrz(normal,if_p=True)