'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-07-15 23:36:56
Version: v1
File: 
Brief: 
'''
from server import Server 
from dtsam import DTSAM
from utils.lib_io import *
import os

root_dir = r'E:\realman-robot\open_door\data\lever_handle_2'

## os
names = get_filenames(folder=root_dir,is_base_name=False,filter='png')

## init
server = Server.init_from_yaml(cfg_path=f'cfg/cfg_server.yaml')
dtsam = DTSAM.init_from_yaml(cfg_path=f'cfg/cfg_dtsam.yaml')

## remote
remote_python_path = '/media/datadisk10tb/leo/anaconda3/envs/rm/bin/python'
remote_root_dir = '/media/datadisk10tb/leo/projects/realman-robot/open_door/'
remote_img_dir = '/media/datadisk10tb/leo/projects/realman-robot/open_door/trajectory/remote/'

## dtsam
num = 0
for name in names:
    print(f'Process {os.path.basename(name)} ...')
    rgb_img_path = name
    dtsam.process_images_server(rgb_img_path,server,remote_python_path,remote_root_dir,remote_img_dir)
    num += 1