from server import Server
from dtsam import DTSAM
from utils.lib_io import *

root_dir = r'E:\realman-robot\open_door\data\images'
rename_files_sequentially(folder=root_dir)
names = get_filenames(folder=root_dir,is_base_name=False,filter='png')
print(names)

server = Server.init_from_yaml(cfg_path=f'cfg/cfg_server.yaml')
dtsam = DTSAM.init_from_yaml(cfg_path=f'cfg/cfg_dtsam.yaml')

remote_python_path = '/media/datadisk10tb/leo/anaconda3/envs/rm/bin/python'
remote_root_dir = '/media/datadisk10tb/leo/projects/realman-robot/open_door/'
remote_img_dir = '/media/datadisk10tb/leo/projects/realman-robot/open_door/trajectory/remote/'

for name in names:
    rgb_img_path = name
    x1_2d,y1_2d,orientation,w,h,box = dtsam.get_xy_server(rgb_img_path,server,remote_python_path,remote_root_dir,remote_img_dir)