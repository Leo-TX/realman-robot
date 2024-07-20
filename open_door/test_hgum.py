from hgum import HGUM
from server import Server

## init
server = Server.init_from_yaml(cfg_path=f'cfg/cfg_server.yaml')
hgum = HGUM()

## image and mask
image_path = r'E:\realman-robot\open_door\data\test\trajectory_000\1\rgb.png'
mask_path = r'E:\realman-robot\open_door\data\test\trajectory_000\1\dtsam\center.png'

## remote
remote_python_path: '/media/datadisk10tb/leo/anaconda3/envs/rm/bin/python'
remote_root_dir: '/media/datadisk10tb/leo/projects/realman-robot/open_door/'
remote_img_dir: '/media/datadisk10tb/leo/projects/realman-robot/open_door/trajectory/remote/'

## get dxdyR
dx,dy,R = hgum.get_dxdyR_server(image_path,mask_path,server,remote_python_path,remote_root_dir,remote_img_dir)
print(f'dx: {dx}, dy: {dy}, R: {R}')