import os
import json

from utils.lib_io import *

class HGUM(object):
    def __init__(self):
        pass
    
    # @classmethod
    # def init_from_yaml(cls,cfg_path='cfg/cfg_hgum.yaml'):
    #     cfg = read_yaml_file(cfg_path, is_convert_dict_to_class=True)
    #     return cls()

    def get_dxdyR(self,image_path='',mask_path='',root_dir=''):
        from hgum_package.get_dxdyR import get_dxdyR
        dx,dy,R = get_dxdyR(image_path,mask_path,root_dir=root_dir)
        return dx,dy,R
    
    def get_dxdyR_server(self,image_path,mask_path,server,remote_python_path,remote_root_dir,remote_img_dir):
        local_rgb_img_path = image_path
        remote_rgb_img_path = f'{remote_img_dir}/{os.path.basename(local_rgb_img_path)}'
        local_mask_path = mask_path
        remote_mask_path = f'{remote_img_dir}/{os.path.basename(local_mask_path)}'
        
        # transfer the input files to the server
        server.exec_cmd(f'mkdir -p {remote_img_dir}/hgum/')
        server.transfer_file_local2remote(local_rgb_img_path,remote_rgb_img_path)
        server.transfer_file_local2remote(local_mask_path,remote_mask_path)

        # hgum
        remote_hgum_script_dir = f'{remote_root_dir}/hgum_package/'
        remote_hgum_script_path = f'get_dxdyR.py'
        hgum_cmd = f'cd {remote_hgum_script_dir}; {remote_python_path} {remote_hgum_script_path} -i {remote_rgb_img_path} -m {remote_mask_path}'
        server.exec_cmd(hgum_cmd)

        # transfer the output dir to the server
        server.transfer_folder_remote2local(f'{remote_img_dir}/hgum/', f'{os.path.dirname(local_rgb_img_path)}/hgum/')

        # open hgum_result.json to get dx,dy,R
        with open(f'{os.path.dirname(local_rgb_img_path)}/hgum/hgum_result.json','r') as f:
            data = json.load(f)
            dx = data['dx']
            dy = data['dy']
            R = data['R']

        return dx,dy,R

if __name__ == "__main__":
    hgum = HGUM()
    image_path = r'/media/datadisk10tb/leo/projects/realman-robot/open_door/data/test/trajectory_000/1.png'
    mask_path = r'/media/datadisk10tb/leo/projects/realman-robot/open_door/data/test/trajectory_000/1/dtsam/center.png'
    dx, dy, R = hgum.get_dxdyR(image_path,mask_path,root_dir='./hgum_package/')