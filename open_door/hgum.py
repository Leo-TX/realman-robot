import torch
from hgum_package.handle_grasp_unlock_model import HandleGraspUnlockModel

from utils.lib_io import *

class HGUM(object):
    def __init__(self,root_dir='./',model_load_path='checkpoints/hgum.pth'):
        self.root_dir = root_dir
        self.model_load_path = f'{root_dir}/{model_load_path}'
        self.model = HandleGraspUnlockModel(resnet_depth=RESNET_DEPTH, pretrained=True).to(DEVICE)
        self.model.load_state_dict(torch.load(self.model_load_path))
    
    @classmethod
    def init_from_yaml(cls,cfg_path='cfg/cfg_hgum.yaml'):
        cfg = read_yaml_file(cfg_path, is_convert_dict_to_class=True)
        return cls(cfg.root_dir,cfg.model_load_path)

    def get_dxdyR(self,image_path='',vis=False,if_p=False):
        from hgum_package.get_dxdyR import get_dxdyR
        dx,dy,R = get_dxdyR(self.root_dir,self.model,vis)
        return dx,dy,R
    
    def get_dxdyR_server(self,image_path,mask_path,server,remote_python_path,remote_root_dir,remote_img_dir,vis=False,if_p=False):
        local_rgb_img_path = image_path
        remote_rgb_img_path = f'{remote_img_dir}/{os.path.basename(local_rgb_img_path)}'
        local_mask_path = mask_path
        remote_mask_path = f'{remote_img_dir}/{os.path.basename(local_mask_path)}'
        
        # transfer the input files to the server
        server.exec_cmd(f'mkdir -p {remote_img_dir}/hgum/')
        server.transfer_file_local2remote(local_rgb_img_path,remote_rgb_img_path)
        server.transfer_file_local2remote(local_mask_path,remote_mask_path)

        # hgun
        remote_hgum_script_dir = f'{remote_root_dir}/hgum_package/'
        remote_hgum_script_path = f'get_dxdyR.py'
        hgum_cmd = f'cd {remote_hgum_script_dir}; {remote_python_path} {remote_hgum_script_path} -rgb {remote_rgb_img_path} -d {remote_mask_path} -cfg {remote_config_file_path} -camera {remote_camera_info_file_path}'
        server.exec_cmd(hgum_cmd)

        # transfer the output dir to the server
        server.transfer_folder_remote2local(f'{remote_img_dir}/hgum/', f'{os.path.dirname(local_rgb_img_path)}/hgum/')

        # open hgum_result.json to get rx and ry and rz
        with open(f'{os.path.dirname(local_rgb_img_path)}/hgum/hgum_result.json','r') as f:
            data = json.load(f)
            normal = data['normal']
            weights = data['weights']
            _3d_center = data['3d_center']
            _2d_center = data['2d_center']
            mask_color = data['mask_color']
        return normal,weights,_3d_center,_2d_center,mask_color

        return dx,dy,R

if __name__ == "__main__":
    hgum = HGUM.init_from_yaml('cfg/cfg_hgum.yaml')
    image_path = 'your_image_path.jpg'
    dx, dy, R = hgum.get_dxdyR(image_path,vis=True,if_p=True)