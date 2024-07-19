'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-07-19 13:42:57
Version: v1
File: 
Brief: 
'''
import json
from PIL import Image
import torch
from torchvision import transforms
from handle_grasp_unlock_model import HandleGraspUnlockModel

import sys
root_dir = "../"
sys.path.append(root_dir)
from utils.lib_rgbd import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESNET_DEPTH = 18

def get_dxdyR(root_dir='./',model=None,image_path='',vis=False,if_p=False):
    ## model
    if not model:
        model_load_path = f'{root_dir}/checkpoints/hgum.pth'
        model = HandleGraspUnlockModel(resnet_depth=RESNET_DEPTH, pretrained=True).to(DEVICE)
        model.load_state_dict(torch.load(model_load_path))
    model.eval()

    mask_path = image_path.replace('.png', '_mask.png')

    ## forward
    dx,dy,R = model.hgum_api(image_path,mask_path,if_p=if_p)

    if vis:
    ## vis_grasp
        with open(image_path.replace('.png','.json'), 'r') as f:
            data = json.load(f)
        Cx = data['Cx']
        Cy = data['Cy']
        orientation = data['orientation']
        x1_2d, y1_2d = Cx+dx, Cy+dy
        angle = 90
        x2_2d, y2_2d, Ox, Oy = rotate_point(x1_2d, y1_2d, R, orientation, angle)
        vis_grasp(image_path, dx, dy, x1_2d, y1_2d, x2_2d, y2_2d, Ox, Oy, R, orientation, angle, save_path=image_path.replace('.png','_vis_predicted.png'))
        
    return dx,dy,R

if __name__ == '__main__':
    root_dir = '../'
    image_path = '/media/datadisk10tb/leo/projects/realman-robot/images/lever.png'
    dx,dy,R = get_dxdyR(image_path,vis=True)