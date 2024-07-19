import os
import json
import re
import torch
from handle_grasp_unlock_dataset import HandleGraspUnlockDataset
from handle_grasp_unlock_model import HandleGraspUnlockModel

import sys
root_dir = "../"
sys.path.append(root_dir)
from utils.lib_rgbd import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESNET_DEPTH = 18

def test():
    ## model
    model_load_path = r'../checkpoints/hgum.pth'
    model = HandleGraspUnlockModel(resnet_depth=RESNET_DEPTH, pretrained=True).to(DEVICE)
    model.load_state_dict(torch.load(model_load_path))
    model.eval()

    root_dir = r'./data/lever_handle/test/'
    image_files = sorted([f for f in os.listdir(root_dir) if re.match(r'.*_.*_\d+\.png$', f)])
    
    for i in range(len(image_files)):
        print(f'[Num]: {i}')
        image_file = image_files[i]
        image_path = os.path.join(root_dir, image_file)
        mask_path = image_path.replace('.png', '_mask.png')

        ## forward
        dx,dy,R = model.hgum_api(image_path,mask_path,if_p=False)

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