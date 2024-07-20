'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-07-19 13:42:57
Version: v1
File: 
Brief: 
'''
import os
import json
import argparse
from PIL import Image
import torch
from torchvision import transforms

from handle_grasp_unlock_model import HandleGraspUnlockModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESNET_DEPTH = 18

def get_dxdyR(image_path='',mask_path='',model_path='./checkpoints/hgum.pth',if_p=False):
    ## model
    model = HandleGraspUnlockModel(resnet_depth=RESNET_DEPTH, pretrained=True).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    ## image dir
    image_dir = os.path.dirname(image_path)+'/hgum'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    ## mask path
    if not mask_path:
        mask_path = image_path.replace('.png', '_mask.png')

    ## forward
    dx,dy,R = model.hgum_api(image_path,mask_path,if_p=if_p)

    ## save to hgum/hgum_result.json
    result_save_path = image_dir+'/hgum_result.json'
    result = {"dx":dx,
              "dy":dy,
              "R":R,
    }
    with open(result_save_path, 'w') as file:
        json.dump(result, file, indent=4)
    
    return dx,dy,R

def main(args):
    get_dxdyR(args.image_path,args.mask_path,args.model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", type=str, default="", help="Input image path.")
    parser.add_argument("-m", "--mask_path", type=str, default="", help="Input mask path.")
    parser.add_argument("-model", "--model_path", type=str, default="./checkpoints/hgum.pth", help="Input model path.")
    main(parser.parse_args())