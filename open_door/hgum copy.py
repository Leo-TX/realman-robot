import os
import json
import re
import time
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt

from utils.lib_rgbd import *
from utils.lib_io import *

BATCH_SIZE = 16
RESNET_DEPTH = 18
LR = 1e-4
N_EPOCHS = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HandleGraspUnlockModel(nn.Module):
    def __init__(self, resnet_depth=18, pretrained=True):
        super(HandleGraspUnlockModel, self).__init__()

        ## reset
        self.resnet = models.__dict__[f'resnet{resnet_depth}'](pretrained=pretrained)
        
        self.image_encoder = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove the last FC layer
        self.mask_encoder = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove the last FC layer

        self.feature_dim = self.resnet.fc.in_features

        self.predictor = nn.Sequential(
            nn.Linear(2 * self.feature_dim, 512),  # merge tow resnet features
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # output: dx, dy, R
        )

    def forward(self, image, mask):
        image_features = self.image_encoder(image.to(DEVICE)).squeeze() # batch_size * 512
        mask_features = self.mask_encoder(mask.to(DEVICE)).squeeze() # batch_size * 512
        
        # check dimension
        if len(image_features.shape) == 1:
            image_features = image_features.unsqueeze(0).flatten(start_dim=1)  # batch_size * 512
        if len(mask_features.shape) == 1:
            mask_features = mask_features.unsqueeze(0).flatten(start_dim=1)  # batch_size * 512
            
        # print(f'image_features.shape: {image_features.shape}')
        # print(f'mask_features.shape: {mask_features.shape}')

        features = torch.cat((image_features, mask_features), dim=1)
        output = self.predictor(features)

        return output
    
    @classmethod
    def init_from_yaml(cls,cfg_path='cfg/cfg_hgum.yaml'):
        cfg = read_yaml_file(cfg_path, is_convert_dict_to_class=True)
        model = cls(resnet_depth=RESNET_DEPTH, pretrained=True).to(DEVICE)
        model.load_state_dict(torch.load(cfg.model_load_path))
        return model

    def hgum_api(self,image_path,mask_path,if_p=False):
        self.eval()
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        ## transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image) # 3 * 224 * 224
        mask = transform(mask) # 3 * 224 * 224

        ## add batch dimension
        image = image.unsqueeze(0) # 1 * 3 * 224 * 224
        mask = mask.unsqueeze(0) # 1 * 3 * 224 * 224

        ## forward
        with torch.no_grad():
            output = self.forward(image,mask)
        dx, dy, R = output[0].cpu().numpy()

        if if_p:
            print(f'[HGUM Result] dx: {dx}, dy: {dy}, R: {R}')
        
        return dx,dy,R
    
    def hgum_api_server(self,image_path,mask_path,if_p=False,server,remote_python_path,remote_root_dir,remote_img_dir):
        local_rgb_img_path = image_path
        remote_rgb_img_path = f'{remote_img_dir}/{os.path.basename(local_rgb_img_path)}'
        local_mask_path = mask_path
        remote_mask_path = f'{remote_img_dir}/{os.path.basename(local_mask_path)}'
        
        # transfer the input files to the server
        server.exec_cmd(f'mkdir -p {remote_img_dir}/hgum/')
        server.transfer_file_local2remote(local_rgb_img_path,remote_rgb_img_path)
        server.transfer_file_local2remote(local_mask_path,remote_mask_path)

        # ransac
        remote_ransac_script_dir = f'{remote_root_dir}/ransac_package/'
        remote_ransac_script_path = f'plane_detector.py'
        ransac_cmd = f'cd {remote_ransac_script_dir}; {remote_python_path} {remote_ransac_script_path} -rgb {remote_rgb_img_path} -d {remote_mask_path} -cfg {remote_config_file_path} -camera {remote_camera_info_file_path}'
        server.exec_cmd(ransac_cmd)

        # transfer the output dir to the server
        server.transfer_folder_remote2local(f'{remote_img_dir}/hgum/', f'{os.path.dirname(local_rgb_img_path)}/hgum/')

        # open ransac_result.json to get rx and ry and rz
        with open(f'{os.path.dirname(local_rgb_img_path)}/hgum/hgum_result.json','r') as f:
            data = json.load(f)
            normal = data['normal']
            weights = data['weights']
            _3d_center = data['3d_center']
            _2d_center = data['2d_center']
            mask_color = data['mask_color']
        return normal,weights,_3d_center,_2d_center,mask_color

        return dx,dy,R


def train():
    train_dataset_dir = r'./data/lever_handle/train'
    model_load_path = r'./checkpoints/hgum.pth'
    loss_save_path = r'./checkpoints/loss.png'

    ## dataset and dataloader
    train_dataset = HandleGraspUnlockDataset(root_dir=train_dataset_dir)
    print(train_dataset)
    train_dataset = [data for data in train_dataset if data is not None]
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    ## model
    model = HandleGraspUnlockModel(resnet_depth=RESNET_DEPTH, pretrained=True).to(DEVICE)
    model.train()

    ## loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    plt.ion()
    fig, ax = plt.subplots()

    start_time = time.time()

    for epoch in range(N_EPOCHS):
        for i, (images, masks, targets) in enumerate(train_dataloader):
            ## forward
            outputs = model(images, masks)
            
            ## backward
            loss = loss_fn(outputs, targets.to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ## print
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{N_EPOCHS}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')
                ## vis
                losses.append(loss.item())
                ax.clear()
                ax.plot(losses)
                ax.set_title('Training Loss')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Loss')
                plt.pause(0.1)
        plt.ioff()
        fig.savefig(loss_save_path)

        # ## save model
        torch.save(model.state_dict(), model_load_path)
        print(f'[All Time] {time.time()-start_time}')

def eval():
    eval_dataset_dir = r'./data/lever_handle/eval' 
    model_load_path = r'./checkpoints/hgum.pth'
    eval_vis_path = r'./checkpoints/'

    ## dataset and dataloader
    eval_dataset = HandleGraspUnlockDataset(root_dir=eval_dataset_dir)
    print(eval_dataset)
    eval_dataset = [data for data in eval_dataset if data is not None]
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    ## model
    model = HandleGraspUnlockModel(resnet_depth=RESNET_DEPTH, pretrained=True).to(DEVICE)
    model.load_state_dict(torch.load(model_load_path))
    model.eval()
    
    total_loss = 0.0
    loss_fn = nn.MSELoss()

    relative_errors = []
    all_real_values = []
    all_predicted_values = []

    with torch.no_grad():
        for i, (images, masks, targets) in enumerate(eval_dataloader):
            outputs = model(images, masks)

            loss = loss_fn(outputs, targets.to(DEVICE))
            total_loss += loss.item()

            all_real_values.extend(targets.cpu().numpy())
            all_predicted_values.extend(outputs.cpu().detach().numpy())

            for j in range(targets.size(0)):
                real_dx, real_dy, real_R = targets[j].cpu().numpy()

                ## forward
                predicted_dx, predicted_dy, predicted_R = outputs[j].cpu().detach().numpy()

                ## error
                if real_dx != 0:
                    relative_error_dx = abs((predicted_dx - real_dx) / real_dx)
                    print(f'relative_error_dx: {relative_error_dx}')
                else:
                    relative_error_dx = float('inf')
                if real_dy != 0:
                    relative_error_dy = abs((predicted_dy - real_dy) / real_dy)
                else:
                    relative_error_dy = float('inf')
                if real_R != 0:
                    relative_error_R = abs((predicted_R - real_R) / real_R)
                else:
                    relative_error_R = float('inf')

                relative_errors.append([relative_error_dx, relative_error_dy, relative_error_R])

            print(f'Batch [{i+1}/{len(eval_dataloader)}], Loss: {loss.item():.4f}')

    average_loss = total_loss / len(eval_dataloader)
    print(f'Average eval Loss: {average_loss:.4f}')

    # Calculate the average relative error for all samples
    relative_errors = np.array(relative_errors)
    mean_relative_errors = np.mean(relative_errors, axis=0)
    print(f"Mean Relative Errors (dx, dy, R): {mean_relative_errors}")

    labels = ['dx', 'dy', 'R']
    colors = ['blue', 'green', 'red']
    
    plt.figure(figsize=(15, 5))

    for i in range(3):
        plt.figure()
        
        real_values = [val[i] for val in all_real_values]
        predicted_values = [val[i] for val in all_predicted_values]

        x = np.arange(len(real_values))

        plt.bar(x - 0.2, real_values, width=0.4, label='Real', color='red')
        plt.bar(x + 0.2, predicted_values, width=0.4, label='Predicted', color='green')

        plt.xlabel('Data Index')
        plt.ylabel(f'{labels[i]} Value')
        plt.title(f'Real vs. Predicted {labels[i]}')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(eval_vis_path, f"eval_vis_{labels[i]}.png"))
        plt.close()

def test_images():
    ## model
    model_load_path = r'./checkpoints/hgum.pth'
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

def test_image():
    ## model
    model_load_path = r'./checkpoints/hgum.pth'
    model = HandleGraspUnlockModel(resnet_depth=RESNET_DEPTH, pretrained=True).to(DEVICE)
    model.load_state_dict(torch.load(model_load_path))
    model.eval()

    image_path = '/media/datadisk10tb/leo/projects/realman-robot/images/lever.png'
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

def main():
    hgum = HandleGraspUnlockModel.init_from_yaml(cfg_path=f'cfg/cfg_hgum.yaml')
    image_path = r'../images/lever.png'
    mask_path = image_path.replace('.png','_mask.png')
    dx,dy,R = hgum.hgum_api(image_path,mask_path,if_p=True)

if __name__ == "__main__":
    # train()
    # eval()
    # test_images()
    # test_image()
    main()