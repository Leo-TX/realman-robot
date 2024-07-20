'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-07-19 13:34:10
Version: v1
File: 
Brief: 
'''
import time
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader

from handle_grasp_unlock_dataset import HandleGraspUnlockDataset
from handle_grasp_unlock_model import HandleGraspUnlockModel

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
RESNET_DEPTH = 18
LR = 1e-4
N_EPOCHS = 30

def train():
    train_dataset_dir = r'/media/datadisk10tb/leo/projects/data/lever_handle/train'
    model_load_path = r'./checkpoints/hgum2.pth'
    loss_save_path = r'./checkpoints/loss2.png'

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
    
if __name__ == "__main__":
    train()