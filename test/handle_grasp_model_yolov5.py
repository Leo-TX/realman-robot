import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_tensor
from ultralytics import YOLO
import matplotlib.pyplot as plt

device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')


class HandleDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        """
        Initializes the HandleDataset class.

        Args:
            root_dir (str): The root directory containing the images and JSON files.
            transforms (callable, optional): A function/transform that takes in an PIL image
                and a dictionary of targets and returns a transformed version.
        """
        self.root_dir = root_dir
        self.image_files = sorted([
            f for f in os.listdir(self.root_dir) if f.endswith('.png')
        ])
        self.transforms = transforms

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to load.

        Returns:
            tuple: (image, target) where target is a dictionary containing 'boxes', 'labels', and 'regression_targets'.
        """
        image_file = self.image_files[idx]
        image_path = os.path.join(self.root_dir, image_file)
        image = Image.open(image_path).convert("RGB")

        # Load centroid data
        json_file = os.path.splitext(image_file)[0] + '.json'
        json_path = os.path.join(self.root_dir, json_file)

        # Skip if JSON file does not exist
        if not os.path.exists(json_path):
            return None

        with open(json_path, 'r') as f:
            data = json.load(f)
            x_center = data['Cx'] / image.width  # Normalize to [0, 1]
            y_center = data['Cy'] / image.height  # Normalize to [0, 1]
            width = data['w'] / image.width  # Normalize to [0, 1]
            height = data['h'] / image.height  # Normalize to [0, 1]
            dx = data['dx']
            dy = data['dy']
            R = data['R']  # No normalization for R

        # Create target dictionary
        target = {}
        target["boxes"] = torch.as_tensor([[x_center, y_center, width, height]], dtype=torch.float32)
        target["labels"] = torch.zeros((1,), dtype=torch.int64)  # Only one class: grasp point
        target["regression_targets"] = torch.as_tensor([dx, dy, R], dtype=torch.float32)

        # image = to_tensor(image)
        image = np.array(image)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

class HandleGraspModel(nn.Module):
    def __init__(self, yolo_model, regression_head):        
        super().__init__()
        self.yolo_model = yolo_model
        self.regression_head = regression_head

    def forward(self, images):
        # YOLO forward pass
        # yolo_results = self.yolo_model(images)
        # YOLO forward pass, 只获取结果，不进行预测
        yolo_results = self.yolo_model(images, verbose=False)

        # Extract features from the YOLO model
        # features = yolo_results[0].boxes.cls  # You might need to adjust this based on your YOLO model
        features = []
        for result in yolo_results:
            # 使用 .boxes 属性获取边界框坐标
            features.append(result.boxes.xywh.cpu()) 


        # 将 features 列表转换为张量
        features = torch.stack(features, dim=0).to(device) # 将 features 移动到 device 上

        # Regression branch
        regression_preds = self.regression_head(features)

        return yolo_results, regression_preds

def train_one_epoch(model, optimizer, data_loader, device, epoch, all_losses):
    """Trains the model for one epoch and updates the loss plot."""
    # model.train()
    for images, targets in data_loader:
        print("images before:", images) 
        # images = list(image.to(device) for image in images)
        images = [image.cpu().numpy() for image in images] 
        # 将通道维度移动到最后
        # images = [np.moveaxis(image, 0, -1) for image in images] 

        print("images after:", images) 
        
        print("Targets before:", targets) 
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = {k: v.to(device) for k, v in targets.items()}
        print("targets after:", targets) 

        # 清空梯度
        optimizer.zero_grad()   

        # Forward pass
        yolo_results, regression_preds = model(images)
        #  YOLO 模型期望的输入是图片文件路径、PIL Image 对象或 NumPy 数组等类型。

        # Calculate YOLO loss
        print(yolo_results)
        yolo_loss = yolo_results.loss

        # Calculate regression loss
        regression_targets = targets["regression_targets"]  # 不需要再使用 torch.cat
        # regression_targets = torch.cat([t["regression_targets"] for t in targets])  # dx, dy, R
        regression_loss = nn.MSELoss()(regression_preds, regression_targets)


        # Combine losses
        loss = yolo_loss + regression_loss
        all_losses.append(loss.item())  # 将 loss 值添加到列表中

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")  # Print loss for monitoring

    # Update loss plot
    plt.plot(all_losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.pause(0.01) 

def main():
    # Set device
    device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

    # Define dataset and dataloader
    # dataset = HandleDataset(root_dir=r'E:\realman-robot\open_door\data\lever_handle_aug')
    
    dataset = HandleDataset(root_dir=r'/media/datadisk10tb/leo/projects/realman-robot/open_door/data/images3_png_aug')
    dataset = [data for data in dataset if data is not None]
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Load pretrained YOLO model
    # yolo_model = YOLO("cfg/yolov5s.yaml")  # 使用 .yaml 文件初始化模型结构
    yolo_model = YOLO("cfg/yolov5s.pt")

    # Define regression head
    regression_head = nn.Sequential(
        nn.Linear(4, 1024),
        nn.ReLU(),
        nn.Linear(1024, 3),
    )

    # Create model
    model = HandleGraspModel(yolo_model, regression_head)
    model.to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    all_losses = []  # 用于存储所有 loss 值
    plt.ion()  # 开启交互模式，以便实时更新 loss 曲线图
    plt.figure()  # 创建一个图形窗口
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, all_losses)
    plt.ioff()  # 关闭交互模式
    plt.show()  # 显示最终的 loss 曲线图

if __name__ == "__main__":
    main()