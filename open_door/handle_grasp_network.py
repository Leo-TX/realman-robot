import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import json

class HandleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = sorted([f for f in os.listdir(self.root_dir) if f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.root_dir, image_file)
        
         # Load annotation data
        json_file = os.path.splitext(image_file)[0] + '.json'
        json_path = os.path.join(self.root_dir, json_file)

        # Skip if JSON file does not exist
        if not os.path.exists(json_path):
            return None

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(image_path.replace('.png','_vis.png')).convert("RGB")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Load annotations from JSON file
        with open(json_path, 'r') as f:
            annotations = json.load(f)

        cx = annotations["Cx"]
        cy = annotations["Cy"]
        dx = annotations["dx"]
        dy = annotations["dy"]
        R = annotations["R"]
        target = torch.tensor([dx, dy, R], dtype=torch.float32)

        return image, mask, target

# 定义模型
class DualResNetPredictor(nn.Module):
    def __init__(self, resnet_depth=18, pretrained=True):
        super(DualResNetPredictor, self).__init__()

        # 加载预训练的 ResNet 模型
        resnet = models.__dict__[f'resnet{resnet_depth}'](pretrained=pretrained)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])  # 移除最后的全连接层
        self.mask_encoder = nn.Sequential(*list(resnet.children())[:-1])  # 移除最后的全连接层

        # 获取 ResNet 输出特征维度
        feature_dim = resnet.fc.in_features

        # 定义预测网络
        self.predictor = nn.Sequential(
            nn.Linear(2 * feature_dim, 512),  # 拼接两个 ResNet 的特征
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # 输出 dx, dy, R
        )

    def forward(self, image, mask):
        # 提取图像特征
        image_features = self.image_encoder(image).squeeze()
        mask_features = self.mask_encoder(mask).squeeze()

        # 拼接特征
        features = torch.cat((image_features, mask_features), dim=1)

        # 预测
        output = self.predictor(features)
        return output

# --- 训练代码 ---
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集和数据加载器
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = HandleDataset(root_dir=r'/media/datadisk10tb/leo/projects/realman-robot/open_door/data/lever_handle_aug_2', transform=data_transform)
    dataset = [data for data in dataset if data is not None]

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = DualResNetPredictor(resnet_depth=18, pretrained=True).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        for i, (images, masks, targets) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            targets = targets.to(device)

            # 前向传播
            outputs = model(images, masks)

            # 计算损失
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 打印训练信息
            # if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'dual_resnet_predictor.pth')