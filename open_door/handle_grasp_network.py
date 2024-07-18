import os
import json
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HandleGraspDataset(Dataset):
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
        mask = Image.open(image_path.replace('.png','_mask.png')).convert("RGB")

        if not self.transform:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
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

class HandleGraspNetwork(nn.Module):
    def __init__(self, resnet_depth=18, pretrained=True):
        super(HandleGraspNetwork, self).__init__()

        # 加载预训练的 ResNet 模型
        resnet = models.__dict__[f'resnet{resnet_depth}'](pretrained=pretrained)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])  # 移除最后的全连接层
        self.mask_encoder = nn.Sequential(*list(resnet.children())[:-1])  # 移除最后的全连接层

        # 获取 ResNet 输出特征维度
        feature_dim = resnet.fc.in_features

        self.predictor = nn.Sequential(
            nn.Linear(2 * feature_dim, 512),  # 拼接两个 ResNet 的特征
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # output: dx, dy, R
        )

    def forward(self, image, mask):
        image_features = self.image_encoder(image).squeeze()
        mask_features = self.mask_encoder(mask).squeeze()

        features = torch.cat((image_features, mask_features), dim=1)

        output = self.predictor(features)
        return output

def train():
    BATCH_SIZE = 8
    SHUFFLE = True
    RESNET_DEPTH = 18
    LR = 1e-4
    N_EPOCHS = 10

    train_dataset_dir = r'/media/datadisk10tb/leo/projects/realman-robot/open_door/data/lever_handle_aug_2'
    model_save_dir = r'./checkpoints/handle_grasp_model.pth'

    train_dataset = HandleGraspDataset(root_dir=train_dataset_dir)
    train_dataset = [data for data in train_dataset if data is not None]
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    model = HandleGraspNetwork(resnet_depth=RESNET_DEPTH, pretrained=True).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(N_EPOCHS):
        for i, (images, masks, targets) in enumerate(train_dataloader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(images, masks)

            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), model_save_dir)


if __name__ == "__main__":
    train()