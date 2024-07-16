import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from ultralytics import YOLO

class HandleDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.image_files = sorted([
            f for f in os.listdir(self.root_dir) if f.endswith('.png')
        ])
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.root_dir, image_file)
        image = Image.open(image_path).convert("RGB")

        # Load annotation data
        annotation_file = os.path.splitext(image_file)[0] + '.json'
        annotation_path = os.path.join(self.root_dir, annotation_file)

        # Load centroid data
        json_file = os.path.splitext(image_file)[0] + '.json'
        json_path = os.path.join(self.root_dir, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
            x_center = data['Cx']
            y_center = data['Cy']
            width = data['w']
            height = data['h']
            dx = data['dx']
            dx = data['dy']
            R = data['R']

        # Normalize R
        R /= image.width

        # Create target dictionary
        target = {}
        target["boxes"] = torch.as_tensor([[x_center, y_center, width, height]], dtype=torch.float32)
        target["labels"] = torch.zeros((1,), dtype=torch.int64)  # Only one class: grasp point
        target["regression_targets"] = torch.as_tensor([dx, dy, R], dtype=torch.float32)

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
        yolo_results = self.yolo_model(images)

        # Extract features from the YOLO model
        features = yolo_results[0].boxes.cls  # You might need to adjust this based on your YOLO model

        # Regression branch
        regression_preds = self.regression_head(features)

        return yolo_results, regression_preds

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        yolo_results, regression_preds = model(images)

        # Calculate YOLO loss
        yolo_loss = yolo_results[0].loss

        # Calculate regression loss
        regression_targets = torch.cat([t["regression_targets"] for t in targets])  # dx, dy, R
        regression_loss = nn.MSELoss()(regression_preds, regression_targets)

        # Combine losses
        loss = yolo_loss + regression_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define dataset and dataloader
    dataset = HandleDataset(root_dir=r'E:\realman-robot\open_door\data\images3_png_aug')
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Load pretrained YOLO model
    yolo_model = YOLO("yolov5s.pt")  # Load your desired YOLO model

    # Define regression head
    regression_head = nn.Sequential(
        nn.Linear(yolo_model.model.nc, 1024),
        nn.ReLU(),
        nn.Linear(1024, 3),
        nn.Sigmoid()
    )

    # Create model
    model = HandleGraspModel(yolo_model, regression_head)
    model.to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch)

if __name__ == "__main__":
    main()