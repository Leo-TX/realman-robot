import os
import json
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

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
        annotation_file = os.path.splitext(image_file)[0] + '.txt'
        annotation_path = os.path.join(self.root_dir, annotation_file)
        with open(annotation_path, 'r') as f:
            dx, dy, R = map(float, f.read().split())

        # Load centroid data
        json_file = os.path.splitext(image_file)[0] + '.json'
        json_path = os.path.join(self.root_dir, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
            Ox, Oy = data['Cx'], data['Cy']

        # Calculate bounding box coordinates
        x_min = Ox + dx - abs(R)
        y_min = Oy + dy - abs(R)
        x_max = Ox + dx + abs(R)
        y_max = Oy + dy + abs(R)

        # Normalize R
        R /= image.width

        # Create target dictionary
        target = {}
        target["boxes"] = torch.as_tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)
        target["labels"] = torch.zeros((1,), dtype=torch.int64)  # Only one class: grasp point
        target["regression_targets"] = torch.as_tensor([dx, dy, R], dtype=torch.float32)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

class HandleGraspModel(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # Load pretrained Faster R-CNN model
        self.faster_rcnn = fasterrcnn_resnet50_fpn(pretrained=True)

        # Replace classifier
        in_features = self.faster_rcnn.roi_heads.box_predictor.cls_score.in_features
        self.faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Add regression branch
        hidden_dim = 1024  # Hidden layer dimension
        self.regression_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )

    def forward(self, images, targets=None):
        # Faster R-CNN forward pass
        losses = {}
        if self.training:
            # Training mode
            assert targets is not None
            proposals, detector_losses = self.faster_rcnn(images, targets)
            losses.update(detector_losses)
        else:
            # Evaluation mode
            proposals, _ = self.faster_rcnn(images)

        # Regression branch
        box_features = self.faster_rcnn.roi_heads.box_roi_pool(
            proposals, images
        )
        box_features = self.faster_rcnn.roi_heads.box_head(box_features)
        regression_preds = self.regression_head(box_features)

        if self.training:
            # Calculate regression loss
            regression_targets = torch.cat([t["regression_targets"] for t in targets])  # dx, dy, R
            regression_loss = nn.MSELoss()(regression_preds, regression_targets)
            losses.update({"regression_loss": regression_loss})
        
        # Return predictions
        return proposals, losses, regression_preds

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        proposals, losses, _ = model(images, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        losses["loss_classifier"].backward(retain_graph=True)
        losses["loss_box_reg"].backward(retain_graph=True)
        losses["loss_objectness"].backward(retain_graph=True)
        losses["loss_rpn_box_reg"].backward(retain_graph=True)
        losses["regression_loss"].backward()
        optimizer.step()

def main():
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define dataset and dataloader
    dataset = HandleDataset(root_dir=r'E:\realman-robot\open_door\data\images')
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Create model
    model = HandleGraspModel()
    model.to(device)

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch)

if __name__ == "__main__":
    main()