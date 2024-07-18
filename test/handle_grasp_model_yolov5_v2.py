import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

class HandleDataset(Dataset):
    """
    Dataset class for handle grasp points.
    """
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

        # Load annotation data
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

        image = np.array(image)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


class HandleGraspModel(nn.Module):
    """
    Model for predicting handle grasp points.
    """
    def __init__(self, yolo_model, regression_head):        
        super().__init__()
        self.yolo_model = yolo_model
        self.regression_head = regression_head

    def forward(self, images, targets=None):
        """
        Forward pass of the model.

        Args:
            images (list): List of PIL Images.
            targets (list, optional): List of target dictionaries. Required during training.

        Returns:
            tuple: (yolo_results, regression_preds) during evaluation, 
                   (yolo_results, regression_preds, yolo_loss) during training.
        """
        yolo_results = self.yolo_model(images, verbose=False) 

        features = []
        for result in yolo_results:
            features.append(result.boxes.xywh.cpu())

        # Stack features into a single tensor and move to the correct device
        features = torch.stack(features, dim=0).to(next(self.parameters()).device)

        # Regression branch
        regression_preds = self.regression_head(features)

        # Return YOLO loss during training
        if targets is not None:
            yolo_loss = yolo_results.loss
            return yolo_results, regression_preds, yolo_loss
        
        return yolo_results, regression_preds


def train_one_epoch(model, optimizer, data_loader, device, epoch, all_losses):
    """Trains the model for one epoch and updates the loss plot."""
    # model.train()  # Set the model to training mode
    for images, targets in data_loader:
        # 将 Tensor 转换为 NumPy 数组
        images = [img.cpu().numpy() for img in images]
        # 将 NumPy 数组转换为 PIL Image 对象
        images = [Image.fromarray(image) for image in images]

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = {k: v.to(device) for k, v in targets.items()}

        # Forward pass
        yolo_results, regression_preds, yolo_loss = model(images, targets=targets)

        # Calculate regression loss
        regression_targets = torch.cat([t["regression_targets"] for t in targets])
        regression_loss = nn.MSELoss()(regression_preds, regression_targets)

        # Combine losses
        loss = yolo_loss + regression_loss
        all_losses.append(loss.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

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
    dataset = HandleDataset(root_dir=r'/media/datadisk10tb/leo/projects/realman-robot/open_door/data/images3_png_aug')
    dataset = [data for data in dataset if data is not None]
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Load pretrained YOLO model
    yolo_model = YOLO("cfg/yolov8s.pt")  # Use .yaml file to initialize the model structure

    # Define regression head
    regression_head = nn.Sequential(
        nn.Linear(4, 1024),
        nn.ReLU(),
        nn.Linear(1024, 3),
    )

    # Create model
    model = HandleGraspModel(yolo_model, regression_head).to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    all_losses = []  # To store all loss values
    plt.ion()  # Turn on interactive mode to update the loss curve in real time
    plt.figure()  # Create a graph window
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, all_losses)
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Display the final loss curve

if __name__ == "__main__":
    main()