import os
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ultralytics import YOLO

# Define the HandleDataset class
class HandleDataset(Dataset):
    """
    A custom dataset class for loading handle images and annotations.
    """
    def __init__(self, root_dir, transform=None):
        """
        Initializes the HandleDataset.

        Args:
            root_dir (str): The path to the directory containing the images and JSON files.
            transform (callable, optional): A function/transform to apply to the images. Defaults to None.
        """
        self.root_dir = root_dir
        self.image_files = sorted([f for f in os.listdir(self.root_dir) if f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Gets a sample from the dataset.

        Args:
            idx (int): The index of the sample.

        Returns:
            tuple: A tuple containing the image, center coordinates (cx, cy), and target values (dx, dy, R).
                   Returns None if the corresponding JSON file does not exist.
        """
        image_file = self.image_files[idx]
        image_path = os.path.join(self.root_dir, image_file)

        # Load annotation data
        json_file = os.path.splitext(image_file)[0] + '.json'
        json_path = os.path.join(self.root_dir, json_file)

        # Skip if JSON file does not exist
        if not os.path.exists(json_path):
            return None

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load annotations from JSON file
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        cx = annotations["Cx"]
        cy = annotations["Cy"]
        dx = annotations["dx"]
        dy = annotations["dy"]
        R = annotations["R"]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Convert to tensors
        image = torch.from_numpy(np.array(image)).float() # C, H, W
        cx = torch.tensor(cx, dtype=torch.float32)
        cy = torch.tensor(cy, dtype=torch.float32)
        target = torch.tensor([dx, dy, R], dtype=torch.float32)

        return image, cx, cy, target

# Define the ModifiedYOLOv8 class
class ModifiedYOLOv8(torch.nn.Module):
    """
    A modified YOLOv8 model that takes an image and center coordinates as input
    and outputs dx, dy, and R.
    """
    def __init__(self, yolo_model, num_outputs=3):
            super(ModifiedYOLOv8, self).__init__()
            self.yolo_model = yolo_model
            self.num_outputs = num_outputs

    def forward(self, x, cx, cy):
        # x: input image (batch_size, channels, height, width)
        # cx, cy: center coordinates (batch_size)

        detections = self.yolo_model(x)
        detections = detections[0] # Take the first output (assuming batch size of 1)

        # detections will have shape [num_detections, 6 + num_classes]
        # where 6 corresponds to:  xywh, confidence, class_id 

        # Assuming you only have one handle detection per image:
        if detections.shape[0] > 0:
            detection = detections[0]  # Take the first detection

            # Extract xywh
            xywh = detection[0:4]

            # You might need to transform xywh to your desired format
            # For example, if you need center_x, center_y, width, height:
            cx_pred, cy_pred, w_pred, h_pred = xywh

            # Concatenate predicted center coordinates and image features
            features = torch.cat((torch.tensor([cx_pred, cy_pred, w_pred, h_pred]), cx, cy), dim=0)
            
            # Create a linear layer to map features to your desired outputs
            self.regressor = torch.nn.Linear(features.shape[0], self.num_outputs)
            
            # Predict dx, dy, R
            output = self.regressor(features)
            return output
        else:
            # Handle the case where no detections are found
            return torch.tensor([0.0, 0.0, 0.0]) # Return default values or handle appropriately 

# Define the train function
def train(model, dataloader, optimizer, criterion, device, epochs=10):
    """
    Trains the model.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (DataLoader): The data loader.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to train the model on.
        epochs (int, optional): The number of epochs to train for. Defaults to 10.
    """
    # model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, cx, cy, targets) in enumerate(dataloader):
            images = images.to(device)
            cx = cx.to(device)
            cy = cy.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(images.permute(0, 1, 2, 3), cx, cy) # Assuming images are BCHW
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:    # Print every 100 mini-batches
                print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

# Define the test function
def test(model, dataloader, device):
    """
    Tests the model.

    Args:
        model (torch.nn.Module): The model to test.
        dataloader (DataLoader): The data loader.
        device (torch.device): The device to test the model on.
    """
    model.eval()
    with torch.no_grad():
        for images, cx, cy, targets in dataloader:
            images = images.to(device)
            cx = cx.to(device)
            cy = cy.to(device)
            targets = targets.to(device)

            outputs = model(images.permute(0, 3, 1, 2), cx, cy)  # Assuming images are H, W, C
            # Calculate and print evaluation metrics (e.g., MSE)
            print(f'Predictions: {outputs}, Targets: {targets}')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((480, 640)),  # Resize images to match YOLOv8 input
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset and dataloader
dataset = HandleDataset(root_dir=r'/media/datadisk10tb/leo/projects/realman-robot/open_door/data/test', transform=data_transform)
dataset = [data for data in dataset if data is not None]
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load pretrained YOLOv8 model
yolo_model = YOLO("cfg/yolov8x.pt")  # Load the desired YOLOv8 model

# Create the modified YOLOv8 model
model = ModifiedYOLOv8(yolo_model).to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Train the model
train(model, dataloader, optimizer, criterion, device, epochs=10)

# Test the model
# test(model, dataloader, device)