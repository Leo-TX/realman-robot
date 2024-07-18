import torch
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import json
from PIL import Image
import numpy as np
from torchvision import transforms

# Step 1: Define the HandleDataset class
class HandleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = sorted([f for f in os.listdir(self.root_dir) if f.endswith('.png')])
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet stats
                                     std=[0.229, 0.224, 0.225])])
        else:
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

# Step 2: Modify the YOLOv8 model to accept (image, cx, cy) as input
class ModifiedYOLOv8(torch.nn.Module):
    def __init__(self, pretrained=True, model_path="yolov8n.pt"):
        super(ModifiedYOLOv8, self).__init__()
        self.yolo = YOLO(model_path) if pretrained else YOLO("yolov8n")
        
        head_layers = self.yolo.model.yaml["head"]
        last_layer_ch = self.find_last_conv_ch(head_layers) 
        if last_layer_ch is None:
            raise ValueError("Could not determine the channel count of the last convolutional layer.")
        self.fc = torch.nn.Linear(last_layer_ch, 3)  # 3 output features: dx, dy, R

    def find_last_conv_ch(self, layers):
        for i in range(len(layers) - 1, -1, -1):
            layer = layers[i]
            if isinstance(layer, list) and layer[2] == 'C2f':
                return layer[3][0]  # Get the channel count from C2f layer definition
        print('Not found')
        return None  # Not found

    def forward(self, image, cx, cy):
        results = self.yolo(image)  # Get the Results object
        # print(f'results:\n{results}')

        # Check if any detections exist
        if results[0].boxes.cls is not None and len(results[0].boxes.cls) > 0: 
            features = results[0].boxes.cls  
            features = features.mean(dim=(1, 2))
            output = self.fc(features) 
            return output
        else:
            # Handle cases with no detections
            # You might want to return a default value or modify your loss function
            # to handle this situation gracefully. 
            print("No detections found in this batch!")
            return torch.zeros(3)  # Example: Return a tensor of zeros

# Step 3: Define data loaders, optimizer, loss function
train_dataset = HandleDataset(root_dir=r'/media/datadisk10tb/leo/projects/realman-robot/open_door/data/images3_png_aug')
train_dataset = [data for data in train_dataset if data is not None]
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model = ModifiedYOLOv8(pretrained=True,model_path='cfg/yolov8x.pt')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()  # Using Mean Squared Error for regression

# Step 4: Training loop with loss visualization
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, cx, cy, targets) in enumerate(train_loader):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(images, cx, cy)

        # Calculate loss only if there are valid outputs
        if outputs.requires_grad:
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if i % 100 == 99:  # Print every 100 mini-batches
            print(f"Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

    # Append epoch loss for visualization
    losses.append(running_loss)

# Visualize the training loss
plt.plot(range(1, num_epochs + 1), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# Save the trained model
torch.save(model.state_dict(), "handle_grasp_model.pth")
print("Model saved successfully!")