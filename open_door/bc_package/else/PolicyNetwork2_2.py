import os
import json
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models

PROCESS = 'test'
N_EPOCHS = 10
BATCH_SIZE = 1
N_WORKERS = 0
N_ENCODED_FEATURES = 1024
N_FC1 = 512
N_FC2 = 128
LR = 0.001
IF_CROP = True
IF_DROPOUT = True
IF_SHUFFLE = True
VALUE_DROPOUT = 0.3  # 0.2 - 0.5
N_SEQUENCES = 3  # [1,2,4]
TYPES_SEQUENCES = 5  # [None, Grasp, Unlock, Rotate, Open]
N_PARAMS = 3  # Low-level parameter dimension for grasp, others are 1-dimensional

train_data_dir = './data/train_data'
test_data_dir = './data/test_data'
model_save_path = "./checkpoints/PolicyNetwork2.pt"

class PolicyNetwork2(torch.nn.Module):
    def __init__(self):
        super(PolicyNetwork2, self).__init__()

        ## High-level policy
        self.high_level_fc1 = torch.nn.Linear(N_ENCODED_FEATURES, N_FC1)
        self.high_level_relu1 = torch.nn.ReLU()
        self.high_level_dropout1 = torch.nn.Dropout(VALUE_DROPOUT) if IF_DROPOUT else None
        self.high_level_fc2 = torch.nn.Linear(N_FC1, N_FC2)
        self.high_level_relu2 = torch.nn.ReLU()
        self.high_level_dropout2 = torch.nn.Dropout(VALUE_DROPOUT) if IF_DROPOUT else None
        self.high_level_fc3 = torch.nn.Linear(N_FC2, N_SEQUENCES * TYPES_SEQUENCES)
        self.high_level_reshape = torch.nn.Unflatten(dim=1, unflattened_size=(N_SEQUENCES, TYPES_SEQUENCES))
        self.high_level_softmax = torch.nn.Softmax(dim=2)

        ## Low-level policy
        self.low_level_fc1 = torch.nn.Linear(N_ENCODED_FEATURES + N_SEQUENCES, N_FC1)
        self.low_level_relu1 = torch.nn.ReLU()
        self.low_level_dropout1 = torch.nn.Dropout(VALUE_DROPOUT) if IF_DROPOUT else None
        self.low_level_fc2 = torch.nn.Linear(N_FC1, N_FC2)
        self.low_level_relu2 = torch.nn.ReLU()
        self.low_level_dropout2 = torch.nn.Dropout(VALUE_DROPOUT) if IF_DROPOUT else None
        self.low_level_mean_fc = torch.nn.Linear(N_FC2, N_SEQUENCES * N_PARAMS)  # Output mean for all primitives
        self.low_level_std_fc = torch.nn.Linear(N_FC2, N_SEQUENCES * N_PARAMS)  # Output standard deviation for all primitives
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        ## High-level policy
        high_level_x = x.view(-1, N_ENCODED_FEATURES)  # BATCH_SIZE * N_ENCODED_FEATURES
        high_level_x = self.high_level_fc1(high_level_x)  # BATCH_SIZE * N_FC1
        high_level_x = self.high_level_relu1(high_level_x)  # BATCH_SIZE * N_FC1
        high_level_x = self.high_level_dropout1(high_level_x) if self.high_level_dropout1 else high_level_x  # BATCH_SIZE * N_FC1
        high_level_x = self.high_level_fc2(high_level_x)  # BATCH_SIZE * N_FC2
        high_level_x = self.high_level_relu2(high_level_x)  # BATCH_SIZE * N_FC2
        high_level_x = self.high_level_dropout2(high_level_x) if self.high_level_dropout2 else high_level_x  # BATCH_SIZE * N_FC2
        high_level_x = self.high_level_fc3(high_level_x)  # BATCH_SIZE * (N_SEQUENCES*TYPES_SEQUENCES)
        high_level_x = self.high_level_reshape(high_level_x)  # BATCH_SIZE * N_SEQUENCES * TYPES_SEQUENCES
        high_level_x = self.high_level_softmax(high_level_x)  # BATCH_SIZE * N_SEQUENCES * TYPES_SEQUENCES
        
        # Sample high-level actions
        _, high_level_actions = torch.max(high_level_x.data, 2)  # BATCH_SIZE * N_SEQUENCES

        ## Low-level policy
        low_level_x = torch.cat((x.view(-1, N_ENCODED_FEATURES), high_level_actions), dim=1) # BATCH_SIZE * (N_ENCODED_FEATURES+N_SEQUENCES)
        low_level_x = self.low_level_fc1(low_level_x)  # BATCH_SIZE * N_FC1
        low_level_x = self.low_level_relu1(low_level_x)  # BATCH_SIZE * N_FC1
        low_level_x = self.low_level_dropout1(low_level_x) if self.low_level_dropout1 else low_level_x  # BATCH_SIZE * N_FC1
        low_level_x = self.low_level_fc2(low_level_x)  # BATCH_SIZE * N_FC2
        low_level_x = self.low_level_relu2(low_level_x)  # BATCH_SIZE * N_FC2
        low_level_x = self.low_level_dropout2(low_level_x) if self.low_level_dropout2 else low_level_x  # BATCH_SIZE * N_FC2

        # Output mean and standard deviation for low-level parameters
        low_level_mean = self.low_level_mean_fc(low_level_x)  # BATCH_SIZE * (N_SEQUENCES * N_PARAMS)
        low_level_std = self.low_level_std_fc(low_level_x)  # BATCH_SIZE * (N_SEQUENCES * N_PARAMS)

        # Apply activation functions
        low_level_mean = self.tanh(low_level_mean)  # BATCH_SIZE * (N_SEQUENCES * N_PARAMS)
        low_level_std = self.sigmoid(low_level_std)  # BATCH_SIZE * (N_SEQUENCES * N_PARAMS)

        # Reshape for each primitive and each time step
        low_level_mean = low_level_mean.view(-1, N_SEQUENCES, N_PARAMS)  # BATCH_SIZE * N_SEQUENCES * N_PARAMS
        low_level_std = low_level_std.view(-1, N_SEQUENCES, N_PARAMS)  # BATCH_SIZE * N_SEQUENCES * N_PARAMS

        # Sample low-level actions from Gaussian distributions and clip the actions from −1 to 1
        low_level_actions = torch.normal(low_level_mean, low_level_std)  # BATCH_SIZE * N_SEQUENCES * N_PARAMS
        low_level_actions = torch.clamp(low_level_actions, -1, 1)  # BATCH_SIZE * N_SEQUENCES * N_PARAMS

        return high_level_x, high_level_actions, low_level_actions

    def get_params(self, if_p=True):
        params = list(self.parameters())
        if if_p:
            for i, param in enumerate(params):
                print(f"Parameter {i}: {param.size()}")
        return params

    def save(self, save_path, if_dict=False):
        if if_dict:
            torch.save(self.state_dict(), save_path)  # only save parameters
        else:
            torch.save(self, save_path)  # save model

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels_sq, labels_params):
        self.data = data
        self.labels_sq = labels_sq
        self.labels_params = labels_params
    
    def __getitem__(self, index):
        data_item = self.data[index]
        label_sq_item = self.labels_sq[index]
        labels_params_item = self.labels_params[index]

        label_sq_item = torch.tensor(label_sq_item, dtype=torch.long)  # list to torch tensor
        labels_params_item = torch.tensor(labels_params_item, dtype=torch.float)  # list to torch tensor
        # Convert data_item to a PyTorch tensor if needed
        if isinstance(data_item, torch.Tensor):
            data_item = data_item.view(N_ENCODED_FEATURES, 1)  # Reshape to N_ENCODED_FEATURESx1
        else:
            data_item = torch.from_numpy(data_item).view(N_ENCODED_FEATURES, 1)
        return data_item, label_sq_item, labels_params_item

    def __len__(self):
        return len(self.data)

    def __str__(self):
        print(f'=== Custom Dataset ===')
        print(f'data shape: {np.shape(self.data)}')
        print(f'labels_sq shape: {np.shape(self.labels_sq)}')
        print(f'labels_sq shape: {np.shape(self.labels_params)}')
        print(f'======================')
        return ''

def train(dataloader):
    ## init model
    model = PolicyNetwork2()
    model.train()

    ## define the loss function
    high_level_lossfunc = torch.nn.CrossEntropyLoss()
    low_level_lossfunc = torch.nn.MSELoss()

    ## define the optimizer
    # optimizer = torch.optim.SGD(params = model.parameters(), lr = LR)
    optimizer = torch.optim.Adam(params = model.parameters(), lr = LR)

    for epoch in range(N_EPOCHS):
        train_loss = 0.0
        for data, labels_sq, labels_params in dataloader:  # data: # BATCH_SIZE * N_ENCODED_FEATURES * 1; labels_sq: # BATCH_SIZE * N_SEQUENCES; labels_params: BATCH_SIZE * N_SEQUENCES * N_PARAMS
            optimizer.zero_grad()
            high_level_output, high_level_actions, low_level_actions = model(data)  # output: BATCH_SIZE * N_SEQUENCES * TYPES_SEQUENCES
            loss = 0
            for i in range(len(high_level_output)):
                loss += high_level_lossfunc(high_level_output[i], labels_sq[i])  # output[i]: N_SEQUENCES * TYPES_SEQUENCES; labels[i]: N_SEQUENCES
                loss += low_level_lossfunc(low_level_actions[i], labels_params[i]) # loss += low_level_lossfunc(low_level_actions[i], labels[i]) # output[i]: N_SEQUENCES * TYPES_SEQUENCES; labels[i]: N_SEQUENCES
            loss.backward()
            optimizer.step()
            train_loss += loss.item()   
        train_loss = train_loss / len(dataloader.dataset)
        print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
    model.save(save_path=model_save_path)

# def test(dataloader):
#     ## load model
#     model = torch.load(model_save_path)
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data, labels_sq, labels_params in dataloader:  # data: # BATCH_SIZE * N_ENCODED_FEATURES; labels: # BATCH_SIZE * N_SEQUENCES
#             high_level_output, high_level_actions, low_level_actions = model(data)  # output: BATCH_SIZE * N_SEQUENCES * TYPES_SEQUENCES
#             # print(f'output:{output}')
#             print(f'labels_sq:{labels_sq}')
#             _, predicted = torch.max(high_level_output.data, 2)  # predicted: BATCH_SIZE * N_SEQUENCES
#             print(f'predicted:{predicted}')
#             total += labels_sq.size(0)
#             print(f'total:{total}')
#             correct += torch.sum(torch.all(torch.eq(labels_sq, predicted), dim=1)).item()
#             print(f'correct:{correct}')
#     print(f"Accuracy of the network on the test dataset = {correct}/{total} = {correct / total * 100:.2f}%")
#     return correct / total

def your_low_level_accuracy_metric(predicted_params, true_params, threshold=0.1):
    """
    Define your own metric to assess low-level parameter accuracy. 
    This example uses a simple threshold-based approach.

    Args:
        predicted_params (torch.Tensor): Predicted parameters.
        true_params (torch.Tensor): Ground truth parameters.
        threshold (float):  Maximum allowed difference for a parameter to be considered correct.

    Returns:
        bool: True if the prediction is considered correct, False otherwise. 
    """
    return torch.all(torch.abs(predicted_params - true_params) <= threshold).item()

def test(dataloader, model_path=model_save_path):
    model = torch.load(model_path)
    model.eval()  # Set model to evaluation mode

    high_level_correct = 0
    total_samples = 0
    low_level_correct = 0
    total_low_level_preds = 0

    with torch.no_grad():
        for data, labels_sq, labels_params in dataloader:
            high_level_output, high_level_actions, low_level_actions = model(data)

            # High-level accuracy
            _, predicted_sq = torch.max(high_level_output, 2)
            print(f'labels_sq: {labels_sq}')
            print(f'predicted_sq: {predicted_sq}')
            high_level_correct += torch.sum(torch.all(torch.eq(labels_sq, predicted_sq), dim=1)).item()
            print(f'high_level_correct: {high_level_correct}')
            total_samples += labels_sq.size(0)
            print(f'total_samples: {total_samples}')

            # Low-level accuracy (conditional on high-level being correct)
            for i in range(len(high_level_output)):
                if torch.equal(predicted_sq[i], labels_sq[i]):  # Only if high-level prediction is correct
                    print(f'find a correct!!:  \nlow_level_actions[i]: {low_level_actions[i]} \nlabels_params[i]: {labels_params[i]}')
                    for j in range(N_SEQUENCES):
                        # Calculate low-level accuracy (you can define your own metric here)
                        print(f'low_level_actions[i][j]: {low_level_actions[i][j]} \nlabels_params[i][j]: {labels_params[i][j]}')
                        is_low_level_correct = your_low_level_accuracy_metric(low_level_actions[i][j], labels_params[i][j]) 
                        low_level_correct += is_low_level_correct
                        total_low_level_preds += 1

    high_level_accuracy = high_level_correct / total_samples
    low_level_accuracy = low_level_correct / total_low_level_preds if total_low_level_preds > 0 else 0.0

    print(f'High-level Accuracy: {high_level_accuracy:.4f}')
    print(f'Low-level Accuracy (Conditional): {low_level_accuracy:.4f}')

def dataloader(data_dir, shuffle=True):
    encoded_features = []
    sequences = []
    params_list = []
    ## Loop through the files in the folder
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.png'):
            img_path = os.path.join(data_dir, file_name)
            json_path = os.path.join(data_dir, file_name.replace('.png', '.json'))
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                    encoded_feature = resnet18_img2feature(img_path)
                    encoded_features.append(encoded_feature)
                    sequence = json_data['sequence']
                    sequences.append(sequence)
                    params = json_data['params']
                    params_list.append(params)
    
    ## get the dataset
    dataset = CustomDataset(data=encoded_features, labels_sq=sequences, labels_params=params_list)
    print(dataset)
    
    ## get the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=shuffle)
    return dataloader

def resnet18_img2feature(img_path):
    ## Load the pre-trained ResNet18 model
    model = models.resnet18(weights=models.resnet.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    model.fc = torch.nn.Linear(model.fc.in_features, N_ENCODED_FEATURES)  # Adjust output size as needed

    ## Preprocess the input image
    img = Image.open(img_path)  # 1280*720
    # print(f'original img shape:{img.size}')
    transform = []
    if IF_CROP:
        transform.append(transforms.Resize((224, 224)))  # Resize the image to match the input size of ResNet18
    transform.append(transforms.ToTensor())  # Convert the image to a tensor
    transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))  # Normalize the imag
    transform = transforms.Compose(transform)
    input_tensor = transform(img).unsqueeze(0)  # [1,3,224,224]
    # print(f'input_tensor shape:{input_tensor.shape}')

    ## Move the image data to GPU if available for faster processing
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)

    ## Pass the input tensor through the model to get the encoded visual features
    with torch.no_grad():
        encoded_feature = model(input_tensor)  # [1,N_ENCODED_FEATURES]
    # print(f'encoded_feature shape:{encoded_feature.shape}')

    ## postprocess the encoded features
    encoded_feature = encoded_feature.squeeze(0)  # Remove the batch dimension (optional)
    encoded_feature_np = encoded_feature.cpu().numpy()  # [N_ENCODED_FEATURES], Convert the tensor to a NumPy array for further processing (optional)
    # print(f'encoded_feature_np shape:{encoded_feature_np.shape}')

    return encoded_feature_np

if __name__ == '__main__':
    if PROCESS == 'train':
        train_dataloader = dataloader(data_dir=train_data_dir, shuffle=True)
        train(train_dataloader)
    elif PROCESS == 'test':
        test_dataloader = dataloader(data_dir=test_data_dir, shuffle=False)
        test(test_dataloader)