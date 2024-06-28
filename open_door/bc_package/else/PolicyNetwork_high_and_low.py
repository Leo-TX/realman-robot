import os
import json
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# For network
V_RESNET = 34 # ResNet-18 / 34 / 50
N_ENCODED_FEATURES = 1024
N_FC1 = 512
N_FC2 = 128
IF_CROP = True
IF_DROPOUT = True
VALUE_DROPOUT = 0.3  # 0.2 - 0.5
# Primitives Design
N_SEQUENCES = 3  # [1,2,4]
TYPES_SEQUENCES = 5  # [None, Grasp, Unlock, Rotate, Open]
N_PARAMS = 3  # Low-level parameter dimension for grasp, others are 1-dimensional

class PolicyNetworkHigh(torch.nn.Module): 
    def __init__(self):
        super(PolicyNetworkHigh,self).__init__() 
        self.high_level_fc1 = torch.nn.Linear(N_ENCODED_FEATURES, N_FC1)
        self.high_level_relu1 = torch.nn.ReLU()
        self.high_level_dropout1 = torch.nn.Dropout(VALUE_DROPOUT) if IF_DROPOUT else None
        self.high_level_fc2 = torch.nn.Linear(N_FC1, N_FC2)
        self.high_level_relu2 = torch.nn.ReLU()
        self.high_level_dropout2 = torch.nn.Dropout(VALUE_DROPOUT) if IF_DROPOUT else None
        self.high_level_fc3 = torch.nn.Linear(N_FC2, N_SEQUENCES * TYPES_SEQUENCES)
        self.high_level_reshape = torch.nn.Unflatten(dim=1, unflattened_size=(N_SEQUENCES, TYPES_SEQUENCES))
        self.high_level_softmax = torch.nn.Softmax(dim=2)
    
    def forward(self,x):
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
        
        return high_level_x
    
    def get_params(self,if_p=True):
        params = list(self.parameters())
        if if_p:
            for i, param in enumerate(params):
                print(f"Parameter {i}: {param.size()}")
        return params

    def save(self,save_path,if_dict=False):
        if if_dict:
            torch.save(self.state_dict(),save_path) # only save parameters
        else:
            torch.save(self,save_path) # save model

class PolicyNetworkLow(torch.nn.Module):
    def __init__(self):
        super(PolicyNetworkLow, self).__init__()
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
        low_level_x = x.view(-1, N_ENCODED_FEATURES+N_SEQUENCES)  # BATCH_SIZE * (N_ENCODED_FEATURES+N_SEQUENCES)
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
        low_level_params = torch.normal(low_level_mean, low_level_std)  # BATCH_SIZE * N_SEQUENCES * N_PARAMS
        low_level_params = torch.clamp(low_level_params, -1, 1)  # BATCH_SIZE * N_SEQUENCES * N_PARAMS

        return low_level_params

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