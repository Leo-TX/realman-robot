import os
import json
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from _primitive import _Primitive

# For network
V_RESNET = 50 # ResNet-18 / 34 / 50
N_ENCODED_FEATURES = 2048

N_FC1 = 512
N_FC2 = 128
IF_CROP = True
IF_DROPOUT = False
VALUE_DROPOUT = 0.3  # 0.2 - 0.5

# Primitives Design
N_SEQUENCES = 1 
TYPES_SEQUENCES = 8  # [HOME, PREMOVE, GRASP, UNLOCK, ROTATE, OPEN, START, FINISH]
N_PARAMS = 3  # Low-level parameter dimension for grasp, others are 1-dimensional
N_INPUT = N_ENCODED_FEATURES+N_SEQUENCES+N_SEQUENCES*N_PARAMS+N_SEQUENCES


class PolicyNetwork(torch.nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()

        ## High-level policy
        self.high_level_fc1 = torch.nn.Linear(N_INPUT, N_FC1)
        self.high_level_relu1 = torch.nn.ReLU()
        self.high_level_dropout1 = torch.nn.Dropout(VALUE_DROPOUT) if IF_DROPOUT else None
        self.high_level_fc2 = torch.nn.Linear(N_FC1, N_FC2)
        self.high_level_relu2 = torch.nn.ReLU()
        self.high_level_dropout2 = torch.nn.Dropout(VALUE_DROPOUT) if IF_DROPOUT else None
        self.high_level_fc3 = torch.nn.Linear(N_FC2, N_SEQUENCES * TYPES_SEQUENCES)
        self.high_level_reshape = torch.nn.Unflatten(dim=1, unflattened_size=(N_SEQUENCES, TYPES_SEQUENCES))
        self.high_level_softmax = torch.nn.Softmax(dim=2)

        ## Low-level policy
        self.low_level_fc1 = torch.nn.Linear(N_INPUT + N_SEQUENCES, N_FC1)
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
        high_level_x = x.view(-1, N_INPUT)  # BATCH_SIZE * N_INPUT
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
        _, high_level_id = torch.max(high_level_x.data, 2)  # BATCH_SIZE * N_SEQUENCES

        ## Low-level policy
        low_level_x = torch.cat((x.view(-1, N_INPUT), high_level_id), dim=1) # BATCH_SIZE * (N_INPUT+N_SEQUENCES)
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
        low_level_param = torch.normal(low_level_mean, low_level_std)  # BATCH_SIZE * N_SEQUENCES * N_PARAMS
        low_level_param = torch.clamp(low_level_param, -1, 1)  # BATCH_SIZE * N_SEQUENCES * N_PARAMS

        return high_level_id, low_level_param, high_level_x, low_level_mean, low_level_std

    def get_params(self, if_p=True):
        params = list(self.parameters())
        if if_p:
            for i, param in enumerate(params):
                print(f"Parameter {i}: {param.size()}")
        return params

    def save(self, save_path, if_dict=False):
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(save_path)
        if if_dict:
            torch.save(self.state_dict(), save_path)  # only save parameters
        else:
            torch.save(self, save_path)  # save model

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, last_pmt_list, this_pmt_list):
        self.img_list = img_list
        self.last_pmt_list = last_pmt_list
        self.this_pmt_list = this_pmt_list
    
    def __getitem__(self, index):
        img = self.img_list[index]
        last_pmt = self.last_pmt_list[index]
        this_pmt = self.this_pmt_list[index]
        
        ## Convert data_item to a PyTorch tensor if needed
        if isinstance(img, torch.Tensor):
            img = img.view(N_ENCODED_FEATURES, 1)  # Reshape to N_ENCODED_FEATURESx1
        else:
            img = torch.from_numpy(img).view(N_ENCODED_FEATURES, 1)

        ## Convert _Primitive attributes to tensors
        last_id = torch.tensor(last_pmt.id, dtype=torch.long) # torch.Size([])
        last_param = torch.tensor(last_pmt.param, dtype=torch.float) # torch.Size([3])
        last_ret = torch.tensor(last_pmt.ret, dtype=torch.long) # torch.Size([])
        this_id = torch.tensor(this_pmt.id, dtype=torch.long) # torch.Size([])
        this_param = torch.tensor(this_pmt.param, dtype=torch.float) # torch.Size([3])
        this_ret = torch.tensor(this_pmt.ret, dtype=torch.long) # torch.Size([])

        return img, last_id, last_param, last_ret, this_id, this_param, this_ret

    def __len__(self):
        return len(self.img_list)

    def __str__(self):
        print(f'=== Custom Dataset ===')
        print(f'img_list shape: {np.shape(self.img_list)}')
        print(f'last_pmt_list shape: {np.shape(self.last_pmt_list)}')
        print(f'this_pmt_list shape: {np.shape(self.this_pmt_list)}')
        print(f'======================')
        return ''

def gen_dataloader(data_dir, batch_size, num_workers=0, shuffle=True):
    img_list = []
    last_pmt_list = []
    this_pmt_list = []
    
    ## Loop through the files in the folder
    for tjt_name in os.listdir(data_dir):
        tjt_dir = os.path.join(data_dir, tjt_name)
        print(f'Getting the data of {tjt_dir}')
        for file_name in os.listdir(tjt_dir):
            if file_name.endswith('.png'):
                img_path = os.path.join(tjt_dir, file_name)
                json_path = os.path.join(tjt_dir, file_name.replace('.png', '.json'))
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        json_data = json.load(f)

                        img = resnet_img2feature(img_path)
                        img_list.append(img)

                        last_action = json_data['last_action']
                        last_id = json_data['last_id']
                        last_param = json_data['last_param']
                        last_ret = json_data['last_ret']
                        last_error = json_data['last_error']
                        last_pmt = _Primitive(action=last_action,id=last_id,ret=last_ret,param=last_param,error=last_error)
                        last_pmt_list.append(last_pmt)
                        # print(last_pmt)

                        this_action = json_data['this_action']
                        this_id = json_data['this_id']
                        this_param = json_data['this_param']
                        this_ret = json_data['this_ret']
                        this_error = json_data['this_error']
                        this_pmt = _Primitive(action=this_action, id=this_id, ret=this_ret, param=this_param, error=this_error)
                        this_pmt_list.append(this_pmt)
                        # print(this_pmt)

    ## get the dataset
    dataset = CustomDataset(img_list=img_list, last_pmt_list=last_pmt_list, this_pmt_list=this_pmt_list)
    print(dataset)
    
    ## get the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    
    return dataloader

def resnet_img2feature(img_path):
    ## Load the pre-trained ResNet model
    if V_RESNET == 18:
        model = models.resnet18(weights=models.resnet.ResNet18_Weights.IMAGENET1K_V1)
    elif V_RESNET == 34:
        model = models.resnet34(weights=models.resnet.ResNet34_Weights.IMAGENET1K_V1)
    elif V_RESNET == 50:
        model = models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V1)
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
    train_data_dir = './data/train_data'
    dataloader = gen_dataloader(data_dir=train_data_dir,batch_size=8,num_workers=0,shuffle=True)