'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-06-01 18:11:53
Version: v1
File: 
Brief: 
'''

import torch
from PolicyNetwork import PolicyNetwork,gen_dataloader

N_EPOCHS = 100
BATCH_SIZE = 16
LR = 0.01
N_WORKERS = 0
SHUFFLE = True

train_data_dir = './data/train_data'
model_save_path = './checkpoints/PolicyNetwork3.pt'

def train():
    ## get the dataloder
    dataloader = gen_dataloader(data_dir=train_data_dir,batch_size=BATCH_SIZE,num_workers=N_WORKERS,shuffle=SHUFFLE)
    
    ## init model
    model = PolicyNetwork()
    model.train()

    ## define the loss function
    high_level_lossfunc = torch.nn.CrossEntropyLoss()
    low_level_lossfunc = torch.nn.MSELoss()

    ## define the optimizer
    optimizer = torch.optim.Adam(params = model.parameters(), lr = LR) # optimizer = torch.optim.SGD(params = model.parameters(), lr = LR)

    ## every epoch
    for epoch in range(N_EPOCHS):
        train_loss = 0.0
        
        ## get training data
        for img, last_id, last_param, last_ret, this_id, this_param, this_ret in dataloader:
            optimizer.zero_grad()

            # # Reshape tensors for concatenation
            # img shape: (BATCH_SIZE, N_ENCODED_FEATURES, 1)
            last_id = last_id.unsqueeze(-1).unsqueeze(2)      # shape: (BATCH_SIZE, N_SEQUENCES, 1)
            last_ret = last_ret.unsqueeze(-1).unsqueeze(2)    # shape: (BATCH_SIZE, N_SEQUENCES, 1)
            last_param = last_param.unsqueeze(-1)  # shape: (BATCH_SIZE, N_PARAMS, 1)
            this_id = this_id.unsqueeze(-1)    # shape: (BATCH_SIZE, N_SEQUENCES)
            this_param = this_param.unsqueeze(-1).transpose(1, 2)  # shape: (BATCH_SIZE, N_SEQUENCES, N_PARAMS)

            ## get the input (img+last_pmt)
            input_data = torch.cat((img, last_id, last_param, last_ret), dim=1)   # shape: (BATCH_SIZE, (N_ENCODED_FEATURES+N_SEQUENCES+N_PARAMS+N_SEQUENCES), 1)
            
            ## forward
            high_level_id, low_level_param, high_level_x, low_level_mean, low_level_std = model(input_data)
            # high_level_id     torch.Size([BATCH_SIZE, N_SEQUENCES])
            # low_level_param   torch.Size([BATCH_SIZE, N_SEQUENCES, N_PARAMS])
            # high_level_x      torch.Size([BATCH_SIZE, N_SEQUENCES, TYPES_SEQUENCES])
            # low_level_mean    torch.Size([BATCH_SIZE, N_SEQUENCES, N_PARAMS])
            # low_level_std     torch.Size([BATCH_SIZE, N_SEQUENCES, N_PARAMS])

            ## calc loss (for every batch)
            loss = 0
            for i in range(len(high_level_id)):
                loss += high_level_lossfunc(high_level_x[i], this_id[i]) # high_level_x[i]: N_SEQUENCES * TYPES_SEQUENCES; this_id[i]: N_SEQUENCES
                loss += low_level_lossfunc(low_level_param[i], this_param[i]) # low_level_param[i]: N_SEQUENCES * N_PARAMS; this_param[i]: N_SEQUENCES * N_PARAMS
            train_loss += loss.item()

            ## backward            
            loss.backward()
            optimizer.step()
            
        train_loss = train_loss / len(dataloader.dataset)
        
        print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
    
    ## save model
    model.save(save_path=model_save_path)

if __name__ == "__main__":
    train()