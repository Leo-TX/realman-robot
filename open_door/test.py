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

BATCH_SIZE = 1
N_WORKERS = 0
SHUFFLE = False

test_data_dir = './data/test_data'
model_save_path = './checkpoints/PolicyNetwork2.pt'

def test():
    ## get the dataloder
    dataloader = gen_dataloader(data_dir=test_data_dir,batch_size=BATCH_SIZE,num_workers=N_WORKERS,shuffle=SHUFFLE)
    
    ## load model
    model = torch.load(model_save_path)
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        ## get test data
        for batch_idx, (img, last_id, last_param, last_ret, this_id, this_param, this_ret) in enumerate(dataloader):
            if batch_idx == 0:
                ## Reshape tensors for concatenation
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
                
                ## print (for every batch)
                for i in range(len(high_level_id)):
                    print(f'last_id:{last_id[i]}')
                    print(f'last_ret:{last_ret[i]}')
                    print(f'last_param:{last_param[i]}')
                    print()
                    print(f'high_level_id:{high_level_id[i]}')
                    print(f'low_level_param:{low_level_param[i]}')
                    print()
                    print(f'high_level_x:{high_level_x[i]}')
                    print(f'low_level_mean:{low_level_mean[i]}')
                    print(f'low_level_std:{low_level_std[i]}')

if __name__ == "__main__":
    test()