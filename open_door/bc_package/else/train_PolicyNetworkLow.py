import torch
from PolicyNetwork import PolicyNetworkLow,gen_dataloader,N_ENCODED_FEATURES

N_EPOCHS = 10
BATCH_SIZE = 2
LR = 0.001
train_data_dir = './data/train_data'
model_save_path = "./checkpoints/PolicyNetworkLow.pt"

def train():
    ## get the dataloder
    dataloader = gen_dataloader(data_dir=train_data_dir,batch_size=BATCH_SIZE,num_workers=0,shuffle=True)
    
    ## init model
    model = PolicyNetworkLow()
    model.train()

    ## define the loss function
    low_level_lossfunc = torch.nn.MSELoss()

    ## define the optimizer
    optimizer = torch.optim.Adam(params = model.parameters(), lr = LR) # optimizer = torch.optim.SGD(params = model.parameters(), lr = LR)

    for epoch in range(N_EPOCHS):
        train_loss = 0.0
        for data, labels_sq, labels_params in dataloader:  # data: # BATCH_SIZE * N_ENCODED_FEATURES * 1; labels_sq: # BATCH_SIZE * N_SEQUENCES; labels_params: BATCH_SIZE * N_SEQUENCES * N_PARAMS
            optimizer.zero_grad()
            data = torch.cat((data.view(-1, N_ENCODED_FEATURES), labels_sq), dim=1) # data: BATCH_SIZE * (N_ENCODED_FEATURES + N_SEQUENCES)
            low_level_params = model(data)  # output: BATCH_SIZE * N_SEQUENCES * TYPES_SEQUENCES
            loss = 0
            for i in range(len(low_level_params)):
                loss += low_level_lossfunc(low_level_params[i], labels_params[i]) # loss += low_level_lossfunc(low_level_params[i], labels[i]) # output[i]: N_SEQUENCES * TYPES_SEQUENCES; labels[i]: N_SEQUENCES
            loss.backward()
            optimizer.step()
            train_loss += loss.item()   
        train_loss = train_loss / len(dataloader.dataset)
        print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
    model.save(save_path=model_save_path)

if __name__ == "__main__":
    train()