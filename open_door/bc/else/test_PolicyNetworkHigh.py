import torch
from PolicyNetwork import gen_dataloader

BATCH_SIZE = 2
eval_data_dir = './data/eval_data'
model_save_path = "./checkpoints/PolicyNetworkHigh.pt"

def eval():
    ## get the dataloder
    dataloader = gen_dataloader(data_dir=eval_data_dir,batch_size=BATCH_SIZE,num_workers=0,shuffle=False)

    ## load model
    model = torch.load(model_save_path)
    model.eval()  # Set model to evaluation mode

    high_level_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, labels_sq, labels_params in dataloader:
            high_level_output = model(data)
            print(f'high_level_output: {high_level_output}')
            _, predicted_sq = torch.max(high_level_output, 2)
            # print(f'labels_sq: {labels_sq}')
            # print(f'predicted_sq: {predicted_sq}')
            high_level_correct += torch.sum(torch.all(torch.eq(labels_sq, predicted_sq), dim=1)).item()
            # print(f'high_level_correct: {high_level_correct}')
            total_samples += labels_sq.size(0)
            # print(f'total_samples: {total_samples}')

    high_level_accuracy = high_level_correct / total_samples
    print(f'High-level Accuracy: {high_level_accuracy:.4f}')

if __name__ == "__main__":
    eval()