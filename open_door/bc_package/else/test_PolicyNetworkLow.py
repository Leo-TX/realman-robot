import torch
from PolicyNetwork import gen_dataloader,N_ENCODED_FEATURES

BATCH_SIZE = 2
eval_data_dir = './data/eval_data'
model_save_path = "./checkpoints/PolicyNetworkLow.pt"

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

def eval():
    ## get the dataloder
    dataloader = gen_dataloader(data_dir=eval_data_dir,batch_size=BATCH_SIZE,num_workers=0,shuffle=False)

    ## load model
    model = torch.load(model_save_path)
    model.eval()  # Set model to evaluation mode

    low_level_correct = 0
    total_low_level_preds = 0

    with torch.no_grad():
        for data, labels_sq, labels_params in dataloader:
            data = torch.cat((data.view(-1, N_ENCODED_FEATURES), labels_sq), dim=1) # data: BATCH_SIZE * (N_ENCODED_FEATURES + N_SEQUENCES)
            low_level_params = model(data)
            for i in range(len(low_level_params)):
                # print(f'find a correct!!:  \nlow_level_params[i]: {low_level_params[i]}')
                for j in range(len(labels_sq[i])):
                    # print(f'low_level_params[i][j]: {low_level_params[i][j]} \nlabels_params[i][j]: {labels_params[i][j]}')
                    is_low_level_correct = your_low_level_accuracy_metric(low_level_params[i][j], labels_params[i][j]) 
                    low_level_correct += is_low_level_correct
                    total_low_level_preds += 1

    low_level_accuracy = low_level_correct / total_low_level_preds if total_low_level_preds > 0 else 0.0
    print(f'Low-level Accuracy (Conditional): {low_level_accuracy:.4f}')

if __name__ == "__main__":
    eval()