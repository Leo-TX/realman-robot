import torch
from PolicyNetwork import gen_dataloader

BATCH_SIZE = 2
test_data_dir = './data/test_data'
model_save_path = "./checkpoints/PolicyNetworkHighLow.pt"

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

def test():
    ## get the dataloder
    dataloader = gen_dataloader(data_dir=test_data_dir,batch_size=BATCH_SIZE,num_workers=0,shuffle=False)

    ## load model
    model = torch.load(model_save_path)
    model.eval()  # Set model to evaluation mode

    high_level_correct = 0
    total_samples = 0
    low_level_correct = 0
    total_low_level_preds = 0

    with torch.no_grad():
        for data, labels_sq, labels_params in dataloader:
            high_level_output, high_level_sq, low_level_params = model(data)

            ## High-level accuracy
            _, predicted_sq = torch.max(high_level_output, 2)
            # print(f'labels_sq: {labels_sq}')
            # print(f'predicted_sq: {predicted_sq}')
            high_level_correct += torch.sum(torch.all(torch.eq(labels_sq, predicted_sq), dim=1)).item()
            # print(f'high_level_correct: {high_level_correct}')
            total_samples += labels_sq.size(0)
            # print(f'total_samples: {total_samples}')

            ## Low-level accuracy (conditional on high-level being correct)
            for i in range(len(high_level_output)):
                if torch.equal(predicted_sq[i], labels_sq[i]):  # Only if high-level prediction is correct
                    # print(f'find a correct!!:  \nlow_level_params[i]: {low_level_params[i]} \nlabels_params[i]: {labels_params[i]}')
                    for j in range(len(labels_sq[i])):
                        # print(f'low_level_params[i][j]: {low_level_params[i][j]} \nlabels_params[i][j]: {labels_params[i][j]}')
                        is_low_level_correct = your_low_level_accuracy_metric(low_level_params[i][j], labels_params[i][j]) 
                        low_level_correct += is_low_level_correct
                        total_low_level_preds += 1

    high_level_accuracy = high_level_correct / total_samples
    low_level_accuracy = low_level_correct / total_low_level_preds if total_low_level_preds > 0 else 0.0

    print(f'High-level Accuracy: {high_level_accuracy:.4f}')
    print(f'Low-level Accuracy (Conditional): {low_level_accuracy:.4f}')

if __name__ == "__main__":
    test()