import torch
import pandas as pd
from torchmetrics.classification import BinaryAccuracy


# Sigmoid bit
def dice_coefficient(y_pred, y_true, epsilon=1e-6):
    intersection = (y_true * y_pred).sum()
    dice_coeff = (2 * intersection) / torch.sum(y_pred + y_true + epsilon)
    return dice_coeff


def evaluation(device, model, loader):
    model.eval()
    # Set model to evaluation mode
    store = []
    # Accuracy metric
    binary_accuracy = BinaryAccuracy(threshold=0.5).to(device)
    # Stop tracking gradients
    with torch.no_grad():
        for images, masks, patient_id, slice_info in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            # Add a sigmoid to the outputs to convert to probabilities
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()

            for slice_index in range(images.size(0)):
                # Calculate binary accuracy
                accuracy = binary_accuracy(
                    probabilities[slice_index], masks[slice_index]
                )

                # Calculate dice score
                dice_score = dice_coefficient(
                    predictions[slice_index], masks[slice_index]
                )

                # Store indices and dice score in dice store
                store.append(
                    {
                        "patient_id": patient_id[slice_index],
                        "slice_info": slice_info[slice_index],
                        "accuracy": accuracy,
                        "dice_score": dice_score,
                    }
                )

    store = pd.DataFrame(store)
    return store
