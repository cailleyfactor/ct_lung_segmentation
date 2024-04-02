"""
@file eval.py
@brief Evaluate the model on the validation set and calculate metrics.
@details This file contains the code to evaluate the model on the validation set
and calculate metrics such as accuracy and dice score."""
import torch
import pandas as pd
from torchmetrics.classification import BinaryAccuracy


# Sigmoid bit
def dice_coefficient(y_pred, y_true, epsilon=1e-6):
    """
    @param y_pred: Predicted mask
    @param y_true: True mask
    @param epsilon: Coefficient to prevent division by zero
    """
    intersection = (y_true * y_pred).sum()
    dice_coeff = (2 * intersection) / torch.sum(y_pred + y_true + epsilon)
    return dice_coeff


def evaluation(device, model, loader):
    """
    @param device: Device to run the model on
    @param model: Model to evaluate
    @param loader: Data loader for the validation set
    @return store: Dataframe containing the evaluation results"""
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
                # Calculate metrics and convert them to standard Python floats
                accuracy = binary_accuracy(
                    probabilities[slice_index], masks[slice_index]
                ).item()
                dice_score = dice_coefficient(
                    predictions[slice_index], masks[slice_index]
                ).item()
                img_array = (
                    images[slice_index].cpu().numpy()
                )  # Assuming tensors are on a specific device
                mask_array = masks[slice_index].cpu().numpy()
                prediction_array = predictions[slice_index].cpu().numpy()
                # Store indices and dice score in dice store
                store.append(
                    {
                        "patient_id": patient_id[slice_index],
                        "slice_info": slice_info[slice_index].item(),
                        "accuracy": accuracy,
                        "dice_score": dice_score,
                        "image": img_array,  # Storing image as a numpy array
                        "mask": mask_array,  # Storing mask as a numpy array
                        "prediction": prediction_array,
                    }
                )

    store = pd.DataFrame(store)
    return store
