"""
@file loss.py
@brief Create a custom loss function for the model.
@details The custom loss function is a combination of
the Dice loss and the Binary Cross-Entropy loss.
@author Created by C. Factor on 01/03/2024"""
import torch.nn as nn


# Define SoftDiceLoss
class SoftDiceLoss(nn.Module):
    """
    @class SoftDiceLoss
    @param nn.Module: Base class for all neural network modules
    """

    def __init__(self):
        """
        @brief Constructor for the SoftDiceLoss class
        """
        # Calls the constructor of the base class (nn.Module)
        super().__init__()

    def forward(self, y_pred, y_true, epsilon=1e-6):
        """
        @brief Forward pass of the Soft
        @param y_pred: The predicted values
        @param y_true: The true masks
        @param epsilon: The epsilon value to prevent division by zero
        @return dice_loss: The Dice loss"""
        batch_size = y_pred.shape[0]
        dice_loss = 0.0
        for i in range(batch_size):
            # Flattened for faster calculation using view(-1)
            y_true_flat = y_true[i].contiguous().view(-1)
            y_pred_flat = y_pred[i].contiguous().view(-1)
            # Sigmoid the predicted values
            intersection = (y_true_flat * y_pred_flat).sum()
            union = y_true_flat.sum() + y_pred_flat.sum()

            # Calculate Dice score
            dice_score = (2 * intersection) / (union + epsilon)

            # Accumulate the Dice loss
            dice_loss += 1 - dice_score

        # Average the Dice loss over the batch
        dice_loss /= batch_size
        return dice_loss


# Define a Custom loss
class CustomLoss(nn.Module):
    """
    @class CustomLoss
    @param nn.Module: Base class for all neural network modules
    """

    def __init__(self):
        """
        @brief Constructor for the CustomLoss class
        """
        # Calls the constructor of the base class (nn.Module)
        super().__init__()
        self.dice_loss = SoftDiceLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, y_pred, y_true):
        """
        @brief Forward pass of the CustomLoss
        @param y_pred: The predicted masks
        @param y_true: The true masks
        @return custom_loss: The custom loss value, weighting the Dice and BCE losses by 0.5 each
        """
        # Overall loss calculation
        dice_loss = self.dice_loss(y_pred, y_true)
        bce_loss = self.bce_loss(y_pred, y_true)
        custom_loss = 0.5 * dice_loss + 0.5 * bce_loss
        return custom_loss
