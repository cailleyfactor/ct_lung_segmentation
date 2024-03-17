import torch
import torch.nn as nn


# Define SoftDiceLoss
class SoftDiceLoss(nn.Module):
    def __init__(self):
        # Calls the constructor of the base class (nn.Module)
        super().__init__()

    def forward(self, y_pred, y_true, epsilon=1e-6):
        intersection = (y_true * y_pred).sum()
        # Check the dice loss
        dice_loss = 1 - (2 * intersection) / torch.sum(y_pred + y_true + epsilon)
        return dice_loss


# Define a Custom loss
class CustomLoss(nn.Module):
    def __init__(self):
        # Calls the constructor of the base class (nn.Module)
        super().__init__()
        self.dice_loss = SoftDiceLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, y_pred, y_true):
        # Overall loss calculation
        dice_loss = self.dice_loss(y_pred, y_true)
        bce_loss = self.bce_loss(y_pred, y_true)
        custom_loss = dice_loss + bce_loss
        return custom_loss
