import torch.nn as nn


# Define SoftDiceLoss
class SoftDiceLoss(nn.Module):
    def __init__(self):
        # Calls the constructor of the base class (nn.Module)
        super().__init__()

    def forward(self, y_pred, y_true, epsilon=1e-6):
        batch_size = y_pred.shape[0]
        dice_loss = 0.0
        for i in range(batch_size):
            # Flattened for faster calculation using view(-1)
            # ChatGPT recommended that this is how I flatten
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
    def __init__(self):
        # Calls the constructor of the base class (nn.Module)
        super().__init__()
        self.dice_loss = SoftDiceLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, y_pred, y_true):
        # Overall loss calculation
        dice_loss = self.dice_loss(y_pred, y_true)
        bce_loss = self.bce_loss(y_pred, y_true)
        custom_loss = 0.5 * dice_loss + 0.5 * bce_loss
        return custom_loss
