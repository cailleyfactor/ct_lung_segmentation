"""
@file: train.py
@brief: This file contains the function to train the model
@author Created by C. Factor on 01/03/2024
"""
import torch
import torch.optim as optim
from torchmetrics.classification import BinaryAccuracy
from UNet import UNet
from loss import CustomLoss
from tqdm import tqdm


# Function to train the model
def train_model(
    train_loader, val_loader, device, lr, in_channels, out_channels, num_epochs
):
    """
    @brief Train the model
    @param train_loader: DataLoader for the training set
    @param val_loader: DataLoader for the validation set
    @param device: Device to run the model on
    @param lr: Learning rate for the optimizer
    @param in_channels: Number of input channels
    @param out_channels: Number of output channels
    @param num_epochs: Number of epochs to train the model
    @return model: Trained model
    @return train_losses: List of training losses
    @return train_accuracies: List of training accuracies
    @return val_losses: List of validation losses
    @return val_accuracies: List of validation accuracies
    """
    # Initialise loss, model, and optimizer
    model = UNet(in_channels, out_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = CustomLoss()

    # Accuracy metric
    binary_accuracy = BinaryAccuracy(threshold=0.5).to(device)

    # Lists to store training and validation results
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Training and validation loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        train_total_loss = 0.0
        train_total_accuracy = 0.0
        val_total_loss = 0.0
        val_total_accuracy = 0.0

        # Training loop
        for images, masks, _, _ in tqdm(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()  # Clear old gradients in the last step
            outputs = torch.sigmoid(model(images))
            loss = criterion(outputs, masks)  # Forward pass
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters based on gradients
            accuracy = binary_accuracy(
                outputs, masks
            )  # Binary accuracy automatically thresholds my data

            # Accumulate avg batchloss
            accuracy = binary_accuracy(outputs, masks)
            train_total_loss += loss.item()
            train_total_accuracy += accuracy.item()

        # Validation loop
        model.eval()
        with torch.no_grad():
            # Validation loop
            for images, masks, _, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = torch.sigmoid(model(images))
                loss = criterion(outputs, masks)
                accuracy = binary_accuracy(outputs, masks)

                # Accumulate avg batch loss and accuracy
                val_total_loss += loss.item()
                val_total_accuracy += accuracy.item()

        # Calculate average loss and accuracy for this epoch by dividing by the number of batches
        avg_train_loss = train_total_loss / len(train_loader)
        avg_train_accuracy = train_total_accuracy / len(train_loader)
        avg_val_loss = val_total_loss / len(val_loader)
        avg_val_accuracy = val_total_accuracy / len(val_loader)

        # Append the results to the lists
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)

        # Print training and validation results for the epoch
        print(
            f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f},\
               Training Accuracy: {avg_train_accuracy:.4f}, \
                Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}"
        )

    return model, train_losses, train_accuracies, val_losses, val_accuracies
