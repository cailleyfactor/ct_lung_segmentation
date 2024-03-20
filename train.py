import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import BinaryAccuracy
from UNet import UNet
from loss import CustomLoss


# Parameter intialisation
def init_weights(model):
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(model.weight, mode="fan_out", nonlinearity="relu")
        if model.bias is not None:
            nn.init.constant_(model.bias, 0)


# Function to train the model
def train_model(
    train_loader, val_loader, device, lr, in_channels, out_channels, num_epochs
):
    # Initialise loss, model, and optimizer
    model = UNet(in_channels, out_channels).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = CustomLoss()

    # Accuracy metric
    binary_accuracy = BinaryAccuracy(threshold=0.5).to(device)
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
        for images, masks, _, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()  # Clear old gradients in the last step
            outputs = torch.sigmoid(model(images))
            loss = criterion(outputs, masks)  # Forward pass
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters based on gradients
            accuracy = binary_accuracy(
                outputs, masks
            )  # Binary accuracy automatically thresholds my data

            # Accumulate accuracy and loss
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

                # Accumulate accuracy and loss
                val_total_loss += loss.item()
                val_total_accuracy += accuracy.item()

        # Calculate average loss and accuracy for this epoch and append to the respective lists
        avg_train_loss = train_total_loss / len(train_loader)
        avg_train_accuracy = train_total_accuracy / len(train_loader)
        avg_val_loss = val_total_loss / len(val_loader)
        avg_val_accuracy = val_total_accuracy / len(val_loader)

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
