import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
import random
import os
import numpy as np
from dataset import CustomDataset

# from train import train_model
from eval import evaluation
from train import train_model
from UNet import UNet

# Set seed for reproducibility
seed = 42
# Set the seed for generating random numbers in PyTorch
torch.manual_seed(seed)
# Sets the seed for the built-in random module
random.seed(seed)
# Set the seed for generating random numbers in Numpy
np.random.seed(seed)

# Dataset path
images_path = "Dataset/Images"
segmentation_path = "Dataset/Segmentations/"

# List of all the full paths to directories within images_path
dicom_dirs = sorted(
    [
        os.path.join(images_path, d)
        for d in os.listdir(images_path)
        if os.path.isdir(os.path.join(images_path, d))
    ]
)

# List to all of the .npz files in the segmentation_path
mask_paths = sorted(
    [
        os.path.join(segmentation_path, f)
        for f in os.listdir(segmentation_path)
        if f.endswith(".npz")
    ]
)

# Instantiate your dataset
dataset = CustomDataset(dicom_dirs, mask_paths)

# Device configuration
if torch.backends.mps.is_available():
    # Set the device to MPS
    device = torch.device("mps")
    print("Using MPS device")
else:
    # CPU if MPS is not available
    device = torch.device("cpu")
    print("MPS not available, using CPU")

# Define inputs to training loop
batch_size = 3

# Defining train and tsting sets with DataLoader
train_size = int(0.66667 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Define key data inputs
lr = 0.1
in_channels = 1
out_channels = 1
num_epochs = 50

model, training_losses, training_accuracies, val_losses, val_accuracies = train_model(
    train_loader, test_loader, device, lr, in_channels, out_channels, num_epochs
)

# Save the trained model
torch.save(model.state_dict(), "trained_UNet_2.pth")

# Plotting for 2b
# # Plotting for 1b
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(training_losses, label="Training Loss")
# plt.title("Losses")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(training_accuracies, label="Training Accuracy")
# plt.title("Accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.tight_layout()
# plt.show()


# Part 2d
# Code only when not using the train the model part of the code
model = UNet(in_channels, out_channels).to(device)

# Load the weights from a pretrained model into the
model.load_state_dict(torch.load("trained_UNet.pth", map_location=torch.device(device)))

# Run the evaluation method on the model
train_df = evaluation(device, model, train_loader)
test_df = evaluation(device, model, test_loader)

plt.figure(figsize=(14, 6))

# Plotting histograms for accuracy and dice coefficient per slice
plt.subplot(1, 2, 1)
plt.hist(train_df["accuracy"], bins=20, alpha=0.7, label="Train Accuracy")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(train_df["dice_score"], bins=20, alpha=0.7, label="Train DSC")
plt.xlabel("Dice Coefficient")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.show()

# Plotting histograms for accuracy and dice coefficient per slice
plt.subplot(1, 2, 1)
plt.hist(test_df["accuracy"], bins=20, alpha=0.7, label="Test Accuracy")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(test_df["dice_score"], bins=20, alpha=0.7, label="Test DSC")
plt.xlabel("Dice Coefficient")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.show()

for idx in range(20):
    # Print the first 5 rows of the train_df
    image_array = train_df.loc[idx, "image"].squeeze()
    true_mask_array = train_df.loc[idx, "mask"].squeeze()
    prediction_array = train_df.loc[idx, "prediction"].squeeze()
    accuracy = train_df.loc[idx, "accuracy"]
    dice_score = train_df.loc[idx, "dice_score"]
    fig, ax = plt.subplots(
        1, 3, figsize=(15, 5)
    )  # Create a figure and a set of subplots for 3 images

    # Plot original image
    ax[0].imshow(image_array, cmap="gray")
    ax[0].set_title("Original Image")
    ax[0].axis("off")  # Hide axes ticks

    # Plot mask
    ax[1].imshow(true_mask_array, cmap="gray")
    ax[1].set_title("True Mask")
    ax[1].axis("off")  # Hide axes ticks

    # Plot prediction
    title = f"Accuracy: {accuracy:.3f}, Dice Score: {dice_score:.3f}"
    ax[2].imshow(prediction_array, cmap="gray")
    ax[2].set_title(title)
    ax[2].axis("off")  # Hide axes ticks

    plt.show()
