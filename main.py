"""
@file main.py
@brief This file is the main file for the project.
It contains the code for training the model, evaluating the model,
and plotting the results.
@author Created by C. Factor on 01/03/2024
"""
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
import os
from dataset import CustomDataset
from eval import evaluation
from train import train_model
from UNet import UNet

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

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
# Check if MPS is available
if torch.backends.mps.is_available():
    # Set the device to MPS
    device = torch.device("mps")
    print("Using MPS device")
else:
    # CPU if MPS is not available
    device = torch.device("cpu")
    print("Using CPU device")

# Define inputs to training loop
batch_size = 3
lr = 0.1
in_channels = 1
out_channels = 1
num_epochs = 10

# Defining train and testing sets with DataLoader
train_size = int(0.66667 * len(dataset))
val_size = len(dataset) - train_size

# Set the seed for generating random numbers in PyTorch
torch.manual_seed(42)

# Split the dataset into training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Dimensions are torch.Size([3, 1, 512, 512]) for both

# Instantiate the model
model, training_losses, training_accuracies, val_losses, val_accuracies = train_model(
    train_loader, test_loader, device, lr, in_channels, out_channels, num_epochs
)

# Save the trained model
torch.save(model.state_dict(), "results/trained_UNet_model.pth")

# Plotting for 2b
plt.figure(figsize=(12, 5))

# Plot training and validation losses
plt.subplot(1, 2, 1)
plt.plot(training_losses, label="Training Loss", color="blue")
plt.plot(val_losses, label="Validation Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plot training and validation accuracies
plt.subplot(1, 2, 2)
plt.plot(training_accuracies, label="Training Accuracy", color="blue")
plt.plot(val_accuracies, label="Validation Accuracy", color="red")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("results/training_validation_losses.png")

# Part 2d
# Can load in the pre-trained model and evaluate it
model = UNet(in_channels, out_channels).to(device)

# Load the weights from a pretrained model
model.load_state_dict(
    torch.load("results/trained_UNet.pth", map_location=torch.device(device))
)

# Run the evaluation method on the model
train_df = evaluation(device, model, train_loader)
test_df = evaluation(device, model, test_loader)

plt.figure(figsize=(14, 6))
# Plotting histograms for accuracy and dice coefficient per slice for train data
plt.subplot(1, 2, 1)
plt.hist(train_df["accuracy"], bins=20, alpha=0.7)
plt.xlabel("Train Accuracy")
plt.ylabel("Frequency")
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(train_df["dice_score"], bins=20, alpha=0.7)
plt.xlabel("Train DSC")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("results/train_histogram.png")

plt.figure(figsize=(14, 6))
# Plotting histograms for accuracy and dice coefficient per slice for test data
plt.subplot(1, 2, 1)
plt.hist(test_df["accuracy"], bins=20, alpha=0.7)
plt.xlabel("Test Accuracy")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(test_df["dice_score"], bins=20, alpha=0.7)
plt.xlabel("Test DSC")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("results/test_histogram.png")


# GPT helped me on this plotting
test_df_sorted = test_df.sort_values(by="dice_score").reset_index(drop=True)


def plot_images(indices):
    """
    @brief This function plots the original image,
    true mask, and predicted mask for the given indices.
    @param indices: List of indices to plot
    @return None
    """
    i = 0
    fig, axs = plt.subplots(3, 3, figsize=(9, 9))  # Create a 3x3 grid of subplots
    # Loop over the first 3 indices
    for idx in indices:
        # Extract the data for the current index
        image_array = test_df_sorted.loc[idx, "image"].squeeze()
        true_mask_array = test_df_sorted.loc[idx, "mask"].squeeze()
        prediction_array = test_df_sorted.loc[idx, "prediction"].squeeze()
        accuracy = test_df_sorted.loc[idx, "accuracy"]
        dice_score = test_df_sorted.loc[idx, "dice_score"]
        slice_info = test_df_sorted.loc[idx, "slice_info"]
        patient_id = test_df_sorted.loc[idx, "patient_id"]

        # Plot original image
        axs[i, 0].imshow(image_array, cmap="gray")
        axs[i, 0].set_title("Original Image")
        axs[i, 0].axis("off")
        text0 = f"{patient_id},\n Slice location: {slice_info}"
        axs[i, 0].text(
            0.5,
            0.05,
            text0,
            transform=axs[i, 0].transAxes,
            fontsize=9,
            color="white",
            ha="center",
            va="bottom",
        )

        # Plot mask
        axs[i, 1].imshow(true_mask_array, cmap="gray")
        axs[i, 1].set_title("True Mask")
        axs[i, 1].axis("off")

        # Plot prediction with accuracy and dice score in the title
        axs[i, 2].imshow(prediction_array, cmap="gray")
        axs[i, 2].set_title("Prediction Mask")
        # Write accuracy and dice score on the image
        text = f"Accuracy: {accuracy:.3f},\n Dice Score: {dice_score:.3f}"
        axs[i, 2].text(
            0.5,
            0.05,
            text,
            transform=axs[i, 2].transAxes,
            fontsize=9,
            color="white",
            ha="center",
            va="bottom",
        )
        axs[i, 2].axis("off")
        i += 1
    # Generate a unique name for the figure based on the indices
    image_name = f"results/example_slices_{'_'.join(map(str, indices))}.png"
    plt.savefig(image_name)
    print(f"Saved figure as {image_name}")


# Indices for the best, median, and worst predictions
num_entries = len(test_df_sorted)
low_indices = [0, 1, 2]
# Picking not the truly best ones so that they are from different patients
high_indices = [num_entries - 30, num_entries - 20, num_entries - 1]
median_idx = num_entries // 2
median_indices = [median_idx - 1, median_idx, median_idx + 1]

plot_images(low_indices)
plot_images(median_indices)
plot_images(high_indices)
