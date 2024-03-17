import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pydicom
import os
import numpy as np


# Make a Dataset class to inherit from the Dataset class
class CustomDataset(Dataset):
    def __init__(self, dicom_dirs, mask_paths):
        self.slices = []
        self.masks = []
        self.patient_ids = []
        self.slice_location = []

        # Zip to iterate through the lists in parallel
        for dicom_dir, mask_path in zip(dicom_dirs, mask_paths):
            # Store paths to each file name in the directory path
            dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir)]
            # Read file information
            dicom_files = [pydicom.dcmread(f) for f in dicom_files]

            # Sort DICOM files by slice location
            dicom_files.sort(key=lambda x: float(x.SliceLocation))

            # Load mask
            # Assuming masks are stacked in the same order as slices
            mask = np.load(mask_path)["masks"]

            # Check that number of slices and masks match
            assert (
                len(dicom_files) == mask.shape[0]
            ), "Mismatch in number of DICOM files and mask slices"

            # Append each slice and corresponding mask slice
            for dcm, seg in zip(dicom_files, mask):
                self.slices.append(dcm.pixel_array)
                self.masks.append(seg)
                self.patient_ids.append(dcm.PatientID)
                self.slice_location.append(dcm.SliceLocation)

    def __len__(self):
        assert len(self.slices) == len(
            self.masks
        ), "Mismatch in number of DICOM slices and mask slices"
        return len(self.slices)

    def __getitem__(self, idx):
        # Convert numpy arrays to PyTorch tensors
        dicom_tensor = torch.from_numpy(self.slices[idx].astype(np.float32)).float()
        mask_tensor = torch.from_numpy(self.masks[idx].astype(np.float32)).float()

        # Add channel dimension at position 0 to the segmentation mask and dicom image
        dicom_tensor = dicom_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)

        patient_id = self.patient_ids[idx]
        slice_info = self.slice_location[idx]

        return dicom_tensor, mask_tensor, patient_id, slice_info


# Simple UNet architecture - created in Class
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Encoder part
        self.conv1 = self.conv_block(in_channels, 16, 3, 1, 1)
        self.maxpool1 = self.maxpool_block(2, 2, 0)
        self.conv2 = self.conv_block(16, 32, 3, 1, 1)
        self.maxpool2 = self.maxpool_block(2, 2, 0)
        self.conv3 = self.conv_block(32, 64, 3, 1, 1)
        self.maxpool3 = self.maxpool_block(2, 2, 0)

        self.middle = self.conv_block(64, 128, 3, 1, 1)

        self.upsample3 = self.transposed_block(128, 64, 3, 2, 1, 1)
        # Adding the concatenation to the upconv3
        self.upconv3 = self.conv_block(128, 64, 3, 1, 1)
        self.upsample2 = self.transposed_block(64, 32, 3, 2, 1, 1)
        self.upconv2 = self.conv_block(64, 32, 3, 1, 1)
        self.upsample1 = self.transposed_block(32, 16, 3, 2, 1, 1)
        self.upconv1 = self.conv_block(32, 16, 3, 1, 1)

        self.final = self.final_layer(16, 1, 1, 1, 0)

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        convolution = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        return convolution

    # Not doing anything on the number of channels - so doesn't need in/out_channels
    def maxpool_block(self, kernel_size, stride, padding):
        # Only need nn.Sequential for multiple operations in a block
        maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Dropout2d(0.5),
        )
        return maxpool

    def transposed_block(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding
    ):
        transposed = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        return transposed

    def final_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        final = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        return final

    def forward(self, x):
        # downsampling part
        conv1 = self.conv1(x)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        # middle part
        middle = self.middle(maxpool3)

        # upsampling part
        upsample3 = self.upsample3(middle)
        # Add the concats
        upconv3 = self.upconv3(torch.cat([upsample3, conv3], 1))
        upsample2 = self.upsample2(upconv3)
        upconv2 = self.upconv2(torch.cat([upsample2, conv2], 1))
        upsample1 = self.upsample1(upconv2)
        upconv1 = self.upconv1(torch.cat([upsample1, conv1], 1))

        final_layer = self.final(upconv1)

        return final_layer


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
