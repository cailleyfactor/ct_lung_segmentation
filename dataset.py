"""
@file dataset.py
@brief Create a custom dataset class for the model.
@author Created by C. Factor on 01/03/2024
"""

import torch
from torch.utils.data import Dataset
import pydicom
import os
import numpy as np


# Make a Dataset class to inherit from the Dataset class
class CustomDataset(Dataset):
    """
    @class CustomDataset
    @brief This class defines the custom dataset class for the model.
    """

    def __init__(self, dicom_dirs, mask_paths):
        """
        @brief Constructor for the CustomDataset class
        @param dicom_dirs: List of directories containing DICOM files
        @param mask_paths: List of paths to mask files"""
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

            # Sort DICOM files by slice locationm
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
        """
        @brief Get the length of the dataset
        @return The length of the dataset"""
        assert len(self.slices) == len(
            self.masks
        ), "Mismatch in number of DICOM slices and mask slices"
        return len(self.slices)

    def __getitem__(self, idx):
        """
        @brief Get an item from the dataset
        @param idx: The index of the item to get
        @return dicom_tensor: The DICOM tensor
        @return mask_tensor: The mask tensor
        @return patient_id: The patient ID
        @return slice_info: The slice location of the DICOM slice"""
        # Convert numpy arrays to PyTorch tensors
        dicom_tensor = torch.from_numpy(self.slices[idx].astype(np.float32)).float()
        mask_tensor = torch.from_numpy(self.masks[idx].astype(np.float32)).float()

        # Add channel dimension at position 0 to the segmentation mask and dicom image
        dicom_tensor = dicom_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)

        patient_id = self.patient_ids[idx]
        slice_info = self.slice_location[idx]

        return dicom_tensor, mask_tensor, patient_id, slice_info
