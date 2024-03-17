import numpy as np

# import matplotlib.pyplot as plt

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, random_split, Subset
# import torchmetrics
# from torchsummary import summary
# from torchmetrics.classification import BinaryAccuracy

# import random
# import cv2
import pydicom
from pydicom import dcmread
import os
from os import listdir
from os.path import join

# from tqdm import tqdm
# import pandas as pd

# Deidentify the patient data - based on code from class
# Dataset path
images_path = "Dataset/Images"

# Case IDs, which contain personal information
case_ids = ["Case_003", "Case_006", "Case_007"]

# For the case ids with patient information, code to iteratively remove patient information
for case_id in case_ids:
    subdir_path = join(images_path, case_id)

    for file_name in listdir(subdir_path):
        dicom_file = join(subdir_path, file_name)
        metadata = dcmread(dicom_file)

        # Modify the tags that contain patient information
        metadata["PatientID"].value = case_id
        metadata["PatientName"].value = case_id
        metadata["PatientBirthDate"].value = ""

        # PatientBirthTime is optional, and should be deleted if it is present.
        if "PatientBirthTime" in metadata:
            del metadata["PatientBirthTime"]
        metadata.save_as(dicom_file)

# Module 1 - Handling Dicom data
# Convert DICOM dataset into a NumPy array per cas
# Specify data paths
images_path = "Dataset/Images"
segmentation_path = "Dataset/Segmentations/"

dicom_array3D_store = {}
# Sort the Segmentations by case number
for segmentation in sorted(os.listdir("Dataset/Segmentations/")):
    full_segmentation_path = os.path.join(segmentation_path, segmentation)

    # Load npz data
    segmentation_data = np.load(full_segmentation_path)

    # Extract masks and flip vertically
    mask_array = np.flipud(segmentation_data["masks"])

    # Case is 0-7 digits of the segmentation
    Case = segmentation[:8]

    # Full path to current case item
    Case_path = os.path.join(images_path, Case)

    # List all files in the current Case_path that end with .dcm
    dicom_files = [file for file in os.listdir(Case_path) if file.endswith(".dcm")]

    # Sort the DICOM files
    dicom_files.sort()

    dicom_info = []

    # Iterate over all dicom files
    for dicom_file in dicom_files:
        # Open the DICOM file in the case path
        dicom_path = os.path.join(Case_path, dicom_file)
        # Read the DICOM file - imagedata and metadata
        dicom_data = pydicom.dcmread(dicom_path)
        # Extract the slice information
        slice_location = getattr(dicom_data, "SliceLocation", None)

        # Extract the pixel spacing and slice thickness
        slice_thickness = getattr(dicom_data, "SliceThickness", None)
        pixel_spacing = getattr(dicom_data, "PixelSpacing", None)

        if slice_location is not None:
            dicom_info.append((dicom_file, slice_location))

    # Sort the dicom files based on slice location
    dicom_info.sort(key=lambda x: x[1])
    dicom_array = []

    for dicom_file, _ in dicom_info:
        # Open the DICOM file in the case path
        dicom_path = os.path.join(Case_path, dicom_file)

        # Read the DICOM file - imagedata and metadata
        dicom_data = pydicom.dcmread(dicom_path)

        # Image data is extracted into a NumPy array
        arr = dicom_data.pixel_array

        # Rescaling into Hounsfield units for better visualisation
        rescale_slope = dicom_data.RescaleSlope
        rescale_intercept = dicom_data.RescaleIntercept

        # Convert to Hounsfield Units
        hu_arr = arr * rescale_slope + rescale_intercept

        # Append the NumPy arrays
        dicom_array.append(hu_arr)

    # Stack the 3D array and flip vertically
    # (number_of_slices, height_of_each_slice, width_of_each_slice)
    dicom_array_3d = np.flipud(np.stack(dicom_array))

    dicom_array3D_store[Case] = dicom_array_3d

print(dicom_array3D_store)
