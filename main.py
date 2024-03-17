# import numpy as np
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
# import pydicom
from pydicom import dcmread

# import os
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

# Code to check to see that the patient is deidentified
# # Example for "Case_003"
# Case = "Case_003"

# # Full path to current case item
# Case_path = os.path.join(images_path, Case)

# if os.path.isdir(Case_path):
#     # List all files in the current Case_path that end with .dcm
#     dicom_files = [file for file in os.listdir(Case_path) if file.endswith('.dcm')]

#     # Sort the DICOM files
#     dicom_files.sort()
#     print("****************************")
#     print(f"Patient: {Case}")

#     # Iteratively open the DICOM files in the case path
#     for i in range (len(dicom_files)):
#         dicom_path = os.path.join(Case_path, dicom_files[i])

#         # Read the DICOM file - imagedata and metadata
#         dicom_data = pydicom.dcmread(dicom_path)
#         print(dicom_data)
#         print(dicom_data.SliceLocation)
