# Medical Imaging Coursework
## Description
The coursework seeks to use a UNet to segment the lungs from computational tomography images of twelve cases from the Lung CT Segmentation Challenge from the Cancer Imaging Archive. A PDF of a report describing the project in detail is provided in the report folder in the root directory.
Excluding the appendix, the word count for the report is XX words.

# Usage
First clone the repository from git. There are two sets of code to be run, one for handling the DICOM data to convert it to a 3D NumPy array to answer part 1 of the coursework, and another to train and evaluate the UNet model for part 2B of the coursework.

## Saved model
Cloning the repository includes cloning the saved, trained model, which is stored in the "results" folder, as a ".pth" files.

## Running the code to train and evaluate the UNet via main.py
To run the code, a dockerfile is provided in the root directory with the environment needed to run the code, provided in the MI_environment.yml file.

My pre-trained model is coded to be used for the evaluation method. Thus, the code for training the model can be commented out to just evaluate the model with my pre-trained model.

To run the code from the terminal navigate to the root directory and use, e.g.,
$docker build -t [image name of choice] -f dockerfile_main .
$docker run -v .:/cf593_mi -t [image name of choice]
(Make sure to include the periods!)

With the appropriate environment in MI_environment.yml, the code can also be run from the terminal
by navigating into the root directory of the cloned git repository and running the code with the following command
$ python main.py

## Running the code for handling the DICOM files
To run the code, a dockerfile is provided in the root directory with the environment needed to run the code, provided in the MI_environment.yml file.
To run the code from the terminal navigate to the root directory and use, e.g.,
$docker build -t [image name of choice] -f dockerfile_handling_DICOM .
$docker run -v .:/cf593_dicom -t [image name of choice]
(Make sure to include the periods!)

With the appropriate environment in MI_environment.yml, the code can also be run from the terminal
by navigating into the root directory of the cloned git repository and running the code with the following command
$ python handling_DICOM.py

# Documentation
Detailed documentation is available by running the Doxyfile using doxygen in the docs file in the root directory.
This can be run by navigating in the docs file and running doxygen with:
$doxygen

# Auto-generation tool citations
Used ChatGPT 4.0 for the following:
- Saving figures based on the index values for producing examples of 2D slices with their corresponding predicted masks and true masks in `main.py`:
    - Alongside the existing code for plotting the values, used the prompt "how to save these finals, incorporating their index name":

GitHub Copilot was used to help write documentation for docker and comments within the code.

# License
Released 2024 by Cailley Factor.
The License is included in the LICENSE.txt file.
