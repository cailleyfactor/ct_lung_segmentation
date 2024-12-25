# Medical Imaging Coursework
## Description
The projects involves building and training a UNet model to segment the lungs from computational tomography images of twelve cases from the Lung CT Segmentation Challenge from the Cancer Imaging Archive.

# Usage
First clone the repository from git. Cloning the repository includes cloning the trained model, which is stored in the "results" folder, as a ".pth" file. There are two sets of code to be run, one for handling the DICOM data to convert it to a 3D NumPy array to answer part 1 of the coursework, and another to train and evaluate the UNet model for part 2B of the coursework.

Running the code to train and evaluate the UNet via main.py:

My pre-trained model is coded to be used for the evaluation method. Thus, the code for training the model can be commented out to just evaluate the model with my pre-trained model.

To run the main code for 2B, a dockerfile is provided in the root directory with the environment needed to run the code, provided in the MI_environment.yml file.
To run the code from the terminal navigate to the root directory and use, e.g.,
$docker build -t [image name of choice] -f dockerfile_main .
$docker run -v .:/cf593_mi -t [image name of choice]
(Make sure to include the periods!)

With the environment in MI_environment.yml, the code can also be run from the terminal by navigating into the root directory of the cloned git repository and running the code with the following command $ python main.py

Running the code for handling the DICOM files:

To run the code for handling the DICOM file, a dockerfile called dockerfile_handling_DICOM is provided in the root directory.
To run the code from the terminal navigate to the root directory and use, e.g.,
$docker build -t [image name of choice] -f dockerfile_handling_DICOM .
$docker run -v .:/cf593_dicom -t [image name of choice]
(Make sure to include the periods!)

With the environment in MI_environment.yml, the code can also be run from the terminal
by navigating into the root directory of the cloned git repository and running the code with the following command
$ python handling_DICOM.py

# Documentation
Detailed documentation is available by running the Doxyfile in the docs file in the root directory.
This can be run by navigating in the docs file and running doxygen with:
$doxygen

# License
Released 2024 by Cailley Factor.
The License is included in the LICENSE.txt file.
