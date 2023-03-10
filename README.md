# AQUAMI-TGO
Automatic QUAntification of Microscopy Images - Thermally Grown Oxide
<br>
#### Built on technology from MicroNet ([paper](https://www.nature.com/articles/s41524-022-00878-5), [repo](https://github.com/nasa/pretrained-microscopy-models)) and AQUAMI ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0927025617304184), [repo](https://github.com/JStuckner/aquami)). If you find this code useful in your research, please consider citing these sources.
<br>

## Description
Uses machine learning and computer vision to automatically measure microstructure features from images of environmental barrier coatings. Quantifying microstructure is critical to designing better materials by establishing processing-structure-property relationships. Previous measurement techniques relied on manual human measurements which is extremely time consuming, prone to bias, and requires expertise. This software can automatically and accurately measure oxide thickness, roughness, porosity, and crack spacing in a matter of seconds and the results are repeatable and comparable between research groups. The open source GUI and algorithms can
be adapted to perform other types of image analysis and have been applied to analyze many other material microstructures.

## How it works
Thermally grown oxide layers and oxide cracks are segmented with a convolutional neural network (CNN). The CNN has a U-Net decoder with an Inception-ResNet-V2 encoder that was pre-trained on a large dataset of microscopy images called MicroNet. The pores in the oxide layer are segmented with an automatically determined threshold value based on a histogram of pixel intensity values. The oxide thickness is measured from a distance transform of the segmented oxide layer. Crack spacing is determined by the distance between the centroids of adjacent cracks. Roughness is measured on the top and bottom of the segmented oxide layer using standard roughness equations.

## Installation
1. Download this repository.
1. Install [PyTorch](https://pytorch.org/) to a python virtual environment. This is made easy with [Light-the-torch](https://github.com/pmeier/light-the-torch).
<br> - Windows users:
 <br> &emsp;i. `python -m pip install light-the-torch`
 <br> &emsp;ii. `ltt install torch`
1. Navigate to the AQUAMI-TGO folder inside a terminal and install install the dependencies with:
`pip install -r requirements.txt`
1. Download the [segmentation model](https://drive.google.com/uc?export=download&id=1RxzGigTpJQ-k9kYBtc-JYjb6M2z9vbpp) and place it in the aquami/models/ folder.

## Running the program
1. Activate the virtual environmnet in a terminal.
2. Navigate to the aquami folder.
3. Execute `python gui.py` (The first time will be slow).

## Manual
Coming soon.
