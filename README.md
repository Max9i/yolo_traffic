# Project 1: Traffic Sign Recognition




Detailed information can be found in the code. The following is the process of writing and implementing the project code:


## Table of Contents

- [Installation and Environment configuration](#Installation and Environment configuration)
- [Dataset preparation](#Dataset preparation)
- [Models](#Models)
- [Code settings and programming](#Code settings and programming)
- [Default hyperparameters and model network structure adjustment](#Default hyperparameters and model network structure adjustment)
- [Result output and test outcomes](#Result output and test outcomes)


## Installation and Environment configuration

Before implementing the project, it is necessary to install the compatible environment.

Installation of Miniconda: Enter the command "conda --version" in the command line, and it will display the version number of conda.
```bash
conda --version
```
CUDA Installation and CUDNN In the command line, enter the command "nvidia-smi" to check the CUDA version.
```bash
nvidia-smi
```
Environment setup

Enter "conda create -n pytorch python=3.8" to create a new Python environment and name it "pytorch".

Installation of PyTorch: Go to the PyTorch official website. 

## Dataset preparation

Since we are using YOLO to solve the image recognition problem, the dataset format should be as follows:
```
datasets/
│
└── traffic/
    ├── images/
    │   ├── train/
    │   └── val/
    │
    └── labels/
        ├── train
        └── val
```
However, since the labels files for the original training set and test set are named "TsignRecgTest1994Annotation.txt", which is the name of a single file; image width; image height; x-coordinate of the bounding box; y-coordinate of the bounding box; width of the bounding box; height of the bounding box; category label; in this format, they need to be converted to the YOLO format, which is category; x_center; y_center; width; height (normalized values). The conversion code is in the file "transform_to_yolo.py".  

format.ipynb: Convert the images crawled from the website from various formats into JPG formats.
nums.ipynb: Print each number of images in each class

## Models

Reasons for choosing YOLOv8: Compared with lower versions such as (YOLOv3/v5), YOLOv8 has higher accuracy, faster inference speed, and greater task scalability. Compared with higher versions (YOLOv9/Transformer), the accuracy difference is not significant (3%-8%), it has fewer parameters, faster inference speed, lower deployment cost, and lower GPU requirements. All of these make it the preferred choice for this project.

Source code can be obtained from the ultralytics(https://github.com/ultralytics/ultralytics) open-source project, and it can be downloaded.


## Code settings and programming

In the "traffic.yaml" file, set the paths of the train and validation image databases, and name the 58 class names.
```bash
model = YOLO("**.pt")
```
Write the main training code in the "traffic_train.py" file and modify the YOLO function to change the model call.
Modify the model.train function, configure the yaml file, and adjust the hyperparameters.




## Default hyperparameters and model network structure adjustment 

Adjust the default hyperparameters and the model network structure in the files ultralytics/cfg/default.yaml and ultralytics/cfg/models/v8/yolov8.yaml respectively.

## Result output and test outcomes

The training results will be output to the "runs/detect/train" folder. They will include the best and final trained models, confusion matrix, F1 score images, P curve graph, PR curve graph, result table, as well as example images of results before and after training and testing.

The trained model can be tested in the "traffic_test.py" file. The results can be verified. The test image dataset is saved in "/inference/images/".
```bash
model = YOLO("D:\code\yolov8\\runs\detect\\train\weights\\best.pt")
```


