# Instance-Segmentation

This repository contains the code of course project of IIT Bombay EE 782 Advanced Machine Learning course.
The full decription of approach is decribed in blog

## Methods for running the code
1. Download the code in any directory and make one folder in that directory named datasets and download all images from this link https://storage.googleapis.com/openimages/web/download.html and all the relevent files
2. Make one directory name logs in project directory

The project directory will looks like
```
Project_Directory
|--README.md
|--datasets
|   |--train
|   |--train_masks
|   |--validation
|   |--validation_masks
|   |--test
|   |--classes-segmentation.txt
|   |--challenge-2019-train-segmentation-masks.csv
|   |--challenge-2019-validation-segmentation-masks.csv
|   |--challenge-2019-label300-segmentable-hierarchy.json
|   |--challenge-2019-classes-description-segmentable.csv
|--logs
|--coco.py
|--config.py
|--cocoutils.py
|--model.py
|--cocodataset.py
|--cocodatasetval.py
|--cocodatasetL1.py
|--cocodatasetL1val.py
|--utils.py
|--visualize.py
|--parallel_model.py
```
Others code availabe in this repository are of testing and inspecting purpose

## Create datasets
To create datasets for layer0 class
python cocodataset.py -l 0 -m train --img_num 2000

similarly create the validation dataset

The datasets directory will be created as
```
Project_Directory
|--datasets
|  |--coco
|       |--annotations
|             |--instances_train2017.json
|             |--instances_train2017.json
|       |--train2017
|       |--val2017
```
This is coco-based format which we can use on MASK R-CNN implementation mmdetection
## Training
For training use the command
```
python3 coco.py train --dataset datasets/coco --model "path to initial weight"
```
Adjust Number of GPU and images per GPU in coco.py. I have used 2 images on 14GB memory GPU.

Similarly training for layer1 can be done
