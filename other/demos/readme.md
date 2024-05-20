# Brain Tumor Detection

This repository contains inference scripts for two different models designed for the detection of brain tumors: a classification model and a segmentation model.

## Classification

The classification model is trained based on the notebook available at [this link](https://www.kaggle.com/code/seifwael123/brain-tumor-detection-cnn-vgg16?scriptVersionId=167509118). It uses a Convolutional Neural Network (CNN) with the VGG16 architecture to classify whether a brain tumor is present in an image or not.

You can find the inference script for the classification model in [inference.py](classification/inference.py) and the Dockerfile in [dockerfile](classification/dockerfile).

## Segmentation

The segmentation model is trained based on the notebook available at [this link](https://www.kaggle.com/code/abdallahwagih/brain-tumor-segmentation-unet-efficientnetb7). It uses a U-Net with EfficientNetB7 for brain tumor segmentation.

You can find the inference script for the segmentation model in [inference_rmi_tumor_segmentation.py](segmentation/inference_rmi_tumor_segmentation.py) and the Dockerfile in [dockerfile](segmentation/dockerfile).
