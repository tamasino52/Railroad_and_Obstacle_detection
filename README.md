# Project General Outline
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Ftamasino52%2FRailroad_and_Obstacle_detection)](https://hits.seeyoufarm.com)

This project is an ongoing project with the support of Spartan SW project to study **Computer Vision of Soongsil University**. The objective of the project is to improve problem resolution and understanding of the model by using a variety of deep learning models to meet the challenges faced in the field.

**Our topic is to install a camera on the front of the train to detect tracks and obstacles.** The system is designed to minimize casualties and property damage by sending a signal to the engineer when the detected obstacle is on the track and there is a possibility of serious casualties or equipment damage in the event of a collision. For this purpose, the image of the track and the masked data were studied to create a **Segmentation Model**. Also we selected **Object Detection Deep Learning Model** to recognize obstacle.

The data used for learning was prepared with track and train models and taken in a controlled environment. Our ultimate goal is to detect and signal the driver, even when any obstacles are detected, but because this project was designed for demonstration rather than for actual commercialization, we planned to learn only a few pre-selected obstacles and demonstrate them in a controlled environment. And because the number of used data is low, we've use **Augmentation** it in a variety of ways.

## Award & Performance
1. Bronze Prize on Software Contest In Soongsil Univercity 2019.11.07
2. Korean Software Registeration - Railway Obstacle Detection System(RODS) / Railroad Tracker 2019.11.30
3. Patent Registeration - Railroad Obstacle Detection System(RODS) 2019.11.30

## Models
1. Railway Segmentation model : **U-net**
2. Obstacle Detection model : **Faster-RCNN-Inception-V2**

We used some of EdjeElectronics's code to design the model. The original author's Githeub code address is as follows.
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
    
## Workflow:
1. Setting for training
2. Training model(You can skip this to download pretrained models)
3. Run Project

**Time Estimated**: 8 hours

By: Kim Minseok, Department of Software in Soongsil University / 
 Lee Juhee, Department of Software in Soongsil University

## Setting for training
Our project based on **anaconda, tensorflow-gpu, jupyter notebook** and etc. So you have to install these first.
Also, I made this project on **CUDA 10.0** and **cuDNN 7.3** environment. If you install another version, I don't warrant about result.
I recommand to activate this code on virtual anaconda setting.

## How to run
1. Clone our git first
https://github.com/tamasino52/Railroad_and_Obstacle_detection
2. Clone https://github.com/tensorflow/models git
3. Download trained Unet model from (Put file in models/research/object_detection/models )
https://drive.google.com/file/d/18Y_EbJV9s4eJmFDg69uH46xgynHb9vNl/view?usp=sharing
4. Move our all file to 'models/research/object_detection'
5. Download faster_rcnn_inception_v2_coco_2018_01_28 model from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
6. Move model file to 'faster_rcnn_inception_v2_coco_2018_01_28' folder in 'models/research/object_detection'
7. Run 'RailwayTrackingModel_Training.ipynb' file to generate model
8. Run 'ObstacleDetectionModel_Training.ipynb' file to generate seccond model
9. Run 'Run.py' file(for video and image) or 'Run.ipynb' file(only for image)  to activate project
9-1. If you want to use captured image, Run 'Run_capture.py'. After you run, Click and Drag points that you want to capture. Then check your valid yellow box, press 'esc' to activate project.

## Simulation Result
<p align="center">
  <img src="/simulation/test (1).JPG">
</p>
<p align="center">
  <img src="/simulation/test (2).JPG">
</p>
<p align="center">
  <img src="/simulation/test (3).JPG">
</p>
<p align="center">
  <img src="/simulation/test (4).JPG">
</p>
<p align="center">
  <img src="/simulation/test (5).JPG">
</p>
<p align="center">
  <img src="/simulation/test (6).JPG">
</p>
<p align="center">
  <img src="/simulation/test (7).JPG">
</p>
<p align="center">
  <img src="/simulation/test (8).JPG">
</p>
<p align="center">
  <img src="/simulation/test (9).JPG">
</p>
<p align="center">
  <img src="/simulation/test (10).JPG">
</p>
<p align="center">
  <img src="/simulation/test (11).JPG">
</p>

## Postscript
- If you are looking at this repo because you are interested in predicting railway, please refer to it here because a better version has been released than here. This <a href="https://github.com/tamasino52/Railway_detection">repository</a> only predicted with computer vision technology. But this is much faster and more accurate. 
