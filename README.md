# Project General Outline in English 
This project is an ongoing project with the support of Spartan SW project to study **Computer Vision of Soongsil University**. The objective of the project is to improve problem resolution and understanding of the model by using a variety of deep learning models to meet the challenges faced in the field.

**Our topic is to install a camera on the front of the train to detect tracks and obstacles.** The system is designed to minimize casualties and property damage by sending a signal to the engineer when the detected obstacle is on the track and there is a possibility of serious casualties or equipment damage in the event of a collision. For this purpose, the image of the track and the masked data were studied to create a **Segmentation Model**. Also we selected **Object Detection Deep Learning Model** to recognize obstacle.

The data used for learning was prepared with track and train models and taken in a controlled environment. Our ultimate goal is to detect and signal the driver, even when any obstacles are detected, but because this project was designed for demonstration rather than for actual commercialization, we planned to learn only a few pre-selected obstacles and demonstrate them in a controlled environment. And because the number of used data is low, we've use **Augmentation** it in a variety of ways.

Also, the model we used is **Faster-RCNN-Inception-V2** model.

We used some of EdjeElectronics's code to design the model. The original author's Githeub code address is as follows.

https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
    
### We will follow the general workflow:
1. Setting for training
2. Training model
3. Run


**Time Estimated**: 8 hours

By: Kim Minseok, Department of Software in Soongsil University / 
 Lee Juhui, Department of Software in Soongsil University


# Setting for training
Our project based on **anaconda, tensorflow-gpu, jupyter notebook** and etc. So you have to install these first.
Also, I made this project on **CUDA 10.0** and **cuDNN 7.3** environment. If you install another version, I don't warrant about result.
I recommand to activate this code on virtual anaconda setting.

# How to run
1. Clone our git first
https://github.com/tamasino52/Real-time-image-based-obstacle-detection-and-identification-system-using-deep-learning-on-railroad
2. Clone https://github.com/tensorflow/models git
3. Move our all file to 'models/research/object_detection'
4. Download faster_rcnn_inception_v2_coco_2018_01_28 model from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
5. Move model file to 'faster_rcnn_inception_v2_coco_2018_01_28' folder in 'models/research/object_detection'
6. Run 'RailwayTrackingModel_Training.ipynb' file to generate model
7. Run 'ObstacleDetectionModel_Training.ipynb' file to generate seccond model
8. Run 'Run.py' file(for video and image) or 'Run.ipynb' file(only for image)  to activate project

# Simulation
<p align="center">
  <img src="/simulation/test (1).PNG">
</p>
<p align="center">
  <img src="/simulation/test (2).PNG">
</p>
<p align="center">
  <img src="/simulation/test (3).PNG">
</p>

<p align="center">
  <img src="/simulation/test (5).PNG">
</p>
<p align="center">
  <img src="/simulation/test (6).PNG">
</p>
<p align="center">
  <img src="/simulation/test (7).PNG">
</p>
<p align="center">
  <img src="/simulation/test (8).PNG">
</p>
<p align="center">
  <img src="/simulation/test (9).PNG">
</p>
