# Obstacle Detection Model Traning

# Project General Outline in Korean
이 프로젝트는 **숭실대학교** 컴퓨터비전 연구를 위해 스파르탄SW사업의 지원을 받아 진행 중인 과제입니다. 프로젝트의 목적은 실전에서 당면한 과제를 다양한 딥러닝 모델을 사용함으로써 문제해결 능력과 모델에 대한 이해를 향상시키는 데에 있습니다.
 
 저희의 연구 주제는 **열차 전면부에 카메라를 설치하여 선로와 장애물을 탐지하는 것**이며. 탐지한 해당 장애물이 선로 위에 있으며 충돌 시 심각한 인명피해나 장비손상이 발생할 여지가 있는 무시할 수 없는 장애물일 때 기관사에게 신호를 보내 인명 및 재산 피해를 최소화하는 시스템을 기획하였습니다. 이를 위하여 선로의 이미지와 마스킹 된 데이터를 학습시켜  **Segmentation** 모델을 만든 후, **Object detection**을 통해 무시할 수 있는 장애물과 무시할 수 없는 장애물을 선별한 뒤 해당 장애물의 중심부와 선로 사이의 거리를 계산하였습니다.

 학습에 사용된 데이터는 선로와 기차 모형을 준비하여 통제된 환경에서 촬영한 데이터를 사용하였습니다. 저희의 궁극적인 목표는 임의의 장애물이 감지되었을 때에도 이를 감지하여 기관사에게 신호를 보내는 것이지만, 이 프로젝트의 경우 실제 상용화를 위한 것이 아닌 시연을 목적으로 제작되었기 때문에 경제적인 측면을 고려하여 사전에 선별된 몇 가지의 장애물만을 학습시켜 통제된 환경에서 시연을 할 수 있도록만 계획하였습니다. 또한 사용된 데이터의 가짓수가 적기 때문에 다양한 방식으로 데이터를 **Augmentation** 하였습니다.
 
 학습에 사용된 모델은 **Faster-RCNN-Inception-V2** 를 사용하였습니다.
 
 저희는 모델 설계를 위해 EdjeElectronics의 코드를 일부 사용하였습니다. 원작자의 깃허브 코드 주소는 아래와 같습니다.
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
 
 ### 작업 순서:
1. Setting for training
2. Training

**예상학습시간**: 8시간

By: 김민석, 숭실대학교 소프트웨어학부 / 
이주희, 숭실대학교 소프트웨어학부

# Project General Outline in English 
This project is an ongoing project with the support of Spartan SW project to study **Computer Vision of Soongsil University**. The objective of the project is to improve problem resolution and understanding of the model by using a variety of deep learning models to meet the challenges faced in the field.

**Our topic is to install a camera on the front of the train to detect tracks and obstacles.** The system is designed to minimize casualties and property damage by sending a signal to the engineer when the detected obstacle is on the track and there is a possibility of serious casualties or equipment damage in the event of a collision. For this purpose, the image of the track and the masked data were studied to create a **Segmentation Model**. Also we selected **Object Detection Deep Learning Model** to recognize obstacle.

The data used for learning was prepared with track and train models and taken in a controlled environment. Our ultimate goal is to detect and signal the driver, even when any obstacles are detected, but because this project was designed for demonstration rather than for actual commercialization, we planned to learn only a few pre-selected obstacles and demonstrate them in a controlled environment. And because the number of used data is low, we've use **Augmentation** it in a variety of ways.

Also, the model we used is **Faster-RCNN-Inception-V2** model.

We used some of EdjeElectronics's code to design the model. The original author's Githeub code address is as follows.

https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
    
### We will follow the general workflow:
1. Setting for training
2. Training


**Time Estimated**: 8 hours

By: Kim Minseok, Department of Software in Soongsil University / 
 Lee Juhui, Department of Software in Soongsil University


# Setting for training
Our project based on **anaconda, tensorflow-gpu, jupyter notebook** and etc. So you have to install these first.
Also, I made this project on **CUDA 10.0** and **cuDNN 7.3** environment. If you install another version, I don't warrant about result.
I recommand to activate this code on virtual anaconda setting.