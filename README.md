# LicensePlateDetection

Based on MTCNN+STN+LPRNet (Pytorch)

# 介绍

- 电子科技大学--信息与通信工程学院课程--综合课程设计--车牌识别项目
- 本项目采用MTCNN+STN+LPRNet的方案
- MTCNN进行车牌检测及定位，STN进行仿射变换，LPRNet进行字符识别

# 文件描述

- 运行`main`对`data/images`和`data/ccpd_images`内的图片进行测试
- images是老师提供的2023测试集
- ccpd_images是ccpd数据集中的图片

# 使用

`train_Onet.py train_Pnet.py`和LPRNet文件夹内是训练代码

# 需求

- pytorch 1.7及以上
- 详见requirements.txt 或直接运行 ： `pip install -r requirements.txt`

#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#

# Introduction 

- University of Electronic Science and Technology of China -- School of Information and Communication Engineering -- Comprehensive Course Design -- License plate recognition Project
- This project adopts MTCNN+STN+LPRNet scheme
- MTCNN for license plate detection and location, STN for affine transformation, LPRNet for character recognition 

# File description 

- Run 'main' to test images in 'data/images' and' data/ccpd_images'
- images is the 2023 test set provided by the teacher
- ccpd_images Indicates images in the ccpd data set

# Usage

- 'train_Onet.py train_Pnet.py' and the LPRNet folder contain the training code

# Requirements 

- pytorch 1.7 or above
- see requirements.txt for dependencies or run command: `pip install -r requirements.txt`
