# Multi-Task Human Action Recognition

### Project Overview

Human Action Recognition (HAR) involves identifying actions performed by individuals from images or videos. This project focuses on developing a deep convolutional neural network (CNN) to recognize actions from still images. The system is designed to predict both the specific action of a person and whether there are multiple people in the image.


### Key Objectives

- Action Recognition: Identify the action performed by a person in an input RGB image (Multi-Classification of 40 classes)
- Multi-Person Detection: Determine if more than one person is present in the image (Binary Classification)

### Model Architectures

This project features several architectural variants to address different experiments:

- **Simple Architecture**: Utilizes a straightforward dense layer configuration with dropout to handle both multi-class and binary classification tasks.
- **Branch Architecture**: Features two separate branches with distinct dense layer configurations for each task.
- **Default Architecture**: Combines elements from both the simple and branch architectures for a balanced approach.

The base model for this project is VGG16, but the class is flexible to accommodate various pre-trained models for transfer learning. You can experiment with models such as VGG19, Xception, InceptionV3, ResNet50, and MobileNetV2. The class allows for fine-tuning, either by training only the top layers or by fine-tuning the entire network


### Repository Structure

```bash

Multi-Task-Human-Action-Recognition
│
├── README.md
├── conda_list.txt
│
├── notebooks              
│   ├── notebook.ipynb              # Simple exercise notebook
│
└── src
    ├── Modules.py             # Main project modules

```