![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![CNN](https://img.shields.io/badge/Model-Convolutional%20Neural%20Network-purple)

# Deep CNN for Handwritten Digit Recognition

## Overview

This project implements a deep Convolutional Neural Network (CNN) for handwritten digit classification using the MNIST dataset.

The model is built using TensorFlow and Keras and trained on 60,000 labelled images of handwritten digits. 
The goal is to learn discriminative visual features that allow the network to classify unseen handwritten digits with high accuracy.

To improve generalisation and reduce overfitting, the model incorporates data augmentation, batch normalization, and dropout regularisation.

The system demonstrates how deep learning models can be used to perform image classification tasks by automatically learning hierarchical feature representations from raw pixel data.

## Core Concepts

This project demonstrates several key machine learning and deep learning concepts:

- Convolutional Neural Networks (CNNs)
- Image classification
- Feature extraction with convolutional layers
- Data augmentation
- Regularisation techniques (Dropout, Batch Normalization)
- Model evaluation and error analysis

## Model Architecture

Input Image (28x28 grayscale)
        ↓
Conv2D (64 filters, 5x5)
        ↓
Batch Normalization
        ↓
Max Pooling
        ↓
Conv2D (64 filters, 3x3)
        ↓
Batch Normalization
        ↓
Max Pooling
        ↓
Conv2D (128 filters, 3x3)
        ↓
Batch Normalization
        ↓
Max Pooling
        ↓
Flatten
        ↓
Dense Layer (512 units)
        ↓
Dropout (0.6)
        ↓
Softmax Output (10 classes)

## Training Pipeline

MNIST Dataset
      ↓
Image Normalisation
      ↓
Data Augmentation
(rotation, shift, zoom, shear)
      ↓
CNN Training
      ↓
Model Evaluation
      ↓
Misclassification Analysis

## Key Features

- Deep convolutional neural network for image classification
- Data augmentation to improve generalisation
- Batch normalization for stable training
- Dropout regularisation to reduce overfitting
- Training and validation performance visualisation
- Analysis of misclassified examples

## Project Structure

model10.py              - CNN architecture and training script
mnist_model5_tf212.h5   - Saved trained neural network model

keras_tutorial.pdf      - Keras neural network development guide
coursework_instructions.pdf - Coursework requirements and evaluation criteria

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib


---

# Learning Outcomes

This project demonstrates practical implementation of:

- Deep learning for image classification
- Convolutional neural networks
- Model regularisation techniques
- Training and evaluation of neural networks
