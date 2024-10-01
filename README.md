# Car Price Prediction with Image Analysis and Damage Detection

This project aims to predict car prices by combining different methods:
- **Regression** for analyzing numerical and categorical data (like brand, model, year, etc.).
- **Convolutional Neural Networks (CNN)** to assess the car's physical condition using images.
- **YOLO** for detecting visible damages on the car (like scratches and dents).

By using these techniques, we improve the accuracy of price predictions and provide visual information about the car’s condition. The app is deployed using **Flask** to allow interaction via an API.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

The project uses three types of data:
1. **Numerical data**: Car attributes like brand, model, year, etc.
2. **Image data**: Car photos to assess the car's physical condition.
3. **Damage detection**: Car images are used to detect visible damages with bounding boxes around scratches, dents, etc.

By combining this data, the project improves price prediction and provides detailed visual information.

## Project Structure

```bash
.
├── Datasets/        
├── Flask-app/      
├── Notebooks/       
├── Outputs/         
├── README.md        
└── requirements.txt 
```

## Features

- Price Prediction: Predicts car prices using a regression model.
- Image Analysis: Uses CNN to assess car condition from photos.
- Damage Detection: YOLO identifies damages on car images.
- API Deployment: Flask API - GCP for model predictions and damage detection webapp.

## Installation

1. Clone the repository and navigate to the project directory:

git clone https://github.com/Rovat/artefact-project.git
cd artefact-project

2. Install the required Python dependencies:

pip install -r requirements.txt

## Usage

1. Data Exploration: Explore car data and visualize relationships using data-exploration.ipynb.
2. Price Prediction: Train an XGBoost regression model with XGBoost-training.ipynb.
3. Image Analysis: Train a CNN (VGG16) for condition assessment with VGG-training.ipynb.
4. Damage Detection: Detect damages with YOLO with YOLO-training.ipynb.
