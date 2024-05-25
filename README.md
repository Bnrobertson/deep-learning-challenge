### deep-learning-challenge
# Charity Data Prediction

This repository contains a machine learning project that predicts the success of charity applications using a deep neural network. The dataset used is sourced from `charity_data.csv`.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Model Saving](#model-saving)
- [Usage](#usage)


## Overview
The goal of this project is to build a deep learning model to predict whether a charity application will be successful based on various features provided in the dataset. The project involves data preprocessing, model building, training, evaluation, and saving the model.

## Dataset
The dataset contains various features about charity applications, such as application type, classification, and success status.

## Requirements
- pandas
- scikit-learn
- tensorflow
- numpy
  
  You can install the necessary packages using the following command:

pip install pandas scikit-learn tensorflow numpy

## Data Preprocessing
1. **Load the Dataset:** Read the charity_data.csv file into a pandas DataFrame.
2. **Drop Non-beneficial Columns**: Remove the `EIN` and `NAME` columns as they do not contribute to the prediction.
3. **Replace Rare Categories**: Replace less common categories in the `APPLICATION_TYPE` and `CLASSIFICATION` columns with "Other".
4. **Encode Categorical Variables**: Convert categorical variables to numeric using one-hot encoding with `pd.get_dummies`.
5. **Split Data**: Split the preprocessed data into features (`X`) and target (`y`), and then into training and testing sets.
6. **Scale Data**: Normalize the feature data using `StandardScaler`.

## Model Architecture
The neural network model is built using TensorFlow with the following architecture:

- Input Layer: Number of input features.
- First Hidden Layer: 80 units, ReLU activation.
- Second Hidden Layer: 30 units, ReLU activation.
- Output Layer: 1 unit, Sigmoid activation.
## Training and Evaluation
The model is compiled with binary cross-entropy loss and the Adam optimizer, and then trained for 100 epochs. The trained model is evaluated using the test data to calculate the loss and accuracy.

## Model Saving
The trained model is saved to an HDF5 file named AlphabetSoupCharityOptimized.h5.

## Usage
To use this code, follow these steps:
git clone https://github.com/Bnrobertson/deep-learning-challenge.git
cd deep-learning-challenge

## References

IRS. Tax Exempt Organization Search Bulk Data Downloads. https://www.irs.gov/ Links to an external site.

