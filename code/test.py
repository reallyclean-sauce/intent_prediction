from __future__ import print_function, division
"""
Calculate the accuracy of each part of the model
Head Detection Model
    Mean Square Error
Object Recognition Model
    Categorical Cross Entropy
    MAE
Gaze Pathway Prediction Model
    ???
Heatmap Pathway Prediction Model
    ???
"""


import os
import torch
import pandas as pd
from skimage import io, transform
from sklearn.metrics import mean_squared_error
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Import my libraries
from head_dataset import HeadPositionDataset, showHeadPosition
from object_dataset import ObjectPositionDataset
from intent_prediction import IntentPredictionNetwork

# Returns class and
# def get_head_pred(network, dataset):

def evaluateModel(network, dataset):
    # fig = plt.figure()

    # Extract the ground truth
    y_true = np.array([[]])
    y_pred = np.array([[]])
    start_time = time.time()
    for i in range(len(dataset)):
        sample = dataset[i]

        # Preprocess the data
        x,y,_ = sample['image'].shape

        # Append the ground truth
        truth = sample['head_pos']
        y_true = np.append(y_true, [float(truth[0][0]/x), float(truth[0][1]/y)])
        # y_true = np.append(y_true, [float(truth[0][0]), float(truth[0][1])])

        # Get the prediction then append
        pred = network.headDetect(sample['image'])
        y_pred = np.append(y_pred, [float(pred[0]/x), float(pred[1]/y)])
        # y_pred = np.append(y_pred, [float(pred[0]), float(pred[1])])

        if not i % 10:
          runtime = time.time() - start_time
          percent_completion = float(i / len(dataset)) * 100
          print(f'Total time taken: {runtime}s  Percent Completion: {percent_completion}%')

        # if i > 5:
        #   break

    runtime = time.time() - start_time
    percent_completion = float(i / len(dataset)) * 100
    print(f'Total time taken: {runtime}s  Percent Completion: {percent_completion}%')


    # Comparison model
    y_true = y_true.astype(float).reshape(-1,2)
    y_pred = y_pred.astype(float).reshape(-1,2)

    loss = mean_absolute_error(y_true, y_pred)

    print(f'Loss is {loss}.')

    return loss, y_true, y_pred

def main():

    head_pos_dataset = HeadPositionDataset( \
        '../dsp_intent_analyzer_dataset/head_data.csv',  \
        '../dsp_intent_analyzer_dataset/head_data')
    architecture = IntentPredictionNetwork()
    accuracy = evaluateModel(architecture, head_pos_dataset)

    print(accuracy)




if __name__ == '__main__':
    main()
    print("Testing!!")



