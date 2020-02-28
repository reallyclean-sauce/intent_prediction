from __future__ import print_function, division
"""
Calculate the accuracy of each part of the model
Head Detection Model
    Mean Absolute Error
Object Recognition Model
    IoU for determining if TP TN FP FN
    draw the precision/recall curve
    gets the mean average precision
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


def get_object_gt(dataset):
    """
    Args:
        dataset: gets the ground truth for the object
            elements: 'uid', 'y_cls', 'y_offset'
    """
    gt_list = []
    for i in range(len(dataset)):
        sample = dataset[i]

        uid = sample['uid']
        image = sample['image']
        y_cls = sample['class']
        y_offset = sample['offset']

        gt_sample = {
            'uid': uid,
            'class': y_cls,
            'offset': y_offset
        }

        gt_list.append(gt_sample)

    return gt_list

def get_object_pred(network, dataset):
    """
    Args:
        dataset: gets the ground truth for the object
            elements: 'uid', 'y_cls', 'y_offset'
    """
    pred_list = []
    for i in range(len(dataset)):
        sample = dataset[i]

        uid = sample['uid']
        image = sample['image']
        y_cls = sample['class']
        y_offset = sample['offset']

        output = network.objectRecog(image)

        pred_list.append(gt_sample)


def evaluateHeadModel(network, dataset):
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

    return loss


def main():

    csv_file = '../dsp_intent_analyzer_dataset/object_data.csv'
    root_dir = '../dsp_intent_analyzer_dataset/object_data'

    object_pos_dataset = ObjectPositionDataset(csv_file,root_dir)
    architecture = IntentPredictionNetwork()

    for i in range(len(object_pos_dataset)):
        sample = object_pos_dataset[i]

        uid = sample['uid']
        image = sample['image']
        y_cls = sample['class']
        y_offset = sample['offset']

        # bounding box coordinates ground truth
        xmin = y_offset[0][0]
        ymin = y_offset[0][1]
        xmax = y_offset[1][0]
        ymax = y_offset[1][1]

        # Predict the class and offset




    print(accuracy)




if __name__ == '__main__':
    main()



