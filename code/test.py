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
from dataset import HeadPositionDataset


def evaluateHeadDetection(network, predict, data):
    """
    Evaluate the model using MSE
    Why MSE? (Research
    predict is a numpy array with (:,2) shape
    data is a numpy array with (:,2) shape
    """
    return mean_squared_error(predict,data)




def showHeadPosition(image, head_pos):
    """
    Show the position of the head
    in the image
    """
    plt.imshow(image)
    plt.scatter(head_pos[:,0], head_pos[:,1], s=10, marker='.', c='r')
    plt.pause(0.001) # For loading the image

def extractGroundTruth(dataset):
    # fig = plt.figure()

    # Extract the ground truth
    data = np.array([[]])
    start_time = time.time()
    for i in range(len(dataset)):
        sample = dataset[i]

        # print(i, sample['image'].shape, sample['head_pos'])

        # # Visualize the video
        # ax = plt.subplot(1,2,i+1)
        # plt.tight_layout()
        # ax.set_title(f'Sample #{i}')
        # ax.axis('off')
        # show_head_position(**sample)

        # if i == 1:
        #     plt.show()
        #     break

        data = np.append(data, [sample['head_pos']])
        runtime = time.time() - start_time
        print(runtime)


def main():

    head_pos_dataset = HeadPositionDataset( \
        '../dsp_intent_analyzer_dataset/head_data.csv',  \
        '../dsp_intent_analyzer_dataset/head_data')

    y_true = extractGroundTruth(head_pos_dataset)

    print(y_true)




if __name__ == '__main__':
    main()
    print("Testing!!")
    # Testing
    # y_true = np.asarray([[0.5, 1],[-1, 1],[7, -6]])
    # y_true = y_true.astype(float).reshape(-1,2)
    # y_pred = np.asarray([[0, 2],[-1, 2],[8, -5]])
    # y_pred = y_pred.astype(float).reshape(-1,2)
    # # print(y_pred)
    # print(mean_squared_error(y_true, y_pred))

