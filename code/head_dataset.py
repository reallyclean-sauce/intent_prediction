from __future__ import print_function, division
"""
Contains a dataset class for
validating the accuracy of the models
in the Intent Prediction Architecture
"""

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def showHeadPosition(name, image, head_pos):
    """
    Show the position of the head
    in the image
    """
    print(name)
    plt.imshow(image)
    plt.scatter(head_pos[:,0], head_pos[:,1], s=10, marker='.', c='r')
    plt.pause(0.001) # For loading the image

# Inherit features of torch Dataset
class HeadPositionDataset(Dataset):
    """
    Middle of the Eyes Position Dataset
    To be specific
    """

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): the file path of the csv dataset
            root_dir (string): Directory with all the images
            transform (callable, optional): Optional transform
                This can be resizing the image into 224x224
        """
        self.head_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    # Need to be created;
    def __len__(self):
        return len(self.head_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Variables
        file_idx = 0 # Contains the filename of the image
        attr_idx = 5 # Contains the head position data

        # Extract Image
        img_name = self.head_data.iloc[idx, file_idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = io.imread(img_path)

        # Extract Head Position
        head_pos = json.loads(self.head_data.iloc[idx, attr_idx])
        head_pos = [head_pos['cx'],head_pos['cy']]
        head_pos = np.asarray(head_pos)
        head_pos = head_pos.astype('float').reshape(-1,2)

        sample = {
            'name': img_name,
            'image': image,
            'head_pos': head_pos
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == '__main__':
    csv_path = '../dsp_intent_analyzer_dataset/head_data.csv'
    head_data_dir = '../dsp_intent_analyzer_dataset/head_data'

    dataset = HeadPositionDataset(csv_path,head_data_dir)

    # Apply each of the above transforms on sample.
    fig = plt.figure()
    # sample = face_dataset[65]
    for i in range(len(dataset)):
        sample = dataset[i]

        print(sample['uid'])

        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        showHeadPosition(**sample)

        if i == 2:
            break

    plt.show()
