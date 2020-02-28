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


# Inherit features of torch Dataset
class ObjectPositionDataset(Dataset):
    """
    Middle of the Eyes Position Dataset
    To be specific
    """

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            json_file (string): the file path of the csv dataset
            root_dir (string): Directory with all the images
            transform (callable, optional): Optional transform
                This can be resizing the image into 224x224
        """
        self.object_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    # Need to be created;
    def __len__(self):
        return len(self.object_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Index Variables
        file_idx = 0 # Contains the filename
        cls_idx = 6 # Contains the class
        offset_idx = 5 # Contains the offset

        # Extract Image Data
        img_name = self.object_data.iloc[idx,file_idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = io.imread(img_path)

        # Extracting the class
        region_cls = json.loads(self.object_data.iloc[idx,cls_idx])
        y_cls = int(region_cls['backpack'])

        # Extracting the object
        region_offset = json.loads(self.object_data.iloc[idx,offset_idx])
        xmin = int(region_offset['x'])
        ymin = int(region_offset['y'])
        xmax = xmin + int(region_offset['width'])
        ymax = ymin + int(region_offset['height'])
        origin = (xmin,ymin)
        endpt  = (xmax,ymax)
        y_offset = [origin,endpt]

        sample = {
            'uid': f'{img_name}_{y_cls}',
            'image': image,
            'class': y_cls,
            'offset': y_offset
        }

        if self.transform:
            sample = self.transform(sample)

        return sample




if __name__ == '__main__':
    csv_path = '../dsp_intent_analyzer_dataset/object_data.csv'
    object_data_dir = '../dsp_intent_analyzer_dataset/object_data'

    dataset = ObjectPositionDataset(csv_path,object_data_dir)


    for i in range(len(dataset)):
        sample = dataset[i]

        # Extract data
        print(sample['uid'])
        image = sample['image']
        y_cls = sample['class']
        y_offset = sample['offset']
        print(y_cls,y_offset)

        xmin = y_offset[0][0]
        ymin = y_offset[0][1]
        xmax = y_offset[1][0]
        ymax = y_offset[1][1]

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.imshow(image, cmap='gray')
        plt.title(str(y_cls))

        rect = mpatches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        plt.show()

        if i > 5:
            break




