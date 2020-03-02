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

# COCO_Dataset to Detectron2 class label
numcoco2label = {
    25: 'backpack',
    26: 'umbrella',
    39: 'tennis racket',
    40: 'bottle',
    42: 'cup',
    43: 'fork',
    44: 'knife',
    45: 'spoon',
    46: 'bowl',
    47: 'banana',
    48: 'apple',
    49: 'sandwich',
    50: 'orange',
    64: 'laptop',
    65: 'mouse',
    67: 'keyboard',
    68: 'cell phone',
    74: 'book',
    75: 'clock',
    80: 'toothbrush'
}

# Inherit features of torch Dataset
class ObjectPositionDataset(Dataset):
    """
    Middle of the Eyes Position Dataset
    To be specific
    """

    def __init__(self, json_file, root_dir, transform=None):
        """
        Args:
            json_file (string): the file path of the csv dataset
            root_dir (string): Directory with all the images
            transform (callable, optional): Optional transform
                This can be resizing the image into 224x224
        """
        with open(json_file) as file:
            file = json.load(file)
        self.object_data = pd.DataFrame().from_dict(file['_via_img_metadata']).T
        self.root_dir = root_dir
        self.transform = transform

    # Need to be created;
    def __len__(self):
        return len(self.object_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Index Variables
        file_idx = 0
        region_idx = 2

        # Get the required data
        img_name = self.object_data.iloc[idx, file_idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = io.imread(img_path)

        regions = self.object_data.iloc[idx, region_idx]

        sample = {
            'name': img_name,
            'image': image,
            'regions': regions
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def looping_json():
    json_file = '../dsp_intent_analyzer_dataset/object_data.json'
    with open(json_file) as file:
        file = json.load(file)
        # print(file)

    print("Getting the filename: ", file['_via_img_metadata']['001_gaze_undetermined.png1437982']['filename'])
    print("Getting the offset: ", file['_via_img_metadata']['001_gaze_undetermined.png1437982']['regions'][0]['shape_attributes']['x'])
    print("Getting the class", file['_via_img_metadata']['001_gaze_undetermined.png1437982']['regions'][0]['region_attributes']['backpack'])

    # print("Looping across the different files: ")
    # for instance in file['_via_img_metadata']:
    #     print(file['_via_img_metadata'][instance]['filename'])

    print("Looping across different classes and offsets")
    for instance in file['_via_img_metadata']:
        print(file['_via_img_metadata'][instance]['filename'])
        for region in file['_via_img_metadata'][instance]['regions']:
            xmin = region['shape_attributes']['x']
            ymin = region['shape_attributes']['y']
            xmax = xmin + region['shape_attributes']['width']
            ymax = ymin + region['shape_attributes']['height']
            ob_cls = int(region['region_attributes']['backpack'])
            print("Class:", ob_cls, "  Offsets:", xmin,ymin,xmax,ymax)

def main():
    json_file = '../dsp_intent_analyzer_dataset/object_data.json'
    object_data_dir = '../dsp_intent_analyzer_dataset/object_data'

    dataset = ObjectPositionDataset(json_file,object_data_dir)

    for i in range(len(dataset)):
        sample = dataset[i]

        # Extract data
        image = sample['image']
        regions = sample['regions']

        # Extract all classes for the image
        start = False
        for region in regions:
            xmin = region['shape_attributes']['x']
            ymin = region['shape_attributes']['y']
            xmax = xmin + region['shape_attributes']['width']
            ymax = ymin + region['shape_attributes']['height']
            ob_cls = region['region_attributes']['backpack']

            if not start:
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.imshow(image, cmap='gray')
                plt.title(str(ob_cls))

                start = True

            rect = mpatches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            plt.show(block=False)
            plt.pause(0.1)
        break

if __name__ == '__main__':
    main()

    # Testing json look
    # looping_json()



