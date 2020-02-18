#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 18:40:04 2020

@author: jericolinux
"""
# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import important libraries
import sys
import time
import os
import numpy as np
import json
import cv2
from PIL import Image, ImageOps
import random
from tqdm import tqdm
import operator
import itertools
from scipy.io import  loadmat
import logging

# import math libraries
import numpy as np
import cv2
import random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt

# import torch libraries
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel

# Import gazeFollow Libraries
import gazeFollow_functions as gFF



class intentClassifier:

    def __init__(self):
        # Predict Object Classifications of objects
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the detectron2:// shorthand
        cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
        self.objectRec_cfg = cfg
        self.object_recog = DefaultPredictor(cfg)

        # Predict Keypoints of humans
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"
        self.humanKP_cfg = cfg
        self.human_keypoint = DefaultPredictor(cfg)
        
        self.debug = 0 # default to not debug
        
    # Orig size img -> 224x224 image
    def preprocess(self, img):
        dim = (224, 224)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
        
        return resized
        

    # Output: head_pos, head_img
    def head_detect(self, img):
        # Preprocess the image first
        
        
        # Use detectron2 to extract human keypoints
        outputs = self.human_keypoint(img)
        
        if self.debug:
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.humanKP_cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            # cv2_imshow(v.get_image()[:, :, ::-1])
            plt.imshow(v.get_image()[:, :, ::-1], CMAP='gray')
            # print(v.get_image().shape)

        # Extract the position from the human keypoints
        # !!!! Check this with pixel_pos_sampler !!!!
        b = outputs["instances"][0:1].pred_keypoints
        x = b.int()
        head_pos = (x[0][0][0:2].to("cpu").numpy())/(224) # Position of nose

        # For position of middle of eye
        # diff1 = np.linalg.norm(x[0][0][0:2].to("cpu").numpy().all(),x[0][1][0:2].to("cpu").numpy().all()) # nose -> first eye
        # diff2 = np.linalg.norm(x[0][0],x[0][2]) # nose -> second eye
        # ave_diff = (diff1+diff2)/2

        ## Add head extraction code here
        left_shoulder = x[0][5][0:2].to("cpu").numpy()
        right_shoulder = x[0][6][0:2].to("cpu").numpy()
        left_eye = x[0][1][0:2].to("cpu").numpy()
        right_eye = x[0][2][0:2].to("cpu").numpy()

        # head_pos = [left_eye,
        #             right_eye,
        #             left_shoulder,
        #             right_shoulder]
        
        eye = (left_eye+right_eye)/(224*2)
        # eye = head_pos
        
        print(head_pos, eye)
        
        # crop face
        x_c, y_c = head_pos
        x_0 = x_c - 0.15
        y_0 = y_c - 0.15
        x_1 = x_c + 0.15
        y_1 = y_c + 0.15
        if x_0 < 0:
            x_0 = 0
        if y_0 < 0:
            y_0 = 0
        if x_1 > 1:
            x_1 = 1
        if y_1 > 1:
            y_1 = 1
    
        h, w = img.shape[:2]
        head_img = img[int(y_0 * h):int(y_1 * h), int(x_0 * w):int(x_1 * w), :]
        

        return head_pos, head_img

    # Output: Gaze Pathway Probability Map
    # Size: 224x224
    def predict_pathway(self, img, head_pos, head_img):
        # Insert code here

        return gaze_pathway

    # Output: Gaze Area
    # Size: 224x224 image
    def predict_heatmap(self, img, gaze_pathway):
        # Insert code here

        return gaze_area

    # Output: gazed_object_label, gazed_object_image
    def object_recog(self, img, heatmap_pathway):
        # Use detectron2 to extract objects in the image
        outputs = self.object_recog(img)

        # Extract the Gazed Object
        ## This depends on output of heatmap pathway

        return gazed_object_label, gazed_object_image

    # Filter
    # Binary Output
    # Check if subject is moving
    def position_filter(self, img, head_pos_arr):
        # Insert filter code here

        return decision

    # Filter
    # Binary Output
    # Check if gaze is towards the area/objets
    def gaze_filter(self, img, gaze_direction):
        # Insert filter code here

        return decision
    
# def main():
classifier = intentClassifier()
classifier.debug = 1

im = cv2.imread('./imgs/input.jpg')
plt.imshow(im, CMAP='gray')
# print((im.shape))
# pause;

# One loop:
im_processed = classifier.preprocess(im)

eye_pos, head = classifier.head_detect(im_processed)

print(eye_pos)
plt.imshow(head)

# if __name__ == '__main__':
#     main()

