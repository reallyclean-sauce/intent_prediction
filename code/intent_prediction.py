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
import cv2
from PIL import Image, ImageOps
import random
from tqdm import tqdm
import operator
import itertools
from scipy.io import  loadmat
import logging
import subprocess

# import math libraries
import numpy as np
import cv2
import random
from skimage import io, transforms

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
import utils

class IntentPredictionNetwork():

    def __init__(self):
        # Predict Object Classifications of objects
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg = get_cfg()
        # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.3
        # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the detectron2:// shorthand
        # Object detection https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_101_FPN_3x/138363263/model_final_59f53c.pkl
        # cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/retinanet_R_101_FPN_3x/138363263/model_final_59f53c.pkl"
        self.objectRec_cfg = cfg
        self.object_recog = DefaultPredictor(cfg)

        # Predict Keypoints of humans
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"
        self.humanKP_cfg = cfg
        self.human_keypoint = DefaultPredictor(cfg)

        # For head position filter
        self.head_pos_arr = np.array([[0,0]])

        # For Getting Gazed Objects
        self.OOIs = []

        # default to not debug
        self.debug = 0 # For frame-by-frame debugging
        self.debug_vid = 0 # For whole video debugging

    # Input: Video
    # Output: Task Category
    def predictTask(self, vid, destvid='../dsp_intent_analyzer/draw_vids/tmp', new_fps=3):

        # Initialize loop
        vid = cv2.VideoCapture(vidpath) # Get the video
        fps = vid.get(cv2.CAP_PROP_FPS) # Get the FPS
        rotateCode = utils.getRotateCode(vidpath) # For correcting rotation
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) # For total loops

        if self.debug_vid:
            stateIndicator = utils.Drawer() # Initialize the drawing tool

        # Loop for getting the head detection for all frames
        start_time = time.time()
        for i in range(total_frames-1):
            # Get new image and Increment counter
            success, img = vid.read()

            # Desired Output FPS
            if not (i % int(fps/new_fps)):

                # Monitor progress of loop
                diff_time = time.time() - start_time
                completion = 100*float(i/total_frames)
                print(f"Percent Completion: {completion}%  for {diff_time}s passed.")

                # Image preprocessing
                new_img = utils.rotateIMG(img, rotateCode)

                # Apply head detection
                head_pos = self.headDetect(new_img)

                # Head Position Filter
                enable = self.posFilter(head_pos)

                # Append head position for filtering
                self.head_pos_arr = np.append(self.head_pos_arr, [head_pos], axis=0)

                # For visualization
                if self.debug_vid:
                    # Apply human keypoint for visualization
                    outputs = self.human_keypoint(new_img)

                    # Get visualization
                    v = Visualizer(new_img[:, :, ::-1], MetadataCatalog.get(self.humanKP_cfg.DATASETS.TRAIN[0]), scale=1.2) # Model Prediction
                    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                    imOut = v.get_image()[:, :, ::-1]

                    # Visualize Head Position Filter
                    if int(i / int(fps/new_fps)) > 5:
                        if enable:
                            current_task = 'undet'
                            new_img = stateIndicator.drawTask(imOut, 'undet')
                        else:
                            current_task = 'spont'
                            new_img = stateIndicator.drawTask(imOut, 'spont')
                    else:
                        current_task = 'spont'
                        new_img = stateIndicator.drawTask(imOut, 'spont')



        # Save the video output
        if self.debug_vid:
            stateIndicator.getRecording(destvid, new_fps)

        task_category = 'spont'
        return task_category

    # Output: head_pos, head_img
    def headDetect(self, img):
        # Preprocess the image first
        resized = self.preProcess(img)

        # Use detectron2 to extract human keypoints
        outputs = self.human_keypoint(resized)

        # No head detected
        if len(outputs["instances"].pred_classes) == 0:
            head_pos = [0,0]
            return head_pos

        # Extract the position from the human keypoints
        b = outputs["instances"][0:1].pred_keypoints

        x = b.int()
        head_pos = (x[0][0][0:2].to("cpu").numpy())/(224) # Position of nose

        # Eye positions
        left_eye = x[0][1][0:2].to("cpu").numpy()
        right_eye = x[0][2][0:2].to("cpu").numpy()

        # For ablation study: Check which has better accuracy
        eye_pos = (left_eye+right_eye)/(224*2)

        # Shows the output image from model
        if self.debug:
            v = Visualizer(resized[:, :, ::-1], MetadataCatalog.get(self.humanKP_cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            plt.title("Keypoint Prediction Output")
            plt.imshow(v.get_image()[:, :, ::-1], CMAP='gray')
            plt.show(block=True)

        return eye_pos

    # Output: Gaze Pathway Probability Map
    # Size: 224x224
    def predictGazeDir(self, img, head_pos):
        # Get head img
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

        # Insert code here
        # Provide visualization for debugging of gaze direction


        return gaze_pathway


    # Output: Gaze pt
    # Size: 224x224 image
    def predictHeatMap(self, img, gaze_pathway):
        # Insert code here

        # Provide visualization for debugging of gaze pt
        # And Heat map

        return gaze_pt

    # Output: List of object labels and positions
    # [0,81]: Labels
    # (xmin,ymin),(xmax,ymax): Bounding Box
    def objectRecog(self, img):
        """
        1. Predict the image
        2. Get the classes of each object
        3. Remove if 'person'
        4. Get the labels and their corresponding bbox
        5. Simplify the labels for thesis Purpose
        6. Append to Objects List
        7. Repeat for all objects
        """

        # Use detectron2 to extract objects in the image
        outputs = self.object_recog(img)

        # Shows sample output of image from model
        if self.debug:
            v = Visualizer(resized[:, :, ::-1], MetadataCatalog.get(self.objectRec_cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            plt.title("Object Recognition Output")
            plt.imshow(v.get_image()[:, :, ::-1], CMAP='gray')
            plt.show(block=True)

        # Insert Code
        # Incomplete for the mean time
        for objects in outputs:
            if self.objects['instances'].pred_classes == 0:
                continue

            objectOfInterest = {}
            objectOfInterest['label'] = 'dummy'
            originpt = (0,0) # (xmin,ymin)
            finalpt = (0,0) # (xmax,ymax)
            objectOfInterest['boundbox'] = [originpt, finalpt]

        self.OOIs.append(objectOfInterest)

    # Output:
    # Gaze Label
    def predictGazedObject(self, gaze_pt):
        """
        1. Get all bounding box loc for objects
        2. Get the gaze point
        3. Get in which bounding box is the gaze pt inside
        4. Determine the corresponding label of the bounding box
        """

        return gazed_object_label

    # Orig size img -> 224x224 image
    def preProcess(self, img):
        dim = (224, 224)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) # resize image
        cv2.imwrite('./imgs/processed.png', resized) # New img

        return resized


    # Filter
    # Binary Output
    # Check if subject is moving
    def posFilter(self, head_pos):
        # Insert filter code here
        mean = np.mean(self.head_pos_arr, axis=0)
        std_dev = np.std(self.head_pos_arr, axis=0)


        # mean = np.mean(self.head_pos_arr[len(self.head_pos_arr)-5:len(self.head_pos_arr)], axis=0)
        # std_dev = np.std(self.head_pos_arr[len(self.head_pos_arr)-5:len(self.head_pos_arr)], axis=0)
        # print(self.head_pos_arr[len(self.head_pos_arr)-3:len(self.head_pos_arr)])
        # raise SystemError("HALLOOOO")
        # print("Hallooo", mean, std_dev)
        # print("Add mean and std_dev ", (mean + std_dev)[0])
        # print("head_pos ", head_pos, type(head_pos))

        if head_pos[0] < (mean + std_dev)[0]:
            decision = True
        else:
            decision = False

        return decision

    # Filter
    # Binary Output
    # Check if gaze is towards the area/objets
    def gazeFilter(self, img, gaze_direction):
        # Insert filter code here
        return decision

def main():
    # vidpath = './raw_vids/001_Task5_2.MOV'

    # Initialize model
    classifier = intentClassifier()

    # # Output video is saved in "vidss" folder
    # classifier.predictTask(vidpath, '../dsp_intent_analyzer/recogOut', 3)

    # Object Recognition
    imgpath = '../dsp_intent_analyzer/head_data/004_gaze_undetermined.png'
    img = io.imread(imgpath)

    output = object_recog(img)
    y_offset = output['instances'].pred_boxes

    for xmin,ymin,xmax,ymax in y_offset:
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.imshow(image, cmap='gray')
        plt.title(str(y_cls))

        rect = mpatches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        plt.show()



if __name__ == '__main__':
    main()

