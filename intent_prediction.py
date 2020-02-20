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
import ffmpeg

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
from visualizingUtils import drawer, getRotateCode, rotateIMG
import gazeFollow_functions as gFF
from intentUtils import getRotateCode, rotateIMG



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

        self.head_pos_arr = np.array([[]])


    # Filter
    # Binary Output
    # Check if subject is moving
    def posFilter(self, head_pos):
        # Insert filter code here
        mean = np.mean(self.head_pos_arr, axis=2)
        std_dev = np.std(self.head_pos_arr, axis=2)
        print(mean, std_dev, head_pos)

        if head_pos < (mean + std_dev):
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

    # Input: Video
    # Output: Task Category
    def predictTask(self, vid):
        # Get the FPS
        fps = vid.get(cv2.CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(int(fps)))

        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

        posFilter_ans = []
        count = 0
        frame_count = 0
        success = True
        while count != total_frames:

            # Validate if working
            success,img = vid.read()
            if count == 0 and not success:
                raise SystemError("Video does not exist!")

            # Increment to monitor vid reading
            count += 1

            # print(f"Count = {count} {success} {img[0][1][1]}")
            if int(count) == 10000:
                completion = 100*float(count/total_frames)
                print(f"Percent Completion: {completion}%")
                break

            # Get predict gazed object of the image
            if not count % (int(fps/10)): # Get 10 frames per total fps

                # progress bar
                completion = 100*float(count/total_frames)
                print(f"Percent Completion: {completion}%")


                # Apply Head Detection
                head_pos = self.headDetect(img)

                if head_pos[0] < 0:
                    continue


                # Append head pos to head pos array
                self.head_pos_arr = np.append(self.head_pos_arr,[head_pos],axis=1)

                if frame_count > 5:
                    # Apply filter
                    decision = self.posFilter(head_pos)
                    # Append answer
                    posFilter_ans.append(decision)

                frame_count += 1



                # Visualize prediction
                # if decision:
                #     self.drawer.drawTask(img, 'undet')
                # else:
                #     self.drawer.drawTask(img, 'spont')

            # Increment to monitor vid reading
            count += 1





        print(posFilter_ans)
        print(head_pos_arr)
        # Get the visualized video
        # self.drawer.getVid()

        task_category = 'spont'
        return task_category

    def drawHeadDetect(self, vidpath, destvid, new_fps):
        # Initialize loop
        vidOut = [] # For writing the video
        stateIndicator = drawer() # Initialize the drawing tool
        fps = vid.get(cv2.CAP_PROP_FPS) # Get the FPS
        rotateCode = getRotateCode(vidpath) # For correcting rotation
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) # For total loops
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.humanKP_cfg.DATASETS.TRAIN[0]), scale=1.2) # Model Prediction
        
        # Loop for getting the head detection for all frames
        start_time = time.time()
        for i in range(50):
            
            # Desired Output FPS
            if not (i % int(fps/new_fps)): 
                
                # Monitor progress of loop
                diff_time = time.time() - start_time
                completion = 100*float(count/total_frames)
                print(f"Percent Completion: {completion}%  for {diff_time}s passed.")
    
                # Apply head detection
                outputs = self.human_keypoint(img)
                
                # Get visualization
                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                imOut = v.get_image()[:, :, ::-1]
    
                # Append the output image
                vidOut.append(imOut)
    
                # Get new image and Increment counter
                success, img = vid.read()

        # Save the output
        drawer.getVid(vidOut, destvid)


    # Orig size img -> 224x224 image
    def preProcess(self, img):
        dim = (224, 224)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) # resize image
        cv2.imwrite('./imgs/processed.png', resized) # New img

        return resized

    # Output: head_pos, head_img
    def headDetect(self, img):
        # Preprocess the image first
        resized = self.preProcess(img)

        # Use detectron2 to extract human keypoints
        outputs = self.human_keypoint(resized)

        if len(outputs["instances"].pred_classes) == 0:
            head_pos = (-1,-1)
            print("No head")
            return head_pos

        # Set for debugging: Shows the output image from model
        if self.debug:
            v = Visualizer(resized[:, :, ::-1], MetadataCatalog.get(self.humanKP_cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            plt.imshow(v.get_image()[:, :, ::-1], CMAP='gray')


        # Extract the position from the human keypoints
        b = outputs["instances"][0:1].pred_keypoints

        x = b.int()
        head_pos = (x[0][0][0:2].to("cpu").numpy())/(224) # Position of nose

        # Eye positions
        left_eye = x[0][1][0:2].to("cpu").numpy()
        right_eye = x[0][2][0:2].to("cpu").numpy()

        # For ablation study: Check which has better accuracy
        eye = (left_eye+right_eye)/(224*2)
        # eye = head_pos

        # print(head_pos, eye)

        return head_pos

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


        return gaze_pathway


    # Output: Gaze Area
    # Size: 224x224 image
    def predictHeatMap(self, img, gaze_pathway):
        # Insert code here
        return gaze_area

    # Output: gazed_object_label, gazed_object_image
    def objectRecog(self, img, heatmap_pathway):
        # Use detectron2 to extract objects in the image
        outputs = self.object_recog(img)

        # Extract the Gazed Object
        ## This depends on output of heatmap pathway

        return gazed_object_label, gazed_object_image

# https://stackoverflow.com/questions/53097092/frame-from-video-is-upside-down-after-extracting
def checkRotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
        rotateCode = cv2.ROTATE_90_CLOCKWISE
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
        rotateCode = cv2.ROTATE_180
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

    return rotateCode


# def main():
#     classifier = intentClassifier()
#     classifier.debug = 1

#     im = cv2.imread('./imgs/pauline_bag.png')
#     plt.imshow(im, CMAP='gray')
#     # print((im.shape))
#     # pause;

#     # One loop:
#     im_processed = classifier.preProcess(im)

#     eye_pos = classifier.headDetect(im_processed)


# if __name__ == '__main__':
    # main().

vidpath = './vidss/001_Task5_2.MOV'

# # Debugging
# vid = cv2.VideoCapture(vidpath)
# length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
# success,img = vid.read()
# rotateCode = checkRotation(vidpath)
# cv2.rotate(img,rotateCode, new_img)
# plt.imshow(img, cmap='gray')
# time.sleep(4)
# plt.imshow(new_img, cmap='gray')
# classifier = intentClassifier()


# Output video is saved in "vidss" folder
classifier.drawHeadDetect(vidpath, "kpOut", 3)



