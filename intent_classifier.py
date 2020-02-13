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

# import some common libraries
import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


class intentClassifier:
    
    def __init__(self):
        # Predict Object Classifications of objects
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the detectron2:// shorthand
        cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
        self.object_recog = DefaultPredictor(cfg)

        # Predict Keypoints of humans
        cfg = get_cfg()
        cfg.merge_from_file("./detectron2_repo/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"
        self.human_keypoint = DefaultPredictor(cfg)


    def head_detect(self, img):
        # Use detectron2 to extract human keypoints
        outputs = self.human_keypoint(img)

        # Extract the position from the human keypoints
        # !!!! Check this with pixel_pos_sampler !!!!
        b = outputs["instances"][0:1].pred_keypoints 
        x = b.int()
        head_pos = x[0][0][0:2].to("cpu").numpy() # Position of nose
        
        # For position of middle of eye
        # diff1 = np.linalg.norm(x[0][0][0:2].to("cpu").numpy().all(),x[0][1][0:2].to("cpu").numpy().all()) # nose -> first eye
        # diff2 = np.linalg.norm(x[0][0],x[0][2]) # nose -> second eye
        # ave_diff = (diff1+diff2)/2

        ## Add head extraction code here
        left_shoulder = x[0][5][0:2].to("cpu").numpy()
        right_shoulder = x[0][6][0:2].to("cpu").numpy()
        left_eye = x[0][1][0:2].to("cpu").numpy()
        right_eye = x[0][2][0:2].to("cpu").numpy()
        
        head_pos = [left_eye,
                    right_eye,
                    left_shoulder,
                    right_shoulder]

        

        return head_pos, head_img

    def object_recog(self, img, heatmap_pathway_out):
        # Use detectron2 to extract objects in the image
        outputs = self.object_recog(img)

        # Extract the Gazed Object
        ## This depends on output of heatmap pathway

        return gazed_object_label, gazed_object_image
