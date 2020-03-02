#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:02:04 2020

@author: jericolinux
"""

import subprocess
import cv2
import time
import os
from skimage import io
import numpy as np
import matplotlib.pyplot as plt


# Get the total rotation of an image
# based on metadata
# https://stackoverflow.com/questions/53097092/frame-from-video-is-upside-down-after-extracting
def getRotateCode(vidpath):
    string = f"script -c 'ffmpeg -i {vidpath}'"
    command = string.split()
    ffmpegOut = subprocess.check_output(['script', '-c', f'ffmpeg -i {vidpath}'])
    rotateCode = int(str(ffmpegOut).split('rotate')[1].split()[1].strip('\\r\\n'))
    return rotateCode


# Rotates the image
def rotateIMG(img, rotateCode):
    if rotateCode == 90:
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotateCode == 180:
        rotated_img = cv2.rotate(img, cv2.ROTATE_180)
    else:
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return rotated_img


class Drawer():
    """
    class drawer contains functions that draws visualization
    for the head and gaze of each subject.
    Thus, allowing visualization for multiple subjects.
    """

    # Get a subject's head_pos for tracking
    def __init__(self, fps):
        # self.head_pos = head_pos
        self.fps_out = fps
        self.frames = []


    # Visually keeps track the model prediction
    def drawTask(self, img, task, object_label='bottle'):
        """
        1. Draw the bounding box
        2. Draw the task classification
        3. Draw another bounding box
        4. Draw the gazed object label
        """

        # Decide task
        if task == 'spont':
            text = "S P O N T"
        elif task == 'undet':
            text = "U N D E T"
        else:
            text = task



        # Initialize variables needed for the text
        height, width, layer = img.shape
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 255)
        thickness = 5
        textSize = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
        origin = (int(0.03*width),int(0.03*height))
        bgPoint = [(origin[0]-5, origin[1]-5),
                        (origin[0]+textSize[0]+5,origin[1]+textSize[1]+5)]
        borderPoint = [(bgPoint[0][0]-8, bgPoint[0][1]-8),
                        (bgPoint[1][0]+8, bgPoint[1][1]+8)]

        # Draw the Bounding box
        bgColor = (0,255,0)
        borderColor = (0,128,0)

        img = np.copy(img) # Fix the problem
        # print(type(img))
        # io.imshow(img)
        # print(bgPoint)
        # print(borderPoint)
        # plt.show(block=True)

        cv2.rectangle(img, bgPoint[0], bgPoint[1], bgColor, -1)
        cv2.rectangle(img, borderPoint[0], borderPoint[1], borderColor, 10)

        # Draw the Text
        origin = (origin[0], origin[1]+textSize[1]-2) # top-left
        cv2.putText(img, text, origin, font_face, font_scale, color, thickness)

        # Future, draw the Gaze to Object
        # Insert Code here
        # drawGaze(img, head_pos, gaze_pt)

        self.frames.append(img)
        return img

    # Visualize the GazeFollow
    def drawGaze(self, head_pos, gaze_pt, object_label):
        """
        1. Create head_pos
        2. Create gaze_pt
        3. Connect
        4.
        """


    # Combine image frames into video
    def getRecording(self, filename, fps=3):
        # get Size
        size = (self.frames[0].shape[1], self.frames[0].shape[0])

        # Destination folder
        dest = '../dsp_intent_analyzer_dataset/draw_vids'
        if not os.path.exists(dest):
            os.mkdir(dest)

        # Create Video based from the file
        filepath = os.path.join(dest, filename)
        out = cv2.VideoWriter(f'{filepath}.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)
        for i in range(len(self.frames)):
            # Append the visualized image to the output video
            out.write(self.frames[i])
        out.release()

        # return out ## Comment-out since video out is not needed

# Sample
def main():
    path = './imgs/processed.png'
    img = cv2.imread(path)
    visualized_out = drawTask(img, 'undet')
    cv2.imwrite('./imgs/boxed_proc.png', visualized_out)



if __name__ == '__main__':
    main()


