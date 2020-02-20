#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:02:04 2020

@author: jericolinux
"""

import subprocess
import cv2
import time
import matplotlib.pyplot as plt


# Get the total rotation of an image
# based on metadata
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


class drawer:
    # def drawBox(self, img, textSize):
    #     height, width, layer = img.shape

    #     # x1, y1 = point1[0], point1[1] # top-left
    #     # x2, y2 = point1[0], point2[1] # btm-left
    #     # x3, y3 = point2[0], point1[1] # top-right
    #     # x4, y4 = point2[0], point2[1] # btm-right
    #     # # cv2.circle(img, (x1, y1), 3, (255, 0, 255), -1)    #-- top_left
    #     # # cv2.circle(img, (x2, y2), 3, (255, 0, 255), -1)    #-- bottom-left
    #     # # cv2.circle(img, (x3, y3), 3, (255, 0, 255), -1)    #-- top-right
    #     # # cv2.circle(img, (x4, y4), 3, (255, 0, 255), -1)    #-- bottom-right
    #     # cv2.line(img, (x1, y1), (x3,y3), lineColor, 2)    #-- top-left -> top-right
    #     # cv2.line(img, (x1, y1), (x2,y2), lineColor, 2)    #-- top-left -> btm-left
    #     # cv2.line(img, (x4, y4), (x3,y3), lineColor, 2)    #-- btm-righat -> top-right
    #     # cv2.line(img, (x4, y4), (x2,y2), lineColor, 2)    #-- btm-right -> btm-left
    #     return img
    def drawText(self, img, text):
        height, width, layer = img.shape
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 255)
        thickness = 5
        textSize = cv2.getTextSize(text, font_face, font_scale, thickness)[0]

        # Draw the bounding box of text
        origin = (int(0.03*width),int(0.03*height))
        bgPoint = [(origin[0]-5, origin[1]-5),
                        (origin[0]+textSize[0]+5,origin[1]+textSize[1]+5)]
        borderPoint = [(bgPoint[0][0]-8, bgPoint[0][1]-8),
                        (bgPoint[1][0]+8, bgPoint[1][1]+8)]

        bgColor = (0,255,0)
        borderColor = (0,128,0)
        cv2.rectangle(img, bgPoint[0], bgPoint[1], bgColor, -1)
        cv2.rectangle(img, borderPoint[0], borderPoint[1], borderColor, 10)

        # Draw the text
        # point, _ = [(int(0.01*width),  int(0.05*height)),
        #                 (int(0.4*width),int(0.2*height))]
        origin = (origin[0], origin[1]+textSize[1]-2) # top-left
        cv2.putText(img, text, origin, font_face, font_scale, color, thickness)
        return img
    def drawTask(self, img, task):
        # Decide task
        if task == 'spont':
            text = "S P O N T"
        elif task == 'undet':
            text = "U N D E T"
        else:
            text = task
        # Draw the text
        text_proc = self.drawText(img, text)
        # # Draw the box
        # boxed_proc =  self.drawBox(img, textSize)
        return text_proc
    # Combine image frames into video
    @staticmethod
    def getVid(frames, name, fps):
        size = (frames[0].shape[1], frames[0].shape[0])
        out = cv2.VideoWriter(f'./vidss/{name}.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)
        for i in range(len(frames)):
            # Append the visualized image to the output video
            out.write(frames[i])
        out.release()

if __name__ == '__main__':
    path = './imgs/processed.png'
    img = cv2.imread(path)
    visualized_out = drawTask(img, 'undet')
    cv2.imwrite('./imgs/boxed_proc.png', visualized_out)


