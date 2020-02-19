#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:02:04 2020

@author: jericolinux
"""

import cv2

TASK_BOX_LOC = [(10, 11),
            (60,30)]
class drawer:

    # def __init__(self):
        
    #     vid
    #     success,image = vid.read()
    #     if not success:
    #         raise SystemExit("Video does not exist!")
        
        
        # fps = vid.get(cv2.CAP_PROP_FPS)
        # height, width, layers = img.shape
        # self.size = (width,height)
        
    #     self.out = cv2.VideoWriter('./vidss/project.avi',cv2.VideoWriter_fourcc(*'DIVX'), int(fps/10), self.size)
        
        
    
    def drawBox(self, img, point1, point2):
        x1, y1 = point1[0], point1[1] # top-left
        x2, y2 = point1[0], point2[1] # btm-left
        x3, y3 = point2[0], point1[1] # top-right
        x4, y4 = point2[0], point2[1] # btm-right
    
        # cv2.circle(img, (x1, y1), 3, (255, 0, 255), -1)    #-- top_left
        # cv2.circle(img, (x2, y2), 3, (255, 0, 255), -1)    #-- bottom-left
        # cv2.circle(img, (x3, y3), 3, (255, 0, 255), -1)    #-- top-right
        # cv2.circle(img, (x4, y4), 3, (255, 0, 255), -1)    #-- bottom-right
        lineColor = (0,255,0)
        cv2.line(img, (x1, y1), (x3,y3), lineColor, 2)    #-- top-left -> top-right
        cv2.line(img, (x1, y1), (x2,y2), lineColor, 2)    #-- top-left -> btm-left
        cv2.line(img, (x4, y4), (x3,y3), lineColor, 2)    #-- btm-right -> top-right
        cv2.line(img, (x4, y4), (x2,y2), lineColor, 2)    #-- btm-right -> btm-left
    
    
        return img
    
    def drawText(self, img, text, point):
        
        origin = point[0]+4, point[1]+10 # top-left
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.2
        color = (0, 0, 255)
        thickness = 1
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
        
        # Draw the box
        boxed_proc =  drawBox(img, TASK_BOX_LOC[0], TASK_BOX_LOC[1])
        
        # Draw the text
        text_proc = drawText(img, text, TASK_BOX_LOC[0])
        
        
        
        return text_proc
    
    @staticmethod
    def getVid(frames, name):
        size = (224, 224)
        
        out = cv2.VideoWriter(f'./vidss/{name}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
        
        
        height, width, layers = frames[0].shape
        size = (width,height)
        
        for frame in frames:
            # Append the visualized image to the output video
            out.write(frame)
        
        out.release()
    

if __name__ == '__main__':
    path = './imgs/processed.png'
    img = cv2.imread(path)
    visualized_out = drawTask(img, 'undet')
    cv2.imwrite('./imgs/boxed_proc.png', visualized_out)
    
    
    