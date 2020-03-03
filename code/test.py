from __future__ import print_function, division
"""
Calculate the accuracy of each part of the model
Head Detection Model
    Mean Absolute Error
Object Recognition Model
    IoU for determining if TP TN FP FN
    draw the precision/recall curve
    gets the mean average precision
Gaze Pathway Prediction Model
    ???
Heatmap Pathway Prediction Model
    ???
"""


import os
import torch
import pandas as pd
from skimage import io, transform
from sklearn.metrics import mean_absolute_error
import numpy as np
import json
import time
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Import my libraries
from head_dataset import HeadPositionDataset, showHeadPosition
from object_dataset import ObjectPositionDataset
from intent_prediction import IntentPredictionNetwork

numcoco2label = {
    25: 'backpack',
    26: 'umbrella',
    39: 'racket',
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
    68: 'phone',
    74: 'book',
    75: 'clock',
    80: 'toothbrush'
}

def create_object_gt(dataset, root_dir):
    """
    Args:
        dataset: gets the ground truth for the object
        root_dir: root directory of images
    """
    gt_dir = os.path.join(root_dir, 'groundtruths')

    if not os.path.exists(gt_dir):
        os.mkdir(gt_dir)


    for i in range(len(dataset)):
        sample = dataset[i]

        filename = sample['name']
        image = sample['image']
        regions = sample['regions']

        # Replace .png with .txt
        filename = filename.split('.')[0]
        filename = f'{filename}.txt'

        # Prepare the gt text file path
        gt_file = os.path.join(gt_dir, filename)

        # Delete file if it exist
        if os.path.exists(gt_file):
            os.remove(gt_file)

        # Create new file for appending
        with open(gt_file, mode='x') as file:
            print(f'Recreated {gt_file}')

        # Loop across all predictions
        for region in regions:
            # Get the prediction
            y_cls = int(region['region_attributes']['backpack'])
            y_cls = numcoco2label[y_cls]
            left = region['shape_attributes']['x']
            top = region['shape_attributes']['y']
            right = left + region['shape_attributes']['width']
            bottom = top + region['shape_attributes']['height']

            data_format = f'{y_cls} {int(left)} {int(top)} {int(right)} {int(bottom)}'
            # Write the predictions into the file
            with open(gt_file, mode='a+') as f:
                # print(data_format)
                f.write(data_format)
                f.write('\n')


def create_object_pred(dataset, root_dir, network):
    """
    Args:
        dataset: gets the ground truth for the object
        root_dir: root directory of images
        network: network used for predictions
    """
    pred_dir = os.path.join(root_dir, 'detections')

    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    for i in range(len(dataset)):
        sample = dataset[i]

        # Required data
        filename = sample['name']
        image = sample['image']
        output = network.object_recog(image)
        classes = output['instances'].pred_classes
        offsets = output['instances'].pred_boxes
        scores = output['instances'].scores

        # Replace .png with .txt
        filename = filename.split('.')[0]
        filename = f'{filename}.txt'

        # Prepare the gt text file path
        pred_file = os.path.join(pred_dir, filename)

        # Delete file if it exist and create new
        if os.path.exists(pred_file):
            os.remove(pred_file)

        # Create new file for appending
        with open(pred_file, mode='x') as file:
            print(f'Recreated {pred_file}')

        # Loop across all predictions
        for y_cls,y_offset,y_score in zip(classes,offsets,scores):
            if torch.is_tensor(y_offset):
                y_offset = y_offset.tolist()

            # Get the prediction
            left = y_offset[0]
            top = y_offset[1]
            right = y_offset[2]
            bottom = y_offset[3]

            y_cls = int(y_cls.to('cpu'))+1
            if not y_cls in numcoco2label:
                # print("Not needed")
                continue

            y_cls = numcoco2label[y_cls]
            y_score = y_score.to('cpu')

            data_format = f'{y_cls} {y_score} {int(left)} {int(top)} {int(right)} {int(bottom)}'
            # Write the predictions into the file
            with open(pred_file, mode='a') as f:
                f.write(data_format)
                f.write('\n')
                # print(data_format)



# For checking the functionality of the object_detection network
def view_objectRecog(network):
    # Initialize model
    network = IntentPredictionNetwork()

    # Object Recognition
    imgpath = '../dsp_intent_analyzer_dataset/head_data/019_gaze_utensils.png'
    image = io.imread(imgpath)

    # output = network.object_recog(image)
    # classes = output['instances'].pred_classes
    # offsets = output['instances'].pred_boxes


    outputs = network.objectRecog(image)

    pred_list = []
    name_list = []


    start = False
    for pred in outputs:
        y_cls = pred['class']
        y_offset = pred['offset']

        xmin,ymin,xmax,ymax = y_offset
        # print(y_cls.to('cpu'),y_offset.to('cpu'))

        if not start:
            # Visualize
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.imshow(image, cmap='gray')
            plt.title(str(y_cls.to('cpu')))

            start = True

        rect = mpatches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        plt.show(block=False)
        plt.pause(0.1)

        # Extract Prediction
        y_offset = y_offset.to('cpu')
        y_cls = y_cls.to('cpu')

        pred = {
            'uid': imgpath,
            'class': y_cls,
            'offset': y_offset
        }

        pred_list.append(pred)


    plt.show(block=True)

    return pred_list

def val_objectRecog(dataset, root_dir, network, dst_dir):
    create_object_pred(dataset, root_dir, network)
    create_object_gt(dataset, root_dir)

    # Get src directory
    src_pred = os.path.join(root_dir, 'detections')
    dst_pred = os.path.join(dst_dir, 'detections')

    # Get all files in src directory
    for file in os.listdir(src_pred):

        # Copy files in src directory to dst directory
        src_file = os.path.join(src_pred, file)
        shutil.copy(src_file,dst_pred)
        # print(src_file, dst_dir)



    # Get src directory
    src_pred = os.path.join(root_dir, 'groundtruths')
    dst_pred = os.path.join(dst_dir, 'groundtruths')

    # Get all files in src directory
    for file in os.listdir(src_pred):

        # Copy files in src directory to dst directory
        src_file = os.path.join(src_pred, file)
        shutil.copy(src_file,dst_pred)
        # print(src_file, dst_dir)


def view_headDetect(network, dataset):
    """
    Here are the ground truth.
    """

    start_time = time.time()
    for i in range(len(dataset)):
        sample = dataset[i]

        # print(sample['head_pos'], type(sample['head_pos']))

        # Append the ground truth and the prediction
        pred = network.headDetect(sample['image'])
        sample['head_pos'] = np.append(sample['head_pos'], [pred], axis=0)
        sample['head_pos'] = sample['head_pos'].astype(float).reshape(-1,2)

        # Visualize the video
        plt.figure()
        ax = plt.subplot(1,1,1)
        plt.tight_layout()
        ax.set_title(f'Sample #{i}')
        ax.axis('off')
        showHeadPosition(**sample)
        plt.show()
        plt.pause(0.001)

        runtime = time.time() - start_time
        percent_completion = float(i / len(dataset)) * 100
        # print(f'Total time taken: {runtime}s  Percent Completion: {percent_completion}%')

        if i == 3:
          plt.show()
          break




def val_headDetect(network, dataset):
    # fig = plt.figure()

    # Extract the ground truth
    y_true = np.array([[]])
    y_pred = np.array([[]])
    start_time = time.time()
    for i in range(len(dataset)):
        sample = dataset[i]

        # Preprocess the data
        x,y,_ = sample['image'].shape

        # Get sample of ground truth and prediction
        truth = sample['head_pos']
        pred = network.headDetect(sample['image'])

        # Remove the extremities
        if mean_absolute_error(pred,[truth[0][0], truth[0][1]]) > 50.0:
            print(pred, [truth[0][0], truth[0][1]])
            continue

        # Append the sample of gt and pred
        y_true = np.append(y_true, [float(truth[0][0]/y), float(truth[0][1]/x)])
        y_pred = np.append(y_pred, [float(pred[0]/y), float(pred[1]/x)])

        if not i % 10:
          runtime = time.time() - start_time
          percent_completion = float(i / len(dataset)) * 100
          print(f'Total time taken: {runtime}s  Percent Completion: {percent_completion}%')


    runtime = time.time() - start_time
    percent_completion = float(i / len(dataset)) * 100
    print(f'Total time taken: {runtime}s  Percent Completion: {percent_completion}%')


    # Comparison model
    y_true = y_true.astype(float).reshape(-1,2)
    y_pred = y_pred.astype(float).reshape(-1,2)

    loss = mean_absolute_error(y_true, y_pred)

    print(f'Loss is {loss}.')

    return loss

def main():

    # Get head data and model
    csv_file = '../dsp_intent_analyzer_dataset/random_head.csv'
    root_dir = '../dsp_intent_analyzer_dataset/head_data'

    head_pos_dataset = HeadPositionDataset(csv_file,root_dir)
    architecture = IntentPredictionNetwork()

    print("TESTING THE HEAD DETECTION MODEL")
    time.sleep(1)
    print("Visualizing the head detection model prediction.")
    view_headDetect(architecture, head_pos_dataset)
    time.sleep(3)

    # Get accuracy of head detection
    print("Getting the accuracy of the model for the overall dataset.")
    accuracy = val_headDetect(architecture, head_pos_dataset)
    print(f'Head detection model has an error of {accuracy} using mean absolute error.')
    time.sleep(3)

    # Get data from object
    json_file = '../dsp_intent_analyzer_dataset/object_data.json'
    root_dir = '../dsp_intent_analyzer_dataset/object_data'

    object_pos_dataset = ObjectPositionDataset(json_file, root_dir)

    # Visualize the object detection
    print("TESTING THE OBJECT RECOGNITION MODEL")
    time.sleep(1)
    print("Visualizing the object detection model prediction.")
    view_objectRecog(architecture)
    time.sleep(3)


    # src_dir = '../dsp_intent_analyzer_dataset/object_data'
    dst_dir = '../Object-Detection-Metrics'
    val_objectRecog(object_pos_dataset, root_dir, architecture, dst_dir)
    print("Finished creating object detection outputs. Now proceeding to evaluation...")
    time.sleep(3)

    # Testing Object Detection (WARNING BAD CODING HERE)
    os.system('python ../Object-Detection-Metrics/pascalvoc.py -gtformat xyrb -detformat xyrb -t 0.50')
    time.sleep(3)

    print("TESTING THE INTENT PREDICTION MODEL")
    time.sleep(1)
    vidpath = '../dsp_intent_analyzer_dataset/raw_vids/001_Task5_2.MOV'
    destvid = 'recogOut'
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))
    timestamp = timestamp.replace(' ', '_')
    architecture.debug_vid = 1
    architecture.predictTask(vidpath, timestamp, 3)
    time.sleep(3)

    # Transferring to local machine for viewing
    print("Transferring to Local Machine")
    #os.system(f'scp ../dsp_intent_analyzer_dataset/draw_vids/{timestamp}.avi jericolinux@10.80.65.162:~/workspace/codes/thesis/intent_prediction/dsp_intent_analyzer_dataset/draw_vids')




if __name__ == '__main__':
    main()

