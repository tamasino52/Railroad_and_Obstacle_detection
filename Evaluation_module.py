# ## Run session
# Perform the actual detection by running the model with the image as input
# Danger index coloring function
# ## Import modules
import numpy as np
import math
import tensorflow as tf
import cv2
import os
import matplotlib.image as mpimg
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
import win32gui, win32ui, win32con, win32api
from object_detection.utils import label_map_util
import visualization_utils as vis_util

def box_to_color_map(boxes=None, scores=None, final_img=None, min_score_thresh=0.75, danger_thresh=120, caution_thresh=50, classes=None):
    '''
    This function makes colormap for boxes and labels.
    If there are boxes that have score over min_score_thersh, this function evalutes its danger measure.
    Also, It classifies danger measure to 3 parts
    1. Danger (over danger_tresh)  2. Caution (over caution_tresh)  3. Fine

    :param boxes: Detection boxes
    :param scores: Detection score
    :param final_img: Mask image predicted railway
    :param min_score_thresh: Showing boxes that have scores over min_score_thresh
    :param danger_thresh: Classifying to danger object
    :param caution_thresh: Classifying to caution object
    :param W: Weight
    :return: Colormap by boxes
    '''
    if final_img is None:
        print('ERROR::frame is None')
        return None
    if boxes is None:
        print('ERROR::boxes is None')
        return None
    if scores is None:
        print('ERROR::scores is None')
        return None
    im_height, im_width = final_img.shape[0], final_img.shape[1]
    color_map = [''] * np.squeeze(boxes).__len__()
    for i in range(np.squeeze(boxes).__len__()):
        if np.squeeze(scores)[i] > min_score_thresh:
            (ymin, xmin, ymax, xmax) = np.squeeze(boxes)[i]
            (top, left, bottom, right) = (ymin * im_height, xmin * im_width,  ymax * im_height, xmax * im_width)
            (top, left, bottom, right) = (top.astype(np.int32), left.astype(np.int32),
                                          bottom.astype(np.int32), right.astype(np.int32))

            pixel_sum = 0.0
            for bar_iter in range(left, right):
                try:
                    pixel_sum += final_img[bottom-1][bar_iter]
                except IndexError:
                    print('index Error :: bottom ', bottom, ' / bar_iter ', bar_iter)

            # Feature List P, I, S, Y, L, U, W
            # P : Pixel Average Value
            P = pixel_sum / max(right - left, 1)
            # L :
            if abs(right-left) > abs(bottom - top):
                L = 1.0
            else:
                L = 0.7
            # I : Width of Image
            I = im_width
            # S : Average Size of Box
            S = (abs(right-left) + abs(bottom - top)) / 2
            # Y : Distance of bottom of Image to bottom of Box
            Y = im_height - bottom
            # U : Ultimate Width
            U = (S + (2 * Y / math.tan(math.pi / 3))) / I
            # W : Weight
            if np.squeeze(classes)[i] == 1.0:  # person
                W = 2.0
            elif np.squeeze(classes)[i] == 2.0:  # web
                W = 1.0
            elif np.squeeze(classes)[i] == 3.0:  # tree
                W = 2.0
            elif np.squeeze(classes)[i] == 4.0:  # box
                W = 1.5
            else:
                W = 1.0

            # R value
            R = 100 * P * math.log(U + 2.78) * L * W / math.log((Y / I) + 2.78)

            if R > danger_thresh:
                danger_color = 'blue'  # Danger color
            elif R > caution_thresh:
                danger_color = 'skyblue'  # Caution color
            else:
                danger_color = 'white'  # Fine color
            color_map[i] = danger_color

            print('class ', np.squeeze(classes)[i], '/ score ', np.squeeze(scores)[i], '/ R ', R)

    return color_map