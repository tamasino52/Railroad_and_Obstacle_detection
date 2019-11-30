#!/usr/bin/env python
# coding: utf-8
# # Railway obstacle detector with Tensorflow and Keras
'''
Last modified on 2019.11.20
Author - Kim Minseok (Software department in Soongsil University)
'''

# ## Import modules
import numpy as np
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

# # Obstacle Detection Model
# ## Load lable map
# Load the label map.
def load_lables(PATH_TO_LABELS,PATH_TO_CKPT, NUM_CLASSES):
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=NUM_CLASSES,
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # ## Load the tf model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Input tensor is the image. Output tensors are the detection boxes, scores, and classes
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    return detection_boxes, detection_scores, detection_classes, num_detections, sess, image_tensor, category_index