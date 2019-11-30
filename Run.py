# ## Import modules
import numpy as np
import tensorflow as tf
import cv2
import os
import argparse
import matplotlib.image as mpimg
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
import win32gui, win32ui, win32con, win32api

# Util modules
from object_detection.utils import label_map_util
import visualization_utils as vis_util
print('UTILITY MODULE PATH :: vis_util=', vis_util.__file__, ', label_map_util=', label_map_util.__file__)
import Capture_module
import ObstacleDetection_module
import Railtracking_module
import Evaluation_module

# Define args property
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()

group.add_argument('-r', '--railway', action="store_true", help='Run Railway Tracking Module only')
parser.add_argument('--weight', default='./railway_detection/weights.hdf5', help='Input weight path')
parser.add_argument('--model', default='./railway_detection/model.h5', help='Input model path')
parser.add_argument('--ckpt', default='./inference_graph/frozen_inference_graph.pb',  help='Input ckpt path')
parser.add_argument('--labels', default='./training/labelmap.pbtxt',  help='Input inference_graph path')
parser.add_argument('--train', default='./training', help='Input training path')
parser.add_argument('--test', default=0, help='Input training path')

args = parser.parse_args()

# ## Set properties
INPUT_SHAPE = (256, 256, 3)
VALID_BOX_LT = None
VALID_BOX_RB = None
WEIGHT_PATH = args.weight
MODEL_PATH = args.model
BATCH_SIZE = 1
PATH_TO_CKPT = args.ckpt
PATH_TO_LABELS = args.labels
NUM_CLASSES = 4

# Rail tracking model build
model = Railtracking_module.build_railtracking_model(MODEL_PATH, INPUT_SHAPE)

# Obstacle detection model load
if args.railway is not True:
    detection_boxes, detection_scores, detection_classes, num_detections, sess, image_tensor, category_index = \
        ObstacleDetection_module.load_lables(PATH_TO_LABELS, PATH_TO_CKPT, NUM_CLASSES)

frame_count = 0
cam = Capture_module.VideoCamera(args.test)
while True:
    # Frame read
    frame = cam.get_frame()
    print('FRAME NUMBER :: ', frame_count, ' =================================')
    frame_count += 1
    im_width, im_height = frame.shape[1], frame.shape[0]

    # Convert frame to input data
    squared_img = cv2.resize(frame, dsize=(256, 256), interpolation=cv2.INTER_AREA) / 255
    input_img = np.expand_dims(squared_img, axis=0)
    predicted_label = model.predict(input_img)[0]
    final_img = cv2.resize(predicted_label, dsize=(im_width, im_height), interpolation=cv2.INTER_AREA)
    zeros = np.zeros((final_img.shape[0], final_img.shape[1]), dtype="uint8")
    backtorgb = cv2.merge([zeros, (final_img * 100).astype(np.uint8), zeros])

    # Mix original image and predicted segmentation mask
    mixed_img = cv2.add(backtorgb, frame)

    if args.railway is not True:
        # Run obstacle detection session
        frame_expanded = np.expand_dims(frame, axis=0)
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Visualize detection boxes
        vis_util.visualize_boxes_and_labels_on_image_array(
            mixed_img,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            color=Evaluation_module.box_to_color_map(boxes=boxes, scores=scores, final_img=final_img, classes=classes),
            use_normalized_coordinates=True,
            line_thickness=3,
            min_score_thresh=0.75);
    cv2.imshow('Display', mixed_img)
    cv2.waitKey(1)
cv2.destroyAllWindows()
