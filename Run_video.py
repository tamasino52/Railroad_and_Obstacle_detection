#!/usr/bin/env python
# coding: utf-8
# # Railway obstacle detector with Tensorflow and Keras
'''
Last modified on 2019.09.03
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
#Util modules
from object_detection.utils import label_map_util
import visualization_utils as vis_util
print('UTILITY MODULE PATH :: vis_util=', vis_util.__file__, ', label_map_util=', label_map_util.__file__)

# ## Set properties
INPUT_SHAPE = (256, 256, 3)
IMAGE_PATH = None
VIDEO_PATH = './complex7.mp4'
WEIGHT_PATH = './railway_detection/weights.hdf5'
MODEL_PATH = './railway_detection/model.h5'
MODEL_NAME = 'inference_graph'
BATCH_SIZE = 1
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join('training', 'labelmap.pbtxt')
NUM_CLASSES = 4

# ## Model build function
def conv_block(input_tensor, num_filters):
  encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
  encoder = layers.BatchNormalization()(encoder)
  encoder = layers.Activation('relu')(encoder)
  encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
  encoder = layers.BatchNormalization()(encoder)
  encoder = layers.Activation('relu')(encoder)
  return encoder


def encoder_block(input_tensor, num_filters):
  encoder = conv_block(input_tensor, num_filters)
  encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
  return encoder_pool, encoder


def decoder_block(input_tensor, concat_tensor, num_filters):
  decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
  decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  return decoder


# ## Model compile function
def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


# ## Build or load model
if not os.path.exists(MODEL_PATH):
    inputs = layers.Input(shape=INPUT_SHAPE)  # 256
    encoder0_pool, encoder0 = encoder_block(inputs, 32)  # 128
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)  # 64
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)  # 32
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)  # 16
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)  # 8
    center = conv_block(encoder4_pool, 1024)  # center
    decoder4 = decoder_block(center, encoder4, 512)  # 16
    decoder3 = decoder_block(decoder4, encoder3, 256)  # 32
    decoder2 = decoder_block(decoder3, encoder2, 128)  # 64
    decoder1 = decoder_block(decoder2, encoder1, 64)  # 128
    decoder0 = decoder_block(decoder1, encoder0, 32)  # 256
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)
    model = models.Model(inputs=[inputs], outputs=[outputs])
    print('New model built')
else:
    model = models.load_model(MODEL_PATH, custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_loss': dice_loss})
    print('Model loaded :: ', MODEL_PATH)


# ## Compile model
model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss])
model.summary()
print('Model compiled :: opimizer=adam')


# ## Show image result
if IMAGE_PATH is not None:
    test_img =mpimg.imread(IMAGE_PATH)
    print('Image read :: ', IMAGE_PATH)
    squared_img = cv2.resize(test_img, dsize=(256, 256), interpolation=cv2.INTER_AREA) / 255
    input_img = np.expand_dims(squared_img, axis=0)
    predicted_label = model.predict(input_img)[0]
    print('predicted_Iabel image made')
    resize_img = cv2.resize(predicted_label,(test_img.shape[1], test_img.shape[0]))
    cv2.imshow('image', resize_img)
    cv2.waitKey(0)


# # Obstacle Detection Model
# ## Load lable map
# Load the label map.
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


# ## Run session
# Perform the actual detection by running the model with the image as input
# Danger index coloring function
def box_to_color_map(boxes=None, scores=None, final_img=None, min_score_thresh=0.75, danger_tresh=0.7, caution_tresh=0.3):
    '''
    This function makes colormap for boxes and labels.
    If there are boxes that have score over min_score_thersh, this function evalutes its danger measure.
    Also, It classifies danger measure to 3 parts
    1. Danger (over danger_tresh)  2. Caution (over caution_tresh)  3. Fine

    :param boxes: Detection boxes
    :param scores: Detection score
    :param final_img: Mask image predicted railway
    :param min_score_thresh: Showing boxes that have scores over min_score_thresh
    :param danger_tresh: Classifying to danger object
    :param caution_tresh: Classifying to caution object
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
    color_map = ['white'] * np.squeeze(boxes).__len__()
    for i in range(np.squeeze(boxes).__len__()):
        if np.squeeze(scores)[i] > min_score_thresh:
            (ymin, xmin, ymax, xmax) = np.squeeze(boxes)[i]
            (top, left, bottom, right) = (ymin * im_height, xmin * im_width,  ymax * im_height, xmax * im_width)
            (top, left, bottom, right) = (top.astype(np.int32), left.astype(np.int32),
                                          bottom.astype(np.int32), right.astype(np.int32))
            pixel_sum = 0.0

            print('\nProperty------------------------------------')
            print('score ', np.squeeze(scores)[i])
            print(final_img[bottom-1][left:right])
            for bar_iter in range(left, right):
                try:
                    pixel_sum += final_img[bottom-1][bar_iter]
                except IndexError:
                    print('index Error :: bottom ', bottom, ' / bar_iter ', bar_iter)
            print('im_height ', im_height, ' / im_width ', im_width)
            print('top ', top, ' / left ', left, ' / bottom ', bottom, ' / right ', right)
            print('xmin ', xmin, ' / xmax ', xmax, ' / ymin ', ymin, ' / ymax ', ymax)
            print('pixel_sum ', pixel_sum, ' / average value ', pixel_sum/(right-left))

            danger_score = pixel_sum / max(right - left  + 1, 1)  # Average value
            if danger_score > danger_tresh:
                danger_color = 'blue'  # Danger color
            elif danger_score > caution_tresh:
                danger_color = 'deepskyblue'  # Caution color
            else:
                danger_color = 'cyan'  # Fine color
            color_map[i] = danger_color
    return color_map


# # Video result
cap = cv2.VideoCapture(VIDEO_PATH)
print('Video loaded :: ', VIDEO_PATH)
frame_count = 0

while cap.isOpened():
    # Frame read
    ret, frame = cap.read()
    print('Frame Number :: ', frame_count)
    frame_count = frame_count + 1
    if not ret:
        break
    if frame.shape[0] > 1000:  # Downsizing
        frame = cv2.resize(frame, dsize=(int(frame.shape[1]/2), int(frame.shape[0]/2)), interpolation=cv2.INTER_AREA)
    im_width, im_height = frame.shape[1], frame.shape[0]

    # Convert frame to input data
    squared_img = cv2.resize(frame, dsize=(256, 256), interpolation=cv2.INTER_AREA) / 255
    input_img = np.expand_dims(squared_img, axis=0)
    predicted_label = model.predict(input_img)[0]
    final_img = cv2.resize(predicted_label, dsize=(im_width, im_height), interpolation=cv2.INTER_AREA)
    backtorgb = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)
    backtorgb = (backtorgb * 255).astype(np.uint8)

    # Mix original image and predicted segmentation mask
    mixed_img = cv2.add(backtorgb, frame)

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
        color=box_to_color_map(boxes, scores, final_img),
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.75);

    cv2.imshow('mixed', mixed_img)
    cv2.waitKey(1)
cv2.destroyAllWindows()
