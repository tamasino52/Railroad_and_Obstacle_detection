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
import argparse
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
import Capture_module

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
def build_railtracking_model(MODEL_PATH, INPUT_SHAPE):
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
    return model

if __name__ == '__main__':
    # Define args property
    parser = argparse.ArgumentParser()

    parser.add_argument('--weight', default='./railway_detection/weights.hdf5', help='Input weight path')
    parser.add_argument('--model', default='./railway_detection/model.h5', help='Input model path')
    parser.add_argument('--test', default=0, help='Input training path')

    args = parser.parse_args()

    # ## Set properties
    INPUT_SHAPE = (256, 256, 3)
    WEIGHT_PATH = args.weight
    MODEL_PATH = args.model

    # Rail tracking model build
    model = build_railtracking_model(MODEL_PATH, INPUT_SHAPE)
    cam = Capture_module.VideoCamera(args.test)

    while True:
        # Frame read
        frame = cam.get_frame()
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

        # show the frame
        cv2.imshow('Display', mixed_img)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
