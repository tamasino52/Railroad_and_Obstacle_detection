import cv2 as cv
import numpy as np
import os
import sys

VIDEO_PATH = 'complex7.mp4'
CUT_VIDEO = True
FRAME_INTERVAL = 2
RESIZE_RATIO = 1
NAME_COUNT = 3744
SAVE_PATH = './image'

xmin, ymin, xmax, ymax = 0, 0, 0, 0
click_count = 0
frame_count = 0
cap = cv.VideoCapture(VIDEO_PATH)

print('------Video2image Setting------')
print('Video loaded :: ', VIDEO_PATH)
print('CUT_VIDEO :: ', CUT_VIDEO)
if FRAME_INTERVAL is 0:
    print('FRAME_INTERVAL must larger than 0')
    sys.exit(1)
else:
    print('FRAME_INTERVAL :: ', FRAME_INTERVAL)
print('RESIZE_RATIO :: ', RESIZE_RATIO)
print('NAME_COUNT :: ', NAME_COUNT, '.jpg')
print('SAVE_PATH :: ', SAVE_PATH)
print('-------------------------------\n')

def mouse_callback(event, x, y, flags, param):
    global click_count, xmin, xmax, ymin, ymax
    if event is cv.EVENT_LBUTTONDOWN:
        if click_count is 0:
            xmin = x
            ymin = y
            print('xmin, ymin :: ', xmin, ', ', ymin)
            click_count += 1
        elif click_count is 1:
            xmax = x
            ymax = y
            print('xmax, ymax :: ', xmax, ', ', ymax)
            if xmin > xmax or ymin > ymax:
                print('Wrong selection')
                sys.exit(1)
            cv.destroyWindow('Cut Preview')

while cap.isOpened():
    # Frame read
    ret, frame = cap.read()
    frame_count += 1
    if frame_count % FRAME_INTERVAL is not 0:
        continue
    if not ret:
        print('Finished')
        break
    if RESIZE_RATIO is not 1:  # Resize
        frame = cv2.resize(frame,
                           dsize=(int(frame.shape[1] * RESIZE_RATIO), int(frame.shape[0] * RESIZE_RATIO)),
                           interpolation=cv2.INTER_AREA)
    if CUT_VIDEO is True:
        if click_count is 0:
            print('------CUT_VIDEO :: TRUE------')
            # setting event listener
            cv.imshow('Cut Preview', frame)
            cv.setMouseCallback('Cut Preview', mouse_callback)
            cv.waitKey(0)
            print('CUT_SETTING :: ', xmin, ', ', ymin, ', ', xmax, ', ', ymax)
            print('-------------------------------\n')
        if click_count is not 0:
            frame = frame[ymin:ymax, xmin:xmax]
    cv.imwrite(SAVE_PATH + '/' + str(NAME_COUNT) + '.jpg', frame)
    print('SAVE :: ', SAVE_PATH + '/' + str(NAME_COUNT) + '.jpg')
    NAME_COUNT += 1





