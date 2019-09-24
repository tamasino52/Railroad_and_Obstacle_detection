import cv2 as cv
import numpy as np
import os
import sys


# Defining seeds and colors of each seed
n_seeds = 2
colors = [(0, 0, 0), (255, 255, 255)]
mouse_pressed = False
seeds_updated = False
current_seed = 1
seeds=[]

# File loading
IMAGE_PATH = '.\image'
MASK_PATH = '.\mask'
FILE_LIST = os.listdir(IMAGE_PATH)
FILE_LIST.sort()






# Declare variables
file_count = 0
frame = cv.imread(os.path.join(IMAGE_PATH, FILE_LIST[file_count]))
show_img = np.copy(frame)
print('FILE OPENED :: ', os.path.join(IMAGE_PATH, FILE_LIST[file_count]))
segmentation = np.full(frame.shape, 0, np.uint8)
seeds = np.full(frame.shape, 0, np.uint8)
last_segmentation = cv.imread(os.path.join(MASK_PATH, FILE_LIST[file_count]))
if last_segmentation is None:
    print('MASK NOT FOUND')
    last_segmentation = np.full(frame.shape, 0, np.uint8)
last_segmentation = cv.addWeighted(show_img, 0.5, last_segmentation, 0.5, 0)


# Declare mouse event listener
def mouse_callback(event, x, y, flags, param):
    global mouse_pressed, seeds_updated, current_seed
    if event == cv.EVENT_LBUTTONDOWN:
        current_seed = 1
        mouse_pressed = True
        cv.circle(seeds, (x, y), 2, (current_seed), cv.FILLED)
        cv.circle(show_img, (x, y), 2, colors[current_seed - 1], cv.FILLED)
        seeds_updated = True
    elif event == cv.EVENT_RBUTTONDOWN:
        current_seed = 2
        mouse_pressed = True
        cv.circle(seeds, (x, y), 2, (current_seed), cv.FILLED)
        cv.circle(show_img, (x, y), 2, colors[current_seed - 1], cv.FILLED)
        seeds_updated = True
    elif event == cv.EVENT_MOUSEMOVE:
        if mouse_pressed:
            cv.circle(seeds, (x, y), 2, (current_seed), cv.FILLED)
            cv.circle(show_img, (x, y), 2, colors[current_seed - 1], cv.FILLED)
            seeds_updated = True
    elif event == cv.EVENT_LBUTTONUP:
        mouse_pressed = False
    elif event == cv.EVENT_RBUTTONUP:
        mouse_pressed = False


# Set event listener
cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.setMouseCallback('image', mouse_callback)


# Loop
while True:
    cv.imshow('segmentation', segmentation)
    cv.imshow('image', show_img)
    cv.imshow('last_segmentation', last_segmentation)
    k = cv.waitKey(10)

    if k == 27:
        break
    elif k == ord('c') or k == ord('C'):  # Cancel
        show_img = np.copy(frame)
        seeds = np.full(frame.shape[0:2], 0, np.uint8)
        segmentation = np.full(frame.shape, 0, np.uint8)
    elif k > 0 and chr(k).isdigit():  # Seed selection
        n = int(chr(k))
        if 1 <= n <= n_seeds and not mouse_pressed:
            current_seed = n
    elif k == ord('n') or k == ord('N'):  # New file
        file_count += 1
        if len(FILE_LIST) < file_count:
            print('FINISHED')
            sys.exit(1)
        print('FILE OPENED :: ', os.path.join(IMAGE_PATH, FILE_LIST[file_count]))
        frame = cv.imread(os.path.join(IMAGE_PATH, FILE_LIST[file_count]), cv.IMREAD_COLOR)
        show_img = np.copy(frame)
        seeds = np.full(frame.shape[0:2], 0, np.uint8)
        segmentation = np.full(frame.shape, 0, np.uint8)
        last_segmentation = cv.imread(os.path.join(MASK_PATH, FILE_LIST[file_count]))
        if last_segmentation is None:
            last_segmentation = np.full(frame.shape, 0, np.uint8)
        last_segmentation = cv.addWeighted(show_img, 0.5, last_segmentation, 0.5, 0)
    elif k == ord('s') or k == ord('S'):  # Save mask
        print('MASK FILE SAVED :: ' + os.path.join(MASK_PATH, FILE_LIST[file_count]))
        cv.imwrite(os.path.join(MASK_PATH, FILE_LIST[file_count]), segmentation)
    elif k == ord('l') or k == ord('L'):  # Last image
        file_count -= 1
        print('OLD FILE OPENED :: ' + os.path.join(IMAGE_PATH, FILE_LIST[file_count]))
        frame = cv.imread(os.path.join(IMAGE_PATH, FILE_LIST[file_count]), cv.IMREAD_COLOR)
        show_img = np.copy(frame)
        seeds = np.full(frame.shape[0:2], 0, np.uint8)
        segmentation = np.full(frame.shape, 0, np.uint8)
        last_segmentation = cv.imread(os.path.join(MASK_PATH, FILE_LIST[file_count]))
        if last_segmentation is None:
            last_segmentation = np.full(frame.shape, 0, np.uint8)

    if seeds_updated and not mouse_pressed:
        cv.watershed(show_img, seeds)
        segmentation = np.full(frame.shape, 0, np.uint8)

        #coloring segmentation
        for m in range(n_seeds):
            segmentation[seeds == (m + 1)] = colors[m]
        seeds_updated = False
print('All Task finished')
sys.exit(1)
