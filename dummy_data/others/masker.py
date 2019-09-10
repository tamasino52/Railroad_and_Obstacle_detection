import cv2
import numpy as np
import os
import glob
#defining seeds and colors of each seed
n_seeds = 2
colors = [(0,0,0),(255,255,255)]
mouse_pressed = False
current_seed = 1
seeds_updated = False

count = 444 #이미지시작번호
path_name = 'normal_left' #원본이미지 불러오기 경로
img_save_path = 'train' #이미지 저장경로
mask_save_path = 'train_mask' #마스크 저장경로
file_names = []
for i in range(count, count+1000):
    file_names.append(str(i)+'.jpg') #저장 파일명 지정
load_file_names = []
load_file_names = glob.glob(path_name + '/*.jpg')

#making copies that saving seeds into
print(load_file_names[0]) #샘플 불러오기
frame = cv2.imread(load_file_names[0])
show_img = np.copy(frame)
seeds = np.full(frame.shape[0:2], 0, np.int32)
segmentation = np.full(frame.shape, 0, np.uint8)
last_segmentation = cv2.imread(mask_save_path + '/' + file_names[0])
try:
    last_segmentation = cv2.addWeighted(show_img, 0.5, last_segmentation, 0.5, 0)
except:
    last_segmentation = cv2.imread(load_file_names[0])


#마우스 이벤트 리스너
def mouse_callback(event,x,y,flags,param):
    global mouse_pressed, seeds_updated, current_seed
    if event == cv2.EVENT_LBUTTONDOWN:
        current_seed = 2
        mouse_pressed = True
        cv2.circle(seeds, (x, y), 2, (current_seed), cv2.FILLED)
        cv2.circle(show_img, (x, y), 2, colors[current_seed - 1], cv2.FILLED)
        seeds_updated = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        current_seed = 1
        mouse_pressed = True
        cv2.circle(seeds, (x, y), 2, (current_seed), cv2.FILLED)
        cv2.circle(show_img, (x, y), 2, colors[current_seed - 1], cv2.FILLED)
        seeds_updated = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            cv2.circle(seeds, (x, y), 2, (current_seed), cv2.FILLED)
            cv2.circle(show_img, (x, y), 2, colors[current_seed - 1], cv2.FILLED)
            seeds_updated = True
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
    elif event == cv2.EVENT_RBUTTONUP:
        mouse_pressed = False

# setting event listener
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

count = 0
while True:
    #이미지 창 3개 띄우기
    cv2.imshow('segmentation', segmentation)
    cv2.imshow('image', show_img)
    cv2.imshow('last_segmentation', last_segmentation)

    k = cv2.waitKey(1)

    if k == 27:
        break
    elif k == ord('c') or k == ord('C'): #c 누르면 다시그리기
        show_img = np.copy(frame)
        seeds = np.full(frame.shape[0:2], 0, np.int32)
        segmentation = np.full(frame.shape, 0, np.uint8)
    elif k == ord('n') or k == ord('N'): #n 누르면 다음 그림
        count += 1
        print('load new img :: ' + load_file_names[count]);
        frame = cv2.imread(load_file_names[count], cv2.IMREAD_COLOR)
        show_img = np.copy(frame)
        seeds = np.full(frame.shape[0:2], 0, np.int32)
        segmentation = np.full(frame.shape, 0, np.uint8)
        last_segmentation = cv2.imread(mask_save_path + '/' + file_names[count])
        try:
            last_segmentation = cv2.addWeighted(show_img, 0.5, last_segmentation, 0.5, 0)
        except:
            last_segmentation = cv2.imread(load_file_names[count])
    elif k == ord('s') or k == ord('S'): #s 누르면 저장하기
        print('save img :: ' + mask_save_path + '/' + file_names[count])
        cv2.imwrite(mask_save_path + '/' + file_names[count], segmentation)
        cv2.imwrite(img_save_path + '/' + file_names[count], frame)
    elif k == ord('l') or k == ord('L'):
        if count <= 0:
            continue
        count -= 1
        print('load old img :: ' + load_file_names[count]);
        frame = cv2.imread(load_file_names[count], cv2.IMREAD_COLOR)
        show_img = np.copy(frame)
        seeds = np.full(frame.shape[0:2], 0, np.int32)
        segmentation = np.full(frame.shape, 0, np.uint8)
        last_segmentation = cv2.imread(mask_save_path + '/' + file_names[count])
        try:
            last_segmentation = cv2.addWeighted(show_img, 0.5, last_segmentation, 0.5, 0)
        except:
            last_segmentation = cv2.imread(load_file_names[count])

    if seeds_updated and not mouse_pressed:
        seeds_copy = np.copy(seeds)
        cv2.watershed(frame, seeds_copy)
        segmentation = np.full(frame.shape, 0, np.uint8)

        #coloring segmentation
        for m in range(n_seeds):
            segmentation[seeds_copy == (m + 1)] = colors[m]
        seeds_updated = False
