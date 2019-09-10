# Run-Length Encode and Decode

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time

# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    print(runs[1::2])
    print(runs[2::2])

    runs[1::2] -= runs[2::2]
    return runs

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


import numpy as np
from PIL import Image
import os, imageio
from scipy.misc import imread
input_path = '.'
masks = [f for f in os.listdir(input_path) if f.endswith('.jpg')]

encodings = []
N = 100     # process first N masks
for i,m in enumerate(masks[:N]):
    if i % 10 == 0: print('{}/{}'.format(i, len(masks)))
    img = imageio.imread(m)
    print(rle_encode(img))

#check output
conv = lambda l: ' '.join(map(str, l)) # list -> string
subject, img = 1, 1
print('\n{},{},{}'.format(subject, img, conv(encodings[0])))
