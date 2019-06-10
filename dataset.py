import os
import cv2
import numpy as np
from random import sample, shuffle
from keras.utils import np_utils

# Global variables
INPUT_SIZE = 64
DATASET_PATH = './dataset'
NUM = 7
LABEL = ['Leonard', 'Sheldon', 'Penny', 'Howard', 'Raj', 'Amy', 'Bernadette']

def load_grey_image(d, n):
    impath = os.path.join(DATASET_PATH, d, n)
    return np.expand_dims(cv2.imread(impath, 0), axis=2)

def load_dataset(id):
    x_train = []
    y_train = []    
    x_test = []
    y_test = []

    dirs = next(os.walk(DATASET_PATH))[1]

    for d in dirs:
        files = os.listdir(os.path.join(DATASET_PATH, d))
        shuffle(files)

        if d == LABEL[id]:
            for i in range(3):
                x_test.append(load_grey_image(d, files[i]))
                y_test.append(1)
            for i in range(3, len(files)):
                x_train.append(load_grey_image(d, files[i]))
                y_train.append(1)
        else:
            for i in range(3):
                x_train.append(load_grey_image(d, files[i]))
                y_train.append(0)
            for i in range(3,5):
                x_test.append(load_grey_image(d, files[i]))
                y_test.append(0)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = np_utils.to_categorical(y_train, 2)
    y_test = np_utils.to_categorical(y_test, 2)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    return x_train, y_train, x_test, y_test 