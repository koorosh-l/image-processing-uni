import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import collections
import tensorflow as tf
import os
from tensorflow.keras import layers, models
prefix = os.getenv("IMG")
#fruits data set can be found here
#https://www.kaggle.com/datasets/afsananadia/fruits-images-dataset-object-detection
def list_map(proc, items):
    return list(map(proc, items))

def list_files_in_directory(directory):
    directory = directory
    files = os.listdir(directory)
    files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    return files

def find_bounding_rectangles(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_image,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    bounding_rectangles = []
    print(contours)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_rectangles.append((x, y, w, h))
    return bounding_rectangles

def pad(arr):
    p = (0,0,0,0)
    l = len(arr)
    if l >= 4:
        return arr[0:4]
    while l != 4:
        arr.append(p)
        l = l + 1
    return arr

def create_model():
    model = models.Sequential([layers.Input(input_shape=(16,)),
                               layers.Dense(100, activation='relu'),
                               layers.Dense(10,  activation='relu')])
    # decide the last layer

#data prepreation
test_path = "fruits/test/"
train_path = "fruits/train/"
train_names = list_map(lambda str: prefix + train_path + str,
                       list_files_in_directory(prefix + train_path))
test_names  = list_map(lambda str: prefix + test_path + str,
                       list_files_in_directory(prefix + test_path))
train_images = list_map(cv2.imread,
                        train_names)
test_images  = list_map(cv2.imread,
                        test_names)
# train_labels =
# test_labels  =
data = np.array(list_map(pad(find_bounding_rectangles(d)), train_images))
#np.array(pad([(0, 0, 2827, 2827)])).flatten()
model = create_model()
model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics='accuracy')
