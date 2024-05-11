import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import collections
import tensorflow as tf
import os
from tensorflow.keras import layers, models
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
prefix = os.getenv("IMG")
#https://www.kaggle.com/datasets/afsananadia/fruits-images-dataset-object-detection

def list_files_in_directory(directory):
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
    model = models.Sequential([layers.Input=(16,),
                               layers.Dense(100, activation='relu'),
                               layers.Dense(10,  activation='relu')])
    # decide the last layer

#data prepreation
train_names = list_files_in_directory(prefix + "./fruits/train")
test_names  = list_files_in_directory(prefix + "./fruits/test")

data = np.array(map((lambda d:
                     np.array(pad(find_bounding_rectangles(d)))),
                    data))
#np.array(pad([(0, 0, 2827, 2827)])).flatten()
model = create_model()
model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics='accuracy')
