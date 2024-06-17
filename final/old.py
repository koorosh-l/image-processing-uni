import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import collections
import tensorflow as tf
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn import datasets
import gc

# variables for configuration
#rec_padding = 40
data_set = "opt"
plt.matplotlib.use('TkAgg')
os.environ["IMG"] = '/home/quasikote/proj/img/project1/'

# os.environ["IMG"] = '<path-repo>/img/project1/'

prefix = os.getenv("IMG")
# utils
def list_files_in_directory(directory):
    directory = directory
    files = os.listdir(directory)
    files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    return files
def list_map(proc, items, cc_pair = [lambda a: False, lambda a: a]):
    # for i in items:
    #     res = proc(i)
    #     if(cc_pair[0](res)):
    #         cc_pair[1](res)
    #     i = res
    # return items
 return list(map(proc, items))
#raw data accusition
def raw_fruit_data():
    test_path    = "fruits/test/"
    train_path   = "fruits/train/"
    train_names  = list_map(lambda str: prefix + train_path + str,
                            list_files_in_directory(prefix + train_path))
    test_names  = list_map(lambda str: prefix + test_path + str,
                           list_files_in_directory(prefix + test_path))
    train_images = list_map(cv2.imread,
                            train_names)
    test_images  = list_map(cv2.imread,
                            test_names)
    train_names  = list_map(lambda str:
                            os.path.basename(str).split('_')[1].split('.')[0].lower()
                            ,train_names)
    test_names = list_map(lambda str:
                          os.path.basename(str).split('_')[1].split('.')[0].lower()
                          ,test_names)
    return train_images, train_names, test_images, test_names
def optical_chars():
    digits = datasets.load_digits()
    imgs = digits.images
    lbls = digits.target
    train_images, test_images, train_names, test_names = train_test_split(imgs, lbls, test_size=0.2, shuffle=False)
    return train_images, train_names, test_images, test_names
def get_data():
    if data_set == "fruits":
        return raw_fruit_data()
    else:
        return optical_chars()

#general oprations

def create_model(i_size, cl_no):
    model = models.Sequential([layers.Flatten(input_shape=(i_size,)),
                               layers.Dense(300, activation='relu'),
                               layers.Dense(cl_no,  activation='softmax')])
    return model

def prep_classes(lbls):
    return list(set(lbls))
def prep_lables(lbls, clz):
    return list(map(lambda a:
                    clz.index(a)
                    ,lbls))

def run_model(get_data):
    train_data, train_labels, test_data, test_labels = get_data()
    clz = len(prep_classes(test_labels))
    model = create_model(len(train_data[0]), clz)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.fit(train_data,train_labels, epochs=100)
    tl, tc = model.evaluate(test_data, test_labels, verbose=2)
    return tc

# midterm

def prep_datav1():
    train_images, train_labels, test_images, test_labels = get_data()
    clz = prep_classes(test_labels)
    if data_set == 'fruits' :
        mmnts = (lambda i:
                 np.array(list(cv2.moments(cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)).values())))
    else:
        mmnts = (lambda i:
                 np.array(list(cv2.moments(i).values())))
    train_images = np.array(list_map(mmnts, train_images))
    test_images = np.array(list_map(mmnts, test_images))
    return train_images, prep_lables(train_labels, clz), test_images, prep_lables(test_labels, clz)

# makeup
def pad(arr, size):
    p = [0,0,0,0]
    l = len(arr)
    if l >= size:
        return arr[0:size]
    while l != size:
        arr.append(p)
        l = l + 1
    return arr

def find_bounding_rectangles(image):
    if data_set == "fruits":
        gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image.astype(np.uint8)
    _, threshold_image = cv2.threshold(gray_image, 0, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(gray_image,
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    bounding_rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_rectangles.append([x, y, w, h])
    return bounding_rectangles

#main
def prep_datav2():
    train_images, train_labels, test_images, test_labels = get_data()
    clz = prep_classes(test_labels)
    data = list_map(find_bounding_rectangles, train_images)
    m = max(list_map(len, data))
    data = np.array(list_map(lambda rects: np.array(pad(rects, m)).flatten(), data))
    # train_images = np.array(list_map(lambda d:
    #                                  np.array(list_map(lambda i: list(i),pad(find_bounding_rectangles(d)))).flatten()
    #                                  ,train_images))
    test_data = list_map(find_bounding_rectangles, test_images)
    test_data = np.array(list_map(lambda rects: np.array(pad(rects, m)).flatten(), test_data))
    # test_images = np.array(list_map(lambda d:
    #                                 np.array(list_map(lambda i: list(i),pad(find_bounding_rectangles(d)))).flatten()
    #                                 ,test_images))
    return data, prep_lables(train_labels, clz), test_data, prep_lables(test_labels, clz)
#testing and training

tc_1 = tc_2 = "NaN"
tc_1 = run_model(prep_datav1)
tc_2 = run_model(prep_datav2)
# train_images, train_labels, test_images, test_labels = prep_datav1()
# tc_1 = run_model(train_images, train_labels, test_images, test_labels,input_size)
print("\nmidterm acc is: ", tc_1)
print("\nmakup acc is: ", tc_2)

# def find_and_draw_cnt(img):
#     i = np.copy(img)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     _, t = cv2.threshold(img, 20, 255, cv2.THRESH_TRIANGLE)
#     cnt, _ = cv2.findContours(t, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(i, cnt, -1, (0,255,0), 2)
#     plt.imshow(i)
#     plt.show()
