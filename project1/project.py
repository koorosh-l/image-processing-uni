import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import collections
import tensorflow as tf
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
digits = datasets.load_digits()
images = digits.images
features = []
for img in images:
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img, 4, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(np.uint8(thresh), 0, 0)
    for cnt in contours:
        M = cv2.moments(cnt)
        area = M['m00']
        if area > 0:
            features.append([area, M['m10']/area, M['m01']/area, M['m20']/area, M['m02']/area, M['m11']/(area**2)])
# Convert features to numpy array
data = np.array(features)

# compelte here

df_data = pd.DataFrame(data)
df_data = df_data.astype('float32')
print("aaaaaaaaaaaaaaaaaaaa")

X_train, X_test, y_train, y_test = train_test_split(df_data, digits.target, test_size=0.2, shuffle=False)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(df_data.shape[1],)),  # Adjusted input shape
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # Softmax activation for multiclass classification
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print('\nTest accuracy:', test_acc)
# # totalItems = len(images)
# # for i in range(totalItems):
# #     images[i] = cv2.threshold(images[i],10, 255, cv2.THRESH_BINARY)[1]
# # flattened_images = digits.images.reshape(totalItems, -1)
# features = []
# for img in images:
#     # Convert image to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # Thresholding
#     ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     # Find contours
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # Moments calculation
#     for cnt in contours:
#         M = cv2.moments(cnt)
#         area = M['m00']
#         if area > 0:
#             # Feature extraction
#             features.append([area, M['m10']/area, M['m01']/area, M['m20']/area, M['m02']/area, M['m11']/(area**2)])

# # Convert features to numpy array
# data = np.array(features)

# df_data = pd.DataFrame(flattened_images)
# df_data = df_data.astype('float32')

# X_train, X_test, y_train, y_test = train_test_split(df_data, digits.target, test_size=0.2, shuffle=False)

# model = tf.keras.Sequential([tf.keras.layers.Dense(128, activation='relu', input_shape=(64,)),
#                              tf.keras.layers.Dense(10, activation='softmax')])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(X_train, y_train, epochs=100)
# test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
# print("Test Accuracy: ", test_acc)
# print("Test Loss: ", test_loss)
