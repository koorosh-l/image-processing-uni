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
    thresh = cv2.threshold(img, 4, 255, cv2.THRESH_TRUNC)[1]
    contours, hierarchy = cv2.findContours(np.uint8(thresh), 0, 0)
    for cnt in contours:
        M = cv2.moments(cnt)
        area = M['m00']
        area += 1
        features.append([area, M['m10']/area, M['m01']/area, M['m20']/area, M['m02']/area, M['m11']/(area**2)])

data = np.array(features)

df_data = pd.DataFrame(data)
df_data = df_data.astype('float32')
print("--------------------")
print(len(features))
print(len(digits.target))
print("--------------------")
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
