import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import collections
import tensorflow as tf
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split

plt.matplotlib.use('TkAgg')
digits = datasets.load_digits()
images = digits.images
features = []
for img in images:
    thresh = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(np.uint8(thresh), 0, 3)
    blank_img = np.zeros_like(img)
    M = 0
    for cnt in contours:
        M += cv2.HuMoments(cv2.moments(cnt)).flatten()
    # redrawn = cv2.drawContours(blank_img, contours, -1, (255, 255, 255), thickness=cv2.FILLED).flatten()
    # features.append(np.concatenate(redrawn, M))
    features.append(M)

data = np.array(features)

df_data = pd.DataFrame(data)
df_data = df_data.astype('float32')
print("--------------------")
print(len(features))
print(len(digits.target))
print("--------------------")
X_train, X_test, y_train, y_test = train_test_split(df_data, digits.target, test_size=0.2, shuffle=False)

model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(df_data.shape[1],)),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print('\nTest accuracy:', test_acc)
