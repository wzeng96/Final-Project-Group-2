# %% --------------------------------------- Imports -------------------------------------------------------------------

import os
import cv2
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf

# references https://www.kaggle.com/rmishra258/counting-crowd-with-cnn-social-distancing-project
# %% -------------------------------------- Load Data ------------------------------------------------------------------

# read the labels.cvs to get the labels and set columns name: 'image_id' , 'num_of_people'
df_of_labels = pd.read_csv('/home/ubuntu/test0/final-project/dataset/labels.csv')
df_of_labels.columns = ['image_id', 'num_of_people']

# get the label_num of the people
labels = np.array(df_of_labels['num_of_people'])
print("labels.shape:\n", labels.shape)

# read the data of image, check the dimension
DATA_DIR = "/home/ubuntu/test0/final-project/dataset/all_Images/"
RESIZE_TO = 50

x = []
for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".jpg"]:
    x.append(cv2.resize(cv2.imread(DATA_DIR + path), (RESIZE_TO, RESIZE_TO)))
x = np.array(x)
print("x.shape:\n", x.shape)

# %% -------------------------------------- Data Prep ------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.3)
x_train, x_test = x_train.astype('float') / 255, x_test.astype('float') / 255
print("x_train.shape:\n", x_train.shape, "\nx_test.shape:\n", x_test.shape)
print("y_train.shape:\n", y_train.shape, "\ny_test.shape:\n", y_test.shape)

# %% -------------------------------------- Model Prep ------------------------------------------------------------------

model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(16, (5, 5), input_shape=(50, 50, 3), activation=tf.keras.activations.relu),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.relu),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(400, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)

])

model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam(), metrics=['mae'])

# %% -------------------------- Train Model with Different LR-----------------------------------------------------------

# add a learning rate monitor to get the plot of loss with different learning rate
LR_monitor = tf.keras.callbacks.LearningRateScheduler(
    lambda epochs: 1e-6 * 10 ** (epochs / 20))

history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                    epochs=75, batch_size=32, callbacks=[LR_monitor])

# plot loss with different learning rate
xs_orginal = history.history['lr']
ys_original = history.history['loss']

xs = np.array(xs_orginal)
ys = np.array(ys_original)
xs = xs[::10]
ys = ys[::10]

plt.semilogx(xs_orginal, ys_original)
plt.plot(xs, ys, 'bo')  # 'bo'/'go'/'yo' means blue/green/yellow color, round points, (solid lines)
plt.title("Loss vs LR (batch_size=32)")
plt.xlabel("learning Rate")
plt.ylabel("Loss")

plt.show()

