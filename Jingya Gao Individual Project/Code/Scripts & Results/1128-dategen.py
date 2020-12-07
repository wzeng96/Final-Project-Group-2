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
from keras_preprocessing.image import ImageDataGenerator

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

# create CNN model (this is a test model, still need to change)
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

model.compile(loss=tf.keras.losses.Huber(),
              optimizer=tf.keras.optimizers.SGD(lr=1e-4), metrics=['mae'])



# add ImageDataGenerator
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
    shear_range=0.5)

datagen.fit(x_train)


# %% -------------------------------------- Training Loop ----------------------------------------------------------

history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                    epochs=100, batch_size=128)

# plot mae
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])

plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('loss in every epoch with ImageDataGenerator')
plt.show()