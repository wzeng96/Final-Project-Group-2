# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

# %% -------------------------------------- Load Data ------------------------------------------------------------------
DATA_DIR=os.getcwd() +"/frames/"
RESIZE_TO=50
x, y = [], []
data=pd.read_csv("labels.csv")
label=data['id']
count=data['count']
for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".jpg"]:
    x.append(cv2.resize(cv2.imread(DATA_DIR + path), (RESIZE_TO, RESIZE_TO)))
    for j in range(len(label)):
        if (label[j]==int(path[4:-4])):
            y.append(count[j])
x, y = np.array(x), np.array(y)
labels = y
print("labels.shape:\n", labels.shape)
print("x.shape:\n", x.shape)

# %% -------------------------------------- Data Prep ------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2)
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
# %% -------------------------------------- Training Loop ----------------------------------------------------------
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=tf.keras.optimizers.SGD(lr=1e-4), metrics=["mae"])

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
history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                    epochs=100, batch_size=128)
# %% -------------------------------------- Saving the model ----------------------------------------------------------
model.save('keras_model.hdf5')
# %% -------------------------------------- Plot the MAE ----------------------------------------------------------
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'validation_loss'])

plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('loss in every epoch with ImageDataGenerator')
plt.show()
# %% -------------------------------------- Print Final Values ----------------------------------------------------------

print("Final loss value",history.history['loss'][-1])
print("Final validation loss", history.history['val_loss'][-1])
