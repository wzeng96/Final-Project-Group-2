# %% --------------------------------------- Imports -------------------------------------------------------------------

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
print("df_of_labels.head：\n", df_of_labels.head())

# read the data of image, check the dimension, check the size(480, 640)
image_data = np.load('/home/ubuntu/test0/final-project/dataset/images.npy')
print("img.shape:\n", image_data.shape)

# get the label_num of the people
labels = np.array(df_of_labels['num_of_people'])

# %% -------------------------------------- Data Prep ------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.3)
x_train, x_test = x_train.astype('float') / 255, x_test.astype('float') / 255
print("x_train.shape:\n", x_train.shape, "\nx_test.shape:\n", x_test.shape)
print("y_train.shape:\n", y_train.shape, "\ny_test.shape:\n", y_test.shape)

# %% -------------------------------------- Model Prep ------------------------------------------------------------------

# create CNN model
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(64, (3, 3), input_shape=(480, 640, 3), activation=tf.keras.activations.relu),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.relu),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)

])

model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(), metrics=['mae'])
print("model.summary():\n", model.summary())

# %% -------------------------------------- Train Model ------------------------------------------------------------------

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
plt.plot(xs, ys, 'bo')  # 'bo-' means blue color, round points, (solid lines)
for x, y in zip(xs, ys):
    label = "{:.3f}".format(y)
    plt.annotate(label,  # this is the text
                 (x, y),  # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(0, 10),  # distance from text to points (x,y)
                 ha='center')  # horizontal alignment can be left, right or center

plt.axis([np.min(xs_orginal), np.max(xs_orginal), np.min(ys_original), np.max(ys_original)])
plt.title("Loss with Different Learning Rate")
plt.xlabel("learning Rate")
plt.ylabel("Loss")

plt.show()

# save the result into dataframe
loss_LR_df = pd.DataFrame(list(zip(xs_orginal, ys_original)),
                          columns=['LR', 'Loss'])
print("loss_LR_df:\n", loss_LR_df)
loss_LR_df.to_csv("loss_LR_df.csv", index=False)
