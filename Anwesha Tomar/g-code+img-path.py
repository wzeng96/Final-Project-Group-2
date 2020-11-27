# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import cv2
import random
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# references https://www.kaggle.com/rmishra258/counting-crowd-with-cnn-social-distancing-project
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
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=0.2)
x_train, x_test = x_train/255, x_test/255

# %% -------------------------------------- Data Prep ------------------------------------------------------------------

print("x_train.shape:\n", x_train.shape, "\nx_test.shape:\n", x_test.shape)
print("y_train.shape:\n", y_train.shape, "\ny_test.shape:\n", y_test.shape)

# %% -------------------------------------- Model Prep ------------------------------------------------------------------

# create CNN model
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(64, (3, 3), input_shape=(50, 50, 3), activation=tf.keras.activations.relu),
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