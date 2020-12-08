import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import *
from tensorflow.keras.applications import resnet50
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# references https://www.kaggle.com/rmishra258/counting-crowd-with-cnn-social-distancing-project
# %% -------------------------------------- Load Data ------------------------------------------------------------------
DATA_DIR = os.getcwd() + "/frames/"
# RESIZE_TO = 50
x, y = [], []
label = pd.read_csv("labels.csv")
label['image_name'] = label['id'].map('seq_{:06d}.jpg'.format)
lb = label['id']
count = label['count']

for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".jpg"]:
    x.append(cv2.resize(cv2.imread(DATA_DIR + path), (128, 96)))
    for j in range(len(lb)):
        if lb[j] == int(path[4:-4]):
            y.append(count[j])
x, y = np.array(x), np.array(y)
# print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=0.2)
x_train, x_test = x_train / 255, x_test / 255

resize = 64
batch = 256

# ImageDataGenerator - with defined augmentaions
datagen = ImageDataGenerator(
    rescale=1. / 255,
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2,

    preprocessing_function=resnet50.preprocess_input
)

flow_params = dict(
    dataframe=label,
    directory=DATA_DIR,
    x_col="image_name",
    y_col="count",
    weight_col=None,
    target_size=(128, 96),
    color_mode='rgb',
    class_mode="raw",
    batch_size=batch,
    shuffle=True,
    seed=0
)

# The dataset is split to training and validation sets
train_generator = datagen.flow_from_dataframe(
    subset='training',
    **flow_params
)
valid_generator = datagen.flow_from_dataframe(
    subset='validation',
    **flow_params
)

model = load_model('resnet50_cc.hdf5')
valid_generator.reset()
all_labels = []
all_pred = []
for i in range(len(valid_generator)):
    x = next(valid_generator)
    pred_i = model.predict(x[0])[:, 0]
    labels_i = x[1]
    all_labels.append(labels_i)
    all_pred.append(pred_i)

# print('pred: ', all_pred)
# print('label:', all_labels)

test_labels = np.concatenate(all_labels)
test_pred = np.concatenate(all_pred)

print(test_pred)
print(test_labels)

# ynew = model.predict(x_test)
plt.imshow(x_test[1])
plt.title("Actual Count: " + str(y_test[1]) +
          "\nPredicted Count: " + str(int(test_pred[1])))
plt.show()

plt.imshow(x_test[3])
plt.title("Actual Count: " + str(y_test[3]) +
          "\nPredicted Count: " + str(int(test_pred[3])))
plt.show()

plt.imshow(x_test[36])
plt.title("Actual Count: " + str(y_test[36]) +
          "\nPredicted Count: " + str(int(test_pred[36])))
plt.show()

plt.imshow(x_test[63])
plt.title("Actual Count: " + str(y_test[63]) +
          "\nPredicted Count: " + str(int(test_pred[63])))
plt.show()
plt.imshow(x_test[2])
plt.title('Actual Count: ' + str(test_labels[2]) +
          '\nPredicted Count: ' + str(int(test_pred[2])))
plt.show()

test_pred_int = list(map(int, test_pred))
test_labelss = list(test_labels)
# print(test_labelss)
# print(test_pred_int)

error = [(test_labelss[i] - test_pred_int[i]) for i in range(len(test_labelss))]
print(np.mean(error))
plt.hist(error, bins=15, rwidth=0.8)
plt.title('ResNet 50 Histogram of Residuals in Validation Set')
plt.show()

correct = []
for w in error:
    if w == 0:
        correct.append(w)

print('The model correctly predict {} images'.format(len(correct)))
