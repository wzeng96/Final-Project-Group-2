# %% --------------------------------------- Imports -------------------------------------------------------------------

import random
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import load_model

# references https://www.kaggle.com/rmishra258/counting-crowd-with-cnn-social-distancing-project

# %% --------------------------------------- Set-Up --------------------------------------------------------------------

SEED = 64
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %% --------------------------------------- Load-Data -----------------------------------------------------------------

df_of_labels = pd.read_csv('labels.csv')
df_of_labels.columns = ['image_id', 'num_of_people']

# get the label_num of the people
labels = np.array(df_of_labels['num_of_people'])
print("labels.shape:\n", labels.shape)

# read the data of image, check the dimension
DATA_DIR = os.getcwd()+"/frames/"
# set image size to (128, 96)
x = []
for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".jpg"]:
    x.append(cv2.resize(cv2.imread(DATA_DIR + path), (128, 96)))
x = np.array(x)
print("x.shape:\n", x.shape)

# %% -------------------------------------- Data Prep ------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2)
x_train, x_test = x_train.astype('float') / 255, x_test.astype('float') / 255
print("x_train.shape:\n", x_train.shape, "\nx_test.shape:\n", x_test.shape)
print("y_train.shape:\n", y_train.shape, "\ny_test.shape:\n", y_test.shape)


# %% -------------------------------------- Test Model --------------------------------------------------------

model = load_model('keras-CNN-128-96.hdf5')
predict_y = model.predict(x_test)
# print(predict_y)
model_eval = model.evaluate(x_test, y_test)
print("model_eval:\n", model_eval)


plt.imshow(x_test[1])
plt.title("Actual Count: " + str(y_test[1]) +
          "\nPredicted Count: " + str(int(predict_y[1])))
plt.show()

plt.imshow(x_test[3])
plt.title("Actual Count: " + str(y_test[3]) +
          "\nPredicted Count: " + str(int(predict_y[3])))
plt.show()

plt.imshow(x_test[36])
plt.title("Actual Count: " + str(y_test[36]) +
          "\nPredicted Count: " + str(int(predict_y[36])))
plt.show()

plt.imshow(x_test[63])
plt.title("Actual Count: " + str(y_test[63]) +
          "\nPredicted Count: " + str(int(predict_y[63])))
plt.show()

pred = [int(i) for i in predict_y]
error = [y_test[i] - pred[i] for i in range(len(y_test))]
plt.hist(error, bins=15, rwidth=0.8)
plt.title('CNN Histogram of Residuals in Validation Set')
plt.show()

correct = []
for i in error:
    if i == 0:
        correct.append(i)

print('The model correctly predict {} images'.format(len(correct)))
