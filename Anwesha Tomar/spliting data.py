# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import cv2
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import cohen_kappa_score, f1_score
import matplotlib.pyplot as plt

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_NEURONS = 300
N_EPOCHS = 10
BATCH_SIZE = 512
DROPOUT = 0.2

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
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
x_train, x_test = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
x_train, x_test = x_train/255, x_test/255
print(np.shape(x_train))
#split data 5 ways:
print(len(x_train))
split=len(x)
s1=int(len(x_train)*0.2)
s2=int(len(x_train)*0.4)
s3=int(len(x_train)*0.6)
s4=int(len(x_train)*0.8)
print(s1,s2,s3,s4)
x1=x_train[:s1]
x2=x_train[s1:s2]
x3=x_train[s2:s3]
x4=x_train[s3:s4]
x5=x_train[s4:]
print(len(x1),len(x2),len(x3),len(x4),len(x5))
