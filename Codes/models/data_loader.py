# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# %% -------------------------------------- Load Data ------------------------------------------------------------------
if "frames" not in os.listdir():
    os.system("wget https://storage.googleapis.com/speechrecogdata/frames.zip")
    os.system("unzip frames.zip")
if "labels.csv" not in os.listdir():
    os.system("wget https://storage.googleapis.com/speechrecogdata/labels.csv")
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
# %% -------------------------------------- Explore Data Labels --------------------------
plt.hist(y)
plt.show()