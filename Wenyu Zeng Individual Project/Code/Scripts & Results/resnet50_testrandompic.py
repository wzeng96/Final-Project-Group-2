import os
import cv2
import random
import scipy
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import *


model = load_model('resnet50_cc.hdf5')

# tester.jpg
dir = '/home/ubuntu/Deep-Learning/Final_Project/tester.jpg'
tester = cv2.resize(cv2.imread(dir), (128, 96))
plt.imshow(tester)
plt.show()

print(tester.shape)
tester = np.expand_dims(tester, axis=0)
print(tester.shape)
tester_pred = model.predict(tester)
print(tester_pred)

mall = cv2.resize(cv2.imread('/home/ubuntu/Deep-Learning/Final_Project/random_mall.jpg'), (128, 96))
plt.imshow(mall)
plt.show()
mall = np.expand_dims(mall, axis=0)

print(model.predict(mall))