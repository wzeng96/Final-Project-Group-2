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
# %% -------------------------------------- Training Prep ----------------------------------------------------------
# fit model on dataset
def fit_model(trainX, trainy):
	# define model
	model = Sequential()
	model.add(Dense(25, input_dim=7500, activation='relu'))
	model.add(Dense(1, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit model
	model.fit(trainX, trainy, epochs=500, verbose=0)
	return model

# %% -------------------------------------- Training Loop ----------------------------------------------------------
# create directory for models
os.makedirs('models',exist_ok=True)
# fit and save models
n_members = 5
for i in range(n_members):
	# fit model
	model = fit_model(x_train, y_train)
	# save model
	filename = 'models/model_' + str(i + 1) + '.h5'
	model.save(filename)
	print('>Saved %s' % filename)

# load models from file
def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = 'models/model_' + str(i + 1) + '.h5'
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models
# load all models
n_members = 5
members = load_all_models(n_members)
print('Loaded %d models' % len(members))
# evaluate standalone models on test dataset
for model in members:
	_, acc = model.evaluate(x_test, y_test, verbose=0)
	print('Model Accuracy: %.3f' % acc)