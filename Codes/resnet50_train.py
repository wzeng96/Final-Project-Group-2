import os
import cv2
import random
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import *
from keras.layers import *
from sklearn.metrics import mean_squared_error
from tensorflow.keras.applications import resnet50
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# references https://www.kaggle.com/rmishra258/counting-crowd-with-cnn-social-distancing-project
# %% -------------------------------------- Load Data ------------------------------------------------------------------
DATA_DIR = os.getcwd() + "/frames/"
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

batch = next(train_generator)

base_model = resnet50.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 96, 3),
    pooling='avg'
)

# Change the top (the last parts) of the network.
x = base_model.output
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='linear')(x)
model = Model(inputs=base_model.input, outputs=predictions)
k = -7
for layer in model.layers[:k]:
    layer.trainable = False
print('Trainable:')
for layer in model.layers[k:]:
    print(layer.name)
    layer.trainable = True


optimizer = Adam(learning_rate=3e-4)

model.compile(
    optimizer=optimizer,
    loss="mean_squared_error",
    metrics=['mean_absolute_error']
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_mean_absolute_error',
    patience=3,
    verbose=1,
    factor=0.2,
    min_lr=0.000001
)

model_check = ModelCheckpoint('resnet50_cc.hdf5', monitor='val_loss', verbose=2, save_best_only=True)

# Fit the model
history = model.fit(
    train_generator,
    epochs=90,
    validation_data=valid_generator,
    verbose=2,
    callbacks=[learning_rate_reduction, model_check]
)
print('\nTraining Done')

# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(history.history['loss'], color='b', label="Training loss")
ax.plot(history.history['val_loss'], color='r', label="Validation loss", axes=ax)
ax.set_ylim(top=np.max(history.history['val_loss']) * 1.2, bottom=0)
legend = ax.legend(loc='best', shadow=True)
ax.set_title('Training Loss vs Validation Loss')

plt.show()

# Predict on entire validation set, to be able to review the predictions manually
valid_generator.reset()
all_labels = []
all_pred = []
for i in range(len(valid_generator)):
    x = next(valid_generator)
    pred_i = model.predict(x[0])[:, 0]
    labels_i = x[1]
    all_labels.append(labels_i)
    all_pred.append(pred_i)

test_labels = np.concatenate(all_labels)
test_pred = np.concatenate(all_pred)
df_predictions = pd.DataFrame({'True values': test_labels, 'Predicted values': test_pred})
ax = df_predictions.plot.scatter('True values', 'Predicted values', alpha=0.5, s=14, figsize=(9, 9))
# ax.plot(df_predictions['True values'], df_predictions['Predicted values'], color='r')
ax.grid(axis='both')
ax.set_title('Testing set')

plt.show()

# ----------- Evaluation ---------------
mse = mean_squared_error(*df_predictions.T.values)
pearson_r = scipy.stats.pearsonr(*df_predictions.T.values)[0]

#print(f'MSE on testing: {mse:.1f}\nPearson r between True Values and Predicted Values: {pearson_r:.1f}')
print("MSE Train Set Loss value", history.history['loss'][-1])
print("MSE Validation Set Loss value", history.history['val_loss'][-1])
print("MAE Train Set Loss value",history.history['mean_absolute_error'][-1])
print("MAE Validation Set Loss value",history.history['val_mean_absolute_error'][-1])

