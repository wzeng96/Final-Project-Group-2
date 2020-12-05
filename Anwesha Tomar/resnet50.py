import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import *
from keras.layers import *
from tensorflow.keras.applications import resnet50
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import tensorflow as tf
# %% -------------------------------------- Load Data ------------------------------------------------------------------
DATA_DIR = os.getcwd() + "/frames/"
RESIZE_TO = 50
x, y = [], []
label = pd.read_csv("labels.csv")
label['image_name'] = label['id'].map('seq_{:06d}.jpg'.format)
lb = label['id']
count = label['count']

for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".jpg"]:
    x.append(cv2.resize(cv2.imread(DATA_DIR + path), (RESIZE_TO, RESIZE_TO)))
    for j in range(len(lb)):
        if lb[j] == int(path[4:-4]):
            y.append(count[j])
x, y = np.array(x), np.array(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
x_train, x_test = x_train / 255, x_test / 255

resize = 64
batch = 256
# %% -------------------------------------- Data Prep------------------------------------------------------------------

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
    directory='/home/anwesha/ml2-project/frames',
    x_col="image_name",
    y_col="count",
    weight_col=None,
    target_size=(resize, resize),
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
    input_shape=(resize, resize, 3),
    pooling='avg'
)

# Change the top of the network, the end of the network.
x = base_model.output
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(1, activation='linear')(x)
model = Model(inputs=base_model.input, outputs=predictions)
#make first 3 layers untrainable
k = -3
for layer in model.layers[:k]:
    layer.trainable = False
print('Trainable:')
for layer in model.layers[k:]:
    print(layer.name)
    layer.trainable = True


# %% --------------------------------------Model Training  ------------------------------------------------------------------

optimizer = Adam(learning_rate=3e-4)

model.compile(
    optimizer=optimizer,
    loss="mean_absolute_error",
    metrics=['mean_absolute_error', 'mean_squared_error']
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_mean_squared_error',
    patience=3,
    verbose=1,
    factor=0.2,
    min_lr=0.000001
)

# Fit the model
history = model.fit(
    train_generator,
    epochs=5,
    validation_data=valid_generator,
    verbose=2,
    callbacks=[learning_rate_reduction]
)
print('\nDone.')

model.save('model_resnet50.h5')

# %% -------------------------------------- Validation loss Curve  ------------------------------------------------------------------

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(history.history['loss'], color='b', label="Training loss")
ax.plot(history.history['val_loss'], color='r', label="Validation loss", axes=ax)
ax.set_ylim(top=np.max(history.history['val_loss']) * 1.2, bottom=0)
legend = ax.legend(loc='best', shadow=True)
plt.show()

valid_generator.reset()
all_labels = []
all_pred = []
for i in range(len(valid_generator)):
    x = next(valid_generator)
    pred_i = model.predict(x[0])[:, 0]
    labels_i = x[1]
    labels_i=int(labels_i)
    all_pred.append(pred_i)

xs_orginal = history.history['lr']
ys_original = history.history['loss']
xs = np.array(xs_orginal)
ys = np.array(ys_original)
# %% --------------------------------------Prediction  ------------------------------------------------------------------

loss_LR_df = pd.DataFrame(list(zip(xs_orginal, ys_original)),
                          columns=['LR', 'Loss'])
print("loss_LR_df:\n", loss_LR_df)
loss_LR_df.to_csv("loss_LR_df.csv", index=False)
cat_labels = np.concatenate(all_labels)
cat_pred = np.concatenate(all_pred)
print("Final loss value",history.history['loss'][-1])
print("Final validation loss", history.history['val_loss'][-1])
