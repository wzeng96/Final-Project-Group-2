# Description and order of execution of codes:

data_loader.py- Is the code for loading, preprocessing and exploring the data.

```
python3 data_loader.py
```

cnn_train.py- Is the code for training the custom CNN model. 

```
python3 cnn_train.py
```

resnet50_train.py- Is the code for training and fine tuning the ResNet50 model.

```
python3 resnet50_train.py
```

##### After training the models we have two models saved, one for CNN and ResNet50 (models provided in zip file on BB)

cnn_test and resnet50_test: code for getting the prediction values and prints the image with true labels.

```
python3 cnn_test.py
```
```
python3 resnet50_test.py
```
