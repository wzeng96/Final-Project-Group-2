# Wenyu Zeng Individual File
For this project, my part mainly focus on building ResNet 50 model. By building this pretrained 50 layers model, we've improved the accuracy from the basic CNN model.
The last 7 layers of this model is removed to prevent overfit, one Dense layer is also added so that it predicts the number of people in the image. As a result, the validation 
loss is smaller than the CNN model's validation loss, and errors relatively follow the normal distribution with mean around 5. For future improvements, one thing is finding 
more data, and we could also try the MCNN model.
