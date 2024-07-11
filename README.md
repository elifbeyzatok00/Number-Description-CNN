# Number Description CNN (Rakam Tanima CNN)

This project aims to develop a Convolutional Neural Network (CNN) model to recognize handwritten digits using the MNIST dataset. The model is built using Keras and TensorFlow libraries and is trained and evaluated on the MNIST dataset.

## Table of Contents

- [Project Overview](#project-overview)
- [Setup](#setup)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [References](#references)

## Project Overview

The goal of this project is to create a CNN that can classify handwritten digits (0-9) with high accuracy. The MNIST dataset, which contains 60,000 training images and 10,000 testing images, is used for this purpose.

## Setup

### Colab Setup

To use Google Colab for this project, follow these steps:

1. Mount Google Drive:

   ```python
   from google.colab import drive
   drive.mount('/gdrive/')
   ```

2. Install necessary libraries:

   ```python
   !pip install -q keras
   !pip install -q tensorflow
   !pip install keras.utils
   ```

3. Import required libraries:
   ```python
   from __future__ import print_function
   import keras
   import tensorflow
   from keras.datasets import mnist
   from keras.models import load_model, Sequential
   from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
   from tensorflow.keras.utils import to_categorical
   from tensorflow.keras import optimizers
   from keras import backend as K
   import matplotlib.pyplot as plt
   ```

## Data

The MNIST dataset is loaded and split into training and testing sets:

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

The training set contains 60,000 images and the test set contains 10,000 images. Each image is 28x28 pixels.

## Model Architecture

The model is constructed using the Sequential API in Keras. It includes the following layers:

- Two Conv2D layers
- One MaxPooling2D layer
- One Dropout layer
- One Flatten layer
- Two Dense layers

```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
```

## Training

The model is compiled and trained using the following parameters:

```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=6,
          verbose=1,
          validation_data=(x_test, y_test))
```

## Evaluation

The model is evaluated using the test set:

```python
score = model.evaluate(x_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])
```

## Results

After training, the model achieves a certain level of accuracy and loss, which is printed out.

## Usage

To use the trained model for prediction on a random test image:

```python
model_test = load_model('save_models/mnist_model.h5')
test_image = x_test[32]
plt.imshow(test_image.reshape(28,28), cmap='gray', vmin=2, vmax=255)
test_data = x_test[32].reshape(1,28,28,1)
pred = model_test.predict(test_data)
import numpy as np
print(np.argmax(pred))
```

## References

- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
