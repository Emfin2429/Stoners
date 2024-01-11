# Import necessary libraries
import tensorflow as tf  # TensorFlow is an open-source machine learning framework
from tensorflow import keras  # Keras is an API for building and training deep learning models
import numpy as np  # NumPy is a library for numerical operations in Python
import random  # Python library for generating random numbers

# model that will create a CNN
def createCNN(train_images, train_labels, numEpochs):
    # Create a sequential model, which is a linear stack of layers
    model = keras.Sequential([
        # Convolutional layer with 32 filters, each of size (3, 3), using ReLU activation
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        
        # Max pooling layer with a pool size of (2, 2)
        keras.layers.MaxPooling2D(2, 2),
        
        # Convolutional layer with 64 filters, each of size (3, 3), using ReLU activation
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Max pooling layer with a pool size of (2, 2)
        keras.layers.MaxPooling2D(2, 2),
        
        # Flatten layer to convert the 2D matrix data to a vector for the fully connected layers
        keras.layers.Flatten(),
        
        # Dense (fully connected) layer with 128 neurons and ReLU activation
        keras.layers.Dense(128, activation='relu'),
        
        # Dense (fully connected) layer with 10 neurons (output layer for classification) and softmax activation
        keras.layers.Dense(10, activation='softmax')
    ])


    # Compile the model with an optimizer, loss function, and evaluation metric
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    model.fit(train_images.reshape(-1, 28, 28, 1), train_labels, epochs= numEpochs)

    #returned trained model
    return model