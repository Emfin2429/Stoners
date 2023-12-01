# Import necessary libraries
import numpy as np  # NumPy is a library for numerical operations in Python
import tensorflow as tf  # TensorFlow is an open-source machine learning framework
from tensorflow import keras  # Keras is an API for building and training deep learning models
import requests  # Requests is a library for making HTTP requests in Python
requests.packages.urllib3.disable_warnings()  # Disable SSL warnings in requests
import ssl  # SSL (Secure Sockets Layer) is a protocol for secure communication over a computer network

# Define a function for generating adversarial examples using the Fast Gradient Sign Method (FGSM)
def fgsm_attack(model, images, labels, epsilon=0.01):
    # Set the model to not be trainable during the attack
    model.trainable = False
    
    # Use GradientTape to compute the gradient of the loss with respect to the input images
    with tf.GradientTape() as tape:
        # Convert input images to TensorFlow tensors of type float32
        images = tf.convert_to_tensor(images, dtype=tf.float32)
        tape.watch(images)  # Watch the input images for computing gradients
        predictions = model(images)  # Obtain model predictions for the input images
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)  # Compute the cross-entropy loss

    # Get the gradient of the loss with respect to the input images
    gradient = tape.gradient(loss, images)

    # Generate adversarial examples by adding epsilon times the sign of the gradient to the input images
    perturbed_images = images + epsilon * tf.sign(gradient)

    # Clip the perturbed images to ensure they are within the valid pixel value range [0, 1]
    perturbed_images = tf.clip_by_value(perturbed_images, 0, 1)

    # Convert the TensorFlow tensor to a NumPy array and return the perturbed images
    return perturbed_images.numpy()
