# Import necessary libraries
import tensorflow as tf  # TensorFlow is an open-source machine learning framework
from tensorflow import keras  # Keras is an API for building and training deep learning models
import requests  # Requests is a library for making HTTP requests in Python
requests.packages.urllib3.disable_warnings()  # Disable SSL warnings in requests
import ssl  # SSL (Secure Sockets Layer) is a protocol for secure communication over a computer network
from FGSA import fgsm_attack  # Import the Fast Gradient Sign Method (FGSM) attack function from a custom module
from createModel import createCNN #Import createModel function
from perturbedData import perturbedDataEntries # gets perturbedDataEntries function
import numpy as np  # NumPy is a library for numerical operations in Python
import random  # Python library for generating random numbers
import matplotlib.pyplot as plt  # Matplotlib is a plotting library for Python


# This code is to address an error in calls for the Mac Book
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# Load the MNIST dataset from keras
mnist = tf.keras.datasets.mnist

# create two data sets: training 6/7 testing 1/7
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# Preprocess the data by scaling pixel values to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

initialModel = createCNN(train_images, train_labels, 5)

attackedTrainingImage, attackedTrainingLabel = perturbedDataEntries(initialModel, train_images, train_labels, .1)

attackedModel = createCNN(attackedTrainingImage, attackedTrainingLabel, 5)

attackedtestImage, attackedtestLabel = perturbedDataEntries(initialModel, test_images, test_labels, .1)


#test the accuracy of og data set
test_loss, test_acc = initialModel.evaluate(attackedtestImage.reshape(-1, 28, 28, 1),attackedtestLabel)
print("Test accuracy of OG File:", test_acc)

#test the accuracy on new data set
test_loss2, test_acc2 = attackedModel.evaluate(attackedtestImage.reshape(-1, 28, 28, 1), attackedtestLabel)
print("Test accuracy of changed File:", test_acc2)

"""
# Evaluate the model on the original and perturbed images
perturbed_test_images = test_images.copy()
perb_test_labels = test_labels.copy()
DisplayIndex = 0

# Loop through every image
for index in range(len(perturbed_test_images)):
    numRand = random.randint(1, len(perturbed_test_images))
    
    # should randomly perturb roughly 20% of the testing data set. The testing set is 10000.
    if numRand <= 6000:
        DisplayIndex = index

        #extracts the image
        imageToMutate = perturbed_test_images[index]
        #extracts image label
        label = np.array([perb_test_labels[index]])
        #sizes the image
        #original_pred = model.predict(imageToMutate.reshape(1, 28, 28, 1))
        #perturbs the image
        perturbed_image = fgsm_attack(model, imageToMutate.reshape(1, 28, 28, 1), label, .3)
        #replaces OG image with new perturbed image
        perturbed_test_images[index] = perturbed_image.squeeze()

#test the accuracy of og data set
test_loss, test_acc = model.evaluate(test_images.reshape(-1, 28, 28, 1), test_labels)
print("Test accuracy of OG File:", test_acc)

#test the accuracy on new data set
test_loss2, test_acc2 = model.evaluate(perturbed_test_images.reshape(-1, 28, 28, 1), perb_test_labels)
print("Test accuracy of changed File:", test_acc2)

#Code to visually inspect what we just did
# This gets the manipulated Image and label
image = perturbed_test_images[DisplayIndex]
label = np.array([perb_test_labels[DisplayIndex]])

# Tests to see how the model does on the manipulated image
perturbed_pred = model.predict(image.reshape(1, 28, 28, 1))
predicted_label = np.argmax(perturbed_pred)

# Display the original image
plt.imshow(test_images[DisplayIndex], cmap='gray')
plt.title(f"Original MNIST Digit: {test_labels[DisplayIndex]}")
plt.show()

# Display the perturbed image
plt.imshow(image, cmap='gray')
plt.title(f"Perturbed MNIST Digit: {predicted_label}")
plt.show()
"""
