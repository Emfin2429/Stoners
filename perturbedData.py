# Import necessary libraries
import tensorflow as tf  # TensorFlow is an open-source machine learning framework
from tensorflow import keras  # Keras is an API for building and training deep learning models
import numpy as np  # NumPy is a library for numerical operations in Python
import random  # Python library for generating random numbers
from FGSA import fgsm_attack  # Import the Fast Gradient Sign Method (FGSM) attack function from a custom module
from createModel import createCNN #Import createModel function


def perturbedDataEntries(model, images, labels, percent):
    # Evaluate the model on the original and perturbed images
    perturbed_test_images = images.copy()
    perb_test_labels = labels.copy()
    DisplayIndex = 0

    # Loop through every image
    for index in range(len(perturbed_test_images)):
        numRand = random.randint(1, len(perturbed_test_images))
        
        # should randomly perturb roughly 20% of the testing data set. The testing set is 10000.
        if numRand <= percent*len(perturbed_test_images):
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
    return perturbed_test_images, perb_test_labels

   