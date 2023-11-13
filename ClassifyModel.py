import tensorflow as tf
from tensorflow import keras
import requests
requests.packages.urllib3.disable_warnings()
import ssl
from FGSA import fgsm_attack
import numpy as np
import random

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


# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images.reshape(-1, 28, 28, 1), train_labels, epochs=5)



# Choose a random test image and label
"""
index = np.random.randint(0, len(test_images))
image = test_images[index]
label = np.array([test_labels[index]])
"""

# Evaluate the model on the original and perturbed images
perturbed_test_images = test_images.copy()
perb_test_labels = test_labels.copy()

# Loop through every image
for index in range(len(perturbed_test_images)):
    numRand = random.randint(1, len(perturbed_test_images))
    
    # should randomly perturb roughly 10% of the training set.
    if numRand <= 500:
        #extracts the image
        imageToMutate = perturbed_test_images[index]
        #extracts image label
        label = np.array([perb_test_labels[index]])
        #sizes the image
        original_pred = model.predict(imageToMutate.reshape(1, 28, 28, 1))
        #perturbs the image
        perturbed_image = fgsm_attack(model, imageToMutate.reshape(1, 28, 28, 1), label, .8)
        #replaces OG image with new perturbed image
        perturbed_test_images[index] = perturbed_image.squeeze()


"""
        for i in range(1, 100):
            epsilon = i / 100.0
            perturbed_image = fgsm_attack(model, imageToMutate.reshape(1, 28, 28, 1), label, epsilon)
            new_Perb_image = np.argmax(model.predict(perturbed_image))
            og_image = np.argmax(original_pred)




            if new_Perb_image != og_image:
                final_perturbed_image = perturbed_image
                final_epsilon = epsilon
                listOfEpsilons.append(final_epsilon)
                numChanged += 1 
                perturbed_test_images[index] = final_perturbed_image.squeeze()
                break
"""
#test the accuracy of og data set
test_loss, test_acc = model.evaluate(test_images.reshape(-1, 28, 28, 1), test_labels)
print("Test accuracy of OG File:", test_acc)

#test the accuracy on new data set
test_loss2, test_acc2 = model.evaluate(perturbed_test_images.reshape(-1, 28, 28, 1), perb_test_labels)
print("Test accuracy of changed File:", test_acc2)


# Code for testing a single image with a fast gradient sign attack
"""
# Evaluate the model on the original and perturbed images
original_pred = model.predict(image.reshape(1, 28, 28, 1))
final_perturbed_image = 0
final_epsilon = 0


for i in range(1, 100):
    epsilon = i / 100.0
    perturbed_image = fgsm_attack(model, image.reshape(1, 28, 28, 1), label, epsilon)
    new_Perb_image = np.argmax(model.predict(perturbed_image))
    og_image = np.argmax(original_pred)

    if new_Perb_image != og_image:
        final_perturbed_image = perturbed_image
        final_epsilon = epsilon
        break

# Evaluate the model on the original and perturbed images
perturbed_pred = model.predict(final_perturbed_image)

# Display results
print("Original Prediction:", np.argmax(original_pred))
print("Perturbed Prediction:", np.argmax(perturbed_pred ))
print("Final epsilon: ", final_epsilon)

#print("Number of training samples:", len(train_images))
#print("Number of testing samples:", len(test_images))
"""
