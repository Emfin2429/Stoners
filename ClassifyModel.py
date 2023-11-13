import tensorflow as tf
from tensorflow import keras
import requests
requests.packages.urllib3.disable_warnings()
import ssl
from FGSA import fgsm_attack
import numpy as np

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
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



# Evaluate the model
#test_loss, test_acc = model.evaluate(test_images.reshape(-1, 28, 28, 1), test_labels)
#print("Test accuracy of OG File:", test_acc)


# Choose a random test image and label
index = np.random.randint(0, len(test_images))
image = test_images[index]
label = np.array([test_labels[index]])

# Generate adversarial example
epsilon = 0.01
perturbed_image = fgsm_attack(model, image.reshape(1, 28, 28, 1), label, epsilon)

# Evaluate the model on the original and perturbed images
original_pred = model.predict(image.reshape(1, 28, 28, 1))
perturbed_pred = model.predict(perturbed_image)

# Display results
print("Original Prediction:", np.argmax(original_pred))
print("Perturbed Prediction:", np.argmax(perturbed_pred))
