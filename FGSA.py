import numpy as np

def fgsm_attack(model, images, labels, epsilon=0.01):
    # Set the model to trainable
    model.trainable = False
    
    # Use GradientTape to compute the gradient of the loss with respect to the input images
    with tf.GradientTape() as tape:
        images = tf.convert_to_tensor(images, dtype=tf.float32)
        tape.watch(images)
        predictions = model(images)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)

    # Get the gradient of the loss with respect to the input
    gradient = tape.gradient(loss, images)

    # Generate adversarial examples by adding epsilon times the sign of the gradient to the input
    perturbed_images = images + epsilon * tf.sign(gradient)

    # Clip the perturbed images to ensure they are within the valid range [0, 1]
    perturbed_images = tf.clip_by_value(perturbed_images, 0, 1)

    return perturbed_images.numpy()

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
