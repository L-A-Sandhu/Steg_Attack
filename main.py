import os
import math
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------
# 1. Download and Preprocess the Imagenette Dataset
# ---------------------------
# We use the "imagenette" dataset (10 classes) with a train/validation split.
# IMPORTANT: We will keep the pixel values in [0,255] (float32) so that the MobileNetV2
# preprocessing (which expects images in that range) can be applied in the model.
def preprocess_image(image, label):
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)  # Keep in [0,255]
    return image, label

# Load training and validation splits.
train_ds = tfds.load('imagenette', split='train', as_supervised=True,data_dir='./')
val_ds   = tfds.load('imagenette', split='validation', as_supervised=True,data_dir='./')

train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_ds   = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# Batch and prefetch for performance.
batch_size = 32
train_ds = train_ds.cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

# For later evaluation, convert the entire validation set to arrays.
val_images = []
val_labels = []
for image, label in tfds.as_numpy(val_ds.unbatch()):
    val_images.append(image)
    val_labels.append(label)
val_images = np.array(val_images)  # shape: (N, 224, 224, 3), values in [0,255]
val_labels = np.array(val_labels)

# ---------------------------
# 2. Modify Validation Images: Edge Detection and LSB Toggle
# ---------------------------
def modify_image(image):
    """
    Given an image with pixel values in [0,255] (float32, shape (224,224,3)),
    detect its edges using Canny and toggle the two least significant bits
    (using XOR with 3) at the edge pixels.
    Returns the modified image (float32, still in [0,255]).
    """
    # Convert image to uint8
    img_uint8 = image.astype(np.uint8)
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    # Apply Canny edge detection (adjust thresholds if necessary)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    # Copy image to modify
    mod_img = img_uint8.copy()
    # Create boolean mask for edge pixels
    edge_mask = edges > 0
    # For each channel, toggle the two LSBs using XOR with 3
    for c in range(3):
        mod_img[:, :, c][edge_mask] = mod_img[:, :, c][edge_mask] ^ 3
    # Return modified image as float32 in [0,255]
    return mod_img.astype(np.float32)

print("Modifying validation images...")
val_images_modified = np.array([modify_image(img) for img in val_images])
print("Modification complete.")

# ---------------------------
# 3. Build and Train a MobileNetV2–Based Model on the Original Training Data
# ---------------------------
num_classes = 10
input_shape = (224, 224, 3)

# Create a model using MobileNetV2 as the base. The MobileNetV2 preprocess_input function
# expects inputs in [0,255] and converts them to [-1,1]. We include it as the first layer.
inputs = tf.keras.Input(shape=input_shape)
# Preprocess inputs for MobileNetV2.
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
# Use MobileNetV2 as the base model.
base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_tensor=x)
base_model.trainable = False  # Freeze base model for initial training.
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# Train the model on the original training set.
print("Training the model on the original training data...")
epochs = 5  # For demonstration; you can increase epochs if desired.
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

# ---------------------------
# 4. Evaluate the Model on Original and Modified Validation Sets
# ---------------------------
# Get predictions on the original validation set.
preds_orig = np.argmax(model.predict(val_images), axis=1)
# Get predictions on the modified validation set.
preds_mod  = np.argmax(model.predict(val_images_modified), axis=1)
true_labels = val_labels.flatten()  # Ensure labels are 1D

# Classification Report and Confusion Matrix for Original Validation Set
print("\nClassification Report for Original Validation Set:")
print(classification_report(true_labels, preds_orig))

cm_orig = confusion_matrix(true_labels, preds_orig)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_orig, annot=True, fmt="d", cmap='Blues')
plt.title("Confusion Matrix - Original Validation Set")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Classification Report and Confusion Matrix for Modified Validation Set
print("\nClassification Report for Modified Validation Set:")
print(classification_report(true_labels, preds_mod))

cm_mod = confusion_matrix(true_labels, preds_mod)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_mod, annot=True, fmt="d", cmap='Blues')
plt.title("Confusion Matrix - Modified Validation Set")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
