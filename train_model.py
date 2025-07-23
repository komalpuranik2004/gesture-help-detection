import os
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import shutil

# CONFIG
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_PATH = 'dataset'
TEMP_DATASET_PATH = 'temp_dataset'
MAX_IMAGES_PER_CLASS = 300

# Clean/create temp dataset
if os.path.exists(TEMP_DATASET_PATH):
    shutil.rmtree(TEMP_DATASET_PATH)

def copy_limited_images(src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    images = sorted(os.listdir(src_dir))[:MAX_IMAGES_PER_CLASS]
    for img in images:
        shutil.copy(os.path.join(src_dir, img), os.path.join(dest_dir, img))

for split in ['train', 'test', 'valid']:
    for category in ['help', 'normal']:
        src = os.path.join(DATASET_PATH, split, category)
        dest = os.path.join(TEMP_DATASET_PATH, split, category)
        copy_limited_images(src, dest)

# Image generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(TEMP_DATASET_PATH, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(TEMP_DATASET_PATH, 'valid'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(TEMP_DATASET_PATH, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# CNN model
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure(figsize=(10, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_performance.png')  # Saves the graph
    plt.show()

# Evaluate
loss, acc = model.evaluate(test_generator)
print(f"Test accuracy: {acc*100:.2f}%")


plot_training_history(history)

# Save
model.save('gesture_help_model.keras')
