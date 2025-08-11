# =============================================================================
#          Multiclass Fish Image Classification - Training Script
# =============================================================================
# This script handles data loading, model building, training, and saving
# for a custom CNN model designed to classify fish species.
#
# To run: python train_fish_classifier.py
# =============================================================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# --- 1. Configuration and Parameters ---
# Define the core parameters for the project.
print("Step 1: Configuring parameters...")

# Path to the dataset directory.
# IMPORTANT: Update this path to where your dataset is located.
DATASET_DIR = 'D:\multi class\dataset' 

# Model and image processing parameters.
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 25

# --- 2. Data Preparation and Augmentation ---
# Load images from directories and apply transformations.
print("\nStep 2: Preparing data generators...")

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        f"The dataset directory was not found at {DATASET_DIR}. "
        "Please update the 'DATASET_DIR' variable with the correct path."
    )

# Create an ImageDataGenerator for training with augmentation.
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Rescale pixel values from [0, 255] to [0, 1].
    rotation_range=40,         # Randomly rotate images.
    width_shift_range=0.2,     # Randomly shift images horizontally.
    height_shift_range=0.2,    # Randomly shift images vertically.
    shear_range=0.2,           # Apply shear transformations.
    zoom_range=0.2,            # Randomly zoom in on images.
    horizontal_flip=True,      # Randomly flip images horizontally.
    fill_mode='nearest',       # Strategy for filling in newly created pixels.
    validation_split=0.2       # Reserve 20% of data for validation.
)

# Create a generator for training data.
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'          # Specify this is the training set.
)

# Create a generator for validation data (only rescaling is applied).
validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'        # Specify this is the validation set.
)

# Get the number of classes from the generator.
NUM_CLASSES = train_generator.num_classes
print(f"Found {train_generator.samples} images for training.")
print(f"Found {validation_generator.samples} images for validation.")
print(f"Number of classes: {NUM_CLASSES}")

# --- 3. Build the CNN Model ---
# Define the model architecture using a function for modularity.
print("\nStep 3: Building the CNN model...")

def build_custom_cnn(input_shape, num_classes):
    """Builds, compiles, and returns a custom CNN model."""
    model = Sequential([
        # Convolutional Base
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Classifier Head
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5), # Regularization to prevent overfitting.
        Dense(num_classes, activation='softmax') # Output layer for probabilities.
    ])

    # Compile the model.
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy', # For multi-class classification.
        metrics=['accuracy']
    )
    return model

# Create an instance of the model.
input_shape = IMAGE_SIZE + (3,)
cnn_model = build_custom_cnn(input_shape=input_shape, num_classes=NUM_CLASSES)

# Display the model's architecture.
cnn_model.summary()

# --- 4. Train the Model ---
# Start the training process using the prepared data generators.
print("\nStep 4: Starting model training...")

history = cnn_model.fit(
    train_generator,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    verbose=1 # Show progress bar.
)

# --- 5. Save the Trained Model ---
# Save the final model for later use in the Streamlit application.
print("\nStep 5: Saving the trained model...")

cnn_model.save('fish_classifier_cnn_model.h5')

print("\n-----------------------------------------")
print("✅ Training complete and model saved as 'fish_classifier_cnn_model.h5'")
print("-----------------------------------------")