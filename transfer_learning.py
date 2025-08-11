# =============================================================================
#      Multiclass Fish Classification - Transfer Learning Script
# =============================================================================
# This script uses MobileNetV2, a pre-trained model, to classify fish images.
# It demonstrates the process of feature extraction and optional fine-tuning.
#
# =============================================================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import os

# --- 1. Configuration and Parameters ---
print("Step 1: Configuring parameters...")

# IMPORTANT: Update this path to where your dataset is located.
DATASET_DIR = 'D:\multi class\dataset' 

# Model and image processing parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 15 # Transfer learning often requires fewer epochs

# --- 2. Data Preparation ---
# Load images from directories using ImageDataGenerator
print("\nStep 2: Preparing data generators...")

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Dataset directory not found at {DATASET_DIR}")

# Create a data generator with augmentation for the training set
# For validation, we only apply rescaling.
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2 # Reserve 20% for validation
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

NUM_CLASSES = train_generator.num_classes
print(f"Found {train_generator.samples} training images belonging to {NUM_CLASSES} classes.")
print(f"Found {validation_generator.samples} validation images.")

# --- 3. Build Transfer Learning Model ---
print("\nStep 3: Building the transfer learning model with MobileNetV2...")

def build_transfer_model(input_shape, num_classes):
    """
    Builds a transfer learning model using MobileNetV2 as the base.
    """
    # Load the pre-trained MobileNetV2 model without its top classification layer.
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False, # Exclude the final Dense layer
        weights='imagenet' # Use weights pre-trained on ImageNet
    )

    # Freeze the layers of the base model so they are not updated during initial training.
    base_model.trainable = False

    # Create a new model on top
    inputs = base_model.input
    x = base_model.output
    # Add a pooling layer to reduce dimensions
    x = GlobalAveragePooling2D()(x)
    # Add a fully-connected layer for learning
    x = Dense(1024, activation='relu')(x)
    # Add a dropout layer for regularization
    x = Dropout(0.5)(x)
    # Add the final prediction layer
    outputs = Dense(num_classes, activation='softmax')(x)

    # Combine the base model and the new top layers
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model with a low learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Create an instance of the model
input_shape = IMAGE_SIZE + (3,)
transfer_model = build_transfer_model(input_shape=input_shape, num_classes=NUM_CLASSES)

# Display the model's architecture
transfer_model.summary()

# --- 4. Train the Model ---
print("\nStep 4: Starting model training (feature extraction)...")

history_transfer = transfer_model.fit(
    train_generator,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator
)

# --- 5. Save the Trained Model ---
print("\nStep 5: Saving the trained model...")
transfer_model.save('fish_classifier_transfer_model.h5')

print("\n---------------------------------------------------------")
print("✅ Training complete and model saved as 'fish_classifier_transfer_model.h5'")
print("---------------------------------------------------------")