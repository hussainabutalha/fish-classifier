# =============================================================================
#                       Model Evaluation Script
# =============================================================================
# This script loads pre-trained models and evaluates their performance on
# the validation dataset by generating reports and confusion matrices.
#
# To run: python evaluate_models.py
# =============================================================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- 1. Configuration ---
# IMPORTANT: Update these paths to match your project structure.
CNN_MODEL_PATH = 'fish_classifier_cnn_model.h5'
TRANSFER_MODEL_PATH = 'fish_classifier_transfer_model.h5' # Assuming this is your transfer model
DATASET_DIR = 'D:\multi class\dataset'

# Use the same parameters as in your training script
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# --- 2. Data Preparation ---
# We only need the validation generator to evaluate the models.
# IMPORTANT: Only apply rescaling. Do not use data augmentation here.
print("Step 1: Preparing validation data generator...")
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Dataset directory not found at {DATASET_DIR}")

validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

validation_generator = validation_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Important: Do not shuffle for evaluation
)

CLASS_LABELS = list(validation_generator.class_indices.keys())

# --- 3. Model Loading ---
print("\nStep 2: Loading trained models...")
# Load the Custom CNN model
if os.path.exists(CNN_MODEL_PATH):
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    print(f"✅ Successfully loaded model from {CNN_MODEL_PATH}")
else:
    raise FileNotFoundError(f"Model file not found at {CNN_MODEL_PATH}")

# Load the Transfer Learning model
if os.path.exists(TRANSFER_MODEL_PATH):
    transfer_model = tf.keras.models.load_model(TRANSFER_MODEL_PATH)
    print(f"✅ Successfully loaded model from {TRANSFER_MODEL_PATH}")
else:
    raise FileNotFoundError(f"Model file not found at {TRANSFER_MODEL_PATH}")


# --- 4. Evaluation Function ---
def evaluate_model(model, model_name, generator):
    """Generates and displays a classification report and confusion matrix."""
    print(f"\n--- Evaluating {model_name} ---")

    # Predict classes for the validation set
    y_pred_probs = model.predict(generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = generator.classes

    # Generate and print the classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_LABELS))

    # Generate and display the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()


# --- 5. Run Evaluations ---
print("\nStep 3: Running evaluations...")
evaluate_model(cnn_model, "Custom CNN", validation_generator)
evaluate_model(transfer_model, "MobileNetV2 Transfer Learning", validation_generator)

print("\n✅ Evaluation complete.")