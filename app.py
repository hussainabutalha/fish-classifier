# ====================================================================
#          Streamlit Web App for Fish Image Classification
# ====================================================================
# This script creates a web interface where users can upload a fish
# image and get a classification prediction from a trained model.
#
# To run: streamlit run app.py
# ====================================================================

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- Configuration ---
# Set the title and icon for the browser tab
st.set_page_config(page_title="Fish Classifier", page_icon="🐟", layout="centered")

# IMPORTANT: Define the path to your trained model
MODEL_PATH = 'fish_classifier_transfer_model.h5'

# IMPORTANT: Define the class names in the correct order
# Check the order from your training script's `train_generator.class_indices`
CLASS_NAMES = ['Black Sea Sprat', 'Gilt-Head Bream', 'Hourse Mackerel', 'Red Mullet', 'Red Sea Bream', 'Sea Bass', 'Shrimp', 'Striped Red Mullet', 'Trout'] # Example names, update with your actual fish species


# --- Model Loading ---
# Cache the model to avoid reloading it on every interaction
@st.cache_resource
def load_model(model_path):
    """Loads the trained Keras model, with error handling."""
    if not os.path.exists(model_path):
        # Display a user-friendly error if the model file is not found
        st.error(f"Model file not found at {model_path}")
        st.stop() # Stop the app execution
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model(MODEL_PATH)


# --- Image Preprocessing ---
def preprocess_image(image):
    """Preprocesses the uploaded image to match the model's input requirements."""
    # Resize to the model's expected input size (e.g., 224x224)
    img = image.resize((224, 224))
    # Convert image to numpy array
    img_array = np.array(img)
    # Ensure it's 3-channel (RGB)
    if img_array.ndim == 2: # Grayscale
        img_array = np.stack([img_array]*3, axis=-1)
    # Remove alpha channel if it exists
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    # Expand dimensions to create a batch of 1 and normalize
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch / 255.0


# --- App Layout and Logic ---
st.title("🐟 Fish Species Classifier")
st.write("Upload an image of a fish, and the app will predict its species.")

# File uploader allows user to add their own image
uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Preprocess the image and make a prediction
    with st.spinner('Classifying...'):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction)
        
        # Ensure the index is within the bounds of your CLASS_NAMES list
        if predicted_class_index < len(CLASS_NAMES):
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence_score = np.max(prediction)
            
            # Display the prediction result
            st.success(f"**Predicted Species:** {predicted_class_name}")
            st.info(f"**Confidence Score:** {confidence_score:.2%}")
        else:
            st.error("Prediction index is out of range. Please check your CLASS_NAMES list.")

else:
    st.info("Please upload an image file to get a prediction.")