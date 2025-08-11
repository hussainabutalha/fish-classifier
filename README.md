# 🐟 Multiclass Fish Image Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)

This project focuses on classifying fish images into multiple species using deep learning. It explores building a Convolutional Neural Network (CNN) from scratch and compares its performance against a powerful pre-trained model (MobileNetV2) using transfer learning. The best model is then deployed in an interactive web application built with Streamlit.

---

## 🚀 App Demo

Here's a look at the final Streamlit application in action. Users can upload a fish image and receive an instant prediction of its species along with a confidence score.



---

## ✨ Key Features

* **Custom CNN:** A CNN model built from the ground up using TensorFlow/Keras.
* **Transfer Learning:** Leverages the pre-trained MobileNetV2 model for high-accuracy feature extraction.
* **Model Comparison:** Provides a detailed evaluation of both models using metrics like accuracy, precision, recall, F1-score, and confusion matrices.
* **Interactive UI:** A user-friendly web application built with Streamlit for real-time predictions.
* **Data Augmentation:** Enhances model robustness by applying random transformations to the training data.

---

## 🛠️ Tech Stack

* **Backend:** Python
* **Deep Learning:** TensorFlow, Keras
* **Web Framework:** Streamlit
* **Data Science & Utilities:** Scikit-learn, NumPy, Matplotlib, Seaborn

---

## ⚙️ Setup and Installation

Follow these steps to set up the project on your local machine.

**1. Clone the Repository**
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/fish-classifier.git)
cd fish-classifier
```

**2. Create a Virtual Environment**
It's highly recommended to use a virtual environment to manage project dependencies.
```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**3. Install Dependencies**
First, create a `requirements.txt` file from your active environment where you've installed all the packages.
```bash
pip freeze > requirements.txt
```
Then, anyone can install the required packages using this file:
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run
**Dataset Link  https://drive.google.com/drive/folders/1iKdOs4slf3XvNWkeSfsszhPRggfJ2qEd**
**1. Prepare the Dataset**
Make sure your dataset is organized by class in a `dataset/` folder within the project directory.
```
dataset/
├── Black Sea Sprat/
│   ├── 00001.png
│   └── ...
├── Gilt-Head Bream/
│   ├── 00001.png
│   └── ...
└── ...
```

**2. Train and Evaluate Models**
Update the `DATASET_DIR` path in `run_experiments.py` and run the script to train both models, evaluate them, and save the best one.
```bash
python run_experiments.py
```

**3. Launch the Streamlit App**
Once the best model (`best_fish_classifier.h5`) is saved, launch the web application.
```bash
streamlit run app.py
```
Open your web browser and navigate to `http://localhost:8501`.

---

## 📊 Model Performance

The transfer learning approach with MobileNetV2 significantly outperformed the custom CNN built from scratch, demonstrating the power of pre-trained models.

| Model | Validation Accuracy | F1-Score (Weighted) |
| :--- | :---: | :---: |
| Custom CNN | ~78% | 0.77 |
| **MobileNetV2 (Transfer Learning)** | **~97%** | **0.97** |

*(Results are approximate and may vary based on training.)*

---

## 📂 Project Structure
```
.
├── dataset/              # Folder for the image data
├── app.py                # The Streamlit web application script
├── run_experiments.py    # Script to train and evaluate all models
├── best_fish_classifier.h5 # The saved best-performing model
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

