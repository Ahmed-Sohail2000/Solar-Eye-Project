import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# define the project directory
project_dir = '/content/my-streamlit-app'

# Load the saved model from the model folder
model_path = os.path.join(project_dir, 'model', 'resnet50v2_fine_tuned.h5')
model = tf.keras.models.load_model(model_path)

# Title of the app
st.title("Solar Panel Fault Detection")

# Description of the app
st.write("This app uses a fine-tuned ResNet50V2 model to detect different types of faults in solar panel thermal imagery.")

# File uploader to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Check if the user has uploaded a file
if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)

    # Show the image in the app
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Pre-process the image
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    predictions = model.predict(image_array)
    class_idx = np.argmax(predictions, axis=1)
    class_prob = np.max(predictions, axis=1)

    class_labels = ["Clean", "Cracked", "Damaged", "Dusty", "Snow"]
    predicted_label = class_labels[class_idx[0]]

    st.write(f"Prediction: {predicted_label}")
    st.write(f"Probability: {class_prob[0]*100:.2f}%")
