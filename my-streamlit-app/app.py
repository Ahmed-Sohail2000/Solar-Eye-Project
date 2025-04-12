import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the saved model from the model folder
model_path = os.path.join("my-streamlit-app", "model", "resnet50v2_fine_tuned.h5")
model = tf.keras.models.load_model(model_path)

# Title of the app
st.title("Solar Panel Fault Detection")

# Description of the app
st.write("This app uses a fine-tuned ResNet50V2 model to detect different types of faults in solar panels' thermal imagery.")

# File uploader to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Check if the user has uploaded a file
if uploaded_file is not None:
    
    # Open the image
    image = Image.open(uploaded_file)

    # Show the image in the app
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Pre-process the image for the model
    image = image.resize((224, 224))  # Resize the image to match model's input size (224x224 for ResNet)
    image_array = np.array(image) / 255.0  # Normalize the image (ResNet requires values between 0 and 1)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict with the model
    predictions = model.predict(image_array)
    
    # Print prediction values for debugging
    st.write(f"Predictions: {predictions}")

    # Get the class with the highest probability
    class_idx = np.argmax(predictions, axis=1).flatten()

    # Debugging output
    st.write(f"class_idx shape: {class_idx.shape}")
    st.write(f"class_idx value: {class_idx}")

    # Ensure the predictions are valid
    if predictions.size > 0 and len(class_idx) > 0:
        # Map class index to labels (assuming you have a list of labels)
        class_labels = ["Clean", "Cracked", "Damaged", "Dusty", "Snow"]  # Example class labels
        predicted_label = class_labels[class_idx[0]]

        # Display the prediction result
        st.write(f"Prediction: {predicted_label}")
        st.write(f"Probability: {np.max(predictions)*100:.2f}%")
    else:
        st.error("No valid prediction could be made.")
