import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the saved model from the model folder
model_path = os.path.join('my-streamlit-app', 'model', 'resnet50v2_fine_tuned.h5')
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
    
    # Pre-process the image for the model
    image = image.resize((224, 224))  # Resize the image to match model's input size (224x224 for ResNet)
    image_array = np.array(image) / 255.0  # Normalize the image (ResNet requires values between 0 and 1)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict with the model
    predictions = model.predict(image_array)
    
    # Get the class with the highest probability
    class_idx = np.argmax(predictions, axis=1)
    class_prob = np.max(predictions, axis=1)

    # Map class index to labels (using your class names)
    class_labels = ['Cell', 'Cell-Multi', 'Cracking', 'Diode', 'Diode-Multi', 'Hot-Spot', 'Hot-Spot-Multi', 
                    'No-Anomaly', 'Offline-Module', 'Shadowing', 'Soiling', 'Vegetation']  # Replace with your actual labels
    predicted_label = class_labels[class_idx[0]]

    # Display the prediction result
    st.write(f"Prediction: {predicted_label}")
    st.write(f"Probability: {class_prob[0]*100:.2f}%")
