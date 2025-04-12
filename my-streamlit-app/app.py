import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the model
model = load_model("my-streamlit-app/model/resnet50v2_fine_tuned.h5")

# Define class labels
class_names = [
    'Cell', 'Cell-Multi', 'Cracking', 'Diode', 'Diode-Multi',
    'Hot-Spot', 'Hot-Spot-Multi', 'No-Anomaly', 'Offline-Module',
    'Shadowing', 'Soiling', 'Vegetation'
]

st.title("🔍 Solar Panel Thermal Fault Detection")

uploaded_file = st.file_uploader("Upload a thermal image of a solar panel (grayscale or RGB)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Check if image is grayscale
    if img.mode == "L":
        st.warning("Grayscale image detected. Converting to RGB...")
        img = img.convert("RGB")
    else:
        st.info("RGB or color image detected. Proceeding with standard preprocessing...")

    # Resize and preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize

    # Make prediction
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = preds[0][class_idx]
    predicted_label = class_names[class_idx]

    st.success(f"🔎 Predicted Fault: **{predicted_label}**")
    st.info(f"🧠 Confidence: **{confidence * 100:.2f}%**")
