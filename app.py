
# Import Streamlit for UI
import streamlit as st

# Import numpy
import numpy as np

# Import OpenCV for image processing
import cv2

# Import model loader
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("dr_model.h5")

# Define class labels
classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

# App title
st.title("Diabetic Retinopathy Detection")

# Upload image widget
file = st.file_uploader("Upload Fundus Image")

# If image is uploaded
if file:

    # Convert file to numpy array
    img = np.frombuffer(file.read(), np.uint8)

    # Decode image
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    # Resize image
    img = cv2.resize(img, (224,224))

    # Normalize pixel values
    img = img / 255.0

    # Reshape for model input
    img = np.reshape(img, (1,224,224,3))

    # Predict DR class
    pred = model.predict(img)

    # Get predicted class index
    result = np.argmax(pred)

    # Show uploaded image
    st.image(img[0])

    # Display prediction
    st.success(f"Prediction: {classes[result]}")
