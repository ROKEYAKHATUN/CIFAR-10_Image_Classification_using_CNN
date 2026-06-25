import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image


class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🖼️", layout="centered")

st.title("CIFAR-10 Image Classifier")
st.write("Upload an image and the trained CNN model will predict its class.")

@st.cache_resource
def load_trained_model():
    return load_model("cifar10_cnn.keras")  

model = load_trained_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image = image.resize((32, 32))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict"):
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(predictions[0])) * 100

        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")