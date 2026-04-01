import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# load model
model = load_model("model/model_1.keras")

# class names (IMPORTANT: match training order)
class_names = [
    "Anushka_Sharma",
    "Barack_Obama",
    "Bill_Gates",
    "Dalai_Lama",
    "Indira_Nooyi",
    "Melinda_Gates",
    "Narendra_Modi",
    "Sundar_Pichai",
    "Vikas_Khanna",
    "Virat_Kohli",
]

st.title("Face Recognition App 🔍")
st.write("Upload an image to identify the person")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # preprocess image
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # prediction
    pred = model.predict(img_array)
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
