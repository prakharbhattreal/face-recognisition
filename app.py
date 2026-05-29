import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import json

st.set_page_config(page_title="Face Classifier", layout="centered")

st.title("Face Classifier")
st.caption("This project is a face recognition system built using MobileNetV2 and Streamlit.\n It identifies personalities like Bill Gates, Jack Ma, Narendra Modi, Elon Musk, and Donald Trump from uploaded images.\n The model predicts the detected face with confidence scores in a simple interactive interface.")

@st.cache_resource
def load_classifier():
    return load_model("model.keras")

@st.cache_data
def load_class_names():
    try:
        with open("class_indices.json") as f:
            idx = json.load(f)
        return {v: k for k, v in idx.items()}
    except FileNotFoundError:
        # folder order: gates, jack, modi, musk, trump
        return {0: "Bill Gates", 1: "Jack Ma", 2: "Narendra Modi",
                3: "Elon Musk", 4: "Donald Trump"}

try:
    model = load_classifier()
    class_names = load_class_names()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded:
    img_pil = Image.open(uploaded).convert("RGB")
    display_img = img_pil.resize((450, 450))

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            display_img,
            caption="Uploaded Image",
            use_container_width=True
        )


    img_resized = img_pil.resize((224, 224))
    img_array   = keras_image.img_to_array(img_resized) / 255.0
    img_array   = np.expand_dims(img_array, axis=0)

    preds    = model.predict(img_array, verbose=0)[0]
    top_idx  = int(np.argmax(preds))
    top_name = class_names[top_idx]
    top_conf = float(preds[top_idx]) * 100

    with col2:
        st.subheader(top_name)
        st.write(f"Confidence: **{top_conf:.1f}%**")
        st.divider()
        st.write("**All probabilities**")
        for i, prob in enumerate(preds):
            st.text(class_names[i])
            st.progress(float(prob), text=f"{prob*100:.1f}%")
