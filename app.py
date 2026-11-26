import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Plant Leaf Disease Detector", page_icon="🌿")
st.title("🌿 Plant Leaf Disease Detector")
st.write(
    "Upload a photo of a Pepper, Potato, or Tomato leaf to detect its disease using MobileNetV2."
)


# 1. Load your trained model once and cache it so predictions stay fast
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("plant_disease_model.keras")


model = load_trained_model()

# 2. Exact alphabetical list of your 15 PlantVillage classes
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

# 3. File uploader UI
uploaded_file = st.file_uploader(
    "Choose a leaf image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    # Preprocess to 128x128 (matches your training setup)
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Analyzing leaf..."):
        preds = model.predict(img_array, verbose=0)[0]

    # Show top 3 predictions
    top_3_indices = np.argsort(preds)[-3:][::-1]
    best_name = (
        CLASS_NAMES[top_3_indices[0]]
        .replace("___", " - ")
        .replace("__", " - ")
        .replace("_", " ")
    )
    best_conf = preds[top_3_indices[0]] * 100

    st.success(f"**Prediction:** {best_name} ({best_conf:.2f}% confidence)")

    st.subheader("Top 3 Probabilities")
    for idx in top_3_indices:
        clean_name = (
            CLASS_NAMES[idx]
            .replace("___", " - ")
            .replace("__", " - ")
            .replace("_", " ")
        )
        st.write(f"**{clean_name}:** {preds[idx] * 100:.2f}%")
        st.progress(float(preds[idx]))
