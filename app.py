import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# 1. Wide layout so everything fits cleanly on one screen
st.set_page_config(
    page_title="Plant Leaf Disease Detector",
    page_icon="🌿",
    layout="wide",
)

# 2. Compact vertical spacing to eliminate unnecessary scrolling
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1rem;
            max-width: 1100px;
        }
        h1 {
            margin-bottom: 0.2rem !important;
        }
        .stProgress > div > div > div > div {
            height: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# 3. Load trained model once (unchanged backend)
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("plant_disease_model.keras")


model = load_trained_model()

# 4. Exact alphabetical list of 15 PlantVillage classes (unchanged)
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


def format_label(name):
    return name.replace("___", " - ").replace("__", " - ").replace("_", " ")


# 5. Compact header
st.title("🌿 Plant Leaf Disease Detector")
st.caption(
    "Upload a Pepper, Potato, or Tomato leaf photo—or pick a sample—to detect its disease using MobileNetV2."
)

# 6. Auto-detect optional sample images from a "samples" folder
SAMPLES_DIR = "samples"
sample_files = []
if os.path.isdir(SAMPLES_DIR):
    sample_files = sorted(
        f
        for f in os.listdir(SAMPLES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

# 7. Top Input Row: Upload on the left, Sample Selector on the right
input_col1, input_col2 = st.columns([2, 1], gap="medium")

with input_col1:
    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

with input_col2:
    if sample_files:
        selected_sample = st.selectbox(
            "Or try a sample image",
            ["-- Select a sample leaf --"] + sample_files,
            label_visibility="collapsed",
        )
    else:
        selected_sample = st.selectbox(
            "Or try a sample image",
            ["No sample images in /samples folder"],
            disabled=True,
            label_visibility="collapsed",
        )

# 8. Decide which image to analyze (uploaded image takes priority)
image = None
image_caption = ""

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_caption = f"Uploaded: {uploaded_file.name}"
elif sample_files and selected_sample != "-- Select a sample leaf --":
    sample_path = os.path.join(SAMPLES_DIR, selected_sample)
    image = Image.open(sample_path).convert("RGB")
    image_caption = f"Sample: {selected_sample}"

# 9. Main Side-by-Side Result Area (Preview Left | Prediction Right)
if image is not None:
    # Exact same preprocessing & prediction logic as before
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing leaf..."):
        preds = model.predict(img_array, verbose=0)[0]

    top_3_indices = np.argsort(preds)[-3:][::-1]
    best_name = format_label(CLASS_NAMES[top_3_indices[0]])
    best_conf = preds[top_3_indices[0]] * 100

    left_col, right_col = st.columns([1, 1.3], gap="large")

    # Left Card: Clean, reasonably sized image preview
    with left_col:
        with st.container(border=True):
            preview = image.copy()
            preview.thumbnail((340, 300))
            st.image(preview, caption=image_caption)

    # Right Card: Prediction + Top 3 Confidence Bars
    with right_col:
        with st.container(border=True):
            st.markdown("#### Diagnosis Result")
            st.success(f"**{best_name}** — **{best_conf:.2f}%** confidence")

            st.markdown("##### Top 3 Probabilities")
            for idx in top_3_indices:
                clean_name = format_label(CLASS_NAMES[idx])
                prob_pct = preds[idx] * 100
                st.markdown(f"**{clean_name}** ({prob_pct:.2f}%)")
                st.progress(float(preds[idx]))
else:
    st.info(
        "Upload a leaf photo above (or choose a sample image) to view the preview and disease prediction side by side."
    )
