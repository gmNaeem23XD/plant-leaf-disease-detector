import os
from datetime import datetime
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# 1. Wide layout with clean vector page icon
st.set_page_config(
    page_title="Plant Leaf Disease Detector",
    page_icon=":material/eco:",
    layout="wide",
)

# 2. Compact spacing + subtle fade-in, hover states, and custom icon styling
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1rem;
            max-width: 1120px;
        }
        .app-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.25rem;
        }
        .app-logo {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            background: rgba(46, 204, 113, 0.12);
            border: 1px solid rgba(46, 204, 113, 0.35);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        .app-title {
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
            line-height: 1.2;
            letter-spacing: -0.02em;
        }
        .stProgress > div > div > div > div {
            height: 10px;
        }
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            animation: fadeInUp 0.35s ease-out;
            transition: border-color 0.25s ease, box-shadow 0.25s ease;
            border-radius: 12px !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(46, 204, 113, 0.55) !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        }
        .step-card {
            background-color: rgba(46, 204, 113, 0.05);
            border: 1px solid rgba(46, 204, 113, 0.2);
            border-radius: 12px;
            padding: 1.15rem 1rem;
            text-align: center;
            animation: fadeInUp 0.4s ease-out;
            transition: border-color 0.25s ease, transform 0.25s ease;
        }
        .step-card:hover {
            border-color: rgba(46, 204, 113, 0.55);
            transform: translateY(-2px);
        }
        .step-icon {
            width: 44px;
            height: 44px;
            margin: 0 auto 0.65rem auto;
            border-radius: 10px;
            background: rgba(46, 204, 113, 0.12);
            border: 1px solid rgba(46, 204, 113, 0.28);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .step-card h4 {
            margin: 0 0 0.35rem 0;
            font-size: 0.98rem;
            font-weight: 600;
        }
        .step-card p {
            margin: 0;
            font-size: 0.86rem;
            opacity: 0.82;
            line-height: 1.4;
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

# 5. Disease descriptions and practical care tips for all 15 classes
DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "info": "A bacterial infection causing dark, water-soaked spots on leaves and fruit.",
        "tip": "Remove affected leaves, avoid overhead watering, and use copper-based bactericides if needed.",
    },
    "Pepper__bell___healthy": {
        "info": "The pepper leaf shows healthy green color with no visible signs of disease.",
        "tip": "Maintain consistent watering, good airflow, and balanced fertilizer to keep the plant healthy.",
    },
    "Potato___Early_blight": {
        "info": "A fungal disease (Alternaria solani) marked by dark brown spots with concentric rings.",
        "tip": "Prune lower infected leaves, rotate crops yearly, and apply a protective fungicide early.",
    },
    "Potato___Late_blight": {
        "info": "A fast-spreading oomycete disease causing dark, water-soaked lesions that turn brown-black.",
        "tip": "Remove infected plants promptly to protect nearby crops and avoid watering foliage late in the day.",
    },
    "Potato___healthy": {
        "info": "The potato leaf appears healthy with normal texture and no blight or spotting.",
        "tip": "Keep foliage dry, monitor weekly in humid weather, and ensure well-drained soil.",
    },
    "Tomato_Bacterial_spot": {
        "info": "A bacterial disease causing small, dark, greasy spots that may yellow and drop leaves.",
        "tip": "Water at the base of the plant, sanitize garden tools, and avoid working with wet plants.",
    },
    "Tomato_Early_blight": {
        "info": "A common fungal disease producing dark bullseye-shaped spots starting on older lower leaves.",
        "tip": "Mulch around the base to prevent soil splash, remove lower diseased leaves, and improve airflow.",
    },
    "Tomato_Late_blight": {
        "info": "A destructive disease causing large, irregular dark-green to brown blotches in cool, wet weather.",
        "tip": "Remove affected foliage immediately and avoid overhead irrigation to slow its spread.",
    },
    "Tomato_Leaf_Mold": {
        "info": "A fungal infection causing pale yellow patches on top of leaves and olive-green mold underneath.",
        "tip": "Increase ventilation, space plants adequately, and keep greenhouse or garden humidity low.",
    },
    "Tomato_Septoria_leaf_spot": {
        "info": "A fungal disease characterized by many small circular spots with dark borders and gray centers.",
        "tip": "Remove infected lower leaves, mulch the soil, and apply a tomato-safe fungicide if spreading.",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "info": "Tiny sap-sucking pests that cause fine yellow speckling and delicate webbing on leaves.",
        "tip": "Spray leaf undersides with water or insecticidal soap/neem oil and avoid dusty, dry conditions.",
    },
    "Tomato__Target_Spot": {
        "info": "A fungal disease causing small brown lesions with concentric rings that can enlarge and drop leaves.",
        "tip": "Improve air circulation by pruning dense foliage and avoid wetting the leaves during watering.",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "info": "A viral disease spread by whiteflies, causing upward leaf curling, yellowing, and stunted growth.",
        "tip": "Control whiteflies with yellow sticky traps or neem oil, and remove severely infected plants.",
    },
    "Tomato__Tomato_mosaic_virus": {
        "info": "A hardy virus causing mottled light-and-dark green patterns and distorted leaf shapes.",
        "tip": "Wash hands and disinfect tools before handling plants, and remove infected plants to prevent spread.",
    },
    "Tomato_healthy": {
        "info": "The tomato leaf looks healthy with uniform color and no signs of pests or infection.",
        "tip": "Continue regular base watering, adequate sunlight, and weekly leaf checks.",
    },
}


def format_label(name):
    return name.replace("___", " - ").replace("__", " - ").replace("_", " ")


def is_plant_leaf(pil_image, min_foliage_ratio=0.08):
    """
    Lightweight HSV color-space validator to reject obvious non-leaf images
    (e.g., faces, indoor rooms, walls, documents) before running the model.
    """
    hsv_img = pil_image.resize((128, 128)).convert("HSV")
    hsv_arr = np.array(hsv_img)
    h, s, v = hsv_arr[:, :, 0], hsv_arr[:, :, 1], hsv_arr[:, :, 2]

    # Plant foliage hues in PIL HSV (0-255): yellow-green to deep green
    foliage_mask = (h >= 28) & (h <= 115) & (s >= 35) & (v >= 30) & (v <= 245)
    foliage_ratio = np.mean(foliage_mask)
    return foliage_ratio >= min_foliage_ratio


# 6. Compact Header with Custom SVG Leaf Logo + Top-Right Model Info Popover
header_left, header_right = st.columns([5, 1], vertical_alignment="center")

with header_left:
    st.markdown(
        """
        <div class="app-header">
            <div class="app-logo">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2ECC71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/>
                    <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>
                </svg>
            </div>
            <h1 class="app-title">Plant Leaf Disease Detector</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Upload, capture, or select a Pepper, Potato, or Tomato leaf photo for instant AI disease detection."
    )

with header_right:
    with st.popover(":material/info: Model Info", use_container_width=True):
        st.markdown("**Model Architecture**")
        st.write(
            "MobileNetV2 (Transfer Learning) trained on the PlantVillage dataset (128×128 RGB)."
        )
        st.markdown("**Supported Crops (3)**")
        st.write("Bell Pepper, Potato, and Tomato (15 total classes).")
        st.markdown("**Detectable Conditions**")
        for cname in CLASS_NAMES:
            st.markdown(f"- {format_label(cname)}")

# 7. Auto-detect optional sample images from a "samples" folder
SAMPLES_DIR = "samples"
sample_files = []
if os.path.isdir(SAMPLES_DIR):
    sample_files = sorted(
        f
        for f in os.listdir(SAMPLES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

# 8. Top Input Row: Upload / Camera Tabs on the left, Sample Selector on the right
input_col1, input_col2 = st.columns([2, 1], gap="medium")

with input_col1:
    tab_upload, tab_camera = st.tabs(
        [":material/upload_file: Upload Image", ":material/photo_camera: Use Camera"]
    )
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload a leaf image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
    with tab_camera:
        camera_file = st.camera_input(
            "Take a photo of a leaf",
            label_visibility="collapsed",
        )

with input_col2:
    st.markdown("<div style='height: 44px;'></div>", unsafe_allow_html=True)
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

# 9. Decide which image to analyze (Upload > Camera > Sample)
image = None
image_caption = ""

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_caption = f"Uploaded: {uploaded_file.name}"
elif camera_file is not None:
    image = Image.open(camera_file).convert("RGB")
    image_caption = "Captured via Camera"
elif sample_files and selected_sample != "-- Select a sample leaf --":
    sample_path = os.path.join(SAMPLES_DIR, selected_sample)
    image = Image.open(sample_path).convert("RGB")
    image_caption = f"Sample: {selected_sample}"

# 10. Main Side-by-Side Result Area OR Empty-State 3-Step Guide
if image is not None:
    left_col, right_col = st.columns([1, 1.3], gap="large")

    # Left Card: Clean, reasonably sized image preview
    with left_col:
        with st.container(border=True):
            preview = image.copy()
            preview.thumbnail((340, 300))
            st.image(preview, caption=image_caption)

    # Right Card: Validate leaf image BEFORE running the disease model
    with right_col:
        with st.container(border=True):
            if not is_plant_leaf(image):
                st.markdown("#### Invalid Image Detected")
                st.warning(
                    "This image does not appear to be a plant leaf. Please upload or capture a clear, close-up photo of a **Pepper, Potato, or Tomato leaf**.",
                    icon=":material/error:",
                )
                st.caption(
                    "Tip: Make sure the leaf fills a clear portion of the frame with good lighting."
                )
            else:
                # Exact same preprocessing & prediction logic as before
                img_resized = image.resize((128, 128))
                img_array = np.array(img_resized, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=0)

                with st.spinner("Analyzing leaf..."):
                    preds = model.predict(img_array, verbose=0)[0]

                top_3_indices = np.argsort(preds)[-3:][::-1]
                best_raw_name = CLASS_NAMES[top_3_indices[0]]
                best_name = format_label(best_raw_name)
                best_conf = preds[top_3_indices[0]] * 100
                is_healthy = "healthy" in best_raw_name.lower()
                details = DISEASE_INFO.get(
                    best_raw_name,
                    {
                        "info": "No additional description available.",
                        "tip": "Consult a local agricultural extension for detailed guidance.",
                    },
                )

                st.markdown("#### Diagnosis Result")

                if is_healthy:
                    st.success(
                        f"**Healthy:** {best_name} ({best_conf:.2f}% confidence)",
                        icon=":material/check_circle:",
                    )
                else:
                    st.error(
                        f"**Disease Detected:** {best_name} ({best_conf:.2f}% confidence)",
                        icon=":material/warning:",
                    )

                if best_conf < 60:
                    st.warning(
                        "Low confidence (< 60%). For a more reliable result, try a clear close-up leaf photo on a plain background.",
                        icon=":material/info:",
                    )

                st.markdown("##### Top 3 Probabilities")
                top_3_lines = []
                for idx in top_3_indices:
                    clean_name = format_label(CLASS_NAMES[idx])
                    prob_pct = preds[idx] * 100
                    top_3_lines.append(f"- {clean_name}: {prob_pct:.2f}%")
                    st.markdown(f"**{clean_name}** ({prob_pct:.2f}%)")
                    st.progress(float(preds[idx]))

                with st.expander(
                    ":material/healing: Disease Insight & Care Tip", expanded=False
                ):
                    st.markdown(f"**About:** {details['info']}")
                    st.markdown(f"**Care Tip:** {details['tip']}")

                # Build downloadable .txt diagnosis summary
                report_text = (
                    f"Plant Leaf Disease Detection Report\n"
                    f"===================================\n"
                    f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Image: {image_caption}\n\n"
                    f"Primary Diagnosis: {best_name}\n"
                    f"Status: {'Healthy' if is_healthy else 'Disease Detected'}\n"
                    f"Confidence: {best_conf:.2f}%\n\n"
                    f"Top 3 Probabilities:\n"
                    + "\n".join(top_3_lines)
                    + f"\n\nAbout:\n{details['info']}\n\n"
                    f"Recommended Care Tip:\n{details['tip']}\n"
                )

                st.download_button(
                    label=":material/download: Download Diagnosis Report (.txt)",
                    data=report_text,
                    file_name=f"leaf_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
else:
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3, gap="medium")
    with g1:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-icon">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2ECC71" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="17 8 12 3 7 8"/>
                        <line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                </div>
                <h4>1. Upload or Capture</h4>
                <p>Upload a leaf photo, use your camera, or choose a sample image above.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with g2:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-icon">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2ECC71" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="4" y="4" width="16" height="16" rx="2"/>
                        <rect x="9" y="9" width="6" height="6"/>
                        <path d="M15 2v2"/><path d="M15 20v2"/>
                        <path d="M2 15h2"/><path d="M2 9h2"/>
                        <path d="M20 15h2"/><path d="M20 9h2"/>
                        <path d="M9 2v2"/><path d="M9 20v2"/>
                    </svg>
                </div>
                <h4>2. AI Analysis</h4>
                <p>MobileNetV2 scans leaf texture and spots across 15 plant disease classes.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with g3:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-icon">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2ECC71" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                        <path d="m9 12 2 2 4-4"/>
                    </svg>
                </div>
                <h4>3. Diagnosis & Care</h4>
                <p>View confidence scores, practical care tips, and download a summary report.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
