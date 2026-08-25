"""
app.py — Plant Disease Detection Streamlit Web Application

Run locally:
    streamlit run app.py

Requires:
    - Trained model at models/plant_disease_resnet50.pth
    - Class names at models/class_names.json
    - disease_info.json in project root
"""

import json
from pathlib import Path

import streamlit as st
from PIL import Image

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="🌿 PlantGuard — Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hero section ── */
.hero-wrapper {
    text-align: center;
    padding: 3rem 1rem 2rem;
    animation: fadeInDown 0.7s ease;
}
.hero-icon {
    font-size: 4rem;
    animation: pulse 2.5s infinite;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00d4aa 0%, #00aaff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.4rem 0;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #94a3b8;
    font-weight: 300;
    max-width: 560px;
    margin: 0 auto;
}

/* ── Upload zone ── */
.upload-hint {
    text-align: center;
    color: #64748b;
    font-size: 0.9rem;
    margin-top: -0.5rem;
    margin-bottom: 1.5rem;
}

/* ── Result badge ── */
.badge-healthy {
    display: inline-block;
    background: linear-gradient(135deg, #00d4aa22, #00d4aa44);
    border: 1px solid #00d4aa;
    color: #00d4aa;
    border-radius: 100px;
    padding: 0.4rem 1.2rem;
    font-weight: 600;
    font-size: 1rem;
    animation: fadeIn 0.5s ease;
}
.badge-diseased {
    display: inline-block;
    background: linear-gradient(135deg, #f5793522, #f5793544);
    border: 1px solid #f57935;
    color: #f57935;
    border-radius: 100px;
    padding: 0.4rem 1.2rem;
    font-weight: 600;
    font-size: 1rem;
    animation: fadeIn 0.5s ease;
}
.badge-critical {
    display: inline-block;
    background: linear-gradient(135deg, #ef444422, #ef444444);
    border: 1px solid #ef4444;
    color: #ef4444;
    border-radius: 100px;
    padding: 0.4rem 1.2rem;
    font-weight: 600;
    font-size: 1rem;
    animation: fadeIn 0.5s ease;
}

/* ── Confidence score ── */
.confidence-big {
    font-size: 2.4rem;
    font-weight: 700;
    color: #00d4aa;
    line-height: 1;
}
.confidence-label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Info card ── */
.info-card {
    background: linear-gradient(135deg, #0f1829, #131f30);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    animation: fadeIn 0.6s ease;
}
.info-card h4 {
    color: #00d4aa;
    margin-bottom: 0.5rem;
    font-size: 1rem;
}
.info-card p {
    color: #cbd5e1;
    font-size: 0.93rem;
    line-height: 1.6;
    margin: 0;
}

/* ── Section headers ── */
.section-header {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #475569;
    margin-bottom: 0.6rem;
}

/* ── Top-K bar ── */
.topk-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 0.4rem 0;
    animation: fadeIn 0.5s ease;
}
.topk-label {
    font-size: 0.82rem;
    color: #94a3b8;
    min-width: 200px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.topk-pct {
    font-size: 0.82rem;
    color: #00d4aa;
    font-weight: 600;
    min-width: 44px;
    text-align: right;
}

/* ── Severity pill ── */
.pill-none     { color: #00d4aa; background: #00d4aa18; border: 1px solid #00d4aa40; border-radius: 8px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; }
.pill-medium   { color: #f59e0b; background: #f59e0b18; border: 1px solid #f59e0b40; border-radius: 8px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; }
.pill-high     { color: #f57935; background: #f5793518; border: 1px solid #f5793540; border-radius: 8px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; }
.pill-critical { color: #ef4444; background: #ef444418; border: 1px solid #ef444440; border-radius: 8px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: #334155;
    font-size: 0.78rem;
    padding: 2rem 0 1rem;
    border-top: 1px solid #1e293b;
    margin-top: 3rem;
}

/* ── Animations ── */
@keyframes fadeIn      { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
@keyframes fadeInDown  { from { opacity: 0; transform: translateY(-16px); } to { opacity: 1; transform: translateY(0); } }
@keyframes pulse       { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.08); } }

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_resources():
    """Load model and class names once and cache across sessions."""
    from src.model import load_model, get_class_names

    model_path      = Path("models/plant_disease_resnet50.pth")
    class_names_path = Path("models/class_names.json")

    if not model_path.exists():
        return None, None

    model       = load_model(model_path)
    class_names = get_class_names(class_names_path)
    return model, class_names


@st.cache_data
def load_disease_info() -> dict:
    with open("disease_info.json") as f:
        return json.load(f)


def severity_badge(severity: str) -> str:
    labels = {"none": "✅ Healthy", "medium": "⚠️ Medium", "high": "🔶 High", "critical": "🔴 Critical"}
    return f'<span class="pill-{severity}">{labels.get(severity, severity)}</span>'


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-icon">🌿</div>
    <h1 class="hero-title">PlantGuard</h1>
    <p class="hero-subtitle">
        AI-powered plant disease detection using ResNet50 + Grad-CAM.<br>
        Upload a leaf photo and get an instant diagnosis.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Load resources ─────────────────────────────────────────────────────────────
model, class_names = load_resources()
disease_info       = load_disease_info()

MODEL_READY = model is not None

if not MODEL_READY:
    st.info(
        "🔧 **Model not found.** Train the model first (Phase 4) and save it to "
        "`models/plant_disease_resnet50.pth`. The app UI is fully functional.",
        icon="ℹ️",
    )

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown("---")
col_up, col_info = st.columns([1.2, 1], gap="large")

with col_up:
    st.markdown('<p class="section-header">📸 Upload Leaf Image</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        key="leaf_uploader",
    )
    st.markdown('<p class="upload-hint">Supports JPG · PNG · WEBP · Max 200 MB</p>', unsafe_allow_html=True)

with col_info:
    st.markdown('<p class="section-header">ℹ️ How To Use</p>', unsafe_allow_html=True)
    st.markdown("""
    1. **Upload** a clear photo of a single plant leaf
    2. The model will **classify** the disease (or confirm it's healthy)
    3. A **Grad-CAM heatmap** shows which leaf regions influenced the decision
    4. Read the **treatment recommendation** to take action
    """)

# ── Inference & Results ───────────────────────────────────────────────────────
if uploaded is not None:
    pil_image = Image.open(uploaded).convert("RGB")

    st.markdown("---")
    st.markdown('<p class="section-header">🔬 Analysis Results</p>', unsafe_allow_html=True)

    if MODEL_READY:
        from src.model import predict
        from src.gradcam import GradCAM
        import torch

        with st.spinner("Analysing leaf…"):
            result     = predict(model, class_names, pil_image)
            cam        = GradCAM(model)
            class_idx  = class_names.index(result["class_raw"])
            heatmap    = cam.generate(result["image_tensor"], class_idx=class_idx)
            overlay    = cam.overlay(pil_image, heatmap)

        # ── Image columns ───────────────────────────────────────────────────
        img_col, grad_col = st.columns(2, gap="medium")
        with img_col:
            st.markdown('<p class="section-header">Original Image</p>', unsafe_allow_html=True)
            st.image(pil_image, use_container_width=True)
        with grad_col:
            st.markdown('<p class="section-header">Grad-CAM Heatmap</p>', unsafe_allow_html=True)
            st.image(overlay, use_container_width=True, caption="Red = high attention | Blue = low attention")

        # ── Main result ─────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        res_col, conf_col = st.columns([2.5, 1], gap="large")

        with res_col:
            severity = disease_info.get(result["class_raw"], {}).get("severity", "medium")
            badge_class = "badge-healthy" if result["is_healthy"] else (
                "badge-critical" if severity == "critical" else "badge-diseased"
            )
            st.markdown(
                f'<span class="{badge_class}">{result["class_name"]}</span>'
                f'&nbsp;&nbsp;{severity_badge(severity)}',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            # Disease info card
            info = disease_info.get(result["class_raw"])
            if info:
                st.markdown(f"""
                <div class="info-card">
                    <h4>📋 Description</h4>
                    <p>{info["description"]}</p>
                </div>
                <div class="info-card">
                    <h4>💊 Treatment Recommendation</h4>
                    <p>{info["treatment"]}</p>
                </div>
                """, unsafe_allow_html=True)

        with conf_col:
            st.markdown('<p class="section-header">Confidence</p>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="confidence-big">{result["confidence"]*100:.1f}%</p>'
                f'<p class="confidence-label">Top prediction</p>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            # Top-K predictions
            st.markdown('<p class="section-header">Top 5 Predictions</p>', unsafe_allow_html=True)
            for i, item in enumerate(result["top_k"]):
                pct = item["confidence"] * 100
                st.progress(item["confidence"], text=f"{item['class_name']} — {pct:.1f}%")

    else:
        # Model not trained yet — show image only
        st.image(pil_image, caption="Uploaded image (model not yet trained)", width=400)
        st.warning("Train the model first to get predictions.", icon="⚠️")

# ── Accordions ────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🧠 How It Works"):
    st.markdown("""
    **PlantGuard** uses a deep learning pipeline:

    1. **Input preprocessing** — Leaf image is resized to 224×224 and normalised with ImageNet statistics
    2. **ResNet50 feature extraction** — A 50-layer residual network extracts visual features
    3. **Transfer learning** — The model was pretrained on ImageNet (1.2M images), then fine-tuned on PlantVillage (87K leaf images across 38 classes)
    4. **Classification** — A custom fully-connected head maps features to 38 disease categories
    5. **Grad-CAM** — Gradient-weighted Class Activation Mapping hooks into the final convolutional layer to highlight the leaf regions that most influenced the prediction

    **Dataset**: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) — 87,000 images, 38 classes

    **Model accuracy**: ~98-99% on the PlantVillage test set
    """)

with st.expander("🌱 Supported Plants & Diseases (38 classes)"):
    grouped = {}
    for raw_key, info in disease_info.items():
        plant = raw_key.split("___")[0].replace("_", " ").replace(",", "").strip()
        grouped.setdefault(plant, []).append(info["display"])

    cols = st.columns(3)
    for i, (plant, diseases) in enumerate(sorted(grouped.items())):
        with cols[i % 3]:
            st.markdown(f"**{plant}**")
            for d in diseases:
                label = d.split("—")[-1].strip() if "—" in d else d
                st.markdown(f"- {label}")
            st.markdown("")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with PyTorch · ResNet50 · Grad-CAM · Streamlit &nbsp;|&nbsp;
    Dataset: PlantVillage (87K images, 38 classes) &nbsp;|&nbsp;
    <a href="https://github.com" style="color:#00d4aa; text-decoration:none;">GitHub</a>
</div>
""", unsafe_allow_html=True)
