from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.image_reader import read_image_text
from src.review_engine import analyze_frame, analyze_review, load_model, load_reviews, save_model, train_model


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "reviews_dataset.csv"
MODEL = ROOT / "models" / "reviews_model.joblib"
METRICS = ROOT / "models" / "reviews_metrics.json"
STYLE = ROOT / "assets" / "style.css"


def apply_design() -> None:
    if STYLE.exists():
        st.markdown(f"<style>{STYLE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    if MODEL.exists():
        return load_model(MODEL)
    frame = load_reviews(DATASET)
    model, metrics = train_model(frame)
    save_model(model, MODEL)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return model


def get_metrics() -> dict:
    if METRICS.exists():
        return json.loads(METRICS.read_text(encoding="utf-8"))
    return {}


def draw_result(result: dict) -> None:
    sentiment = result["sentiment"]
    icon = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}.get(sentiment, "⚪")
    st.markdown('<div class="ra-card">', unsafe_allow_html=True)
    st.markdown(f"### {icon} {sentiment.capitalize()} review")
    first, second, third = st.columns(3)
    first.metric("Confidence", f"{result['confidence']:.1%}")
    second.metric("Estimated stars", result["estimated_stars"])
    third.metric("Urgency", result["urgency"])
    st.write("Probability distribution")
    chart = pd.DataFrame({"class": list(result["probabilities"].keys()), "score": list(result["probabilities"].values())})
    st.bar_chart(chart, x="class", y="score")
    st.write("Detected aspects")
    pills = "".join(f'<span class="ra-pill">{item}</span>' for item in result["aspects"])
    st.markdown(pills, unsafe_allow_html=True)
    if result["markers"]:
        st.write("Detected wording markers")
        st.json(result["markers"])
    st.markdown("</div>", unsafe_allow_html=True)


st.set_page_config(page_title="Reviews Analyzer", page_icon="🧭", layout="wide")
apply_design()

st.markdown(
    """
    <div class="ra-hero">
        <div class="ra-title">Reviews Analyzer</div>
        <div class="ra-subtitle">Tool for checking customer feedback from text, screenshots and CSV files.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

model = get_model()
metrics = get_metrics()

with st.sidebar:
    st.header("Analyzer settings")
    st.write("Word + character TF-IDF")
    st.write("Calibrated LinearSVC")
    if metrics:
        st.metric("Dataset size", metrics.get("dataset_size", "N/A"))
        st.metric("Accuracy", metrics.get("accuracy", "N/A"))
        st.metric("Macro F1", metrics.get("macro_f1", "N/A"))
    st.divider()
    tesseract_path = st.text_input("Tesseract path", value="")
    ocr_language = st.selectbox("OCR language", ["eng"])

text_tab, image_tab, batch_tab, info_tab = st.tabs(["Text", "Image OCR", "CSV batch", "Project info"])

with text_tab:
    st.markdown('<div class="ra-card">', unsafe_allow_html=True)
    st.subheader("Analyze one review")
    examples = {
        "Positive": "support replied in 10 minutes and actually fixed it",
        "Negative": "the app crashed twice and I lost my work",
        "Neutral": "the order was created yesterday and is still pending",
    }
    choice = st.selectbox("Choose an example", [""] + list(examples.keys()))
    review = st.text_area("Review text", value=examples.get(choice, ""), height=180)
    if st.button("Analyze text", type="primary"):
        try:
            draw_result(analyze_review(model, review))
        except Exception as error:
            st.error(str(error))
    st.markdown("</div>", unsafe_allow_html=True)

with image_tab:
    st.markdown('<div class="ra-card">', unsafe_allow_html=True)
    st.subheader("Analyze review from image")
    uploaded = st.file_uploader("Upload screenshot or photo", type=["png", "jpg", "jpeg", "webp"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded image", use_container_width=True)
        if st.button("Extract and analyze", type="primary"):
            try:
                extracted = read_image_text(image, language=ocr_language, custom_path=tesseract_path)
                st.text_area("Extracted text", value=extracted, height=140)
                draw_result(analyze_review(model, extracted))
            except Exception as error:
                st.error(str(error))
    st.markdown("</div>", unsafe_allow_html=True)

with batch_tab:
    st.markdown('<div class="ra-card">', unsafe_allow_html=True)
    st.subheader("Analyze many reviews")
    uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_csv:
        frame = pd.read_csv(uploaded_csv)
        st.dataframe(frame.head(10), use_container_width=True)
        column = st.selectbox("Column with reviews", list(frame.columns))
        if st.button("Analyze CSV", type="primary"):
            try:
                result = analyze_frame(model, frame, column)
                st.dataframe(result, use_container_width=True)
                st.download_button("Download analysis", result.to_csv(index=False).encode("utf-8"), "reviews_analysis.csv", "text/csv")
            except Exception as error:
                st.error(str(error))
    st.markdown("</div>", unsafe_allow_html=True)

with info_tab:
    st.markdown('<div class="ra-card">', unsafe_allow_html=True)
    st.subheader("About Reviews Analyzer")
    st.write("Reviews Analyzer helps process customer feedback from three sources: manually entered text, images with review text and CSV files with multiple reviews.")
    st.write("For each review, the system determines sentiment, shows confidence, estimates a star rating, identifies topics such as delivery, support, price, app quality or payment, and highlights negative feedback that may require attention.")
    st.write("The project combines a trained text classifier with additional rule-based analysis, so the result is not limited to a simple positive or negative label.")
    st.markdown("</div>", unsafe_allow_html=True)
