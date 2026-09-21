"""Interface Streamlit du démonstrateur de détection NB6."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from config import (
    APP_SUBTITLE,
    APP_TITLE,
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIDENCE,
    DEFAULT_SMALL_OBJECT_RATIO,
    DISPLAY_CLASSES,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    IOU_DISPLAY_VALUE,
    VALIDATION_METRICS,
)
from checkpoint_files import checkpoint_parts, resolve_checkpoint
from inference import filter_detections, run_inference
from model import load_model
from visualization import detections_dataframe, draw_detections, image_to_png_bytes


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(145deg, #08111f 0%, #0d1728 55%, #101b2d 100%);}
    [data-testid="stSidebar"] {background: #0a1322; border-right: 1px solid #22314a;}
    .hero {padding: 1.4rem 1.6rem; border: 1px solid #263954; border-radius: 18px;
           background: linear-gradient(120deg, rgba(20,42,70,.95), rgba(13,27,47,.92));
           margin-bottom: 1.2rem; box-shadow: 0 12px 36px rgba(0,0,0,.22);}
    .hero h1 {margin: 0; color: #f3f7ff; font-size: 2.15rem;}
    .hero h3 {margin: .35rem 0 .6rem; color: #69b7ff; font-weight: 500;}
    .hero p {margin: 0; color: #b8c6da;}
    .notice {padding: .8rem 1rem; border-radius: 10px; background: rgba(255,193,7,.12);
             border-left: 4px solid #ffc107; color: #f5df96;}
    div[data-testid="stMetric"] {background: rgba(21,35,56,.85); border: 1px solid #273a56;
                                 padding: .8rem; border-radius: 12px;}
    .stButton > button {width: 100%; border-radius: 10px; font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.75rem;margin-bottom:1.1rem">
      <div style="padding:.85rem;border:1px solid #263954;border-radius:12px;background:#101c2e">
        <b style="color:#69b7ff">1 · Upload</b><br><span style="color:#aebbd0">Choose a road-scene image</span>
      </div>
      <div style="padding:.85rem;border:1px solid #263954;border-radius:12px;background:#101c2e">
        <b style="color:#69b7ff">2 · Detect</b><br><span style="color:#aebbd0">Run the trained perception model</span>
      </div>
      <div style="padding:.85rem;border:1px solid #263954;border-radius:12px;background:#101c2e">
        <b style="color:#69b7ff">3 · Explore</b><br><span style="color:#aebbd0">Filter, inspect and download results</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def load_model_cached(
    checkpoint_path: str,
    device_name: str,
    checkpoint_mtime_ns: int,
):
    del checkpoint_mtime_ns  # utilisé uniquement pour invalider le cache
    return load_model(checkpoint_path, torch.device(device_name))


def count_class(detections: list[dict], class_name: str) -> int:
    return sum(item["class_name"] == class_name for item in detections)


def average_confidence(detections: list[dict], class_name: str) -> float:
    values = [
        item["confidence"]
        for item in detections
        if item["class_name"] == class_name
    ]
    return sum(values) / len(values) if values else 0.0


device = choose_device()
checkpoint_source = Path(DEFAULT_CHECKPOINT)
checkpoint_path = resolve_checkpoint(checkpoint_source)


def checkpoint_status(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "Checkpoint absent"
    try:
        with path.open("rb") as checkpoint_file:
            prefix = checkpoint_file.read(80)
    except OSError as exc:
        return False, f"Checkpoint illisible : {exc}"
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return False, "Pointeur Git LFS détecté à la place du modèle"
    if path.stat().st_size < 1_000_000:
        return False, "Checkpoint anormalement petit"
    return True, f"Checkpoint disponible ({path.stat().st_size / 1024**2:.1f} MiB)"


checkpoint_ready, checkpoint_message = checkpoint_status(checkpoint_path)
if checkpoint_ready and checkpoint_parts(checkpoint_source):
    checkpoint_message = (
        f"Checkpoint reconstitué depuis {len(checkpoint_parts(checkpoint_source))} morceaux "
        f"({checkpoint_path.stat().st_size / 1024**2:.1f} MiB)"
    )

with st.sidebar:
    st.header("Model Configuration")
    st.subheader("Model Information")
    st.markdown(
        """
        **Architecture:** Faster R-CNN  
        **Backbone:** ViT-S/16  
        **Pretraining:** Driving-Aware I-JEPA  
        **Dataset:** KITTI  
        **Input:** 1024 × 320  
        **Classes:** Pedestrian, Cyclist, Car, Van
        """
    )
    st.divider()
    st.subheader("Detection Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=DEFAULT_CONFIDENCE,
        step=0.05,
    )
    selected_classes = st.multiselect(
        "Classes to display",
        options=DISPLAY_CLASSES,
        default=DISPLAY_CLASSES,
    )
    small_ratio_percent = st.slider(
        "Small-object area threshold (%)",
        min_value=0.1,
        max_value=5.0,
        value=DEFAULT_SMALL_OBJECT_RATIO * 100,
        step=0.1,
        help="Une box sous ce pourcentage de la surface de l'image est considérée petite.",
    )
    small_ratio = small_ratio_percent / 100.0
    st.caption(f"IoU de référence : {IOU_DISPLAY_VALUE:.2f}")
    st.caption(f"Device : {'CUDA' if device.type == 'cuda' else 'CPU'}")
    st.caption(f"Checkpoint : {checkpoint_path.name}")
    if checkpoint_ready:
        st.success(checkpoint_message)
    else:
        st.warning(checkpoint_message)

st.markdown(
    f"""
    <div class="hero">
      <h1>{APP_TITLE}</h1>
      <h3>{APP_SUBTITLE}</h3>
      <p>Upload a road scene to analyze pedestrians, cyclists and vehicles using the trained AI perception model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a road scene",
    type=["jpg", "jpeg", "png"],
    help="Formats acceptés : JPG, JPEG et PNG.",
)

if uploaded_file is None:
    st.info(
        "Start by uploading a KITTI image or another road scene. "
        "The model will look for pedestrians, cyclists, cars and vans."
    )

image = None
image_bytes = None
image_digest = None
if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.getvalue()
        image_digest = hashlib.sha256(image_bytes).hexdigest()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Image invalide ou illisible : {exc}")

if image_digest and st.session_state.get("image_digest") != image_digest:
    st.session_state.pop("inference_result", None)
    st.session_state["image_digest"] = image_digest

run_clicked = st.button(
    "Run AI Detection",
    type="primary",
    disabled=image is None,
    use_container_width=True,
)

if run_clicked and image is not None:
    if not checkpoint_ready:
        st.error(
            f"{checkpoint_message}. Le fichier réel doit être placé dans "
            "`models/best_detector.pt`, puis l'application doit être redémarrée."
        )
    else:
        try:
            with st.spinner("Loading model and running AI inference..."):
                model = load_model_cached(
                    str(checkpoint_path),
                    device.type,
                    checkpoint_path.stat().st_mtime_ns,
                )
                st.session_state["inference_result"] = run_inference(
                    model, image, device
                )
            st.success("Inference completed successfully")
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                st.error(
                    "Mémoire insuffisante pendant l'inférence. Fermez les autres "
                    "applications GPU ou exécutez la démonstration sur une machine plus puissante."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                st.error(f"Erreur pendant l'inférence : {exc}")
        except Exception as exc:  # Streamlit doit rester accessible en démonstration
            st.error(f"Impossible d'exécuter l'inférence : {exc}")

result = st.session_state.get("inference_result")

if image is not None:
    filtered = []
    annotated = image
    if result:
        filtered = filter_detections(
            result["detections"], confidence_threshold, selected_classes
        )
        annotated = draw_detections(image, filtered)

    left, right = st.columns(2)
    with left:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    with right:
        st.subheader("AI Detection")
        st.image(annotated, use_container_width=True)

    if result:
        if not filtered:
            st.info("No objects detected above the selected confidence threshold.")

        st.subheader("Detection Summary")
        metric_columns = st.columns(5)
        summary = [
            ("Total Objects", len(filtered)),
            ("Pedestrians", count_class(filtered, "Pedestrian")),
            ("Cyclists", count_class(filtered, "Cyclist")),
            ("Cars", count_class(filtered, "Car")),
            ("Vans", count_class(filtered, "Van")),
        ]
        for column, (label, value) in zip(metric_columns, summary):
            column.metric(label, value)

        st.subheader("Inference Performance")
        performance_columns = st.columns(4)
        performance_columns[0].metric(
            "Inference Time", f"{result['inference_seconds'] * 1000:.0f} ms"
        )
        performance_columns[1].metric("FPS", f"{result['fps']:.2f}")
        performance_columns[2].metric("Device", result["device"])
        performance_columns[3].metric(
            "Image Resolution",
            f"{result['original_width']} × {result['original_height']}",
        )

        st.subheader("Vulnerable Road Users Analysis")
        pedestrians = count_class(filtered, "Pedestrian")
        cyclists = count_class(filtered, "Cyclist")
        vru_columns = st.columns(5)
        vru_columns[0].metric("Pedestrians", pedestrians)
        vru_columns[1].metric("Cyclists", cyclists)
        vru_columns[2].metric(
            "Avg. pedestrian confidence",
            f"{average_confidence(filtered, 'Pedestrian') * 100:.1f}%",
        )
        vru_columns[3].metric(
            "Avg. cyclist confidence",
            f"{average_confidence(filtered, 'Cyclist') * 100:.1f}%",
        )
        small_pedestrians = sum(
            item["class_name"] == "Pedestrian"
            and item["area_ratio"] < small_ratio
            for item in filtered
        )
        vru_columns[4].metric("Small pedestrians", small_pedestrians)
        if pedestrians or cyclists:
            st.markdown(
                '<div class="notice"><b>Vulnerable road users detected in the scene.</b><br>'
                "This is a model observation for demonstration purposes, not a driving decision.</div>",
                unsafe_allow_html=True,
            )

        st.subheader("Small Object Analysis")
        small_detections = [
            item for item in filtered if item["area_ratio"] < small_ratio
        ]
        small_columns = st.columns(4)
        small_columns[0].metric("Small objects", len(small_detections))
        small_columns[1].metric(
            "Small pedestrians", count_class(small_detections, "Pedestrian")
        )
        small_columns[2].metric(
            "Small cyclists", count_class(small_detections, "Cyclist")
        )
        average_small = (
            sum(item["confidence"] for item in small_detections)
            / len(small_detections)
            if small_detections
            else 0.0
        )
        small_columns[3].metric(
            "Avg. small confidence", f"{average_small * 100:.1f}%"
        )

        st.subheader("Detection Table")
        dataframe = detections_dataframe(filtered, small_ratio)
        if dataframe.empty:
            st.caption("Aucune ligne à afficher avec les filtres actuels.")
        else:
            st.dataframe(
                dataframe,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence", min_value=0.0, max_value=1.0, format="%.3f"
                    )
                },
            )
            st.download_button(
                "Download detection table (CSV)",
                data=dataframe.to_csv(index=False).encode("utf-8"),
                file_name="detections.csv",
                mime="text/csv",
            )
        st.download_button(
            "Download annotated image",
            data=image_to_png_bytes(annotated),
            file_name="ai_detection.png",
            mime="image/png",
        )

with st.expander("Model Information"):
    st.markdown(
        f"""
        - **Model:** Faster R-CNN
        - **Backbone:** ViT-S/16
        - **Feature Pyramid:** Simple Feature Pyramid, 5 levels
        - **SSL Pretraining:** Driving-Aware I-JEPA
        - **Dataset:** KITTI
        - **Detection Resolution:** {INPUT_WIDTH} × {INPUT_HEIGHT}
        - **Checkpoint:** best_detector.pt
        """
    )

with st.expander("Validation Performance", expanded=False):
    metrics_table = pd.DataFrame.from_dict(VALIDATION_METRICS, orient="index")
    metrics_table.index.name = "Class"
    st.caption(
        "Metrics obtained on the 1,497-image KITTI validation split. "
        "They are not recalculated from the uploaded image."
    )
    st.dataframe(metrics_table.style.format("{:.4f}"), use_container_width=True)

st.caption(
    "Academic PFE demonstrator — Predictions must not be interpreted as real-world driving commands."
)
