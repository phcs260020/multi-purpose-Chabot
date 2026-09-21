"""
Interactive Multi-Purpose Chatbot with Image Recognition
=======================================================
Streamlit app that accepts text and image inputs.
Uses a pre-trained YOLOv8 model (COCO) for object detection
and a conversational layer that answers questions about the
detected objects while also supporting general dialogue.
"""

from __future__ import annotations

import hashlib
import io
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths – resolve model relative to this script so it works from any cwd
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = APP_DIR / "yolov8n.pt"

# Simple synonym map so questions like "how many people?" match COCO "person"
SYNONYMS: Dict[str, str] = {
    "people": "person",
    "persons": "person",
    "man": "person",
    "woman": "person",
    "men": "person",
    "women": "person",
    "bike": "bicycle",
    "bikes": "bicycle",
    "motorbike": "motorcycle",
    "motorbikes": "motorcycle",
    "aeroplane": "airplane",
    "plane": "airplane",
    "planes": "airplane",
    "tv": "tv",
    "television": "tv",
    "mobile": "cell phone",
    "phone": "cell phone",
    "cellphone": "cell phone",
    "mobiles": "cell phone",
    "phones": "cell phone",
}

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VisionChat – Image Recognition Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Model loading (cached by model path string)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLOv8 model…")
def load_model(model_path: str) -> YOLO:
    """Load a pre-trained YOLOv8 model. Cached so it loads only once per path."""
    return YOLO(model_path)


# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------
def run_detection(
    image: Image.Image,
    model: YOLO,
    conf_threshold: float = 0.25,
) -> Tuple[Image.Image, List[Dict]]:
    """
    Run object detection on a PIL image.
    Returns the annotated image (RGB) and a list of detection dictionaries.
    Uses only NumPy + PIL (no direct OpenCV calls) for Streamlit Cloud compatibility.
    """
    # YOLO accepts RGB numpy arrays
    img_array = np.array(image.convert("RGB"))

    results = model.predict(img_array, conf=conf_threshold, verbose=False)
    result = results[0]

    # result.plot() returns BGR; flip channels to RGB without cv2
    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1].copy()
    annotated_pil = Image.fromarray(annotated_rgb)

    detections: List[Dict] = []
    if result.boxes is not None and len(result.boxes) > 0:
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls_id]
            xyxy = [float(x) for x in box.xyxy[0].tolist()]
            detections.append(
                {
                    "label": label,
                    "confidence": conf,
                    "bbox": xyxy,
                }
            )
    return annotated_pil, detections


def summarize_detections(detections: List[Dict]) -> str:
    """Create a human-readable summary of detected objects."""
    if not detections:
        return (
            "I couldn't detect any objects in the image with the current "
            "confidence threshold. Try lowering the threshold in the sidebar."
        )

    counts = Counter(d["label"] for d in detections)
    parts = []
    for label, count in counts.most_common():
        confs = [d["confidence"] for d in detections if d["label"] == label]
        avg_conf = sum(confs) / len(confs)
        if count == 1:
            parts.append(f"1 **{label}** (confidence ≈ {avg_conf:.0%})")
        else:
            parts.append(f"{count} **{label}s** (avg confidence ≈ {avg_conf:.0%})")

    return "I detected the following objects:\n• " + "\n• ".join(parts)


def resolve_label(query: str, available_labels: List[str]) -> Optional[str]:
    """
    Map a user query word to a detected COCO label using exact match,
    synonyms, and simple substring checks.
    """
    q = query.lower().strip().rstrip("s")
    # Synonym first
    if q in SYNONYMS:
        candidate = SYNONYMS[q]
        if candidate in available_labels:
            return candidate
    # Exact / substring against available labels
    for lab in available_labels:
        if q == lab or q in lab or lab in q:
            return lab
    # Try original (with trailing s) against synonyms
    q2 = query.lower().strip()
    if q2 in SYNONYMS:
        candidate = SYNONYMS[q2]
        if candidate in available_labels:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Conversational response logic (rule-based + context-aware)
# ---------------------------------------------------------------------------
def generate_response(
    user_text: str,
    detections: Optional[List[Dict]] = None,
) -> str:
    """
    Generate a response based on the user message and current image context.
    Handles both image-related questions and general conversation.
    """
    text = user_text.lower().strip()
    detections = detections or []
    labels = [d["label"] for d in detections]
    unique_labels = sorted(set(labels))
    counts = Counter(labels)

    # --- Greetings ---
    greetings = ("hello", "hi", "hey", "good morning", "good afternoon", "good evening", "hola")
    if any(re.search(rf"\b{re.escape(g)}\b", text) for g in greetings):
        return (
            "Hello! 👋 I'm **VisionChat**. Upload an image and I'll identify the "
            "objects in it, or just ask me anything about what I've seen."
        )

    # --- Help ---
    if any(w in text for w in ("help", "what can you do", "capabilities", "how to use", "commands")):
        return (
            "I can:\n"
            "1. **Detect objects** in any image you upload (YOLOv8 trained on COCO – 80 classes).\n"
            "2. **Answer questions** about the detected objects, e.g.:\n"
            "   - “How many people?” / “How many cars?”\n"
            "   - “Is there a dog?”\n"
            "   - “List everything you see” / “Describe the image”\n"
            "   - “What is the most confident detection?”\n"
            "3. Hold a light conversation.\n\n"
            "Upload an image from the **sidebar**, then ask away!"
        )

    if any(w in text for w in ("thank", "thanks", "appreciate")):
        return "You're welcome! Feel free to upload another image or ask more questions. 😊"

    if any(w in text for w in ("bye", "goodbye", "see you", "exit", "quit")):
        return "Goodbye! Come back anytime you want to analyze more images. 👋"

    # --- No image yet ---
    if not detections:
        if any(
            w in text
            for w in (
                "what",
                "object",
                "detect",
                "see",
                "image",
                "picture",
                "photo",
                "how many",
                "is there",
                "list",
                "describe",
            )
        ):
            return (
                "I don't have an image to analyze yet. Please upload one using the "
                "**sidebar** uploader, and I'll identify the objects for you."
            )
        return (
            "I'm mainly designed to analyze images and answer questions about the "
            "objects I detect. Upload an image and try asking “What do you see?” "
            "or “How many people are there?”. You can also say **help**."
        )

    # --- Count questions: "how many X" ---
    count_match = re.search(r"how many\s+([a-zA-Z\s\-]+?)(?:\s+are|\s+is|\s+do|\?|$)", text)
    if count_match:
        target_raw = count_match.group(1).strip()
        matched = resolve_label(target_raw, unique_labels)
        if matched:
            n = counts[matched]
            plural = "s" if n != 1 and not matched.endswith("s") else ""
            return f"I found **{n}** {matched}{plural} in the image."
        return (
            f"I didn't detect any '{target_raw}' in the current image. "
            f"Detected objects: {', '.join(unique_labels) or 'none'}."
        )

    # --- Existence questions ---
    if any(
        phrase in text
        for phrase in (
            "is there",
            "are there",
            "do you see",
            "can you see",
            "is a ",
            "is an ",
            "any ",
        )
    ):
        # Try to extract the object word after the phrase
        for lab in unique_labels:
            if lab in text or lab.rstrip("s") in text:
                n = counts[lab]
                plural = "s" if n != 1 and not lab.endswith("s") else ""
                return f"Yes – I detected **{n}** {lab}{plural}."
        # Also try synonyms present in the question
        for syn, canonical in SYNONYMS.items():
            if syn in text and canonical in unique_labels:
                n = counts[canonical]
                plural = "s" if n != 1 else ""
                return f"Yes – I detected **{n}** {canonical}{plural}."
        return (
            f"No, I don't see that. The objects I found are: "
            f"{', '.join(unique_labels) or 'none'}."
        )

    # --- List / describe ---
    if any(
        w in text
        for w in (
            "list",
            "what objects",
            "what do you see",
            "describe",
            "what's in",
            "what is in",
            "what are in",
            "show me",
            "tell me about the image",
            "what did you find",
            "summary",
        )
    ):
        return summarize_detections(detections)

    # --- Confidence details ---
    if any(w in text for w in ("confidence", "how sure", "probability", "scores")):
        details = "\n".join(
            f"• **{d['label']}**: {d['confidence']:.1%}"
            for d in sorted(detections, key=lambda x: -x["confidence"])
        )
        return "Here are the detection confidences (highest first):\n" + details

    # --- Most confident / main object ---
    if any(
        w in text
        for w in (
            "main object",
            "most confident",
            "primary",
            "biggest",
            "top detection",
            "highest confidence",
        )
    ):
        top = max(detections, key=lambda d: d["confidence"])
        return (
            f"The most confident detection is **{top['label']}** "
            f"({top['confidence']:.1%})."
        )

    # --- Generic fallback when we have detections ---
    return (
        f"Based on the current image I can see: **{', '.join(unique_labels)}**.\n\n"
        "You can ask me things like:\n"
        "- “How many cars?”\n"
        "- “Is there a person?”\n"
        "- “List all objects”\n"
        "- “What is the most confident detection?”"
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    defaults = {
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "👋 Hi! I'm **VisionChat**, your image-recognition chatbot.\n\n"
                    "Upload an image in the **sidebar** and I'll identify the objects in it. "
                    "Then you can ask me questions about what I found – or just chat!\n\n"
                    "Type **help** to see what I can do."
                ),
            }
        ],
        "detections": [],
        "annotated_image": None,
        "original_image": None,
        "_last_file_hash": None,
        "_last_conf": None,
        "_last_model": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_session() -> None:
    """Reset chat and image state completely."""
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Chat cleared. Upload a new image or ask me something!",
        }
    ]
    st.session_state.detections = []
    st.session_state.annotated_image = None
    st.session_state.original_image = None
    st.session_state._last_file_hash = None
    st.session_state._last_conf = None
    st.session_state._last_model = None


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main() -> None:
    init_session_state()

    # ----- Sidebar -----
    with st.sidebar:
        st.title("⚙️ Controls")
        st.markdown("---")

        model_choice = st.selectbox(
            "YOLO model",
            options=["yolov8n.pt", "yolov8s.pt"],
            index=0,
            help="nano = fastest · small = more accurate (downloads on first use)",
        )
        conf_threshold = st.slider(
            "Confidence threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.25,
            step=0.05,
            help="Higher values show only more confident detections",
        )

        st.markdown("---")
        st.subheader("📷 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image…",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="sidebar_uploader",
        )

        if st.button("🔄 Re-detect (same image)", use_container_width=True):
            # Force re-run by clearing the hash so the next block reprocesses
            st.session_state._last_file_hash = None
            st.session_state._last_conf = None
            st.rerun()

        if st.button("🗑️ Clear chat & image", use_container_width=True):
            clear_session()
            st.rerun()

        st.markdown("---")
        st.caption(
            "Powered by **YOLOv8** (Ultralytics) · COCO 80 classes · "
            "Built with Streamlit"
        )

    # ----- Resolve model path -----
    if model_choice == "yolov8n.pt" and DEFAULT_MODEL.exists():
        model_path = str(DEFAULT_MODEL)
    else:
        # Let ultralytics download / cache other variants
        model_path = model_choice

    model = load_model(model_path)

    # ----- Process uploaded image -----
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        need_detect = (
            st.session_state._last_file_hash != file_hash
            or st.session_state._last_conf != conf_threshold
            or st.session_state._last_model != model_path
        )

        if need_detect:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            with st.spinner("Detecting objects…"):
                annotated, detections = run_detection(image, model, conf_threshold)

            st.session_state.original_image = image
            st.session_state.annotated_image = annotated
            st.session_state.detections = detections
            st.session_state._last_file_hash = file_hash
            st.session_state._last_conf = conf_threshold
            st.session_state._last_model = model_path

            summary = summarize_detections(detections)
            # Avoid spamming the same analysis message on every slider tweak
            # by only appending when the file itself changed
            if st.session_state._last_file_hash == file_hash:
                # Remove previous "Image analyzed" messages for cleaner history
                st.session_state.messages = [
                    m
                    for m in st.session_state.messages
                    if not (
                        m.get("role") == "assistant"
                        and m.get("content", "").startswith("✅ **Image analyzed!**")
                    )
                ]
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"✅ **Image analyzed!**\n\n{summary}",
                        "show_image": True,
                    }
                )

    # ----- Main chat area -----
    st.title("🤖 VisionChat")
    st.caption("Upload an image → I identify objects → Ask me questions about them")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("show_image") and st.session_state.annotated_image is not None:
                st.image(
                    st.session_state.annotated_image,
                    caption="Detected objects",
                    use_container_width=True,
                )

    # Chat input
    if prompt := st.chat_input("Ask about the image or just chat…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = generate_response(
            prompt,
            detections=st.session_state.detections,
        )
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    # Current detection details expander
    if st.session_state.detections:
        with st.expander("📋 Current detection details", expanded=False):
            for i, d in enumerate(st.session_state.detections, 1):
                bbox = [round(x) for x in d["bbox"]]
                st.write(
                    f"{i}. **{d['label']}** – {d['confidence']:.1%}  "
                    f"(bbox: {bbox})"
                )


if __name__ == "__main__":
    main()
