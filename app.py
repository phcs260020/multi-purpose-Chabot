"""
Interactive Multi-Purpose Chatbot with Image Recognition
=======================================================
Streamlit app that accepts text and image inputs.
Uses a pre-trained YOLOv8 model (COCO) for object detection
and a conversational layer that answers questions about the
detected objects while also supporting general dialogue.
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from collections import Counter
import re
import hashlib
import io
from typing import List, Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VisionChat – Image Recognition Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLOv8 model…")
def load_model(model_name: str = "yolov8n.pt") -> YOLO:
    """Load a pre-trained YOLOv8 model. Cached so it loads only once."""
    return YOLO(model_name)


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
    Returns the annotated image and a list of detection dictionaries.
    """
    # Convert PIL → OpenCV (BGR)
    img_array = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    results = model.predict(img_bgr, conf=conf_threshold, verbose=False)
    result = results[0]

    # Annotated image (RGB for display)
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)

    detections = []
    if result.boxes is not None:
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls_id]
            xyxy = box.xyxy[0].tolist()
            detections.append(
                {
                    "label": label,
                    "confidence": conf,
                    "bbox": xyxy,  # [x1, y1, x2, y2]
                }
            )
    return annotated_pil, detections


def summarize_detections(detections: List[Dict]) -> str:
    """Create a human-readable summary of detected objects."""
    if not detections:
        return "I couldn't detect any objects in the image with the current confidence threshold."

    counts = Counter(d["label"] for d in detections)
    parts = []
    for label, count in counts.most_common():
        confs = [d["confidence"] for d in detections if d["label"] == label]
        avg_conf = sum(confs) / len(confs)
        if count == 1:
            parts.append(f"1 {label} (confidence ≈ {avg_conf:.0%})")
        else:
            parts.append(f"{count} {label}s (avg confidence ≈ {avg_conf:.0%})")

    summary = "I detected the following objects:\n• " + "\n• ".join(parts)
    return summary


# ---------------------------------------------------------------------------
# Conversational response logic (rule-based + context-aware)
# ---------------------------------------------------------------------------
def generate_response(
    user_text: str,
    detections: Optional[List[Dict]] = None,
    chat_history: Optional[List[Dict]] = None,
) -> str:
    """
    Generate a response based on the user message and current image context.
    Handles both image-related questions and general conversation.
    """
    text = user_text.lower().strip()
    detections = detections or []
    labels = [d["label"] for d in detections]
    unique_labels = list(set(labels))
    counts = Counter(labels)

    # --- Greetings & general ---
    if any(w in text for w in ["hello", "hi ", "hey", "good morning", "good afternoon", "good evening"]):
        return (
            "Hello! 👋 I'm VisionChat. Upload an image and I'll identify the objects in it, "
            "or just ask me anything about what I've seen."
        )

    if any(w in text for w in ["help", "what can you do", "capabilities", "how to use"]):
        return (
            "I can:\n"
            "1. **Detect objects** in any image you upload (using YOLOv8 trained on COCO).\n"
            "2. **Answer questions** about the detected objects – e.g. “How many people?”, "
            "“Is there a dog?”, “List everything you see”.\n"
            "3. Hold a light conversation.\n\n"
            "Just upload an image from the sidebar or drag one into the chat, then ask away!"
        )

    if any(w in text for w in ["thank", "thanks", "appreciate"]):
        return "You're welcome! Feel free to upload another image or ask more questions. 😊"

    if any(w in text for w in ["bye", "goodbye", "see you", "exit"]):
        return "Goodbye! Come back anytime you want to analyze more images. 👋"

    # --- Image-related questions ---
    if not detections:
        if any(w in text for w in ["what", "object", "detect", "see", "image", "picture", "photo"]):
            return (
                "I don't have an image to analyze yet. Please upload one using the sidebar "
                "or the file uploader, and I'll identify the objects for you."
            )
        # Fall through to generic response

    # Count questions
    count_match = re.search(r"how many\s+(\w+)", text)
    if count_match:
        target = count_match.group(1).rstrip("s")  # crude singularization
        # Find closest matching label
        matched = None
        for lab in unique_labels:
            if target in lab or lab in target:
                matched = lab
                break
        if matched:
            n = counts[matched]
            return f"I found **{n}** {matched}{'s' if n != 1 else ''} in the image."
        else:
            return f"I didn't detect any '{target}' in the current image. Detected objects: {', '.join(unique_labels) or 'none'}."

    # Existence questions
    if any(phrase in text for phrase in ["is there", "are there", "do you see", "can you see", "is a ", "is an "]):
        for lab in unique_labels:
            if lab in text or lab.rstrip("s") in text:
                n = counts[lab]
                return f"Yes – I detected **{n}** {lab}{'s' if n != 1 else ''}."
        # Check common objects even if not present
        return f"No, I don't see that. The objects I found are: {', '.join(unique_labels) or 'none'}."

    # List / describe questions
    if any(w in text for w in ["list", "what objects", "what do you see", "describe", "what's in", "what is in", "show me", "tell me about the image"]):
        return summarize_detections(detections)

    # Confidence / details
    if "confidence" in text or "sure" in text or "probability" in text:
        if not detections:
            return "No detections available."
        details = "\n".join(
            f"• {d['label']}: {d['confidence']:.1%}" for d in sorted(detections, key=lambda x: -x["confidence"])
        )
        return "Here are the detection confidences:\n" + details

    # Most confident object
    if any(w in text for w in ["main object", "most confident", "primary", "biggest"]):
        if detections:
            top = max(detections, key=lambda d: d["confidence"])
            return f"The most confident detection is **{top['label']}** ({top['confidence']:.1%})."
        return "No objects detected."

    # Generic fallback when we have detections
    if detections:
        return (
            f"Based on the current image I can see: {', '.join(unique_labels)}. "
            "You can ask me things like “How many cars?”, “Is there a person?”, "
            "or “List all objects”."
        )

    # Pure text / unknown
    return (
        "I'm mainly designed to analyze images and answer questions about the objects I detect. "
        "Upload an image and try asking “What do you see?” or “How many people are there?”. "
        "You can also say “help” to learn more about my capabilities."
    )


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 Hi! I'm **VisionChat**, your image-recognition chatbot.\n\n"
                    "Upload an image in the sidebar (or below) and I'll identify the objects in it. "
                    "Then you can ask me questions about what I found – or just chat!"
                ),
            }
        ]
    if "detections" not in st.session_state:
        st.session_state.detections = []
    if "annotated_image" not in st.session_state:
        st.session_state.annotated_image = None
    if "original_image" not in st.session_state:
        st.session_state.original_image = None


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main():
    init_session_state()

    # ----- Sidebar -----
    with st.sidebar:
        st.title("⚙️ Controls")
        st.markdown("---")

        model_choice = st.selectbox(
            "YOLO model",
            options=["yolov8n.pt", "yolov8s.pt"],
            index=0,
            help="nano = fastest, small = more accurate",
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

        if st.button("🗑️ Clear chat & image", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Chat cleared. Upload a new image or ask me something!",
                }
            ]
            st.session_state.detections = []
            st.session_state.annotated_image = None
            st.session_state.original_image = None
            st.rerun()

        st.markdown("---")
        st.caption(
            "Powered by **YOLOv8** (Ultralytics) · COCO 80 classes · "
            "Built with Streamlit"
        )

    # ----- Load model -----
    model = load_model(model_choice)

    # ----- Process uploaded image -----
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        # Use a simple hash of the content to detect a new upload
        file_hash = hashlib.md5(file_bytes).hexdigest()
        if getattr(st.session_state, "_last_file_hash", None) != file_hash:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            with st.spinner("Detecting objects…"):
                annotated, detections = run_detection(image, model, conf_threshold)

            st.session_state.original_image = image
            st.session_state.annotated_image = annotated
            st.session_state.detections = detections
            st.session_state._last_file_hash = file_hash

            # Add recognition result to chat
            summary = summarize_detections(detections)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"✅ **Image analyzed!**\n\n{summary}",
                    "image": annotated,  # attach for display
                }
            )

    # ----- Main chat area -----
    st.title("🤖 VisionChat")
    st.caption("Upload an image → I identify objects → Ask me questions about them")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("image") is not None:
                st.image(msg["image"], caption="Detected objects", use_container_width=True)

    # Chat input
    if prompt := st.chat_input("Ask about the image or just chat…"):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate & display response
        response = generate_response(
            prompt,
            detections=st.session_state.detections,
            chat_history=st.session_state.messages,
        )
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    # Optional: show current detections in an expander
    if st.session_state.detections:
        with st.expander("📋 Current detection details", expanded=False):
            for i, d in enumerate(st.session_state.detections, 1):
                st.write(
                    f"{i}. **{d['label']}** – {d['confidence']:.1%} "
                    f"(bbox: {[round(x) for x in d['bbox']]})"
                )


if __name__ == "__main__":
    main()
