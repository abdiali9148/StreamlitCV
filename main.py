import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Live Detection", layout="wide")

st.title("🎥 YOLO Object Detection")

# Sidebar controls
confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
)

device = st.sidebar.selectbox(
    "Inference Device",
    ["cpu", "mps"],
    index=0,  # default to cpu - deployed servers won't have mps
)


# Load model once, cached across reruns
@st.cache_resource
def load_model():
    return YOLO("yolo26n.pt")


model = load_model()

# Browser-based camera capture (works locally AND deployed)
img_file = st.camera_input("Take a photo")

if img_file is not None:
    image = Image.open(img_file)
    frame = np.array(image)

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = model.predict(
        source=frame_bgr,
        device=device,
        conf=confidence,
        verbose=False,
    )

    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    st.image(
        annotated_rgb,
        channels="RGB",
        use_container_width=True,
    )
else:
    st.info("Click the camera button above to take a photo and run detection.")