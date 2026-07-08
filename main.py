import cv2
import streamlit as st
from ultralytics import YOLO

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(page_title="YOLO Live Detection", layout="wide")

st.title("🎥 YOLO Live Object Detection")

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
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
    index=0,
)

# --------------------------------------------------
# Load YOLO Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolo26n.pt")

model = load_model()

# --------------------------------------------------
# Live Webcam
# --------------------------------------------------
start = st.button("Start")
stop = st.button("Stop")
frame_placeholder = st.empty()

if start:
    cap = cv2.VideoCapture(0)

    while not stop:
        ret, frame = cap.read()
        if not ret:
            st.error("Could not access webcam.")
            break

        results = model.predict(
            source=frame,
            conf=confidence,
            device=device,
            verbose=False,
        )

        annotated = results[0].plot()
        frame_placeholder.image(annotated, channels="BGR", use_container_width=True)

    cap.release()