import streamlit as st
import cv2
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Live Detection", layout="wide")

st.title("🎥 Live YOLO Object Detection")

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
    index=1 if cv2.ocl.haveOpenCL() else 0,
)

# Load model once
@st.cache_resource
model = YOLO("yolo26n.pt")

run = st.button("Start Webcam")
stop = st.button("Stop")

frame_placeholder = st.empty()

if run:

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not open webcam.")
        st.stop()

    while cap.isOpened():

        if stop:
            break

        ret, frame = cap.read()

        if not ret:
            st.warning("Failed to grab frame.")
            break

        results = model.predict(
            source=frame,
            device=device,
            conf=confidence,
            verbose=False,
        )

        annotated = results[0].plot()

        # Convert BGR → RGB for Streamlit
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(
            annotated,
            channels="RGB",
            use_container_width=True,
        )

    cap.release()

st.info("Click **Start Webcam** to begin detection.")