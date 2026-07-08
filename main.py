import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(page_title="YOLO Live Detection", layout="wide")

st.title("🎥 YOLO Live Object Detection")

st.markdown("""
<style>
video::-webkit-media-controls {
    display: none !important;
}
video::-webkit-media-controls-enclosure {
    display: none !important;
}
video::-webkit-media-controls-panel {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

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
# Video Processor
# --------------------------------------------------
class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.confidence = 0.5
        self.device = "cpu"

    def recv(self, frame):
        # Convert WebRTC frame to numpy array (BGR)
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO inference
        results = model.predict(
            source=img,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        # Draw detections
        annotated = results[0].plot()

        # Return annotated frame
        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24",
        )

# --------------------------------------------------
# Live Webcam
# --------------------------------------------------
st.write("Click **START** below to begin live detection.")

ctx = webrtc_streamer(
    key="yolo-live",
    video_processor_factory=YOLOProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)

# Sync sidebar controls to processor in real time
if ctx.video_processor:
    ctx.video_processor.confidence = confidence
    ctx.video_processor.device = device