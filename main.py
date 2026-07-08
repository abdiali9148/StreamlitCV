import av
import queue
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
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
# Shared frame queue
# --------------------------------------------------
frame_queue = queue.Queue(maxsize=1)

# --------------------------------------------------
# Video Processor
# --------------------------------------------------
class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.confidence = 0.5
        self.device = "cpu"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model.predict(
            source=img,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        annotated = results[0].plot()

        # Push annotated frame to queue, drop if full
        if not frame_queue.full():
            frame_queue.put(annotated)

        # Return original (unmodified) frame to WebRTC player
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --------------------------------------------------
# WebRTC streamer (hidden via CSS)
# --------------------------------------------------
st.markdown("""
<style>
iframe.stCustomComponentV1 {
    height: 1px !important;
    min-height: 0 !important;
    visibility: hidden !important;
    position: absolute !important;
}
</style>
""", unsafe_allow_html=True)

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

# --------------------------------------------------
# Annotated frame display
# --------------------------------------------------
frame_placeholder = st.empty()

while ctx.state.playing:
    try:
        annotated = frame_queue.get(timeout=1)
        frame_placeholder.image(annotated, channels="BGR", use_container_width=True)
    except queue.Empty:
        continue