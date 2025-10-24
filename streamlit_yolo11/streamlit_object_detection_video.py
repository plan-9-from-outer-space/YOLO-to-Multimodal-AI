
# Import Libraries
import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="YOLO 11 Live Detector", page_icon="🧠", layout="centered")
st.title("🧠 YOLO 11 - Image & Video (Live)")
st.caption("Choose Task (Detection / Segmentation / Pose), then upload an image or video. For video, detections render live frame-by-frame.")

# ---- Task Selection: choose the model family ----
task = st.radio("Select Task", ["Object Detection", "Instance Segmentation", "Pose Estimation"], horizontal=True)

MODEL_MAP = {
    "Object Detection": "yolo11m.pt",
    "Instance Segmentation": "yolo11m-seg.pt",
    "Pose Estimation": "yolo11m-pose.pt"
}
model_path = MODEL_MAP[task]

# Load YOLO model once per chosen task
@st.cache_resource(show_spinner=False)
def get_model(path: str):
    return YOLO(path)

model = get_model(model_path)

# Mode switch
mode = st.radio("Select mode", ["Image", "Video (Live)"], horizontal=True)

# -----------------------
# IMAGE MODE (STATIC)
# -----------------------
if mode == "Image":
    uploaded = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is None:
        st.info("👆 Upload an Image to begin.")
        st.stop()

    img = Image.open(uploaded).convert("RGB")
    with st.spinner(f"Running {task} on image..."):
        result = model.predict(source=np.array(img), save=False, verbose=False)[0]
        annotated = result.plot(pil=True)  # Works for detection / segmentation / pose estimation

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Input")
        st.image(img, use_container_width=True)
    with col2:
        st.subheader("Detections")
        st.image(annotated, use_container_width=True)

# -----------------------
# VIDEO MODE (LIVE)
# -----------------------
else:
    uploaded = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi", "mkv"])
    conf = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
    start = st.button("Start Detection")

    if uploaded is None:
        st.info("👆 Upload a Video to begin.")
        st.stop()

    # Save uploaded video to a temp file so OpenCV can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded.read())
        input_path = tmp_file.name

    st.caption(f"Input File: {uploaded.name}  •  Task: {task}  •  Model: {model_path}")

    if start:
        frame_placeholder = st.empty()
        info_placeholder = st.empty()
        progress_bar = st.progress(0)

        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        processed = 0

        with st.spinner(f"Running {task} on video (live)..."):
            # Loop over the video frames
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok: break

                # Run selected task on every frame
                results = model.predict(frame, conf=conf, verbose=False)[0]
                annotated_bgr = results.plot()  # ndarray (BGR), works for detect/segment/pose

                # Show live frame
                frame_placeholder.image(annotated_bgr, channels="BGR", use_container_width=True)

                processed += 1

                # Update progress bar if total_frames known
                if total_frames:
                    progress_bar.progress(min(processed / total_frames, 1.0))

                # Update info
                info_placeholder.markdown(
                    f"**Frames processed:** {processed} / {total_frames if total_frames else 'unknown'} "
                    f" • **Confidence:** {conf}"
                )

            cap.release()

        st.success("Finished live processing ✅")

    # Clean up temp file
    try:
        os.unlink(input_path)
    except Exception:
        pass
