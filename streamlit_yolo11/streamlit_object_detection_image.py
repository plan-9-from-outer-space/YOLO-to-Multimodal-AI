
# Import All the Required Libraries
import io
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="YOLO 11 Image Detector", page_icon="🧠", layout="centered")
st.title("🧠 YOLO 11 - Image Object Detection (Super Simple)")
st.caption("Upload an Image. I'll run YOLO 11 and show you the result.")

# Load the small YOLO 11 model once and cache it
@st.cache_resource(show_spinner=False)
def get_model():
    return YOLO("yolo11s.pt")

model = get_model()

uploaded = st.file_uploader("Upload an Image (PNG/JPG/JPEG/WebP)", type = ["png", "jpg", "jpeg", "webp"])

if uploaded is None:
    st.info("👆 Upload an Image file to begin.")
    st.stop()

# Read Input image
img = Image.open(uploaded).convert("RGB")

# Run Detection
with st.spinner("Running YOLO 11..."):
    result = model.predict(source=np.array(img), save=True)[0]

# Plot Annotated Image (convert BGR -> RGB)
# annotated = res[0].plot()[:, :, ::-1]
# result = res[0]

# Robust plotting (always RGB)
try:
    annotated_pil = result.plot(pil=True)  # PIL.Image (RGB)
except TypeError:
    annotated_bgr = result.plot()  # ndarray (BGR)
    annotated_rgb = np.ascontiguousarray(annotated_bgr[:, :, ::-1])  # BGR -> RGB
    annotated_pil = Image.fromarray(annotated_rgb)

# Two columns; left = input, right = output
col1, col2 = st.columns(2, gap = "large")
with col1:
    st.subheader("Input")
    st.image(img, use_container_width =True)
with col2:
    st.subheader("Output")
    st.image(annotated_pil, use_container_width =True)

