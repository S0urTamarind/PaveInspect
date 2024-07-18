import streamlit as st
import os
from ultralytics import YOLO
from PIL import Image
curr = os.getcwd()
modelpath = os.path.join(curr, 'pages', 'best.pt')
model = YOLO(modelpath)
st.set_page_config(page_title="Image Demo", page_icon="📷")
st.markdown("# Image Demo")
st.sidebar.header("Image Demo")
st.markdown("Welcome to the Image Demo page of PaveInspect! Here, you can upload images of road surfaces to detect and analyze cracks using our advanced YOLOv8-based detection system.")
confval = st.slider("Confidence Threshold", min_value=0.0,
                    max_value=1.0, value=0.25, step=0.05)


def display_uploaded_image(uploaded_file):
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        result = model(img, conf=confval)
        for r in result:
            im_bgr = r.plot()
            im_rgb = Image.fromarray(im_bgr[..., ::-1])
        st.image(im_rgb, caption="Prediction", use_column_width=True)


uploaded_file = st.file_uploader(
    "Choose an Image", type=["jpg", "png", "jpeg"])

display_uploaded_image(uploaded_file)
