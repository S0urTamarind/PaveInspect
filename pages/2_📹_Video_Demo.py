import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import os
from ultralytics import YOLO
from PIL import Image
import cv2
curr = os.getcwd()
modelpath = os.path.join(curr, 'pages', 'best.pt')
model = YOLO(modelpath)
st.set_page_config(page_title="Video Demo", page_icon="📹")
st.markdown("# Video Demo")
st.sidebar.header("Video Demo")
st.markdown("Welcome to the Video Demo page of PaveInspect! Here, you can utilize the live video feed to detect and analyze cracks on road surfaces in real-time using our advanced YOLOv8-based detection system.")
confval = st.slider("Confidence Threshold", min_value=0.0,
                    max_value=1.0, value=0.25, step=0.05)


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    result = model(img, conf=confval)
    for r in result:
        im_bgr = r.plot()
    return av.VideoFrame.from_ndarray(im_bgr, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
