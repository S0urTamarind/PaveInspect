import streamlit as st
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
st.markdown("To start the video feed, click the 'Start/Stop' button below. To stop the video feed, click the 'Start/Stop' button again.")
confval = st.slider("Confidence Threshold", min_value=0.0,
                    max_value=1.0, value=0.25, step=0.05)
framest = st.empty()
startstop = st.button("Start/Stop")
while startstop:
    vid = cv2.VideoCapture(0)
    ret, frame = vid.read()
    if not ret:
        st.write("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = model(frame, conf=confval)
    for r in result:
        im_bgr = r.plot()
    framest.image(im_bgr, caption="Live Prediction", use_column_width=True)
    if not startstop:
        vid.release()
