import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import av
import os
from ultralytics import YOLO
from PIL import Image
import cv2
import logging
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client
curr = os.getcwd()
modelpath = os.path.join(curr, 'pages', 'best.pt')
model = YOLO(modelpath)
st.set_page_config(page_title="Video Demo", page_icon="📹")
st.markdown("# Video Demo")
st.sidebar.header("Video Demo")
st.markdown("Welcome to the Video Demo page of PaveInspect! Here, you can utilize the live video feed to detect and analyze cracks on road surfaces in real-time using our advanced YOLOv8-based detection system.")
confval = st.slider("Confidence Threshold", min_value=0.0,
                    max_value=1.0, value=0.25, step=0.05)

logger = logging.getLogger(__name__)


def get_ice_servers():
    try:
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."  # noqa: E501
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    try:
        token = client.tokens.create()
    except TwilioRestException as e:
        st.warning(
            f"Error occurred while accessing Twilio API. Fallback to a free STUN server from Google. ({e})"  # noqa: E501
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    return token.ice_servers


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    result = model(img, conf=confval)
    for r in result:
        im_bgr = r.plot()
    return av.VideoFrame.from_ndarray(im_bgr, format="bgr24")


webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": get_ice_servers(),
        "iceTransportPolicy": "relay",
    },
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
