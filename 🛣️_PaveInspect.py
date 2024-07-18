import streamlit as st

st.set_page_config(
    page_title="PaveInspect",
    page_icon="🛣️",
)

st.markdown("# PaveInspect")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    We leverages the power of YOLOv8, a state-of-the-art object detection algorithm, to identify cracks on roads with high accuracy. Whether you upload an image or use live video feed, PaveInspect provides a reliable and efficient solution to monitor road conditions.
    ### Key Features
    - ***Real-time Detection:*** Utilize live video feed to detect road cracks instantly.
    - ***Image Analysis:*** Upload images for a detailed inspection of road surfaces.
    - ***User-Friendly Interface:*** Built on Streamlit, our app is designed for ease of use and accessibility.
    ### Why PaveInspect?
    Maintaining road quality is crucial for safety and longevity. PaveInspect helps in early detection of potential hazards, allowing for timely maintenance and prevention of accidents.
    ### Get Started
    Select a demo from the sidebar to experience PaveInspect in action.
    """
)
