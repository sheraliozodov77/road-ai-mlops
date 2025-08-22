# streamlit_app/main.py

import streamlit as st
import requests
import base64
from PIL import Image
import io
import os
import time

st.set_page_config(page_title="🚧 Road AI Demo", layout="wide")
st.title("🛣️ Road Segmentation & Defect Detection")

# ✅ Use environment variable instead of st.secrets
API_URL = os.getenv("API_URL", "http://localhost:8000")

model_type = st.sidebar.selectbox("Choose Model", ["segformer", "yolov11"])
input_type = st.sidebar.radio("Input Type", ["Image", "Video"])

if input_type == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        with st.spinner("Sending image to API..."):
            files = {"file": uploaded_file.getvalue()}
            data = {"model_type": model_type}
            response = requests.post(f"{API_URL}/predict/image", files=files, data=data)

        if response.status_code == 200 and response.json().get("status") == "success":
            result = response.json()["result"]
            result_image = Image.open(io.BytesIO(base64.b64decode(result)))
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="Original Input", use_column_width=True)
            with col2:
                st.image(result_image, caption="Predicted Output", use_column_width=True)
        else:
            st.error("API Error: " + str(response.text))

elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov"])
    if uploaded_video:
        with st.spinner("Uploading video and starting job..."):
            files = {"file": uploaded_video.getvalue()}
            data = {"model_type": model_type}
            response = requests.post(f"{API_URL}/predict/video", files=files, data=data)

        if response.status_code == 200:
            job_id = response.json()["job_id"]
            st.success(f"Job started: {job_id}")

            with st.spinner("Processing video..."):
                status = "processing"
                while status == "processing":
                    job_status = requests.get(f"{API_URL}/jobs/status", params={"job_id": job_id})
                    status = job_status.json().get("status")
                    time.sleep(2)

                if status == "completed":
                    video_url = job_status.json().get("output_url")
                    st.video(f"{API_URL}{video_url}")
                else:
                    st.error("Job failed or not found.")
        else:
            st.error("API Error: " + str(response.text))

with st.expander("📜 History (last 5)"):
    try:
        hist = requests.get(f"{API_URL}/history")
        if hist.status_code == 200:
            for row in hist.json():
                st.write(f"[{row['timestamp']}] {row['model']} ({row['type']}) — Runtime: {row['runtime']}")
        else:
            st.write("No history found.")
    except Exception as e:
        st.error(f"Failed to fetch history: {e}")
