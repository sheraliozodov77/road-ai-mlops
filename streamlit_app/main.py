import streamlit as st
import streamlit.components.v1 as components
import requests
import base64
from PIL import Image
import io
import os
import time

st.set_page_config(page_title="Road AI Demo", layout="wide")

st.title("🛣️ Road Segmentation & Defect Detection")
st.caption("FastAPI · Streamlit · ONNX · YOLOv11 · SegFormer · MLflow · W&B · Docker · AWS")

API_URL = os.getenv("API_URL", "http://backend:8000")

# =========================
# Class Legend
# =========================
CLASS_LEGEND = {
    "Background": "#000000",
    "Moving Car": "#400080",
    "Building": "#A41400",
    "Human": "#5E6350",
    "Vegetation": "#008000",
    "Static Car": "#C00080",
    "Road": "#804080",
    "Low Vegetation": "#808000"
}

def render_legend(legend_dict):
    html = "<div style='display:flex;flex-wrap:wrap;gap:12px;margin-top:10px;'>"
    for label, color in legend_dict.items():
        html += f"""
        <div style='display:flex;align-items:center;gap:6px;'>
            <div style='width:18px;height:18px;background:{color};border-radius:3px;border:1px solid #555;'></div>
            <div style='font-size:14px;color:#eee;'>{label}</div>
        </div>
        """
    html += "</div>"
    components.html(html, height=120)

# =========================
# Sidebar
# =========================
st.sidebar.header("Model Controls")
model_type = st.sidebar.selectbox("Model", ["segformer", "yolov11"])
input_type = st.sidebar.radio("Input Type", ["Image", "Video"])

# =========================
# IMAGE INPUT
# =========================
if input_type == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    if uploaded_file:
        start = time.time()
        with st.spinner("Running inference..."):
            files = {"file": uploaded_file.getvalue()}
            data = {"model_type": model_type}
            response = requests.post(f"{API_URL}/predict/image", files=files, data=data)
        end = time.time()

        if response.status_code == 200 and response.json().get("status") == "success":
            result = response.json()["result"]
            s3_url = response.json()["url"]
            result_image = Image.open(io.BytesIO(base64.b64decode(result)))

            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="Original Input", use_container_width=True)
            with col2:
                st.image(result_image, caption="Predicted Output", use_container_width=True)
                st.download_button(
                    "Download Prediction",
                    data=base64.b64decode(result),
                    file_name="prediction.png",
                    mime="image/png"
                )
                if model_type == "segformer":
                    render_legend(CLASS_LEGEND)
            st.success(f"✅ Prediction completed in {end - start:.2f} seconds")
        else:
            st.error("❌ API Error: " + str(response.text))

# =========================
# VIDEO INPUT
# =========================
elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov"], label_visibility="collapsed")
    if uploaded_video:
        with st.spinner("Uploading & starting inference..."):
            files = {"file": uploaded_video.getvalue()}
            data = {"model_type": model_type}
            response = requests.post(f"{API_URL}/predict/video", files=files, data=data)

        if response.status_code == 200:
            job_id = response.json()["job_id"]
            st.info(f"Job ID: {job_id}")
            start = time.time()
            status = "processing"
            while status == "processing":
                job_status = requests.get(f"{API_URL}/jobs/status", params={"job_id": job_id})
                status = job_status.json().get("status")
                time.sleep(2)
            end = time.time()

            if status == "completed":
                video_url = job_status.json().get("output_url")
                st.video(f"{API_URL}{video_url}")
                st.success(f"✅ Video inference finished in {end - start:.2f} seconds")
            else:
                st.error("❌ Job failed or not found.")
        else:
            st.error("❌ API Error: " + str(response.text))

# =========================
# History Viewer
# =========================
with st.expander("📜 Inference History"):
    try:
        hist = requests.get(f"{API_URL}/history", timeout=2)
        if hist.status_code == 200:
            rows = hist.json()
            for row in rows[:15]:
                st.markdown(f"**[{row['timestamp']}]** — `{row['model']}` ({row['type']}) | Runtime: `{row['runtime']}`")
        else:
            st.warning("No history found.")
    except:
        st.info("History will be available after your first prediction.")
