from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import shutil
import os
import uuid
import cv2
import numpy as np
import io
import base64
import time
import json
from datetime import datetime

from app.inference.inference import (
    load_onnx_model, run_segformer, run_yolov11
)
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.models import Prediction
from app.utils.s3 import upload_file_to_s3
from app.tracking.logger import log_inference_metrics
from app.monitoring.metrics import (
    router as metrics_router,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)

# Initialize FastAPI
app = FastAPI(title="🚦 Road AI Inference API")

# Register metrics route
app.include_router(metrics_router)

# Prometheus middleware
@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    method = request.method
    endpoint = request.url.path
    REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
    return response

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ONNX models
SEG_MODEL_PATH = "models/segformer/segformer-b4-uavid.onnx"
YOL_MODEL_PATH = "models/yolov11/yolov11m.onnx"

try:
    segformer_model = load_onnx_model(SEG_MODEL_PATH)
    yolov11_model = load_onnx_model(YOL_MODEL_PATH)
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    raise e

def read_image(file: UploadFile):
    from PIL import Image
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    return np.array(image)

@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    db: Session = Depends(get_db)
):
    img = read_image(file)
    filename = file.filename
    job_id = str(uuid.uuid4())

    if model_type == "segformer":
        _, overlay = run_segformer(segformer_model, img, size=1024)
    elif model_type == "yolov11":
        overlay = run_yolov11(yolov11_model, img, size=640)
    else:
        return {"status": "error", "message": "Invalid model_type"}

    _, buffer = cv2.imencode(".png", overlay)
    output_path = f"temp/{job_id}.png"
    with open(output_path, "wb") as f:
        f.write(buffer.tobytes())

    s3_url = upload_file_to_s3(output_path, "outputs")

    prediction = Prediction(
        id=job_id,
        filename=filename,
        input_type="image",
        model_name=model_type,
        status="completed",
        runtime="<1s",
        created_at=datetime.utcnow(),
        output_path=s3_url
    )
    db.add(prediction)
    db.commit()

    log_inference_metrics(model_type, "<1s", "image", filename, output_path)

    return {
        "status": "success",
        "model": model_type,
        "result": base64.b64encode(buffer.tobytes()).decode("utf-8"),
        "url": s3_url
    }

@app.post("/predict/video")
async def predict_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_type: str = Form(...),
    db: Session = Depends(get_db)
):
    job_id = str(uuid.uuid4())
    filename = file.filename
    input_path = f"temp/{job_id}.mp4"
    output_path = f"temp/{job_id}_processed.mp4"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction = Prediction(
        id=job_id,
        filename=filename,
        input_type="video",
        model_name=model_type,
        status="processing",
        runtime=None,
        created_at=datetime.utcnow(),
        output_path=output_path
    )
    db.add(prediction)
    db.commit()

    background_tasks.add_task(process_video, input_path, output_path, model_type, job_id, filename, db)
    return {"job_id": job_id}

@app.get("/jobs/status")
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Prediction).filter(Prediction.id == job_id).first()
    if not job:
        return {"status": "not_found"}
    return {"status": job.status, "output_url": job.output_path}

def process_video(input_path, output_path, model_type, job_id, filename, db):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if model_type == "segformer":
            _, result = run_segformer(segformer_model, frame, size=1024)
        else:
            result = run_yolov11(yolov11_model, frame, size=640)
        out.write(result)

    cap.release()
    out.release()

    s3_url = upload_file_to_s3(output_path, "outputs")

    job = db.query(Prediction).filter(Prediction.id == job_id).first()
    job.status = "completed"
    job.output_path = s3_url
    job.runtime = "~30s"
    db.commit()

    log_inference_metrics(model_type, "~30s", "video", filename, output_path)

@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    rows = db.query(Prediction).order_by(Prediction.created_at.desc()).limit(20).all()
    return [
        {
            "id": row.id,
            "filename": row.filename,
            "type": row.input_type,
            "model": row.model_name,
            "status": row.status,
            "timestamp": row.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime": row.runtime,
            "output_url": row.output_path
        } for row in rows
    ]
