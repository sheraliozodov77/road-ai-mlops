import cv2
import numpy as np
import onnxruntime as ort
import os
import mlflow
import wandb
import time
from .inference_registry import load_model_from_registry_or_local

# =========================
# Constants
# =========================

INDEX_TO_COLOR = np.array([
    [0,   0,   0],     # 0 background
    [64,  0, 128],     # 1 moving car
    [164, 20,  0],     # 2 building
    [94, 99,  80],     # 3 human
    [0, 128,  0],      # 4 vegetation
    [192, 0, 128],     # 5 static car
    [128, 64, 128],    # 6 road
    [128,128,  0],     # 7 low vegetation
], dtype=np.uint8)

YOLO_CLASSES = [
    'longitudinal_crack', 'transverse_crack', 'alligator_crack',
    'block_crack', 'patch', 'pothole'
]

# =========================
# Setup MLflow + W&B once
# =========================

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("RoadAI-Inference")

wandb.init(project="road-ai-inference", name="inference-logs", mode="online")

# =========================
# Load Models (from registry or local)
# =========================

segformer_model = load_model_from_registry_or_local(
    model_name="SegFormer-RoadAI",
    stage="Production",
    local_path="models/segformer/segformer-b4-uavid.onnx",
    s3_key=os.environ.get("SEGFORMER_S3_KEY"),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

yolov11_model = load_model_from_registry_or_local(
    model_name="YOLOv11-RoadAI",
    stage="Production",
    local_path="models/yolov11/yolov11m.onnx",
    s3_key=os.environ.get("YOLOV11_S3_KEY"),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# =========================
# Inference Methods
# =========================

def run_segformer(model_session, img_bgr, size=1024):
    start = time.time()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (size, size))
    x = img_resized.astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0).astype(np.float32)

    ort_inputs = {model_session.get_inputs()[0].name: x}
    ort_outs = model_session.run(None, ort_inputs)
    mask = np.argmax(ort_outs[0], axis=1).squeeze(0).astype(np.uint8)

    mask_resized = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    color_mask = INDEX_TO_COLOR[mask_resized]

    label_map = {
        1: "moving car",
        2: "building",
        3: "human",
        4: "vegetation",
        5: "static car",
        6: "road",
        7: "low vegetation"
    }

    overlay = img_rgb.copy()
    for class_id, label in label_map.items():
        class_mask = (mask_resized == class_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 800:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(overlay, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    final_overlay = cv2.addWeighted(overlay, 0.55, color_mask, 0.45, 0.0)
    latency = time.time() - start

    with mlflow.start_run(run_name="segformer_inference", nested=True):
        mlflow.log_param("model", "SegFormer")
        mlflow.log_param("input_size", size)
        mlflow.log_metric("latency", latency)

    wandb.log({"segformer_latency": latency})
    return mask_resized, cv2.cvtColor(final_overlay, cv2.COLOR_RGB2BGR)

def run_segformer_video(model_session, video_path, output_path, size=1024, frame_skip=4):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps // frame_skip, (width, height))

    frame_idx = 0
    total_latency = 0.0
    total_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            _, overlay = run_segformer(model_session, frame, size)
            out.write(overlay)
            total_frames += 1
        frame_idx += 1

    cap.release()
    out.release()
    avg_latency = total_latency / max(1, total_frames)
    return avg_latency, total_frames

def run_yolov11(model_session, img_bgr, size=640):
    start = time.time()

    img = cv2.resize(img_bgr, (size, size))
    img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)

    ort_inputs = {model_session.get_inputs()[0].name: img_input}
    ort_outs = model_session.run(None, ort_inputs)
    predictions = ort_outs[0].squeeze(0)

    h, w = img.shape[:2]
    latency = time.time() - start
    detections = 0

    for pred in predictions:
        cx, cy, bw, bh = pred[0:4]
        object_conf = pred[4]
        class_scores = pred[5:]
        cls_id = int(np.argmax(class_scores))
        cls_score = float(np.max(class_scores))
        conf = object_conf * cls_score

        if conf < 0.35:
            continue

        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        detections += 1
        label = YOLO_CLASSES[cls_id] if cls_id < len(YOLO_CLASSES) else str(cls_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    with mlflow.start_run(run_name="yolov11_inference", nested=True):
        mlflow.log_param("model", "YOLOv11")
        mlflow.log_param("input_size", size)
        mlflow.log_metric("latency", latency)
        mlflow.log_metric("detections", detections)

    wandb.log({"yolov11_latency": latency, "detections": detections})
    return img

# =========================
# Local ONNX Loader (used by FastAPI)
# =========================

def load_onnx_model(model_path):
    print(f"🔍 Loading ONNX model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")
    session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("✅ ONNX model loaded successfully!")
    return session
