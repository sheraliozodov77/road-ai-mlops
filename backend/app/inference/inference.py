# backend/app/inference/inference.py
import cv2
import numpy as np
import onnxruntime as ort
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
# Load Models from MLflow Registry with Fallback
# =========================

segformer_model = load_model_from_registry_or_local(
    model_name="SegFormer-RoadAI",
    stage="Production",
    local_path="models/segformer/segformer-b4-uavid.onnx"
)

yolov11_model = load_model_from_registry_or_local(
    model_name="YOLOv11-RoadAI",
    stage="Production",
    local_path="models/yolov11/yolov11m.onnx"
)

# =========================
# SegFormer Inference
# =========================

def run_segformer(model_session, img_bgr, size=1024):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (size, size))
    x = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0).astype(np.float32)

    ort_inputs = {model_session.get_inputs()[0].name: x}
    ort_outs = model_session.run(None, ort_inputs)
    mask = np.argmax(ort_outs[0], axis=1).squeeze(0).astype(np.uint8)
    color_mask = INDEX_TO_COLOR[mask]
    overlay = cv2.addWeighted(img_rgb, 0.55, color_mask, 0.45, 0.0)
    return mask, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

# =========================
# YOLOv11 Inference
# =========================

def run_yolov11(model_session, img_bgr, size=640):
    img = cv2.resize(img_bgr, (size, size))
    img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)

    ort_inputs = {model_session.get_inputs()[0].name: img_input}
    ort_outs = model_session.run(None, ort_inputs)
    predictions = ort_outs[0].squeeze(0)

    for pred in predictions:
        conf = pred[4]
        if conf < 0.35:
            continue
        cls_id = int(pred[5])
        label = YOLO_CLASSES[cls_id] if cls_id < len(YOLO_CLASSES) else str(cls_id)
        x1, y1, x2, y2 = map(int, pred[0:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img
