# 🛣️ Road AI – Infrastructure Defect Detection (MLOps Production Deployment)

A full-stack, production-grade AI system for **road segmentation** (SegFormer) and **defect detection** (YOLOv11), purpose-built for **UAV-based road monitoring** and deployed with industry-standard **MLOps best practices**.

---

## 🚀 Key Features

✅ **Streamlit UI** for uploading images/videos & viewing predictions  
✅ **FastAPI backend** with ONNX-optimized inference  
✅ **YOLOv11m** for defect detection, **SegFormer-B4** for road segmentation  
✅ **MLflow** and **Weights & Biases (W&B)** for inference tracking  
✅ **Prometheus + Grafana** for monitoring latency, usage, and alerts  
✅ **S3** for output storage, **RDS PostgreSQL** for prediction logs  
✅ **GitHub Actions** CI/CD pipeline to auto-deploy to **AWS EC2**  
✅ **Fully Dockerized** with separate containers for backend, frontend, MLflow  

---

## 🧠 Model Integration

- ✅ `SegFormer-B4` (semantic segmentation of roads) — 8-class UAVID-style output
- ✅ `YOLOv11m` (object detection of cracks, potholes, etc.)
- ✅ All models run using **ONNXRuntime** for fast, optimized inference
- ✅ Output stored and visualized in real time

---

## 🏗️ Infrastructure Overview

### 🧱 Services (Dockerized)

| Component    | Port   | Description                          |
|--------------|--------|--------------------------------------|
| `backend`    | 8000   | FastAPI ML inference API             |
| `streamlit`  | 8501   | Frontend web UI for predictions      |
| `mlflow`     | 5000   | Experiment tracking & model registry |
| `prometheus` | 9090   | Monitoring metrics endpoint          |
| `grafana`    | 3000   | Visualization of metrics dashboards  |

### ☁️ Cloud Hosting

- EC2 Instance: `t3.medium` or `g4dn.xlarge` (for GPU inference)
- Elastic IP for stable deployment access
- S3 Buckets for outputs, ONNX models
- RDS PostgreSQL for prediction history logging

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/sheraliozodov77/road-ai-mlops.git
cd road-ai-mlops
```

### 2. Configure AWS Infrastructure

Provision:
- ✅ EC2 instance (t3.medium or g4dn.xlarge)
- ✅ RDS PostgreSQL instance
- ✅ S3 buckets for model & output storage

Create a `.env` file with:

```env
POSTGRES_HOST=your-rds-host
POSTGRES_PORT=5432
POSTGRES_DB=roadai
POSTGRES_USER=road_admin
POSTGRES_PASSWORD=your-password

S3_BUCKET=road-ai-prod
S3_MODEL_BUCKET=road-ai-models
SEGFORMER_S3_KEY=models/segformer/segformer-b4-uavid.onnx
YOLOV11_S3_KEY=models/yolov11/yolov11m.onnx

MLFLOW_TRACKING_URI=http://localhost:5000
WANDB_API_KEY=your-wandb-key

AWS_ACCESS_KEY_ID=your-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key

API_URL=http://backend:8000
ENV=production
```

---

## 🐳 Dockerized Deployment

### 1. Build & Run

```bash
docker compose -f docker-compose.prod.yml up --build -d
```

- Backend: `http://<EC2-IP>:8000/docs`
- Frontend: `http://<EC2-IP>:8501`
- MLflow: `http://<EC2-IP>:5000`
- Prometheus: `http://<EC2-IP>:9090`
- Grafana: `http://<EC2-IP>:3000`

---

## 🌐 API Endpoints

| Route              | Method | Description                  |
|--------------------|--------|------------------------------|
| `/predict/image`   | POST   | Upload image & select model  |
| `/predict/video`   | POST   | Upload video & run inference |
| `/jobs/status`     | GET    | Track video job status       |
| `/history`         | GET    | View recent predictions      |
| `/metrics`         | GET    | Prometheus scrape endpoint   |

---

## 📈 Monitoring & Experiment Tracking

- ✅ **MLflow** logs latency, model metadata, artifacts
- ✅ **W&B** logs inference performance in real time
- ✅ **Prometheus + Grafana** dashboards for API request rate, latency, job usage
- ✅ `/metrics` endpoint exposed via FastAPI

---

## 🔁 CI/CD Deployment with GitHub Actions

### ✅ Auto Deployment Flow
On push to `master`:
- SSH into EC2
- Pull latest code
- Rebuild Docker containers
- Register models to MLflow (optional)

Required secrets:
```yaml
EC2_HOST=your.elastic.ip
EC2_SSH_KEY=your-private-ssh-key
```

---

## 🛡️ Resilience & Scaling

- ✅ Robust retry/error handling in API
- ✅ Secrets loaded securely via `.env`
- ✅ `restart: always` in Docker Compose for recovery
- 🔜 Support for GPU inference, async jobs, batch processing

---

## 👨‍💻 Author

**Built by Sherali Ozodov**  
ML Engineer

🔗 GitHub: [github.com/sheraliozodov77](https://github.com/sheraliozodov77)  

---

🗓️ **Last updated:** 2025-08-28  
🏷️ **Tags:** MLOps · Road AI · SegFormer · YOLO · Streamlit · FastAPI · EC2 · Docker · MLflow · Prometheus