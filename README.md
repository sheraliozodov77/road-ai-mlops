# 🛣️ Road AI – Infrastructure Defect Detection

A production-ready AI system for **road segmentation** (SegFormer) and **defect detection** (YOLOv11), designed for UAV-based monitoring and deployed using full MLOps best practices.

---

## 🚀 Features

✅ Streamlit UI for image/video input  
✅ FastAPI backend with ONNX inference (SegFormer + YOLOv11)  
✅ AWS S3 for output storage, RDS for prediction logs  
✅ MLflow & Weights & Biases for experiment tracking  
✅ Prometheus + Grafana monitoring dashboards  
✅ GitHub Actions CI/CD pipeline to EC2 instance  
✅ Downloadable predictions via Streamlit UI

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/sheraliozodov77/road-ai-mlops.git
cd road-ai-mlops
```

### 2. Configure AWS Infrastructure

Provision:
- ✅ EC2 instance (with GPU support, e.g., `g4dn.xlarge`)
- ✅ RDS PostgreSQL instance
- ✅ S3 bucket for predictions

Add environment variables to `.env` or Secrets Manager:

```bash
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
ENV=production

API_URL=http://<EC2_PUBLIC_IP>:8000
```

---

## 🚀 Run Backend

```bash
cd backend
docker compose -f ../docker-compose.prod.yml up --build -d
```

Access API docs at: `http://<EC2-IP>:8000/docs`

---

## 🧠 Run Streamlit Frontend

```bash
cd streamlit_app
streamlit run main.py
```

Access UI at: `http://<EC2-IP>:8501`

---

## 📊 Monitoring Dashboard

- **Prometheus**: `http://<EC2-IP>:9090`
- **Grafana**: `http://<EC2-IP>:3000`  
  Load dashboard from: `grafana/road-ai-fastapi-dashboard.json`

---

## 🧪 API Endpoints

| Route              | Method | Description                  |
|--------------------|--------|------------------------------|
| `/predict/image`   | POST   | Upload image & select model  |
| `/predict/video`   | POST   | Upload video & run inference |
| `/jobs/status`     | GET    | Track video job status       |
| `/history`         | GET    | View recent predictions      |
| `/metrics`         | GET    | Prometheus scrape endpoint   |

---

## ⚙️ GitHub Actions CI/CD

Automatically deploys on push to `master` branch.

- ✅ SSHs into EC2
- ✅ Pulls latest code & restarts backend

Required GitHub Secrets:
- `EC2_HOST`, `EC2_SSH_KEY`

---

## 👨‍💻 Author

**Made by Sherali Ozodov**

---

🗓️ Last updated: 2025-08-27