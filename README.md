# 🛣️ Road AI – Infrastructure Defect Detection

A production-ready AI system for **road segmentation** (SegFormer) and **defect detection** (YOLOv11), designed for UAV footage and deployed with full MLOps best practices.

---

## 🚀 Features

✅ Streamlit UI to upload image/video  
✅ FastAPI backend with ONNX inference  
✅ AWS S3 for storage, RDS for logs  
✅ MLflow + Weights & Biases tracking  
✅ Prometheus + Grafana for monitoring  
✅ GitHub Actions CI/CD to EC2  

---

## 🔧 Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/sheraliozodov77/road-ai-mlops.git
cd road-ai-mlops
```

### 2. Set up AWS

- Create EC2 (GPU), RDS (PostgreSQL), S3 bucket
- Add `.env.prod` or use AWS Secrets Manager

### 3. Build + Run Backend

```bash
cd backend
docker compose -f ../docker-compose.prod.yml up --build -d
```

Accessible at: `http://<EC2-IP>:8000/docs`

---

## 📊 Monitoring

- Prometheus: `http://<EC2-IP>:9090`
- Grafana: `http://<EC2-IP>:3000`
- Dashboard JSON: `grafana/road-ai-fastapi-dashboard.json`

---

## 🧪 API Endpoints

| Route              | Method | Description                  |
|--------------------|--------|------------------------------|
| `/predict/image`   | POST   | Upload image + select model |
| `/predict/video`   | POST   | Upload video + monitor job  |
| `/jobs/status`     | GET    | Check job progress          |
| `/history`         | GET    | Past predictions             |
| `/metrics`         | GET    | Prometheus metrics          |

---

## ⚙️ CI/CD (GitHub Actions)

- Auto-deploy on `main` push
- SSHs into EC2 and restarts backend

Required Secrets:
- `EC2_HOST`, `EC2_SSH_KEY`

---

## 👨‍💻 Authors

Built by Sherali Ozodov