# scripts/register_models.py

import mlflow
import mlflow.onnx
import onnx
import os

# Load secrets from AWS or .env
from app.utils.secrets import load_secrets
load_secrets()

# Set MLflow tracking
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("RoadAI-Models")

def register_model(model_path, model_name, architecture):
    model = onnx.load(model_path)

    with mlflow.start_run(run_name=f"register-{model_name}"):
        mlflow.log_param("architecture", architecture)
        mlflow.log_param("onnx_path", model_path)

        mlflow.onnx.log_model(
            onnx_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )
        print(f"✅ Registered: {model_name} from {model_path}")

if __name__ == "__main__":
    register_model("models/segformer/segformer-b4-uavid.onnx", "SegFormer-RoadAI", "SegFormer-B4")
    register_model("models/yolov11/yolov11m.onnx", "YOLOv11-RoadAI", "YOLOv11-Medium")