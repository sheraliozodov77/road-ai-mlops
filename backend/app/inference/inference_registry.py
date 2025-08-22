import os
import onnxruntime as ort
import mlflow.onnx
import mlflow
from pathlib import Path

def load_model_from_registry_or_local(model_name: str, stage: str, local_path: str):
    try:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        model_uri = f"models:/{model_name}/{stage}"
        print(f"[MLflow] Trying to load model from: {model_uri}")
        model_path = mlflow.artifacts.download_artifacts(model_uri)
        model_file = Path(model_path) / "model.onnx"
        if not model_file.exists():
            raise FileNotFoundError("ONNX model not found in MLflow registry")
        return ort.InferenceSession(str(model_file))
    except Exception as e:
        print(f"[Fallback] Loading local model from: {local_path} (reason: {e})")
        return ort.InferenceSession(local_path)