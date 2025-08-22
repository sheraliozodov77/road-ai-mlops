# backend/app/inference/inference_registry.py
import os
import mlflow
import onnxruntime as ort
from botocore.exceptions import NoCredentialsError
import boto3


def download_from_s3_if_needed(bucket_name: str, s3_key: str, local_path: str):
    if os.path.exists(local_path):
        return local_path

    print(f"[S3] Downloading {s3_key} from {bucket_name} → {local_path}")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        s3 = boto3.client("s3")
        s3.download_file(bucket_name, s3_key, local_path)
        print(f"[S3] Download complete ✅")
    except NoCredentialsError:
        print("[S3] AWS credentials not found. Skipping download.")
    except Exception as e:
        print(f"[S3] Failed to download: {e}")

    return local_path


def load_model_from_registry_or_local(model_name: str, stage: str, local_path: str):
    try:
        print(f"[MLflow] Trying to load {model_name} at stage {stage}...")
        model_uri = f"models:/{model_name}/{stage}"
        model_path = mlflow.artifacts.download_artifacts(model_uri)
        onnx_path = [f for f in os.listdir(model_path) if f.endswith(".onnx")]
        if onnx_path:
            full_path = os.path.join(model_path, onnx_path[0])
            print(f"[MLflow] Model loaded from registry: {full_path}")
            return ort.InferenceSession(full_path)
    except Exception as e:
        print(f"[MLflow] Failed to load model from registry: {e}")

    # Optional S3 fallback
    try:
        s3_bucket = os.environ.get("S3_MODEL_BUCKET")
        s3_key = os.environ.get(f"{model_name.upper()}_S3_KEY")
        if s3_bucket and s3_key:
            local_path = download_from_s3_if_needed(s3_bucket, s3_key, local_path)
    except Exception as e:
        print(f"[S3 Fallback] Error: {e}")

    print(f"[Local] Loading ONNX from disk: {local_path}")
    return ort.InferenceSession(local_path)
