# backend/app/inference/inference_registry.py
import os
import mlflow
import onnxruntime as ort
import boto3
import botocore
import json


def download_model_from_s3(s3_key: str, local_path: str):
    bucket = os.environ.get("S3_MODEL_BUCKET")
    if not bucket:
        raise ValueError("S3_MODEL_BUCKET not set")
    s3 = boto3.client("s3")
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, s3_key, local_path)
        print(f"[✅] Downloaded model from S3: s3://{bucket}/{s3_key}")
    except botocore.exceptions.BotoCoreError as e:
        print(f"[❌] Failed to download model from S3: {e}")
        raise


def load_model_from_registry_or_local(model_name: str, stage: str, local_path: str, s3_key: str):
    try:
        print(f"[MLflow] Trying to load {model_name} at stage {stage}...")
        client = mlflow.tracking.MlflowClient()
        model_uri = f"models:/{model_name}/{stage}"
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print(f"[MLflow] Failed to load model from registry: {e}")
        print(f"[Local] Loading ONNX from disk: {local_path}")
        if not os.path.exists(local_path):
            print("[S3] Attempting to download model from S3...")
            download_model_from_s3(s3_key, local_path)
        return ort.InferenceSession(local_path)
