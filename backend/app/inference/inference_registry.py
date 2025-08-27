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

        # Try downloading model from registry (get actual ONNX file path)
        model_details = client.get_latest_versions(model_name, stages=[stage])[0]
        artifacts = client.list_artifacts(model_details.run_id)
        onnx_artifact = next((a for a in artifacts if a.path.endswith(".onnx")), None)
        if onnx_artifact:
            model_path = mlflow.artifacts.download_artifacts(run_id=model_details.run_id, artifact_path=onnx_artifact.path)
            print(f"[MLflow] ✅ Downloaded ONNX model: {model_path}")
            return ort.InferenceSession(model_path)

        print(f"[MLflow] ⚠️ No ONNX model found in registry for {model_name}. Falling back to local.")

    except Exception as e:
        print(f"[MLflow] Failed to load model from registry: {e}")

    # Local fallback
    if not os.path.exists(local_path):
        print("[S3] Attempting to download model from S3...")
        download_model_from_s3(s3_key, local_path)
    return ort.InferenceSession(local_path)