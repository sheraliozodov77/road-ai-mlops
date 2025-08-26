import mlflow
import wandb
import os
from datetime import datetime

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("RoadAI-Inference")

wandb.login(key=os.environ.get("WANDB_API_KEY"))


def log_inference_metrics(model_name: str, runtime: str, input_type: str, filename: str, output_path: str):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # ✅ Robust runtime parsing
    runtime_clean = runtime.replace("ms", "").replace("~", "").replace("s", "").replace("<", "")
    try:
        runtime_value = float(runtime_clean)
        runtime_ms = runtime_value * 1000 if "s" in runtime else runtime_value
    except ValueError:
        runtime_ms = -1  # Fallback if malformed input

    with mlflow.start_run(run_name=f"{model_name}-{input_type}-{timestamp}"):
        mlflow.log_param("model", model_name)
        mlflow.log_param("input_type", input_type)
        mlflow.log_param("filename", filename)
        mlflow.log_metric("runtime_ms", runtime_ms)

        if os.path.exists(output_path):
            mlflow.log_artifact(output_path, artifact_path="outputs")

    wandb.log({
        "model": model_name,
        "input_type": input_type,
        "filename": filename,
        "runtime_ms": runtime_ms
    })
