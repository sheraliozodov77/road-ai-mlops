import boto3
import os
import json
from datetime import datetime

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name="us-east-1"  # or your region
)

BUCKET = os.environ.get("S3_BUCKET", "road-ai-prod")

def upload_file_to_s3(local_path: str, s3_folder: str) -> str:
    filename = os.path.basename(local_path)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    key = f"{s3_folder}/{timestamp}_{filename}"

    s3.upload_file(local_path, BUCKET, key)

    return f"https://{BUCKET}.s3.amazonaws.com/{key}"
