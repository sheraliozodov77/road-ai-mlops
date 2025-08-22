from dotenv import load_dotenv
import boto3

def load_secrets(secret_name: str = "road-ai-prod-secret", region: str = "us-east-1"):
    try:
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_name)
        secret_dict = json.loads(response['SecretString'])

        for key, value in secret_dict.items():
            os.environ[key] = value
        print("[✅] Loaded secrets from AWS Secrets Manager")
    except Exception as e:
        print(f"[⚠️] AWS SecretsManager failed: {e}")
        print("[ℹ️] Falling back to local .env")
        load_dotenv()  # Loads from .env or .env.prod
