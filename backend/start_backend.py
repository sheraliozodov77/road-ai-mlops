# backend/start_backend.py
import sys
sys.path.append("/app")

import os
from fastapi import FastAPI
import uvicorn
from app.utils.secrets import load_secrets

# ✅ Load secrets from .env or AWS Secrets Manager
load_secrets()

# ✅ Start FastAPI app
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)

