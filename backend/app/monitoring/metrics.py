# ✅ app/monitoring/metrics.py (Corrected)


from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import APIRouter, Request, Response
import time


REQUEST_COUNT = Counter("request_count", "Total API requests", ["method", "endpoint"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "API request latency", ["method", "endpoint"])


router = APIRouter()


@router.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)