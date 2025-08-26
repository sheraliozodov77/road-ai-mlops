# app/monitoring/metrics.py

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from fastapi import APIRouter, Response

router = APIRouter()

# Metric Definitions
REQUEST_COUNT = Counter("request_count", "Total API requests", ["method", "endpoint"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "API request latency", ["method", "endpoint"])

# Safe registration to avoid duplication
def safe_register(metric):
    try:
        REGISTRY.register(metric)
    except ValueError:
        pass

safe_register(REQUEST_COUNT)
safe_register(REQUEST_LATENCY)

# /metrics route for Prometheus scraping
@router.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
