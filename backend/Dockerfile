# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install OS packages
RUN apt-get update && apt-get install -y \
    libgl1 \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*
    
# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Create temp output dir
RUN mkdir -p temp

# Entrypoint (loads secrets, starts FastAPI)
CMD ["python", "start_backend.py"]
