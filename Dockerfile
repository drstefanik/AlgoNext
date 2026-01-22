FROM python:3.11-slim AS base

# Install system dependencies (ffmpeg + ffprobe)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV ULTRALYTICS_AUTOINSTALL=0 \
    ULTRALYTICS_CHECKS=0 \
    YOLO_AUTOINSTALL=0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS api

COPY alembic.ini ./alembic.ini
COPY alembic ./alembic
COPY app ./app

FROM base AS migrate

COPY alembic.ini ./alembic.ini
COPY alembic ./alembic
COPY app ./app

ENV PYTHONPATH=/app

FROM base AS worker

COPY alembic.ini ./alembic.ini
COPY alembic ./alembic
COPY app ./app

RUN python -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"
