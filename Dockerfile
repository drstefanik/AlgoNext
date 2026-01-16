FROM python:3.11-slim AS base

# Install system dependencies (ffmpeg + ffprobe)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

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
