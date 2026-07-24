FROM python:3.11-slim AS base

ARG APP_GIT_SHA=unknown
ARG APP_BUILD_TIME=unknown

# System deps:
# - ffmpeg (video)
# - build toolchain + BLAS/LAPACK (for building lapx wheels on slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    gfortran \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV ULTRALYTICS_AUTOINSTALL=0 \
    ULTRALYTICS_CHECKS=0 \
    YOLO_AUTOINSTALL=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_GIT_SHA=${APP_GIT_SHA} \
    APP_BUILD_TIME=${APP_BUILD_TIME}

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

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

# Prefetch both detector profiles so runtime never depends on external downloads.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11s.pt'); YOLO('yolo11n.pt')"
