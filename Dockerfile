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

# Prefetch the lightweight, person-ReID-specific OSNet checkpoint. The worker
# never downloads identity models while a match is running.
RUN mkdir -p /opt/algonext-models \
    && HF_HUB_DISABLE_XET=1 python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='kaiyangzhou/osnet', filename='osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth', local_dir='/opt/algonext-models')"
