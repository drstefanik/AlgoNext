import logging
import os

import torch
from celery import Celery
from celery.signals import worker_ready, worker_shutdown

from app.core.env import load_env

load_env()

from app.core.evaluation_guard import install_evaluation_guard
from app.core.job_recovery import recover_interrupted_jobs
from app.core.runtime_health import (
    APP_GIT_SHA,
    start_worker_heartbeat,
    stop_worker_heartbeat,
)
from app.reid.runtime import install_windowed_reid

install_evaluation_guard()
install_windowed_reid()

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

logger.info(
    "S3 config: S3_ENDPOINT_URL=%s S3_PUBLIC_ENDPOINT_URL=%s S3_BUCKET=%s",
    os.environ.get("S3_ENDPOINT_URL"),
    os.environ.get("S3_PUBLIC_ENDPOINT_URL"),
    os.environ.get("S3_BUCKET"),
)
logger.info(
    "Torch device: cuda_available=%s device=%s",
    torch.cuda.is_available(),
    "cuda" if torch.cuda.is_available() else "cpu",
)
logger.info("Worker revision: %s", APP_GIT_SHA)

celery = Celery(
    "fnh_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.workers.pipeline", "app.workers.ai_report"],
)

celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    broker_connection_retry_on_startup=True,
    worker_prefetch_multiplier=1,
)


@worker_ready.connect
def _on_worker_ready(sender=None, **_kwargs):
    recover_interrupted_jobs()
    worker_name = getattr(sender, "hostname", None)
    start_worker_heartbeat(str(worker_name) if worker_name else None)


@worker_shutdown.connect
def _on_worker_shutdown(**_kwargs):
    stop_worker_heartbeat()
