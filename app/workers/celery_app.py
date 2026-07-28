import logging
import os
import socket

import torch
from celery import Celery
from celery.signals import worker_init, worker_ready, worker_shutdown

from app.core.env import load_env

load_env()

from app.core.evaluation_guard import install_evaluation_guard
from app.core.job_recovery import recover_interrupted_jobs
from app.core.pipeline_policy import install_worker_pipeline_policy
from app.core.preview_asset_policy import install_worker_preview_asset_policy
from app.core.runtime_health import (
    APP_GIT_SHA,
    inspect_runtime,
    start_worker_heartbeat,
    stop_worker_heartbeat,
)
from app.reid.runtime import install_windowed_reid

logger = logging.getLogger(__name__)

install_evaluation_guard()
install_windowed_reid()

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


@worker_init.connect
def _on_worker_init(**_kwargs):
    """Install policies after import and before the worker begins consuming tasks."""

    tracking_policy_installed = install_worker_pipeline_policy()
    preview_asset_policy_installed = install_worker_preview_asset_policy()
    logger.info(
        "Worker policies installed at init: tracking_only=%s immutable_previews=%s",
        tracking_policy_installed,
        preview_asset_policy_installed,
    )


@worker_ready.connect
def _on_worker_ready(sender=None, **_kwargs):
    # Idempotent safety net in case a custom worker boot sequence skipped the
    # worker_init hook. Both policies must already be active in the normal path.
    tracking_policy_installed = install_worker_pipeline_policy()
    preview_asset_policy_installed = install_worker_preview_asset_policy()
    logger.info(
        "Worker policy safety net: tracking_only=%s immutable_previews=%s",
        tracking_policy_installed,
        preview_asset_policy_installed,
    )
    worker_name = getattr(sender, "hostname", None)
    normalized_worker_name = str(worker_name) if worker_name else socket.gethostname()
    start_worker_heartbeat(normalized_worker_name)
    runtime_snapshot = inspect_runtime()
    heartbeat = runtime_snapshot.get("worker") or {}
    heartbeat_confirmed = bool(
        (runtime_snapshot.get("dependencies") or {}).get("worker") == "ready"
        and str(heartbeat.get("worker_name") or "") == normalized_worker_name
        and str(heartbeat.get("revision") or "") == APP_GIT_SHA
        and str(heartbeat.get("pid") or "") == str(os.getpid())
    )
    if heartbeat_confirmed:
        recover_interrupted_jobs(
            recovery_owner=f"{normalized_worker_name}:{APP_GIT_SHA}",
            recovery_revision=APP_GIT_SHA,
        )
    else:
        logger.warning(
            "Interrupted-job recovery skipped: current worker heartbeat unconfirmed"
        )


@worker_shutdown.connect
def _on_worker_shutdown(**_kwargs):
    stop_worker_heartbeat()
