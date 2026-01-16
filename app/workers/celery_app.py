import logging
import os

from celery import Celery

from app.core.env import load_env

load_env()

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

logger.info(
    "S3 config: S3_ENDPOINT_URL=%s S3_PUBLIC_ENDPOINT_URL=%s S3_BUCKET=%s",
    os.environ.get("S3_ENDPOINT_URL"),
    os.environ.get("S3_PUBLIC_ENDPOINT_URL"),
    os.environ.get("S3_BUCKET"),
)

celery = Celery(
    "fnh_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.workers.pipeline"],  # <-- IMPORTANTISSIMO
)

celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    broker_connection_retry_on_startup=True,
)
