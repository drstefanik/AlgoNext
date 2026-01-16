import os

from celery import Celery

from app.core.env import load_env

load_env()

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

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
