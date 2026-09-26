"""Reclaim only disposable extracted windows for two known terminal jobs.

Requires all workers to report empty active/reserved/scheduled queues. Original
inputs, tracking JSON, object storage, database and Redis files are untouched.
"""

import json
import os
import shutil
import time
from pathlib import Path

from celery import Celery
from app.core.db import SessionLocal
from app.core.models import AnalysisJob

app = Celery(broker=os.environ["CELERY_BROKER_URL"])
inspector = app.control.inspect(timeout=5)
for method in ("active", "reserved", "scheduled"):
    snapshot = getattr(inspector, method)()
    if not snapshot or any(snapshot.values()):
        raise SystemExit("CLEANUP_SKIPPED_WORKER_NOT_CONFIRMED_IDLE")
known = (
    "796f8c0f-94cd-4d2d-b8b1-a0f6ee5a5b60",
    "6520b68b-8de6-43b7-88d2-41375bded0a0",
)
terminal = {"DONE", "COMPLETED", "SUCCEEDED", "FAILED", "WAITING_FOR_PLAYER"}
root = Path("/tmp/fnh_jobs")
removed = []
with SessionLocal() as db:
    for job_id in known:
        job = db.get(AnalysisJob, job_id)
        if job is None or job.status not in terminal:
            continue
        attempts = root / job_id / "attempts"
        if attempts.is_symlink() or not attempts.is_dir():
            continue
        for attempt in attempts.iterdir():
            windows = attempt / "tracking" / "windows"
            if any(
                path.is_symlink()
                for path in (
                    root,
                    attempts.parent,
                    attempts,
                    attempt,
                    windows.parent,
                    windows,
                )
            ):
                continue
            if not windows.is_dir() or time.time() - windows.stat().st_mtime < 600:
                continue
            files = list(windows.iterdir())
            if any(
                file.is_symlink()
                or not file.is_file()
                or not file.name.startswith("window_")
                or file.suffix != ".mp4"
                for file in files
            ):
                continue
            size = sum(file.stat().st_size for file in files)
            shutil.rmtree(windows)
            removed.append({"job_id": job_id, "attempt": attempt.name, "bytes": size})
print(
    json.dumps(
        {
            "removed_temporary_windows": removed,
            "free_bytes": shutil.disk_usage(root).free,
        },
        sort_keys=True,
    )
)
