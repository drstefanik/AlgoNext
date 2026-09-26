"""Temporary video workspace limits; never touches stored inputs or results."""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)
ROOT = Path("/tmp/fnh_jobs")
COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
GIB = 1024**3


class InsufficientWorkspaceError(RuntimeError):
    pass


def require_free_space(path: Path, *, incoming_bytes: int = 0) -> None:
    """Reserve space for Redis/Postgres and refuse growth before disk exhaustion."""
    try:
        reserve = float(os.getenv("WORKSPACE_MIN_FREE_GB", "4"))
        if not 1 <= reserve <= 100:
            reserve = 4
    except ValueError:
        reserve = 4
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    required = int(reserve * GIB) + max(0, int(incoming_bytes))
    if free < required:
        raise InsufficientWorkspaceError(
            "Spazio temporaneo insufficiente: analisi sospesa prima di esaurire il disco "
            f"(liberi {free / GIB:.1f} GB, necessari {required / GIB:.1f} GB)."
        )


def cleanup_tracking_workspace(
    job_id: str, attempt_id: str | None, *, root: Path = ROOT
) -> bool:
    if os.getenv("KEEP_WORKDIR", "0") == "1":
        return False
    components = (str(job_id), str(attempt_id or "legacy"))
    if not all(COMPONENT.fullmatch(value) for value in components):
        return False
    job_dir = root / components[0]
    attempts_dir = job_dir / "attempts"
    target = attempts_dir / components[1]
    # Do not traverse symlinks even if a malformed workspace was left behind.
    if any(path.is_symlink() for path in (root, job_dir, attempts_dir, target)):
        return False
    if not target.is_dir():
        return False
    try:
        shutil.rmtree(target)
        logger.info("Tracking workspace removed job_id=%s attempt_id=%s", *components)
        return True
    except OSError:
        logger.warning(
            "Tracking workspace cleanup failed job_id=%s attempt_id=%s", *components
        )
        return False
