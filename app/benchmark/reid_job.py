from __future__ import annotations

import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_url(*values: Any) -> str | None:
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        parsed = urllib.parse.urlsplit(value.strip())
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return value.strip()
    return None


def unwrap_job_payload(payload: Any) -> dict[str, Any]:
    source = _mapping(payload)
    data = source.get("data") if source.get("ok") is True else source
    if not isinstance(data, Mapping):
        raise ValueError("job API response does not contain an object payload")
    return dict(data)


def discover_job_artifacts(job_payload: Mapping[str, Any]) -> dict[str, str]:
    result = _mapping(job_payload.get("result"))
    tracking = _mapping(result.get("tracking"))
    tracking_asset = _mapping(tracking.get("asset"))
    assets = _mapping(result.get("assets"))
    input_video = _mapping(assets.get("input_video"))
    reid_summary = _mapping(tracking.get("reid_summary"))

    tracking_url = _first_url(
        tracking_asset.get("signed_url"),
        tracking_asset.get("url"),
        tracking.get("tracking_url"),
    )
    video_url = _first_url(
        input_video.get("signed_url"),
        input_video.get("url"),
        assets.get("input_video_url"),
        assets.get("inputVideoUrl"),
        job_payload.get("video_url"),
    )
    job_id = str(
        job_payload.get("job_id") or job_payload.get("id") or ""
    ).strip()
    identity = str(
        reid_summary.get("identity_id")
        or next(
            (
                _mapping(_mapping(segment).get("reid")).get("identity_id")
                for segment in tracking.get("segments") or []
                if _mapping(_mapping(segment).get("reid")).get("identity_id")
            ),
            "",
        )
        or (f"job-{job_id}-selected-player" if job_id else "selected-player")
    ).strip()

    missing = []
    if not job_id:
        missing.append("job_id")
    if not tracking_url:
        missing.append("tracking_url")
    if not video_url:
        missing.append("video_url")
    if missing:
        raise ValueError("job payload is missing " + ", ".join(missing))
    return {
        "job_id": job_id,
        "identity": identity,
        "tracking_url": tracking_url,
        "video_url": video_url,
    }


def fetch_json(url: str, *, timeout_seconds: float = 30.0) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.load(response)


def download_file(
    url: str,
    destination: str | Path,
    *,
    timeout_seconds: float = 120.0,
) -> Path:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"Accept": "*/*"})
    temporary = output.with_suffix(output.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            with temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return output
