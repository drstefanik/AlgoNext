"""Read three cached video windows with staged code, without publishing job state."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

staged = Path(sys.argv[1])
for short in ("association", "jersey_vision", "windowed_tracking"):
    name = "app.reid." + short
    spec = importlib.util.spec_from_file_location(name, staged / (short + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

tracking = sys.modules["app.reid.windowed_tracking"]
from app.reid.team_color_guard import apply_team_color_guard

job_id = "796f8c0f-94cd-4d2d-b8b1-a0f6ee5a5b60"
attempt = "92f4305d-adb7-44b7-ba63-fe629bbc92f8"
inputs = list((Path("/tmp/fnh_jobs") / job_id / attempt).glob("*/input.mp4"))
probe_id = "jersey-probe-20260926"
probe_root = Path("/tmp/fnh_jobs") / probe_id
assert not probe_root.exists(), "Probe workspace already exists"
probe_root.mkdir()
source = probe_root / "input.mp4"
os.environ.update(
    JERSEY_OCR_ENABLED="1",
    JERSEY_OCR_MAX_CALLS="36",
    JERSEY_OCR_MAX_SECONDS="90",
    TRACKING_TIMEOUT_SECONDS="240",
)
tracking.legacy.iter_windows = lambda *a, **k: [
    (1100.0, 1160.0),
    (1155.0, 1215.0),
    (1210.0, 1270.0),
]
tracking.legacy._update_tracking_progress = lambda *a, **k: None
tracking.legacy._mark_tracking_timeout = lambda *a, **k: None
tracking._persist_tracking_output = lambda job, output, **kwargs: output
reference = {
    "t": 1192.607,
    "x": 0.6835180759429932,
    "y": 0.5630893283420139,
    "w": 0.040279293060302736,
    "h": 0.10418120490180122,
}
try:
    if len(inputs) == 1:
        source.hardlink_to(inputs[0])
    else:
        from app.core.workspace import require_free_space
        from app.workers.tracking import _get_s3_client, S3_ENDPOINT_URL

        require_free_space(probe_root, incoming_bytes=2386411460)
        _get_s3_client(S3_ENDPOINT_URL).download_file(
            os.environ["S3_BUCKET"], f"jobs/{job_id}/input.mp4", str(source)
        )
    assert source.stat().st_size == 2386411460
    result = tracking.track_player_windowed_reid(
        probe_id,
        str(source),
        reference,
        [reference],
        analysis_attempt_id="probe",
        jersey_target_number=8,
        video_duration_sec=6559.339,
        window_sec=60,
        overlap_sec=5,
        fps=1,
        max_windows=3,
    )
    guarded = apply_team_color_guard(
        result, input_video_path=source, player_ref=reference
    )
    diagnostic = {
        "raw_tracking_success": result.get("tracking_success"),
        "guarded_tracking_success": guarded.get("tracking_success"),
        "raw_status": result.get("tracking_status"),
        "guarded_status": guarded.get("tracking_status"),
        "jersey": result.get("reid_summary", {}).get("jersey_vision"),
        "windows": [],
    }
    for segment in result.get("segments", []):
        reid = segment.get("reid", {})
        diagnostic["windows"].append(
            {
                "start": segment.get("window_start"),
                "status": reid.get("status"),
                "reasons": reid.get("reason_codes"),
                "candidates": [
                    {
                        "id": c.get("candidate_id"),
                        "appearance": c.get("appearance_similarity"),
                        "combined": c.get("combined_score"),
                        "reasons": c.get("reason_codes"),
                        "jersey": c.get("jersey_evidence"),
                    }
                    for c in reid.get("candidates", [])
                ],
            }
        )
    print("JERSEY_PROBE_RESULT " + json.dumps(diagnostic, sort_keys=True))
    if not guarded.get("tracking_success"):
        raise SystemExit("Probe did not establish autonomous tracking")
finally:
    shutil.rmtree(probe_root, ignore_errors=True)
