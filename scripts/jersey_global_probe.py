"""Exercise independent reacquisition on real video without writing job state."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

staged = Path(sys.argv[1])
for short in ("association", "window_logic", "jersey_vision", "windowed_tracking"):
    name = "app.reid." + short
    spec = importlib.util.spec_from_file_location(name, staged / (short + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

tracking = sys.modules["app.reid.windowed_tracking"]
from app.reid.team_color_guard import apply_team_color_guard
from app.core.workspace import require_free_space
from app.workers.tracking import _get_s3_client, S3_ENDPOINT_URL

job_id = "796f8c0f-94cd-4d2d-b8b1-a0f6ee5a5b60"
probe_id = "jersey-global-probe-" + staged.name
probe_root = Path("/tmp/fnh_jobs") / probe_id
assert not probe_root.exists(), "Probe workspace already exists"
probe_root.mkdir()
source = probe_root / "input.mp4"
runtime_call_limit = os.getenv("JERSEY_OCR_MAX_CALLS")
os.environ.update(
    JERSEY_OCR_ENABLED="1",
    JERSEY_OCR_MAX_CALLS="128",
    JERSEY_OCR_MAX_SECONDS="240",
    TRACKING_TIMEOUT_SECONDS="480",
)
starts = [1100, 1155, 1210, 1265, 1925, 1980, 2035, 3300]
if os.getenv("GLOBAL_PROBE_FOCUSED") == "1":
    starts = [1155, 1265, 1925]
tracking.legacy.iter_windows = lambda *a, **k: [
    (float(s), float(s + 60)) for s in starts
]
tracking.legacy._update_tracking_progress = lambda *a, **k: None
tracking.legacy._mark_tracking_timeout = lambda *a, **k: None
tracking._persist_tracking_output = lambda job, output, **kwargs: output
original_enrich = tracking.JerseyVerifier.enrich


def trace_enrich(self, path, candidates, start, **kwargs):
    result = original_enrich(self, path, candidates, start, **kwargs)
    print(
        "PROBE_READS "
        + json.dumps(
            {
                "start": start,
                "hints": self.dense_hints(result, start),
                "candidates": [
                    {
                        "id": c.candidate_id,
                        "preferred": (c.metadata or {}).get("jersey_preferred_times"),
                        "evidence": (c.metadata or {}).get("jersey_evidence"),
                        "read_boxes": [
                            {"t": d["t"], "bbox": d["bbox"]}
                            for d in (c.metadata or {}).get("tracklet_detections", [])
                            if any(
                                abs(start + d["t"] - r["time_sec"]) <= 0.05
                                for r in (c.metadata or {})
                                .get("jersey_evidence", {})
                                .get("readings", [])
                            )
                        ],
                    }
                    for c in result
                ],
            }
        ),
        flush=True,
    )
    return result


tracking.JerseyVerifier.enrich = trace_enrich
original_build = tracking._build_candidate_profiles


def trace_build(path, track_map, **kwargs):
    result = original_build(path, track_map, **kwargs)
    hints = kwargs.get("sampling_hints") or []
    if hints:
        nearby = []
        for track_id, detections in track_map.items():
            for detection in detections:
                for hint in hints:
                    delta = abs(detection["t"] - hint["t"])
                    if delta <= 0.4:
                        nearby.append(
                            {
                                "id": track_id,
                                "samples": len(detections),
                                "t": detection["t"],
                                "delta": delta,
                                "iou": tracking.bbox_iou(
                                    detection["bbox"], hint["bbox"]
                                ),
                            }
                        )
        print(
            "PROBE_HINT_MATCHES "
            + json.dumps(sorted(nearby, key=lambda x: -x["iou"])[:12]),
            flush=True,
        )
    return result


tracking._build_candidate_profiles = trace_build
reference = {
    "t": 1192.607,
    "x": 0.6835180759429932,
    "y": 0.5630893283420139,
    "w": 0.040279293060302736,
    "h": 0.10418120490180122,
}
try:
    inputs = list((Path("/tmp/fnh_jobs") / job_id).glob("*/*/input.mp4"))
    if inputs:
        source.hardlink_to(inputs[0])
    else:
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
        max_windows=len(starts),
    )
    guarded = apply_team_color_guard(
        result, input_video_path=source, player_ref=reference
    )
    segments = guarded.get("segments", [])
    diagnostics = {
        "runtime_call_limit": runtime_call_limit,
        "status": guarded.get("tracking_status"),
        "jersey": guarded.get("reid_summary", {}).get("jersey_vision"),
        "identity_search": guarded.get("reid_summary", {}).get("identity_search"),
        "windows": [
            {
                "start": s["window_start"],
                "status": s.get("identity_status"),
                "observations": len(s.get("bboxes", [])),
                "reasons": s.get("reid", {}).get("reason_codes"),
                "search_mode": s.get("reid", {}).get("search_mode"),
                "candidates": [
                    {
                        "id": c.get("candidate_id"),
                        "appearance": c.get("appearance_similarity"),
                        "reasons": c.get("reason_codes"),
                        "jersey": c.get("jersey_evidence"),
                    }
                    for c in s.get("reid", {}).get("candidates", [])
                ],
            }
            for s in segments
        ],
    }
    print("GLOBAL_JERSEY_PROBE " + json.dumps(diagnostics, sort_keys=True), flush=True)
    assert diagnostics["identity_search"][
        "global_windows_attempted"
    ], "No global search occurred"
    assert any(
        s.get("bboxes")
        and s.get("reid", {}).get("identity_link") == "JERSEY_REACQUISITION"
        for s in segments
    ), "No independent reacquisition survived the kit and graph guards"
finally:
    shutil.rmtree(probe_root, ignore_errors=True)
