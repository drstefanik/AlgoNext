# PRO Backend Flow (Pick Player → Confirm Target → Enqueue → Result)

## Overview
This flow enforces a **single, unambiguous player/target selection** and guarantees that the final
`job.result` always includes an `overallScore` and `radar` (MVP vision-based scoring when needed).

## 1) Pick player from a preview frame
**Endpoint:** `POST /jobs/{jobId}/pick-player`

**Payload**
```json
{
  "frame_key": "jobs/<id>/frames/frame_0001.jpg",
  "track_id": 9
}
```

**Behavior**
- Locates the preview frame and track (by `frame_key` + `track_id`).
- Saves `job.player_ref` with frame-based metadata.
- Creates a draft `job.target` (confirmed = false) from the chosen track.
- Progress step moves to `WAITING_FOR_TARGET_CONFIRMATION`.

## 2) Confirm the target (coherence enforced)
**Endpoint:** `POST /jobs/{jobId}/target`

**Payload**
```json
{
  "frame_key": "jobs/<id>/frames/frame_0001.jpg",
  "time_sec": 2636.344,
  "bbox": {"x": 0.1, "y": 0.2, "w": 0.2, "h": 0.3}
}
```

**Coherence rules**
- If `job.player_ref.track_id` exists and the frame has tracks:
  - The confirmed bbox must match the selected track with IoU ≥ 0.2.
  - On mismatch, the API returns `409 TARGET_MISMATCH` unless `force=true` is provided.
- On success, `job.target.confirmed = true` and `job.progress.step = READY_TO_ENQUEUE`.

## 3) Enqueue (strict readiness)
**Endpoint:** `POST /jobs/{jobId}/enqueue`

**Readiness checks**
- Missing `player_ref` → `missing: ["player_ref"]`
- Missing confirmed target → `missing: ["target"]`

If any prerequisites are missing, the API returns:
```json
{
  "error": {
    "code": "NOT_READY",
    "missing": ["player_ref", "target"],
    "message": "Missing required selections for enqueue"
  }
}
```

## 4) Final result always populated
At the end of processing, the backend writes:
```json
{
  "overallScore": 0-100,
  "radar": {
    "tracking_quality": 0-100,
    "activity_proxy": 0-100,
    "visibility": 0-100,
    "consistency": 0-100,
    "stability": 0-100
  }
}
```

If traditional football metrics are unavailable, an **MVP vision-based scoring** is computed using
tracking signals (coverage, lost segments, motion continuity, and bbox stability).
