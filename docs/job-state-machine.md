# Job state machine

## Backend endpoints

| Intent | Method + path | Notes |
| --- | --- | --- |
| Create job | `POST /jobs` | Creates job in `WAITING_FOR_SELECTION` and enqueues preview extraction. |
| Get job | `GET /jobs/{job_id}` | Returns status, progress, target, player_ref, results, etc. |
| Candidates | `GET /jobs/{job_id}/candidates` | Returns autodetection status + candidate tracks. |
| Select track | `POST /jobs/{job_id}/select-track` | Saves `player_ref`, resets target, moves to `WAITING_FOR_TARGET`. |
| Select target | `POST /jobs/{job_id}/select-target` | Saves confirmed target (including `frame_key`) and moves to `READY_TO_ENQUEUE`. |
| Save target (legacy) | `POST /jobs/{job_id}/target` or `POST /jobs/{job_id}/selection` | Saves target selections and marks `target.confirmed=true` (also stores `frame_key` when provided). |
| Enqueue | `POST /jobs/{job_id}/enqueue` | Validates prerequisites, sets `QUEUED`, launches `run_analysis`. |
| Enqueue (alias) | `POST /jobs/{job_id}/confirm-selection` | Calls enqueue. |

## State machine overview

```text
WAITING_FOR_SELECTION
  └─ (previews queued/extracted) EXTRACTING_PREVIEWS → PREVIEWS_READY
      └─ (candidates tracking) TRACKING_CANDIDATES
          └─ select-track → WAITING_FOR_TARGET
              └─ select-target OR target save → READY_TO_ENQUEUE
                  └─ enqueue → QUEUED → RUNNING → COMPLETED/PARTIAL/FAILED
```

### States & transitions (high level)

- **WAITING_FOR_SELECTION**
  - Initial state after `POST /jobs`.
  - Preview extraction runs in background (`EXTRACTING_PREVIEWS` → `PREVIEWS_READY`).
  - Candidate tracking may run (`TRACKING_CANDIDATES`).

- **WAITING_FOR_TARGET**
  - Triggered by `POST /jobs/{job_id}/select-track` (player selected).
  - Target selection still required.

- **WAITING_FOR_PLAYER**
  - When target is saved before player reference.

- **READY_TO_ENQUEUE**
  - Triggered by `POST /jobs/{job_id}/select-target` or `POST /jobs/{job_id}/target`.
  - `target.confirmed=true` and `target.selections` are present.

- **QUEUED → RUNNING → COMPLETED/PARTIAL/FAILED**
  - `POST /jobs/{job_id}/enqueue` validates prerequisites, sets `QUEUED` and kicks off the worker.
  - Worker sets `RUNNING`, progresses through pipeline steps, and finishes with `COMPLETED`, `PARTIAL`, or `FAILED`.

## Enqueue prerequisites

`POST /jobs/{job_id}/enqueue` requires:

- **`player_ref`** is present and normalizable.
- **`target.confirmed`** is `true`.
- **`target.selections`** contains at least one selection.

If any prerequisite is missing, the endpoint returns `400`:

```json
{
  "ok": false,
  "error": {
    "code": "NOT_READY",
    "missing": ["player_ref", "target"]
  }
}
```

## Target payload expectations

When a target is saved (select-target or save target), the backend persists:

- `target.confirmed: true`
- `target.selections[]` with `frame_time_sec` and `bbox`
- `frame_key` is stored when available to avoid float-only matching

This ensures the job can be enqueued as soon as all prerequisites are met.
