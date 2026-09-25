# Operational reliability review — 2026-09-25

AlgoNext's API, Redis, database and worker were reachable and reported the same
API/worker revision (`67ab429`) during the production inspection. This change
addresses recoverable workflow failures; it does not validate player ratings.

## Corrected failures

- A saved job could disappear from the client after a queue publish failure or
  a failed follow-up GET. The API now exposes the saved job and an explicit
  retryable preparation failure; the frontend retains a known ID for polling.
- Exhausted preview/candidate dispatch failures could leave preparation pending
  indefinitely. They now finish as FAILED with a preparation retry action.
- Failed preparation can reuse its saved input and preview frames without
  requiring a player selection that has not yet become possible. Retry still
  checks worker readiness, retry limits and the current analysis attempt.
- A delayed initial kickoff cannot adopt a newer preparation attempt. An
  ambiguous queue acknowledgement preserves progress from a started worker.
- Static `/frames/list` and `/frames/overlay` routes now take precedence over
  the filename route. Before the fix, both were incorrectly served as filenames
  and returned 404; their intended contracts are 200 and 410 respectively.
- Zero timestamps and track ID zero survive request normalization. Invalid,
  nonfinite or out-of-frame coordinates yield JSON validation errors instead
  of an internal serialization exception.
- The paired frontend change isolates frames and selections when switching
  jobs, tolerates unavailable browser storage, applies deadlines to retry and
  profile requests, rejects truncated JSON, and preserves bodyless proxy
  responses and backend diagnostic headers.

## Verification

The executable test steps from `api-contract.yml` and `runtime-resilience.yml`
passed locally with the declared FastAPI/Pydantic versions: 268 test executions,
including 22 new reliability tests. Runtime tests replace model inference and
external services with controlled fixtures; they do not certify YOLO/ReID
accuracy or production infrastructure.

The frontend production build, existing selection/outcome checks and 12 new
transport/session/proxy tests passed. The new backend tests are included in API
contract CI, and frontend reliability checks run before every production build.

For a deployed revision, verify API and worker SHA agreement with `/ready`,
the frame routes above, and one short owned-video preparation job before
declaring deployment verified. No database migration is required.

## Remaining acceptance boundary

The existing full-match fixture
`6520b68b-8de6-43b7-88d2-41375bded0a0` completed 108 processing windows but
remained `WAITING_FOR_PLAYER`. Its diagnostics found both reference anchors
before filtering, but did not establish autonomous identity across shots.
This is a failed identity acceptance result, not a completed player evaluation.

Cross-shot identity, calibrated physical measures and event-based player
ratings still require separate validation against annotated match evidence.
`match_rating_10` and `impact_100` must remain unavailable until their evidence
requirements pass. Existing truth gates and identity thresholds are unchanged.
LGI integration data was outside this verification scope.
