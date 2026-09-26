# Selected-player tracking diagnosis — 26 September 2026

## Verified runtime evidence

Read-only production diagnostics were collected in GitHub Actions run
`36230912676`, job `108373769486`.

- The recovered attempt `38f55b60-871a-4bfe-8a56-f4e3c273989d` ended at
  07:28:58 UTC with `ANCHOR_ONLY` and `RESELECT_PLAYER`.
- A subsequent attempt `6d709f98-c44a-4184-8226-35aa562f407c` ran from
  07:45:54 to 08:05:36 UTC and ended with the same tracking failure.
- The latest result reports 120/120 windows processed and two manual anchors
  matched before the final guard. No autonomous identity evidence was accepted.
- The current job status is `WAITING_FOR_PLAYER`; the analysis outcome is
  `pipeline_state=DONE`, `tracking_state=FAILED`, `evaluation_status=TRACKING_FAILED`.
- The API and worker are on revision `057735655f76a6429cceddd6bd23903dc5ecded0`.
  Database, Redis, and worker readiness checks pass.
- At 08:49 UTC the server had approximately 8 GiB free (95% used). The earlier
  disk incident remains a capacity concern, but the latest completed attempts
  ended with an identity rejection rather than the earlier persistence error.

## Reference inconsistency

The primary reference is at video time 198.768 seconds (03:18). Its frame shows
a night-time sequence with the scoreboard ordered FIO–BOL. The second reference
is at 1192.607 seconds (19:52). It shows a daytime sequence with the scoreboard
ordered BOL–FIO; the selected box contains a white-shirted player with a visible
number 8. The field-clock reading in that second image is 07:43.

Both selection previews were checked against frames extracted directly from
the AlgoNext-owned cached input video at the same timestamps. Their content
matches the input video. Preview/source mix-up is therefore not indicated by
these two checks. Additional previews at 13:15 and 16:33 also show the daytime
BOL–FIO sequence.

This is evidence of inconsistent reference contexts, not proof that it is the
sole cause of the association failure. The desired player is not identified in
the saved profile, so clarification of the intended team and shirt number is
needed before replacing the primary reference.

## Limits and next action

`TEAM_COLOR_GUARD_UNVERIFIED_FAILURE_OUTPUT` is added when an already-failed
tracking result still contains unverified selected-player data. It does not,
by itself, prove an actual shirt-colour mismatch. The guard preserves anchor
diagnostics but clears per-window candidate scores and rejection details.

No thresholds were loosened, no unverified scores were exposed, and no further
analysis was enqueued during this investigation. First correct the reference
context for the intended player, then evaluate whether autonomous tracking is
actually demonstrated. The operational recovery alone does not establish that
the player-recognition module is reliable.
