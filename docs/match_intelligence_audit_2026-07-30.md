# Match-intelligence capability audit — 2026-07-30

## Scope

This audit compares the capability contract returned to the result UI with the
backend paths that actually produce evidence. A capability is not promoted to
validated merely because code exists: runtime wiring, sufficient observations
and a locked real-match benchmark are separate gates.

## Findings and implementation status

| Capability | Before | Implemented in this change | Public status |
| --- | --- | --- | --- |
| Person detection | YOLO person detections | Reused as the common observation pass | Available |
| Short-term tracking | ByteTrack within a window | Reused for people and ball observations | Available |
| Cross-shot player ReID | HSV kit-colour descriptor; insufficient real canary evidence | OSNet x0.25 person-ReID embedding, hybrid colour signal, lazy CPU inference, build-time weight prefetch and rollback controls | Experimental until the locked cross-shot benchmark passes |
| Camera-motion compensation | Standalone shot/geometry analysis, not used by player motion | Robust median displacement from at least three shared person tracks; camera-compensated selected-player path per continuous window | Experimental; translation-only and not benchmark-validated |
| Pitch calibration | Validated homography and calibrated-kinematics libraries; no semantic landmark model | Semantic pitch-keypoint benchmark contract and quality gate integrated from the existing calibration work | Foundation; automatic runtime calibration is not yet claimed |
| Ball tracking | No runtime output | COCO sports-ball detections collected in the existing YOLO/ByteTrack pass, aggregated across overlapping windows | Experimental |
| Event recognition | No runtime output | Auditable selected-player/ball proximity sequences | Experimental; not a tactical event classifier |
| Athletic metrics | Calibrated-kinematics diagnostic existed but had no automatic calibration input | Capability contract now distinguishes foundation, experimental and validated states | Foundation until pitch calibration, identity and timing are validated together |
| Technical/tactical evaluation | Legacy heuristic scores suppressed by the truth gate | Ball/event evidence foundation only; no unvalidated player score is emitted | Unavailable |

## Root causes

1. The result contract used static booleans, so completed backend foundations
   always appeared as unavailable.
2. The existing appearance descriptor was dominated by kit colour and did not
   provide a learned person-ReID signal across camera changes.
3. People, camera motion and ball evidence were not composed in one auditable
   per-window contract.
4. Calibration math and benchmark tooling existed independently of runtime
   observability.
5. There is not yet enough labelled real-match evidence to call cross-shot
   identity, ball events, physical metrics or player scoring validated.

## Safety and rollout controls

- All new observation payloads use `match-observability-v1`.
- Experimental outputs carry `validated: false` and cannot bypass the player
  evaluation truth gate.
- `PLAYER_REID_DESCRIPTOR_BACKEND=hsv` rolls ReID back without rebuilding.
- `PLAYER_REID_LEARNED_FAIL_OPEN=1` keeps tracking alive if the learned model is
  unavailable; mixed descriptor samples abstain conservatively rather than
  crashing a window.
- `BALL_TRACKING_ENABLED=0` disables ball collection.
- The UI renders `Available`, `Experimental`, `Foundation` and `Unavailable`
  separately instead of flattening all states into a misleading boolean.

## Promotion gates

Before changing any experimental capability to validated:

1. Freeze labelled multi-camera matches containing cuts, replay, occlusion and
   visually similar kits.
2. Pass the existing tracking/ReID quality gates on enough autonomous windows,
   observations and coverage.
3. Add ball detection and proximity-event precision/recall gates on labelled
   match clips.
4. Validate semantic pitch keypoints and one homography per camera segment.
5. Validate physical metrics only on segments with verified identity, timing
   and calibration.
6. Define a labelled technical/tactical event taxonomy before deriving any
   player rating.
