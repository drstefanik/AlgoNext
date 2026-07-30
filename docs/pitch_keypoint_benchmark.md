# Semantic pitch-keypoint benchmark

## Status

This benchmark defines what an automatic football-pitch keypoint model must prove
before AlgoNext can turn model output into a pitch homography. It does not ship a
validated learned model and does not enable automatic calibration.

The production capabilities therefore remain:

```json
{
  "pitch_calibration": false,
  "athletic_metrics": false,
  "technical_tactical_scoring": false
}
```

## Why line intersections are not enough

The camera-analysis stage can find white lines and geometric intersections. It
cannot safely decide that an intersection is, for example, the left penalty-area
top corner rather than a goal-area corner or the corresponding point on the
opposite side of the pitch.

A geometrically clean but semantically reversed set of labels can create a
plausible, wrong homography. Semantic labels therefore have their own dataset,
metrics and release gate.

## Versioned contracts

### `pitch-keypoint-annotation-v1`

Each annotated frame declares:

- stable `frame_id`, `video_id` and `shot_id`;
- timestamp and frame dimensions;
- whether the frame is a pitch view;
- dataset split: `development`, `validation` or `test`;
- visible or occluded canonical landmarks in normalized image coordinates.

Non-pitch frames must contain no pitch keypoints. They are required to measure
false positives on close-ups, crowd shots, benches, graphics and replay inserts.

### `pitch-keypoint-prediction-v1`

Each prediction declares:

- the exact frame metadata;
- `model_version` and `configuration_hash`;
- an explicit `abstained` decision;
- reason codes when abstaining;
- at most one prediction per canonical landmark;
- normalized coordinates and confidence.

An abstained prediction cannot contain keypoints. Unknown landmarks, duplicate
landmarks, unknown fields, non-finite coordinates and unsupported schema versions
are rejected.

## Canonical landmark vocabulary

The vocabulary is shared with the pitch-calibration model and currently includes:

- four pitch corners;
- top and bottom halfway-line intersections;
- centre spot and centre-circle horizontal tangencies;
- penalty-area and goal-area line intersections on both sides;
- both penalty spots.

Left and right are canonical pitch-space labels, not screen-space labels. A model
must resolve field orientation rather than silently mirroring the pitch.

## Semantic and localization metrics

Predictions below the configured confidence threshold are treated as absent.
Remaining predictions are matched by semantic label. A prediction is a true
positive only when its normalized localization error is within the configured
match radius.

The report includes:

- semantic precision, recall and F1;
- mean, median and p95 normalized localization error;
- PCK at 1%, 2% and 5% of frame diagonal;
- pitch-frame prediction coverage;
- frame-level false-positive rate on non-pitch views;
- abstention rate.

A correctly named point far from its annotation is counted as both a false
positive and a false negative. Missing labels cannot be hidden by a high PCK on
the small subset that was predicted.

## Calibration-readiness diagnostic

For pitch frames with enough high-confidence semantic points, the benchmark fits
an image-to-field homography using the canonical landmark coordinates. It then
measures:

- number of semantic points;
- RANSAC inlier ratio;
- RMSE and p95 error in metres;
- convex-hull coverage in image space;
- convex-hull coverage in pitch space;
- solved and validated calibration rates.

This is a benchmark diagnostic. Production still routes accepted correspondences
through the separately validated `pitch-calibration-request-v1` solver.

`build_calibration_request` returns a request only when:

- the prediction did not abstain;
- at least six points pass confidence;
- image-space coverage is sufficient;
- all labels belong to the canonical vocabulary.

Model provenance remains in the prediction artifact. The strict calibration
request itself contains only fields accepted by its versioned contract.

## Initial quality gate

| Metric | Initial gate |
|---|---:|
| Semantic F1 | >= 0.75 |
| PCK@0.02 | >= 0.65 |
| p95 normalized error | <= 0.035 |
| Non-pitch frame false-positive rate | <= 0.05 |
| Validated calibration rate among attempts | >= 0.70 |
| Pitch-frame prediction coverage | >= 0.60 |

These are engineering gates for beginning controlled experiments. They are not a
claim that a model is accurate enough for production or that athletic metrics are
validated.

## CLI

```bash
python scripts/evaluate_pitch_keypoints.py \
  --annotations dataset/annotations \
  --predictions runs/model-v1 \
  --json-out reports/model-v1.json \
  --fail-on-gate
```

A file can be compared with a file. For directories, annotation and prediction
JSON filenames must match exactly. Extra prediction frames are rejected; missing
prediction frames are treated as explicit abstention.

## Dataset protocol

A serious dataset should contain independent videos and clubs across splits. The
locked test split must never be used to choose thresholds. Coverage should include:

- wide main-camera shots;
- partial pitch views;
- pan, tilt and zoom;
- both pitch orientations;
- youth and senior pitch dimensions;
- shadows, rain, floodlights and worn markings;
- replay, close-up, crowd, bench and graphic negatives;
- centre-circle, penalty-area and goal-area ambiguities;
- severe perspective and partly occluded landmarks.

At least two annotators should label a reviewed subset. Inter-annotator
localization and semantic disagreement should be reported before model scores are
interpreted.

## Activation requirements

Automatic calibration remains disabled until all of the following hold:

1. a learned or otherwise validated semantic keypoint model is versioned;
2. development and validation thresholds are frozen;
3. the locked test split passes the gate by video, not only in aggregate;
4. left/right orientation errors are reviewed separately;
5. false positives on non-pitch shots remain below gate;
6. generated calibration requests pass the homography quality gate;
7. per-shot calibration error is measured in metres across the visible pitch;
8. model, weights, configuration and threshold hashes are persisted;
9. rollout is protected by a kill switch and production monitoring.

Even after this gate, player scoring remains suspended until ball events and the
scoring model are independently validated.
