# Tracking benchmark

This benchmark measures whether AlgoNext keeps the **same annotated person**
associated with a stable tracker identity. It does not measure football ability,
athletic performance, or tactical quality.

## Why this exists

A deployment that returns bounding boxes is not automatically a reliable player
tracker. The benchmark makes failures measurable:

- missed players and false detections;
- low box overlap;
- track fragmentation;
- identity switches;
- poor identity association despite good detection recall;
- long stretches in which the annotated player is lost.

The benchmark output is versioned as `tracking-benchmark-report-v1`.

## Evaluation set

Use clips or complete matches for which a human annotator has verified the target
identity. An annotation frame is part of the scored set. Prediction frames that
are not present in the annotation file are reported as unscored and do not become
false positives. This permits sparse annotation, but benchmark reports must state
their annotation density.

Recommended dataset split:

1. **Development**: visible during implementation.
2. **Validation**: used for threshold selection.
3. **Test**: locked until a release candidate is evaluated.

Do not tune thresholds on the test set.

The first useful dataset should deliberately include:

- stable wide-camera sequences;
- camera pans and zooms;
- cuts between cameras;
- replay transitions;
- player crossings;
- partial and full occlusions;
- substitutions or long absences;
- similar uniforms and small player boxes.

## Annotation contract

Coordinates are normalized image coordinates in `[0, 1]`. A box is
`x, y, w, h`, where `x, y` is the top-left corner.

```json
{
  "schema_version": "tracking-annotation-v1",
  "video_id": "match-001-player-8",
  "fps": 5,
  "frames": [
    {
      "frame_index": 0,
      "time_sec": 0.0,
      "objects": [
        {
          "identity": "player-8",
          "ignore": false,
          "bbox": {"x": 0.12, "y": 0.24, "w": 0.04, "h": 0.16}
        }
      ]
    }
  ]
}
```

Rules:

- `video_id` must be unique in the dataset.
- `frame_index` must be unique in a sequence.
- one non-ignored box per identity per annotated frame;
- boxes must remain inside the normalized frame;
- use `ignore: true` for regions where detections should not count as false
  positives, such as an intentionally unlabelled referee or crowd region;
- do not invent an identity when the target cannot be verified. Omit the object or
  mark the frame outside the scored set.

## Prediction contract

```json
{
  "schema_version": "tracking-prediction-v1",
  "video_id": "match-001-player-8",
  "frames": [
    {
      "frame_index": 0,
      "time_sec": 0.0,
      "tracks": [
        {
          "track_id": "segment-0001/track-6",
          "confidence": 0.91,
          "bbox": {"x": 0.12, "y": 0.24, "w": 0.04, "h": 0.16}
        }
      ]
    }
  ]
}
```

AlgoNext `tracking.json` artifacts can be converted with:

```bash
python scripts/convert_tracking_output.py \
  --input tracking.json \
  --video-id match-001-player-8 \
  --output predictions/match-001-player-8.json
```

For windowed tracking, ByteTrack IDs are namespaced by window. Numeric ID `1` in
two windows is **not** assumed to be the same player. A future ReID layer must earn
that association in the benchmark instead of receiving it for free.

## Running the benchmark

One sequence:

```bash
python scripts/evaluate_tracking.py \
  --annotations annotations/match-001-player-8.json \
  --predictions predictions/match-001-player-8.json
```

A directory:

```bash
python scripts/evaluate_tracking.py \
  --annotations annotations/ \
  --predictions predictions/ \
  --json-out artifacts/tracking-benchmark.json \
  --fail-on-gate
```

The fixture smoke test is:

```bash
python scripts/evaluate_tracking.py \
  --annotations tests/fixtures/tracking_benchmark/annotations \
  --predictions tests/fixtures/tracking_benchmark/predictions \
  --fail-on-gate
```

## Metrics

Frame matching is one-to-one. The matcher maximizes the number of valid matches
first and total IoU second. A match requires IoU at or above the configured
threshold, `0.50` by default.

### Detection metrics

- **Precision**: matched predictions / scored predictions.
- **Recall**: matched ground-truth boxes / ground-truth boxes.
- **Detection F1**: harmonic mean of precision and recall.
- **Mean matched IoU**: average overlap of matched boxes.
- **Track coverage**: matched target observations / annotated target observations.

### Identity metrics

The evaluator builds a global one-to-one mapping between annotated identities and
predicted track IDs that maximizes matched observations.

- **ID precision**, **ID recall**, **IDF1** use that transparent global mapping.
- **ID switch**: the predicted track ID changes between two matched observations
  of the same annotated identity. A gap does not erase the previous identity.
- **Fragmentation**: a matched identity becomes unmatched and is later reacquired.
- **Mostly tracked**: at least 80% of annotated observations matched.
- **Mostly lost**: at most 20% matched.

### HOTA-style diagnostic

`hota_style_at_threshold` is:

```text
DetA = TP / (TP + FP + FN)
AssA = mean association Jaccard over matched observations
HOTA-style = sqrt(DetA * AssA)
```

This deliberately exposes the detection/association trade-off. It is a
transparent engineering diagnostic at one IoU threshold. It is **not claimed as
an official TrackEval HOTA leaderboard result**. Before publishing external
research claims, run the exported data through the canonical TrackEval
implementation and document its version and configuration.

## Initial engineering gates

Defaults:

| Metric | Gate |
|---|---:|
| Detection F1 | >= 0.75 |
| IDF1 | >= 0.65 |
| Track coverage | >= 0.60 |
| ID switches / 100 matches | <= 5.0 |
| HOTA-style at configured IoU | >= 0.55 |

These gates only decide whether the tracking pipeline is good enough to continue
development. Passing them does not validate:

- cross-camera identity in unseen leagues;
- metric distance or speed;
- ball possession or football events;
- technical, tactical, or athletic player scoring.

Thresholds must be revised from held-out benchmark evidence, not product
preference.

## Release evidence

Every tracking release should retain:

- exact annotation and prediction manifests;
- source commit and model hash;
- detector/tracker configuration;
- evaluation FPS and IoU threshold;
- per-sequence metrics, not only aggregate metrics;
- failures sorted by identity switches and longest gaps;
- the JSON report emitted by the CLI.

A release should be blocked when the quality gate fails. Individual bad sequences
must still be reviewed even when aggregate gates pass.
