# Camera-shot segmentation and pitch-geometry proposals

## Status

This module is a conservative computer-vision foundation for separating broadcast
camera shots and finding visible pitch geometry. It does **not** yet assign
canonical football landmarks automatically and therefore does not create a
validated homography on its own.

Public capabilities must remain:

```json
{
  "pitch_calibration": false,
  "athletic_metrics": false,
  "technical_tactical_scoring": false
}
```

The output explicitly returns:

```json
{
  "automatic_calibration_available": false,
  "reason_codes": ["SEMANTIC_PITCH_KEYPOINT_MODEL_REQUIRED"]
}
```

## Why a separate camera stage is required

A broadcast match is not one stable camera. It contains hard cuts, close-ups,
bench shots, crowd shots, replay inserts, graphics, pans, tilts and zooms. A
homography estimated in one projection must not be reused after the projection
changes.

The camera stage therefore performs two distinct tasks:

1. divide the video into auditable shot intervals;
2. decide which intervals contain enough visible pitch geometry to justify
   running a semantic keypoint model.

Non-pitch and very short intervals are excluded before calibration.

## Shot-boundary baseline

Frames are sampled at a configurable rate, normally 2 FPS. Consecutive frames
are compared using four independent signals:

- HSV histogram distance;
- normalized grayscale difference;
- edge-map change ratio;
- mean-colour distance.

The combined score is compared with the larger of:

- an absolute hard-cut floor;
- a robust adaptive threshold computed from the median and median absolute
  deviation of the video sample distances.

Candidate boundaries must also be local maxima and respect a minimum temporal
separation. Gradual brightness drift is deliberately tested not to create a cut.
This baseline is designed for hard cuts; dissolves and fades need a separate
benchmark before they are treated as reliable boundaries.

## Pitch-view evidence

Each sampled frame is evaluated using:

- the largest connected green-field component;
- bright low-saturation pixels inside the field support;
- edge density;
- probabilistic Hough line segments;
- distinct line-orientation families;
- intersections between sufficiently non-parallel lines.

Frames are classified as:

- `PITCH_CANDIDATE`;
- `NON_PITCH`;
- `UNKNOWN`.

A shot becomes a calibration candidate only when enough of its samples are
pitch candidates, its mean pitch probability is sufficient and its duration is
long enough.

## Geometry proposals

For representative frames in each pitch shot, the module returns normalized:

- line segments;
- line confidence and mask support;
- orientation families;
- line-intersection keypoint proposals.

The keypoints deliberately have:

```json
{
  "semantic_landmark": null
}
```

An intersection is not automatically called `centre_spot`, `halfway_top` or a
penalty-area corner. Assigning the wrong semantic identity would allow a clean
but incorrect homography to pass geometric checks.

## Output contract

The versioned output is `camera-analysis-v1`. Its JSON Schema is stored at:

```text
docs/schemas/camera-analysis-v1.schema.json
```

The result contains:

- sampled-shot metadata;
- boundary scores and thresholds;
- contiguous camera-shot intervals;
- pitch/non-pitch decisions;
- geometry proposals for selected frames;
- exclusion reasons;
- an explicit automatic-calibration abstention.

All CLI output is checked for finite numbers and written atomically.

## CLI

```bash
python scripts/analyze_camera_segments.py \
  --video match.mp4 \
  --output camera-analysis.json \
  --sample-fps 2 \
  --fail-if-no-geometry-candidate
```

Use `--include-samples` only for debugging because per-frame evidence increases
the JSON size.

## Validation already covered by tests

The deterministic synthetic test suite checks:

- pitch-line and intersection detection;
- non-pitch rejection;
- black-frame abstention;
- resistance to gradual brightness drift;
- hard cuts and contiguous shot intervals;
- exclusion of non-pitch shots before geometry analysis;
- schema validation;
- real OpenCV video read/write;
- CLI execution and finite JSON output.

Synthetic success is not a real-match validation.

## Next activation gate

Before automatic calibration can be enabled, the next step needs:

1. a labelled camera-shot dataset with hard cuts, dissolves, replay, graphics,
   close-ups and crowd shots;
2. a semantic pitch-keypoint dataset across main-camera views, pan, tilt, zoom,
   rain, shadows, partial fields and youth pitches;
3. per-keypoint precision, recall and localization error;
4. calibration completeness and error in metres on a locked test set;
5. explicit left/right pitch orientation handling;
6. temporal consistency checks across neighbouring frames;
7. an abstention benchmark for ambiguous and partially visible layouts;
8. model and threshold hashes in every generated calibration artifact.

Only semantically labelled keypoints that pass those gates may be converted into
`pitch-calibration-request-v1` and sent to the validated homography solver.
