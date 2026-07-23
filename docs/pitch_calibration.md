# Pitch calibration and metric-space motion

## Status

This module provides a validated mathematical path from normalized image
coordinates to metres on a football pitch. It does **not** yet provide automatic
broadcast-camera calibration. Correspondences must currently come from a manual
annotation tool or a future pitch-keypoint model.

The public capability therefore remains:

```json
{
  "pitch_calibration": false,
  "athletic_metrics": false
}
```

until every camera segment used by a production analysis has a validated
calibration and the full pipeline passes a real-match benchmark.

## Why this is necessary

A bounding-box centre cannot be multiplied by `105 x 68` and interpreted as a
position on the field. Broadcast video includes perspective, pan, tilt, zoom,
camera cuts and replay. The mapping is projective and can change whenever the
camera changes.

The calibration module replaces that shortcut with an explicit homography:

```text
normalized image point (x, y) --H--> pitch point (x_m, y_m)
```

`H` is estimated independently for a declared camera segment. A result is usable
only when its quality gate passes.

## Coordinate conventions

### Image space

- origin: top-left of the frame;
- `x` increases to the right;
- `y` increases downwards;
- both axes are normalized to `[0, 1]`.

### Pitch space

- origin: top-left corner of the canonical pitch diagram;
- `x_m` runs along pitch length;
- `y_m` runs across pitch width;
- units are metres;
- default dimensions are `105 x 68 m` but valid IFAB-style dimensions can be
  supplied in the request.

Player position uses the centre-bottom point of the detected bounding box, not
the box centre. This is a closer image approximation to the point where the
player contacts the ground.

## Request contract

A `pitch-calibration-request-v1` contains:

- a `camera_segment_id`;
- optional `start_sec` and `end_sec`;
- pitch dimensions;
- at least four image-to-field correspondences;
- a source such as `manual`, `model`, or `reviewed_model`.

Correspondences may specify an explicit field point:

```json
{
  "image": {"x": 0.31, "y": 0.42},
  "field": {"x_m": 52.5, "y_m": 34.0}
}
```

or one of the canonical landmarks:

```json
{
  "image": {"x": 0.31, "y": 0.42},
  "landmark": "centre_spot"
}
```

The Python parser rejects duplicate points, invalid dimensions, out-of-frame
image coordinates, out-of-pitch field coordinates, invalid time intervals and
unknown fields. A correspondence must contain exactly one of `field` or
`landmark`.

The request deliberately exposes no user-controlled weight. Every point has the
same influence on quality measurements, so a poor calibration cannot be made to
pass by down-weighting inconvenient reprojection errors.

The JSON Schema is available in `docs/schemas` and is tested against the Python
parser and serialized result objects.

## Homography fitting

The solver uses OpenCV `findHomography` with RANSAC and then evaluates all
reported inliers in metres. Four points are mathematically sufficient to fit a
homography, but the default validation gate requires at least six.

RANSAC uses a recorded fixed seed and a process lock around OpenCV's global RNG.
This makes concurrent server fits reproducible for the same input, OpenCV
version and thresholds.

The result includes:

- image-to-field and field-to-image matrices;
- the complete inlier mask;
- inlier ratio;
- unweighted RMSE in metres;
- median, p95 and maximum inlier error;
- convex-hull coverage in image space;
- convex-hull coverage on the field;
- normalized homography condition number;
- the minimum projective denominator over the frame;
- reason codes and exact thresholds;
- OpenCV version, quality-gate version and RANSAC seed.

The projective denominator is checked analytically at the four frame corners.
Because it is affine in image coordinates, a sign change at the corners proves
that the projective horizon crosses the frame; such a fit is rejected even if a
finite sampling grid misses the exact zero.

## Default quality gate

| Check | Default |
|---|---:|
| Correspondences | >= 6 |
| RANSAC inlier ratio | >= 0.75 |
| RMSE | <= 1.5 m |
| p95 inlier error | <= 3.0 m |
| Image convex-hull coverage | >= 2% |
| Field convex-hull coverage | >= 8% |
| Normalized condition number | <= 1,000,000 |
| Projective denominator | >= 0.005 |

A failed gate returns `status: REJECTED`, `validated: false` and one or more
reason codes. Downstream metric code ignores rejected calibrations. A result
with non-finite matrices, projections or quality values is not serializable.

Point coverage matters because six precise points concentrated around the centre
circle do not constrain the corners of the pitch. Low reprojection error alone
is not sufficient.

## Camera segments

A single homography is valid only while the camera projection remains stable.
Each result can declare:

```json
{
  "camera_segment_id": "main-camera-shot-0042",
  "start_sec": 128.2,
  "end_sec": 136.8
}
```

Motion projection selects a validated calibration by timestamp. It never reuses
a homography outside its declared interval. If several validated calibrations
overlap, the most specific interval with the lowest RMSE is chosen.

No movement transition, smoothing neighbourhood or sprint interval is allowed
to cross from one `camera_segment_id` to another. The first sample after a camera
change starts a new trajectory fragment.

A future shot-boundary and pitch-keypoint stage must generate these segments
automatically. Replay and non-pitch shots must be marked uncalibrated.

## Calibrated motion diagnostics

`calculate_calibrated_motion`:

1. gathers player boxes from accepted tracking segments;
2. discards ReID-abstained windows;
3. projects the centre-bottom point through the matching calibration;
4. removes samples outside the pitch plus a small tolerance;
5. deduplicates overlapping-window samples by timestamp and confidence;
6. applies median smoothing only inside the same camera segment and accepted
   temporal neighbourhood;
7. rejects camera changes, long temporal gaps, implausible speed and implausible
   acceleration;
8. requires continuous accepted duration for a sprint proxy;
9. reports only accepted, observed transitions.

Outputs deliberately use names such as:

- `observed_path_length_m`;
- `average_observed_speed_kmh`;
- `p95_observed_speed_kmh`;
- `sprint_bouts_proxy`.

They are not extrapolated to a full match. `athletic_metric_validated` remains
`false` until calibration stability, frame timing and player identity are
validated together against ground truth.

## Runtime limits

Default motion filters:

| Filter | Default |
|---|---:|
| Maximum sample gap | 1.0 s |
| Maximum accepted speed | 12.5 m/s |
| Maximum accepted acceleration | 12.0 m/s² |
| Sprint proxy threshold | 7.0 m/s |
| Minimum sprint proxy duration | 1.0 s |
| Pitch boundary tolerance | 2.0 m |
| Minimum projected points | 10 |

These are engineering filters, not normative definitions of athletic
performance. Every value is returned with the diagnostic result and is
configurable.

## CLI

Fit and gate a calibration:

```bash
python scripts/solve_pitch_calibration.py \
  --input calibration-request.json \
  --output calibration-result.json \
  --fail-on-gate
```

Apply one or more validated calibrations to a tracking artifact:

```bash
python scripts/apply_pitch_calibration.py \
  --tracking tracking.json \
  --calibration calibration-result.json \
  --output calibrated-motion.json \
  --fail-if-unavailable
```

The calibration argument may contain one result, an array of results, or an
object with a `calibrations` array. Both CLIs persist the exact thresholds used
in their output.

## Validation required before production capability

The next activation gate requires:

1. a pitch-keypoint dataset covering main camera, zoom, pan, rain, shadows,
   partial field views and youth pitches;
2. manually reviewed correspondences on a locked test set;
3. per-shot calibration rather than one homography per video;
4. benchmark error in metres across the full visible pitch;
5. camera-cut and replay exclusion accuracy;
6. comparison of projected player trajectories against an independent reference
   such as tracking provider data, GPS or manually annotated pitch positions;
7. monitoring of rejected segments and projection outliers;
8. documented model, configuration and threshold hashes.

Player scoring remains suspended even after pitch calibration. Ball tracking,
event recognition and a separately validated scoring model are still required.
