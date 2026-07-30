# Player Re-identification foundation

## Status

Player ReID is now connected to the full-match windowed tracker behind a feature
flag. It remains an **experimental association system**, not a validated
cross-camera player identity capability.

The public capability must therefore remain:

```json
{
  "cross_shot_player_reidentification": false
}
```

until the locked tracking benchmark passes with a documented dataset,
configuration and model hash. Enabling the implementation does not make the
capability scientifically validated.

## Runtime activation and rollback

The worker installs the ReID implementation before `pipeline.py` imports
`track_player_windowed`.

```text
PLAYER_REID_ENABLED=1       # install experimental tracker
PLAYER_REID_ENABLED=0       # immediate kill switch
PLAYER_REID_FAIL_OPEN=1     # return to legacy geometry tracker on runtime error
PLAYER_REID_FAIL_OPEN=0     # fail the job instead
```

Docker Compose enables ReID by default for the worker while preserving both
overrides through environment variables.

A `TrackingTimeoutError` never triggers a second legacy run. Other unexpected
exceptions fail open by default and return a `reid_summary.status` of
`FALLBACK_LEGACY`.

## Goal

ByteTrack identifiers are local to a continuous tracking context. A numeric ID
reappearing after a tracker reset, camera cut, replay or separate processing
window is not evidence that the same player was found.

The ReID implementation introduces four explicit concepts:

1. **Appearance descriptor**: a versioned vector derived from player crops.
2. **Identity profile**: a manually anchored player descriptor updated only after
   accepted associations.
3. **Candidate set**: multiple local ByteTrack tracks evaluated in every window.
4. **Association decision**: an auditable `ACCEPTED` or `ABSTAINED` result with
   candidate scores and reason codes.

## Window processing order

The tracker no longer begins blindly at minute zero. It:

1. finds the processing window containing the manually selected anchor;
2. resolves the anchor to one local ByteTrack track;
3. builds the initial identity descriptor from several crops in that track;
4. processes later windows in chronological order;
5. independently processes earlier windows in reverse order;
6. keeps a separate profile history for each direction, both rooted in the
   manual anchor.

This avoids using a weak early-window geometry guess to define the identity for
the entire match.

## Appearance descriptor

Production workers now default to
`osnet-x0.25-msmt17+hsv-torso-v2`, a hybrid descriptor combining:

- a person-ReID-specific OSNet x0.25 embedding trained on MSMT17;
- the existing upper/lower-body kit-colour descriptor;
- a fail-open rollback to `hsv-torso-v1` if the learned model is unavailable.

The worker image prefetches the checkpoint during build. No identity model is
downloaded while a match is running. `PLAYER_REID_DESCRIPTOR_BACKEND=hsv`
provides an immediate runtime rollback.

`hsv-torso-v1` remains the lightweight baseline built from:

- separate upper- and lower-body regions;
- hue, saturation and value histograms;
- per-region colour means and standard deviations;
- crop quality based on size, sharpness, saturation and human-like aspect ratio.

It is intended to make the association contract executable before a learned
person-ReID model is introduced. It is not expected to separate teammates with
identical kits reliably, and that ambiguity must produce abstention.

Descriptors carry:

- `version`;
- normalized `vector`;
- `sample_count`;
- `quality` in `[0, 1]`.

Profiles with incompatible descriptor versions or dimensions cannot be merged.
Descriptor vectors are not written to the public tracking summary; only version,
sample count and quality are persisted.

## Candidate generation

Every window exposes several local tracks rather than only the geometry winner.
Candidates are ranked for descriptor extraction using:

- temporal overlap with the most recent accepted window;
- boundary geometry consistency;
- detection count;
- average detector confidence.

The default maximum is six candidates with up to five temporally distributed
crop samples per candidate. These limits are configurable.

```text
PLAYER_REID_MAX_CANDIDATES=6
PLAYER_REID_SAMPLES_PER_CANDIDATE=5
PLAYER_REID_MIN_TRACK_HITS=3
PLAYER_REID_MIN_INDIVIDUAL_CROP_QUALITY=0.18
```

The sampling routine spreads observations across the track and always preserves
the strongest confidence-by-area crop.

## Association inputs

Each candidate may contribute:

- appearance similarity;
- temporal-overlap consistency at common absolute timestamps;
- geometry consistency at the processing boundary;
- descriptor quality and sample count;
- detection count and audit metadata.

The default normalized score weights are:

| Signal | Weight |
|---|---:|
| Appearance | 0.65 |
| Window overlap | 0.25 |
| Geometry | 0.10 |

Weights are renormalized over available signals, but geometry alone is never
sufficient because a missing appearance descriptor is a hard failure.

## Default hard gates

| Gate | Default | Environment variable |
|---|---:|---|
| Combined score | >= 0.76 | `PLAYER_REID_MIN_COMBINED_SCORE` |
| Appearance similarity | >= 0.78 | `PLAYER_REID_MIN_APPEARANCE_SIMILARITY` |
| Strong overlap exception | >= 0.65 | `PLAYER_REID_STRONG_OVERLAP_SCORE` |
| Best-vs-second margin | >= 0.07 | `PLAYER_REID_MIN_MARGIN` |
| Descriptor quality | >= 0.30 | `PLAYER_REID_MIN_DESCRIPTOR_QUALITY` |
| Descriptor samples | >= 2 | `PLAYER_REID_MIN_DESCRIPTOR_SAMPLES` |

The strong-overlap exception permits a lower appearance score only when adjacent
windows contain compelling temporal overlap. It does not allow an association
without an appearance descriptor.

## Abstention policy

The decision returns `ABSTAINED` for conditions including:

- `NO_CANDIDATES`;
- `MISSING_APPEARANCE_DESCRIPTOR`;
- `DESCRIPTOR_VERSION_MISMATCH`;
- `LOW_DESCRIPTOR_QUALITY`;
- `INSUFFICIENT_DESCRIPTOR_SAMPLES`;
- `LOW_APPEARANCE_SIMILARITY`;
- `LOW_COMBINED_SCORE`;
- `AMBIGUOUS_CANDIDATE_MARGIN`;
- `WINDOW_PROCESSING_FAILED`.

An abstained window receives no selected track and contributes no player boxes.
The last accepted profile is not mutated. This intentionally prefers missing
data over assigning a teammate to the selected player.

Every automatic decision currently contains:

```json
{
  "version": "reid-association-v1",
  "validated": false
}
```

## Tracking artifact

The output keeps `mode: full_match_windowed` for pipeline compatibility and adds:

```json
{
  "identity_mode": "appearance_reid_v1",
  "method": "yolo+bytetrack+appearance_reid",
  "reid_summary": {
    "status": "EXPERIMENTAL",
    "validated": false,
    "accepted_associations": 0,
    "abstained_associations": 0
  }
}
```

Each segment includes its local track, direction, identity status and full
association decision. Coverage is computed from unique sampled timestamps so
overlapping windows are not counted twice.

## Benchmark integration

The tracking artifact adapter continues to namespace local ByteTrack IDs:

```text
segment-0001/track-1
segment-0002/track-1
```

A segment earns a shared benchmark identity only when it contains an explicit:

```json
{
  "reid": {
    "status": "ACCEPTED",
    "identity_id": "selected-player"
  }
}
```

The resulting benchmark track ID is:

```text
identity/selected-player
```

An `ABSTAINED` decision remains local and cannot receive free IDF1 continuity.
An incorrect accepted association is penalized by global identity assignment,
ID switches and the HOTA-style association metric.

## Validation still required

The implementation is deployable but not yet validated. Promotion of
`cross_shot_player_reidentification` to `true` requires all of the following:

1. an annotated development, validation and locked test split;
2. similar-kit teammates, camera cuts, replay, occlusion, zoom and small boxes;
3. threshold selection on validation data only;
4. release-gate success for detection F1, IDF1, coverage, identity switches and
   HOTA-style association;
5. per-sequence review of false accepted associations;
6. production monitoring using the exact validated configuration;
7. documented descriptor/model and configuration hashes.

Player scoring, athletic metrics and technical-tactical claims remain disabled
regardless of ReID status.

## Tests

The test suite verifies:

- clear multi-signal candidate acceptance;
- same-kit ambiguity abstention;
- geometry cannot replace missing appearance;
- low-quality and single-sample descriptors abstain;
- descriptor version mismatch abstains;
- profile updates are explicit;
- synthetic same-uniform crops are closer than different-uniform crops;
- tiny crops are rejected;
- anchor-first bidirectional processing order;
- descriptor sampling preserves temporal spread and the best crop;
- overlap and geometry scoring behaviour;
- coverage deduplication across overlapping windows;
- feature-flag installation, idempotence, fail-open and fail-closed behaviour;
- tracking timeouts never restart a second legacy pass;
- only accepted ReID decisions receive cross-window benchmark continuity.
