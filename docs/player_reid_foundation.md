# Player Re-identification foundation

## Status

This module is an **experimental ReID foundation**, not a validated cross-camera
player identity system. It is deliberately not wired into the production worker
in this change.

The production capability must remain:

```json
{
  "cross_shot_player_reidentification": false
}
```

until the implementation is connected to the windowed tracker and passes the
held-out tracking benchmark with documented model/configuration hashes.

## Goal

ByteTrack identifiers are local to a continuous tracking context. A numeric ID
reappearing after a tracker reset, camera cut, replay, or separate processing
window is not evidence that the same player was found.

The ReID foundation introduces three explicit concepts:

1. **Appearance descriptor**: a versioned vector derived from player crops.
2. **Identity profile**: a manually anchored player descriptor updated only after
   accepted associations.
3. **Association decision**: an auditable `ACCEPTED` or `ABSTAINED` result with
   candidate scores and reason codes.

## Appearance descriptor

`hsv-torso-v1` is a lightweight baseline built from:

- separate upper- and lower-body regions;
- hue, saturation, and value histograms;
- per-region colour means and standard deviations;
- crop quality based on size, sharpness, saturation, and human-like aspect ratio.

It is intended to make the association contract executable before a learned
person-ReID model is introduced. It is not expected to separate teammates with
identical kits reliably, and that ambiguity must produce abstention.

Descriptors carry:

- `version`;
- normalized `vector`;
- `sample_count`;
- `quality` in `[0, 1]`.

Profiles with incompatible descriptor versions or dimensions cannot be merged.

## Association inputs

Each candidate may contribute:

- appearance similarity;
- temporal-overlap consistency between adjacent windows;
- geometry consistency;
- descriptor quality and sample count;
- detection count and arbitrary audit metadata.

The default normalized score weights are:

| Signal | Weight |
|---|---:|
| Appearance | 0.65 |
| Window overlap | 0.25 |
| Geometry | 0.10 |

Weights are renormalized over available signals, but geometry alone is never
sufficient because a missing appearance descriptor is a hard failure.

## Default hard gates

| Gate | Default |
|---|---:|
| Combined score | >= 0.76 |
| Appearance similarity | >= 0.78 |
| Strong overlap exception | >= 0.65 |
| Best-vs-second margin | >= 0.07 |
| Descriptor quality | >= 0.30 |
| Descriptor samples | >= 2 |

The strong-overlap exception permits a lower appearance score only when adjacent
windows contain compelling temporal overlap. It does not allow an association
without an appearance descriptor.

## Abstention reasons

The decision returns `ABSTAINED` for conditions including:

- `NO_CANDIDATES`;
- `MISSING_APPEARANCE_DESCRIPTOR`;
- `DESCRIPTOR_VERSION_MISMATCH`;
- `LOW_DESCRIPTOR_QUALITY`;
- `INSUFFICIENT_DESCRIPTOR_SAMPLES`;
- `LOW_APPEARANCE_SIMILARITY`;
- `LOW_COMBINED_SCORE`;
- `AMBIGUOUS_CANDIDATE_MARGIN`.

Similar-kit candidates with a small score margin must abstain. The profile is
updated only after the caller observes an explicit accepted decision.

Every decision currently contains:

```json
{
  "version": "reid-association-v1",
  "validated": false
}
```

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
An incorrect accepted association is therefore penalized by the global identity
assignment, ID switches, and HOTA-style association metric.

## Production activation checklist

Do not connect this module to `track_player_windowed` until all of the following
are complete:

1. manual anchor crop extraction is deterministic and logged;
2. every window exposes multiple candidate profiles, not only the currently
   selected local track;
3. temporal overlap is computed from common absolute timestamps;
4. ambiguous windows abstain without mutating the identity profile;
5. development and validation sets include similar-kit teammates, cuts, replay,
   occlusion, zoom, and small boxes;
6. thresholds are selected on validation data, not the held-out test split;
7. the locked test split passes the tracking release gate;
8. per-sequence failures are reviewed, especially ID switches and long gaps;
9. the feature is deployed behind a kill switch;
10. capability metadata remains false until production monitoring confirms the
    same validated configuration.

## Tests

The current tests verify:

- clear multi-signal candidate acceptance;
- same-kit ambiguity abstention;
- geometry cannot replace missing appearance;
- low-quality and single-sample descriptors abstain;
- descriptor version mismatch abstains;
- profile updates are explicit;
- synthetic same-uniform crops are closer than different-uniform crops;
- tiny crops are rejected;
- only accepted ReID decisions receive cross-window benchmark continuity.
