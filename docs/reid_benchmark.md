# Player ReID benchmark

AlgoNext must not turn experimental cross-window associations into a player score until identity continuity has been measured against human-reviewed evidence. This benchmark combines two complementary views of the same `tracking.json` artifact:

1. **Frame-level identity tracking** — detection F1, IDF1, track coverage, ID switches, fragmentation and HOTA-style accuracy on annotated frames.
2. **Window-level ReID decisions** — correct links, false links, conservative abstentions, missed associations, candidate-generation misses and processing failures.

Passing this benchmark validates only the tracking/ReID subsystem on the annotated dataset. It does not validate football ability, athletic metrics, ball events, pitch calibration or unseen-domain generalization.

## Annotation contracts

### Frame annotations

Frame boxes use the existing `tracking-annotation-v1` contract documented in [`tracking_benchmark.md`](tracking_benchmark.md). A scored frame may contain:

- one target object with a normalized `x/y/w/h` box;
- an empty `objects` array when the target is verifiably absent;
- no record at all when the frame is uncertain and must be excluded.

### Window annotations

Window decisions use `reid-window-annotation-v1`:

```json
{
  "schema_version": "reid-window-annotation-v1",
  "video_id": "job-id",
  "identity": "selected-player",
  "fps": 1,
  "windows": [
    {
      "window_index": 0,
      "window_start": 0.0,
      "window_end": 60.0,
      "target_visibility": "VISIBLE",
      "candidate_state": "PRESENT",
      "target_candidate_id": "24",
      "selected_track_is_target": true,
      "evidence_frames": [],
      "notes": null
    }
  ]
}
```

`target_visibility` is one of:

- `VISIBLE`: the selected identity is human-verifiable in the window;
- `NOT_VISIBLE`: the identity is not visible;
- `UNCERTAIN`: the identity cannot be verified and the window is excluded.

For visible windows, `candidate_state` is one of:

- `PRESENT`: the correct local candidate ID is verifiable; `target_candidate_id` is required;
- `ABSENT`: the target is visible but missing from the candidate set;
- `UNVERIFIABLE`: the target is visible, but the persisted artifact does not contain enough candidate evidence to map an ID safely.

For every accepted window, reviewers should set `selected_track_is_target` to `true` or `false`. This directly measures false identity links even when historical artifacts lack candidate thumbnails.

## Build the review pack for the preserved production job

The helper downloads the job response, `tracking.json` and the input video, extracts evidence frames with FFmpeg, and writes a self-contained browser review tool:

```bash
python scripts/prepare_reid_job_benchmark.py \
  --api-base https://algonext-frontend.vercel.app/api/backend \
  --job-id d70e0adb-e326-4ce0-b18e-8c49d4d0fccc \
  --output-dir /tmp/reid-d70e0adb
```

To reuse an existing local video instead of downloading it:

```bash
python scripts/prepare_reid_job_benchmark.py \
  --api-base https://algonext-frontend.vercel.app/api/backend \
  --job-id d70e0adb-e326-4ce0-b18e-8c49d4d0fccc \
  --video-path /path/to/videoanalizzare.mp4 \
  --output-dir /tmp/reid-d70e0adb
```

Open:

```text
/tmp/reid-d70e0adb/review-pack/index.html
```

The page autosaves review progress in the browser for that video and provides two downloads:

- `<job-id>.reid-windows.json` for window-level ReID decisions;
- `<job-id>.tracking-frames.json` for frame-level boxes and IDF1.

Move the downloaded files into the benchmark directory, or pass their actual download paths to the evaluation command. The HTML button **Azzera salvataggio locale** deliberately removes the browser recovery copy.

Historical artifacts may not contain candidate bbox evidence for abstained windows. In those cases, select `UNVERIFIABLE`; do not guess a candidate ID. New runs persist a few bbox samples for every ranked candidate so candidate recall can be measured directly.

## Review protocol

Use the following minimum protocol for a release decision:

1. Review every accepted window and mark whether the selected track is the target.
2. Review every abstained/failed window for target visibility.
3. Annotate target boxes on the sampled frames, or mark the target absent.
4. Assign candidate IDs only when the persisted candidate bbox evidence makes the mapping unambiguous.
5. Have a second reviewer audit all accepted windows and at least 20% of abstained windows.
6. Resolve disagreements before generating the final report.

Do not infer identity from score, shirt colour alone or spatial proximity. Replay shots, camera cuts, substitutions and visually similar teammates must be treated conservatively.

## Run the complete benchmark

```bash
python scripts/evaluate_reid_benchmark.py \
  --tracking /tmp/reid-d70e0adb/tracking.json \
  --frame-annotations /path/to/d70e0adb-e326-4ce0-b18e-8c49d4d0fccc.tracking-frames.json \
  --window-annotations /path/to/d70e0adb-e326-4ce0-b18e-8c49d4d0fccc.reid-windows.json \
  --json-out /tmp/reid-d70e0adb/report.json \
  --fail-on-gate
```

The output schema is `reid-benchmark-suite-v1`. The combined gate passes only when both the existing frame-tracking gate and the new window-ReID gate pass.

To evaluate only window decisions:

```bash
python scripts/evaluate_reid_windows.py \
  --tracking /tmp/reid-d70e0adb/tracking.json \
  --annotations /path/to/d70e0adb-e326-4ce0-b18e-8c49d4d0fccc.reid-windows.json \
  --json-out /tmp/reid-d70e0adb/window-report.json \
  --fail-on-gate
```

## Window metrics

| Metric | Meaning |
|---|---|
| Accepted judgement coverage | Fraction of accepted windows explicitly judged by a reviewer |
| Accepted precision | Correct selected identities divided by judged accepts |
| False-link rate | Wrong selected identities divided by judged accepts |
| Association precision given candidate | Correct candidate links divided by accepted candidate-present windows |
| Association recall given candidate | Correct links divided by windows where the correct candidate was present |
| Visible-window recall | Correct accepts divided by all visible target windows |
| Candidate annotation coverage | Visible windows for which candidate presence/absence was verifiable |
| Candidate recall visible | Candidate-present windows divided by candidate-scorable visible windows |
| Non-visible abstention rate | Correct abstentions when the target was absent |
| Processing failure rate | Failed windows divided by scored windows |
| End-to-end window success | Correct accepts plus correct target-absent abstentions divided by scored windows |

## Default release thresholds

### Existing frame-tracking gate

- detection F1 ≥ `0.75`
- IDF1 ≥ `0.65`
- track coverage ≥ `0.60`
- ID switches per 100 matches ≤ `5.0`
- HOTA-style score ≥ `0.55`

### Window-ReID gate

- at least `30` scorable windows;
- accepted judgement coverage ≥ `0.90`;
- accepted precision ≥ `0.95`;
- false-link rate ≤ `0.05`;
- association recall given candidate ≥ `0.60`;
- visible-window recall ≥ `0.45`;
- candidate annotation coverage ≥ `0.70`;
- candidate recall on visible/scorable windows ≥ `0.70`;
- processing failure rate ≤ `0.05`.

These are initial engineering gates, not claims of scientific validation. Before enabling player scoring, repeat the benchmark on multiple matches, teams, lighting conditions, camera systems, kit similarities, occlusions, replay patterns and age categories. Threshold changes must be versioned and justified by reviewed benchmark reports.

## CI smoke test

The API contract workflow runs a deterministic fixture through:

```bash
python scripts/evaluate_reid_benchmark.py \
  --tracking tests/fixtures/reid_benchmark/tracking.json \
  --frame-annotations tests/fixtures/reid_benchmark/frame-annotations.json \
  --window-annotations tests/fixtures/reid_benchmark/window-annotations.json \
  --minimum-scorable-windows 3 \
  --fail-on-gate
```

This protects the metric implementation and contracts. It is not a substitute for the human-reviewed production-job benchmark.
