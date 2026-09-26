# Jersey vision and temporary storage repair

The existing OpenAI key was used by report generation; it had no role in jersey
recognition. This change adds structured vision reads of isolated shirt crops as
supplemental evidence in the experimental CV tracker. The prompt does not disclose
the target number, player, team or video URL. A visible different number vetoes an
association. Two distinct images at least 0.6 seconds apart, with compatible kit
colors, are required for corroboration. Unreadable results cannot establish identity.
OCR never overrides continuity within a tracklet, kit-color or scoring validation
gates. It cannot recover detail absent from a small or blurred source.

## Experimental cut reacquisition

The first deployed version only corroborated physical overlap and therefore left
the user's full run `ANCHOR_ONLY`: 120 windows completed without a storage error,
but neither adjacent window supplied a reliable overlap link. Reacquisition now
requires a legible anchor number, two distinct matching shirt reads on the same
motion-continuous component, compatible kit colors, and the existing appearance
quality/similarity and combined-score thresholds. A unique qualifying candidate
can replace the cross-window overlap requirement and resolve a small appearance
margin. Two qualifying candidates still abstain. Scoring stays `validated=false`.

After one readable match, at most four additional nearby moments (within three
seconds, separated by at least 0.6 seconds) are inspected under the same attempt
budget. The requested number remains absent from every model prompt. Components
use the existing fps-aware motion gate, including 1.001-second sample intervals
from 29.97-fps footage; disconnected raw track IDs cannot be merged by OCR.
If a matching read still cannot establish a link at low sampling frequency, the
window gets one 3-fps CV retry. This is limited to four windows per attempt, with
the same OCR budget and timeout, and never changes the association acceptance gates.

The isolated three-window probe uses only the already-cached AlgoNext input,
does not alter the job or upload its tracking result, and fails its workflow when
the final kit guard does not confirm autonomous tracking. Passing that probe is
not equivalent to full-match identity validation.

On 26 September at 10:15 UTC, isolated production-key probe run **36235093423**
passed. The 3-fps retry read **8** at 1227.017, 1227.684 and 1228.352 seconds on
one continuous component. All five final kit samples were compatible. The same
component with legacy smoothed display coordinates failed: one compatible, two
incompatible and two unknown samples. Jersey-confirmed tracks now retain exact
detection coordinates for identity crops, avoiding that smoothing lag.

The probe made 32 API calls, used 9,882 total tokens and 41.939 seconds of API
time, with zero provider errors. The honest result is
`SPARSE_CROSS_WINDOW_EVIDENCE`, not complete match coverage or a validated rating.

## Bounded operation

- Existing server-side `OPENAI_API_KEY` and optional `OPENAI_BASE_URL` are reused.
- Model snapshot: `gpt-5.4-mini-2026-03-17`; configurable via `JERSEY_OCR_MODEL`.
- Default attempt limits: 64 requests, 240 seconds of API time, 20-second read
  timeout, no implicit retries, circuit opens after three errors.
- Image hashes cache reads within an attempt. No credentials, image payloads,
  signed URLs or raw provider errors appear in logs. Requests specify `store=false`.
- Missing keys, provider errors and exhausted budgets leave CV guards in force.
- Failed tracking retains only bounded scalar rejection and OCR diagnostics;
  selected-player boxes and unverified identity claims remain discarded.

## Disk repair

Window MP4s were written under `/tmp/fnh_jobs/<job>/attempts/<attempt>/tracking`
while the pipeline only removed its separate task delivery directory. Completion,
failure and timeout now also clean the attempt artifact workspace. A 4 GiB free
space reserve stops additional video growth before Redis/Postgres lose disk space.
Inputs/results in object storage are retained. `KEEP_WORKDIR=1` remains an explicit
diagnostic override.

At 09:16 UTC on 26 September, with all workers confirmed idle and the two known
jobs terminal, the canary removed five disposable window directories totaling
13,769,023,797 bytes. Available disk space rose to 22,353,264,640 bytes.

## Real API validation

GitHub Actions run 36232252310 used the existing production key on the cached
AlgoNext frame at 1192.607 seconds from job
`796f8c0f-94cd-4d2d-b8b1-a0f6ee5a5b60`:

| Input | Expected | Observed |
| --- | --- | --- |
| White shirt, visible 8 | 8 | 8 |
| Deliberately unreadable crop | null | null |
| Blank control | null | null |

Three requests completed in 5.938 seconds, 898 total tokens, no errors. This is a
bounded integration check, **not** a jersey accuracy benchmark or full-match
identity validation. The user's original first reference was a different nighttime
sequence in the same cached video; the confirmed target is the daytime white 8.

Official API contracts used:
- https://developers.openai.com/api/docs/guides/images-vision
- https://developers.openai.com/api/docs/guides/structured-outputs
- https://developers.openai.com/api/docs/models/gpt-5.4-mini
