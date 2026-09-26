# Jersey vision and temporary storage repair

The existing OpenAI key was used by report generation; it had no role in jersey
recognition. This change adds structured vision reads of isolated shirt crops as
supplemental evidence in the experimental CV tracker. The prompt does not disclose
the target number, player, team or video URL. A visible different number vetoes an
association. Two distinct images at least 0.6 seconds apart, with compatible kit
colors, are required for corroboration. Unreadable results cannot establish identity.
OCR never overrides the existing motion continuity, ambiguity, kit-color or scoring
validation gates. It cannot recover detail absent from a small or blurred source.

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
