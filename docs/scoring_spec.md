# Scoring system reverse-engineering (code-based)

## Call chain (Celery task → score)

1. **API triggers analysis**: the API enqueues `run_analysis` via Celery (`run_analysis.delay(job.id)`).【F:app/api.py†L1662-L1669】
2. **Celery task entrypoint**: `app.workers.pipeline.run_analysis` loads the job and orchestrates scoring.【F:app/workers/pipeline.py†L1341-L1426】
3. **Video feature extraction**: `extract_video_features` computes `duration_seconds`, `frame_count`, `fps`, `scene_change_count`, and `scene_change_rate`.【F:app/workers/pipeline.py†L427-L442】
4. **Radar skill calculation**: `compute_skill_scores` maps scene-change features to each radar skill (or `None`).【F:app/workers/pipeline.py†L447-L478】
5. **Evaluation aggregation**: `compute_evaluation` sanitizes the radar, applies role-weighted scoring, and falls back to averaging if required keys are missing.【F:app/core/scoring.py†L192-L215】
6. **Persistence & status**: results (overall, role, radar) are saved; missing radar keys trigger `INCOMPLETE_RADAR` and job status `PARTIAL`.【F:app/workers/pipeline.py†L1889-L1947】

## Radar skills (feature → score)

All radar skills are derived *only* from scene-change features computed in `extract_video_features` and `compute_skill_scores`. The only features used are:
- `scene_change_count`
- `scene_change_rate`
- (derived) `activity_score = min(1.0, scene_change_rate / 0.25)`
- (derived) `scene_bonus = min(scene_change_count, 10.0) * 2.0`
- (derived) `_score_value = clamp(round(base + activity_score * 45.0 + scene_bonus), 0, 100)`【F:app/workers/pipeline.py†L443-L478】

### Skill-by-skill conditions and formula
Each skill is only computed if its **minimum condition** is met; otherwise it is set to `None` and marked missing.

| Skill | Minimum condition to be evaluable | Base | Score formula | Missing handling |
| --- | --- | --- | --- | --- |
| Finishing | `scene_change_count >= 2` | 34.0 | `_score_value(34.0, activity_score, scene_bonus)` | `None` + added to `skills_missing` | 
| Shot Power | `scene_change_count >= 2` | 32.0 | `_score_value(32.0, activity_score, scene_bonus)` | `None` + added to `skills_missing` |
| Heading | `scene_change_count >= 3` | 30.0 | `_score_value(30.0, activity_score, scene_bonus)` | `None` + added to `skills_missing` |
| Positioning | `scene_change_rate >= 0.03` | 38.0 | `_score_value(38.0, activity_score, scene_bonus)` | `None` + added to `skills_missing` |
| Off-the-ball Movement | `scene_change_rate >= 0.04` | 36.0 | `_score_value(36.0, activity_score, scene_bonus)` | `None` + added to `skills_missing` |
| Composure | `scene_change_rate >= 0.015` | 40.0 | `_score_value(40.0, activity_score, scene_bonus)` | `None` + added to `skills_missing` |

**Source:** `compute_skill_scores` and `_score_value` implementations.【F:app/workers/pipeline.py†L443-L478】

### Transformations & normalization details
- **Clamping / scaling**:
  - `activity_score = min(1.0, scene_change_rate / 0.25)` (rate scaled to [0–1]).【F:app/workers/pipeline.py†L459-L461】
  - `scene_bonus = min(scene_change_count, 10.0) * 2.0` (cap at 10 scene changes, then scale).【F:app/workers/pipeline.py†L460-L461】
  - `_score_value` clamps to `[0, 100]` and rounds to `int`.【F:app/workers/pipeline.py†L443-L445】
- **Aggregation**: No per-frame aggregation; the scoring uses only global video-level features (`scene_change_count` and `scene_change_rate`).【F:app/workers/pipeline.py†L427-L478】
- **Smoothing / rolling windows**: None in the scoring path for radar skills (only global counts/rates).【F:app/workers/pipeline.py†L427-L478】

### Radar skill missing values
- If a skill does not meet its condition, it is set to `None` and added to `skills_missing`.【F:app/workers/pipeline.py†L463-L478】
- When building the radar for scoring, `None` values are dropped: `radar = {k: v for k, v in skills_computed.items() if v is not None}`.【F:app/workers/pipeline.py†L1838-L1840】

## Role score

### Definition
- **Role score** is computed by `compute_scores` as a **weighted average** of the radar skills required for the role (same formula used for overall score; see below).【F:app/core/scoring.py†L87-L116】
- The result is clamped to `[0, 100]` and rounded to **1 decimal place**.【F:app/core/scoring.py†L102-L116】

### Role profiles (weights)
Role-based weights are defined in `ROLE_WEIGHTS`:
- Striker, Winger, Midfielder, Defender, Goalkeeper each have per-skill weights for the six radar skills.【F:app/core/scoring.py†L3-L43】
- If the role is **unknown or not mapped**, `DEFAULT_WEIGHTS` (all 1.0) are used instead.【F:app/core/scoring.py†L45-L76】

### Missing/unknown role handling
- `keys_required_for_role` uses the role (or default) to define required keys; missing any of these keys causes `compute_scores` to return `None` for both overall and role score.【F:app/core/scoring.py†L78-L103】
- In that case, `compute_evaluation` falls back to averaging the available radar values.【F:app/core/scoring.py†L203-L210】

## Overall score

### Definition
- **Overall score** is computed by the same weighted average as role score in `compute_scores`.【F:app/core/scoring.py†L87-L116】
- **Formula** (for required skills only):
  - `value = clamp( sum(score_i * weight_i) / sum(weight_i), 0, 100 )`
  - `overall_score = round(value, 1)`【F:app/core/scoring.py†L111-L116】
- The output range is explicitly clamped to `[0, 100]` in `compute_scores`.【F:app/core/scoring.py†L102-L116】

### Caps / bonuses / penalties
- **Caps**: skill-level values are clamped to `[0, 100]` before weighting; the weighted result is also clamped to `[0, 100]`.【F:app/core/scoring.py†L102-L116】
- **Bonuses**: the only “bonus”-like term appears inside each radar skill score via `scene_bonus`, which is capped at `10 * 2.0 = 20`.【F:app/workers/pipeline.py†L460-L461】
- **Penalties**: missing required radar keys results in `None` from `compute_scores`, which causes a fallback to unweighted average of available radar values (no explicit numeric penalty beyond loss of weights).【F:app/core/scoring.py†L92-L110】【F:app/core/scoring.py†L203-L210】

### Handling incomplete radar
- If required radar keys are missing, `compute_scores` returns `None` for both scores, and `compute_evaluation` falls back to the **simple average** of available radar values (no weighting).【F:app/core/scoring.py†L92-L110】【F:app/core/scoring.py†L203-L210】
- The pipeline additionally emits a warning `INCOMPLETE_RADAR` and marks the run `PARTIAL` if the radar does not contain all required keys for the role.【F:app/workers/pipeline.py†L1889-L1907】

## RADAR INCOMPLETE (formal definition)

### Minimum conditions for a skill to be “evaluable”
Each skill is evaluable if its explicit threshold condition is met (scene-change count or rate). If not met, it is set to `None` and flagged missing.
- `scene_change_count >= 2`: Finishing, Shot Power.【F:app/workers/pipeline.py†L470-L473】
- `scene_change_count >= 3`: Heading.【F:app/workers/pipeline.py†L474-L474】
- `scene_change_rate >= 0.03`: Positioning.【F:app/workers/pipeline.py†L475-L475】
- `scene_change_rate >= 0.04`: Off-the-ball Movement.【F:app/workers/pipeline.py†L476-L476】
- `scene_change_rate >= 0.015`: Composure.【F:app/workers/pipeline.py†L477-L477】

### Minimum conditions for the radar to be “complete”
A radar is **complete** if all keys required by the role are present in the radar dictionary.
- Required keys are defined by `keys_required_for_role`, which returns all skills from `ROLE_WEIGHTS` (or `DEFAULT_WEIGHTS` for unknown roles).【F:app/core/scoring.py†L78-L81】
- The pipeline emits `INCOMPLETE_RADAR` if the radar is missing any required keys and marks the job `PARTIAL`.【F:app/workers/pipeline.py†L1889-L1907】

### Reasons for radar incompleteness (code-derived)
The code does **not** enumerate “occlusions” or “context” explicitly for skill scoring; the only causes of missing radar skills are:
1. **Insufficient scene changes** (low `scene_change_count`) → Finishing, Shot Power, Heading missing.【F:app/workers/pipeline.py†L470-L474】
2. **Low scene-change rate** (low `scene_change_rate`) → Positioning, Off-the-ball Movement, Composure missing.【F:app/workers/pipeline.py†L475-L477】

Additionally, scoring can fail earlier if **video features cannot be extracted** (e.g., duration/frame count/scene change count invalid), which raises `AnalysisError` and stops scoring entirely (not a partial radar).【F:app/workers/pipeline.py†L427-L468】

### Missing-radar warning and run status
If radar is incomplete (missing any required skill), the pipeline:
- Adds the warning `INCOMPLETE_RADAR` to `job.warnings`.【F:app/workers/pipeline.py†L1889-L1894】
- Sets `status = PARTIAL` when any warnings are present.【F:app/workers/pipeline.py†L1905-L1907】

## Bias / Weak points (code-derived)

1. **All skills depend solely on scene-change metrics** → any footage with low edit rate or long uncut sequences produces missing or low scores regardless of actual play quality.【F:app/workers/pipeline.py†L427-L478】
2. **Hard thresholds on scene-change count/rate** → short or static clips can drop specific skills to `None` and trigger `INCOMPLETE_RADAR`.【F:app/workers/pipeline.py†L470-L477】【F:app/workers/pipeline.py†L1889-L1907】
3. **Global-only aggregation** (no per-frame or positional context) → all radar skills collapse to a function of video edit dynamics, not on-field actions.【F:app/workers/pipeline.py†L427-L478】
4. **Role/overall equivalence** → role score and overall score are identical values from the same weighted average; there is no separate formula, reducing discriminatory power between the two outputs.【F:app/core/scoring.py†L87-L116】
5. **Role unknown uses default equal weights** → role-specific weighting disappears if role is missing/unknown, reducing differentiation and making overall/role scores uniform averages.【F:app/core/scoring.py†L45-L76】

## Code references for scoring status / warnings
- `INCOMPLETE_RADAR` warning and `PARTIAL` status are set in the `finalize_job` section of `run_analysis`.【F:app/workers/pipeline.py†L1889-L1907】
- Score outputs are persisted to `job.result` along with `skills_computed` and `skills_missing`.【F:app/workers/pipeline.py†L1930-L1947】
