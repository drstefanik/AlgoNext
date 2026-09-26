# Football ReID diagnostic, 26 September 2026

The deployed identity descriptor is OSNet x0.25 trained on MSMT17, combined
with torso HSV similarity. It is a general person descriptor. It cannot reliably
separate teammates from the same shirt color alone. This experiment compares it
with OSNet x1.0 trained on SoccerNet Re-Identification by
[SportsReID](https://github.com/shallowlearn/sportsreid).

This branch does not replace production weights, change thresholds, add a model
download, modify production dependencies, or enable football ratings. The model
comparison runs offline. The only proposed tracking change is the short-fragment
confirmation fix described below.

## Reproducible offline comparison

`scripts/compare_crop_reid_models.py` accepts the self-contained
`Verifica_Identita_AlgoNext.html` review file and its exported JSON annotations.
It reads the embedded JSON and JPEGs without executing the HTML. Dataset, crop,
reference and annotation provenance must match. Unknown or uncertain identities
are excluded and counted. Both target and other-player labels are required.

```bash
python scripts/compare_crop_reid_models.py \
  --review-html /path/to/Verifica_Identita_AlgoNext.html \
  --annotations /path/to/AlgoNext_verifica_identita.json \
  --generic-weights /path/to/generic-osnet.pt \
  --football-weights /path/to/sports-osnet-state.pt \
  --output /path/to/crop-reid-comparison.json
```

The runner checks every feature layer and tensor shape and loads checkpoints
with `weights_only=True`. A mismatched architecture or incomplete checkpoint is
an error. Both models use the production RGB 256x128/ImageNet preprocessing and
the same 0.7 learned-similarity + 0.3 HSV formula. The query is one saved anchor
JPEG; production aggregates several anchor crops, so absolute scores from this
diagnostic must not be used to tune production thresholds.

The public SportsReID checkpoint contains training metadata and a large
classifier. A separate PyTorch 2.6 CPU environment was used to extract only the
feature tensors, retaining `weights_only=True` and explicitly allowing the
NumPy scalar/dtype data types required by that checkpoint. No executable custom
classes were allowed. Production PyTorch remained 2.2.2. Use a reviewed
tensor-only export with this runner; do not disable the safe loader to make an
arbitrary checkpoint work. Weight/data usage terms still need confirmation before
any commercial deployment; the repository's code license is not sufficient to
establish rights to every training asset.

| Artifact | SHA-256 |
| --- | --- |
| Production generic checkpoint | `cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18` |
| Original SportsReID OSNet x1.0 checkpoint | `2dcc4bb973b0bacca78f1691e55df56c782130050e104d154c2a3d2e7a183fac` |
| Extracted feature tensor checkpoint used here | `59642c019efbebd95e0c53b9877eed72c19318b2afa9ad8391527755a35c4c69` |
| SportsReID configuration | `06e400a3d70623251e15cc41827a37bbc633b1db11904658acf57595c41cdb5f` |
| Review crop manifest | `8a8b179b2aa6f28e799ae5a4460198adfe2977145d44015c968f02bc54ffb74b` |

## Initial result and its limits

The review contains 96 kit-compatible candidate crops from two 60-second windows
starting at source-video seconds 1925 and 3300. One original manually selected
white-8 crop at 1192.607 seconds is the query. For this initial diagnostic only,
the assistant visually labeled 26 crops with readable jersey numbers: 12 target
crops and 14 other-player crops. **These are not independent human annotations.**
The delivered review file contains no prefilled answers or model predictions.

| Model | Pairwise AUC, pooled | AUC at 1925 s | AUC at 3300 s | Minimum target score | Maximum other-player score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generic OSNet x0.25 | 0.6131 | 0.4583 | 1.0000 | 0.6088 | 0.7928 |
| SoccerNet OSNet x1.0 | 0.7917 | 0.7083 | 1.0000 | 0.5626 | 0.8033 |

AUC measures ranking across the selected positive/negative crop pairs; it is
not the probability that a retained identity is correct. Both score distributions
overlap substantially. The crops are temporally correlated and come from only
two windows in one video. No confidence interval, accepted-identity accuracy,
whole-match recall, or production eligibility is inferred. The diagnostic always
emits `validated=false` and `production_eligible=false`.

The result justifies further evaluation of a football-specific descriptor.
Independent labels, unseen matches and cuts, hard same-team negatives, and the
existing frame/window benchmark are needed before any rollout. Identity
validation would still not validate technical or tactical player ratings.

The CLI reproduced the initial scores exactly. Negative input checks rejected
the wrong dataset, changed crop hash, duplicate identity, unknown annotation
provenance, and an empty review.

## Short-fragment confirmation fix

At 3 fps, a one-second fragment may have detections at 0, 1/3, 2/3 and 1 second.
If both endpoints were read together in one batch, they cannot independently
confirm identity. The previous sampler required every new read to be at least
0.6 seconds away from every previously sampled frame. This excluded both middle
frames, although either middle frame is more than 0.6 seconds away from the
opposite positive endpoint.

The sampler now allows a distinct fresh frame separated by at least 0.6 seconds
from at least one positive read. The downstream checker still requires different
image hashes, different request IDs, the temporal separation, motion continuity,
kit compatibility, unambiguous jersey evidence and matching fixture context.
Two readings in one batch still cannot establish identity. A test reproduces
the short-fragment failure and checks both eligible fresh frames and the
too-close negative case. The 57 focused jersey, association and window-logic
tests pass. No claim of increased full-match coverage is made from this unit test.
