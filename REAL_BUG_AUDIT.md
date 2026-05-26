# Real Bug Audit

> Note: this audit belongs to the previous generated snapshot. Regenerate the
> candidate table before reporting DL-only Xamt++ bug counts.

Snapshot: `2026-05-25 16:00:44 AEST`

Current candidate table: 194 unique DIFF keys / 137 base API groups.

After manual false-positive pass:

- Likely real differential bugs: 190 keys / 133 base groups.
- Strong/reportable without the two borderline numeric cases: 188 keys / 131 base groups.
- Confirmed false positives to drop: 4 keys / 4 base groups.

## Confirmed False Positives

| # | Key | Reason |
| ---: | --- | --- |
| 17 | `conv_transpose3d/nn/8` | Layout-only mismatch: NCDHW vs NDHWC singleton channel; values align after layout normalization. |
| 21 | `huber_loss/loss/4` | Reduction/return-shape semantics mismatch: Chainer returns per-sample vector, MindSpore returns scalar reduced loss. |
| 53 | `split/shape/4` | Adapter-normalization issue: TensorFlow `unstack` drops the split axis while MXNet/Paddle keep a size-1 axis. |
| 141 | `conv_transpose2d/nn` | Layout-only mismatch: NCHW vs NHWC singleton channel; values align after layout normalization. |

## Borderline But Not Dropped

| # | Key | Why borderline |
| ---: | --- | --- |
| 97 | `cond/linalg/component/1` | This is matrix condition-number, not control-flow cond; differences are real numeric variation on ill-conditioned inputs, but needs a minimal oracle before filing. |
| 131 | `normalize/generic/component/1` | Real current DIFF, but magnitude is small and may be due to epsilon/default numerical convention. |

## Recommended Reporting Count

Use **190 likely real differential bugs across 133 base API groups** as the main count.
For a stricter paper number, use **188 strong bugs across 131 base groups** and mention 2 numeric borderline cases separately.
