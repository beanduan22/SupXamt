# Xamt Adapter Fuzz Bug Candidates

> Note: this candidate table comes from a pre-DL-only target snapshot.
> Regenerate it before reporting Xamt++ counts without standalone NumPy/SciPy.

Source run: `/tmp/xamt_adapter_fuzz_9libs_60s.jsonl`
Run ended: `2026-05-21 18:40:12 AEST`
Scope: 9-library `adapter-aware` timed fuzz, 60 seconds per group, non-finite inputs disabled.

## Original Timed Run

| State | Groups |
| --- | ---: |
| PASS | 472 |
| DIFF | 50 |
| ERROR | 54 |
| SKIP | 22 |

## Adapter Fix Pass

After adapter fixes in `tools/diff_static_candidate_groups.py`, the non-timed adapter-aware validation has no ERROR groups:

| State | Groups |
| --- | ---: |
| PASS | 549 |
| DIFF | 16 |
| ERROR | 0 |
| SKIP | 21 |

Notes:
- Former ERROR groups fixed by adapter coverage include missing binary operands, axis/depth/indices/weight/bins arguments, random distribution size plans, MXNet ndarray inputs, and unsupported MXNet dtype/complex wrappers moved to SKIP.
- Current DIFF groups are not automatically confirmed framework bugs; several may still need normalization/tolerance review, especially scalar shape differences and low-confidence matches.

## Pairwise Adapter-Aware Component Expansion

Source components: `/tmp/xamt_pairwise_adapter_components_static_10libs.jsonl`
Detailed report: `PAIRWISE_ADAPTER_SUMMARY.md`

| State | Groups | Unique APIs |
| --- | ---: | ---: |
| PASS | 604 | 3994 |
| DIFF | 46 | 378 |
| ERROR | 0 | 0 |
| SKIP | 0 | 0 |

This expanded pass includes MindSpore and uses connected components of executable pair matches, so groups remain real API groups rather than pair edges. The current pairwise DIFF candidate list is recorded in `PAIRWISE_ADAPTER_SUMMARY.md`.

## Latest Full Combined Timed Run

Source run: `/tmp/xamt_group_timed_pairwise_60s_full_combined.jsonl`
Result file timestamp: `2026-05-22 16:22 AEST`
Scope: 10-library pairwise adapter-aware timed fuzz, 60 seconds per group, sharded full-combined run.

Current comparison tolerance in `tools/diff_static_candidate_groups.py`:
- Numeric arrays: `np.allclose(..., atol=1e-4, rtol=1e-3, equal_nan=True)`.
- Integer, boolean, string/object values: exact equality.
- Shape mismatches fail except scalar-size values, which are reshaped to scalar before comparison.

| State | Groups |
| --- | ---: |
| PASS | 641 |
| DIFF | 6 |
| ERROR | 1 |

### Latest 7 Error Candidates

| # | State | Key | Confidence | Seed | Counts | Notes |
| ---: | --- | --- | --- | ---: | --- | --- |
| 1 | DIFF | `["cond", "linalg", "component", 1]` | high (0.8986) | 389283221 | PASS 22701, DIFF 1 | `jax.numpy.linalg.cond`, `mxnet.numpy.linalg.cond`, `numpy.linalg.cond`, `paddle.linalg.cond`, and `torch.linalg.cond` disagree on the scalar condition number. |
| 2 | DIFF | `["digamma", "generic", "component", 1]` | high (0.8832) | 103271133 | PASS 10613, DIFF 1 | `mindspore.ops.digamma` differs materially from Chainer/JAX/Paddle/TensorFlow/Torch on a large negative value. |
| 3 | DIFF | `["logm", "linalg", "component", 1]` | high (0.8502) | 472260846 | PASS 326, DIFF 1 | `scipy.linalg.logm` and `tensorflow.linalg.logm` return complex results with opposite imaginary signs. |
| 4 | DIFF | `["lstsq", "generic", "component", 1]` | high (0.8750) | 209288881 | PASS 28361, DIFF 1 | `keras.ops.lstsq`, `mindspore.ops.function.lstsq`, and `torch.lstsq` produce different solution values. |
| 5 | DIFF | `["lstsq", "linalg", "component", 1]` | medium (0.7847) | 393260914 | PASS 394, DIFF 1 | `tensorflow.linalg.lstsq` is separated from JAX/MindSpore/MXNet/NumPy/Paddle/SciPy/Torch; other values also vary slightly. |
| 6 | DIFF | `["normalize", "generic", "component", 1]` | high (0.8955) | 564296018 | PASS 35498, DIFF 1 | `chainer.functions.normalize` and `keras.ops.normalize` differ beyond the active tolerance. |
| 7 | ERROR | `["histogram_bin_edges", "generic", "component", 1]` | high (0.9180) | 592260520 | ERROR 1 | `mindspore.numpy.histogram_bin_edges` raises `SystemError: <method reshape of TensorPy objects> returned a result with an exception set`; JAX/MXNet/NumPy/Paddle agree on the output. |

## Expanded CPU Edge-Value Timed Run

Source shards: `/tmp/xamt_edge_round_cpu_20260523_1013/shard*.jsonl`
Combined output: `/tmp/xamt_edge_round_cpu_20260523_1013/combined.jsonl`
Run ended: `2026-05-23 11:34 AEST`
Scope: 10-library pairwise adapter-aware timed fuzz, 60 seconds per group, CPU-only, `--include-edge-values`, `--stop-on-diff`.

Comparison tolerance remains `np.allclose(..., atol=1e-4, rtol=1e-3, equal_nan=True)` for numeric arrays.

| State | Groups |
| --- | ---: |
| PASS | 610 |
| DIFF | 37 |
| ERROR | 0 |
| SKIP | 3 |

DIFF confidence split: high 19, medium 17, low 1.
SKIP groups: `["vmap", "generic", "component", 1]`, `["sequence_mask", "generic", "component", 1]`, `["array_str", "generic", "component", 1]`.

### Expanded DIFF Candidates

| # | Key | Confidence | Seed | Counts |
| ---: | --- | --- | ---: | --- |
| 1 | `["frexp", "generic", "component", 1]` | medium (0.822) | 133260526 | DIFF 1, PASS 3 |
| 2 | `["geqrf", "generic", "component", 1]` | medium (0.7689) | 141260525 | DIFF 1, PASS 2 |
| 3 | `["rsqrt", "generic", "component", 1]` | high (0.8872) | 277260529 | DIFF 1, PASS 6 |
| 4 | `["sgn", "generic", "component", 1]` | medium (0.8465) | 285260528 | DIFF 1, PASS 5 |
| 5 | `["threshold", "generic", "component", 1]` | medium (0.7969) | 317260524 | DIFF 1, PASS 1 |
| 6 | `["flatnonzero", "generic", "component", 1]` | medium (0.8085) | 590260524 | DIFF 1, PASS 1 |
| 7 | `["trim_zeros", "generic", "component", 1]` | high (1.0) | 630260526 | DIFF 1, PASS 3 |
| 8 | `["count_nonzero", "reduction", "component", 1]` | high (0.8847) | 87260523 | DIFF 1 |
| 9 | `["digamma", "generic", "component", 1]` | high (0.8832) | 103260528 | DIFF 1, PASS 5 |
| 10 | `["heaviside", "generic", "component", 1]` | medium (0.7838) | 151260526 | DIFF 1, PASS 3 |
| 11 | `["inv", "linalg", "component", 1]` | high (0.8505) | 167281393 | DIFF 1, PASS 20870 |
| 12 | `["lgamma", "generic", "component", 1]` | high (0.9114) | 191260524 | DIFF 1, PASS 1 |
| 13 | `["sign", "generic", "component", 1]` | high (0.9066) | 287260526 | DIFF 1, PASS 3 |
| 14 | `["cbrt", "generic", "component", 1]` | high (0.8536) | 543260529 | DIFF 1, PASS 6 |
| 15 | `["nonzero", "generic", "component", 1]` | medium (0.7973) | 240260524 | DIFF 1, PASS 1 |
| 16 | `["signbit", "generic", "component", 1]` | high (0.8811) | 288260525 | DIFF 1, PASS 2 |
| 17 | `["lstsq", "linalg", "component", 1]` | medium (0.7847) | 392260837 | DIFF 1, PASS 314 |
| 18 | `["erfcinv", "generic", "component", 1]` | high (0.929) | 464260524 | DIFF 1, PASS 1 |
| 19 | `["logm", "linalg", "component", 1]` | high (0.8502) | 472260846 | DIFF 1, PASS 323 |
| 20 | `["rankdata", "generic", "component", 1]` | high (1.0) | 664260527 | DIFF 1, PASS 4 |
| 21 | `["angle", "generic", "component", 1]` | medium (0.8452) | 33260528 | DIFF 1, PASS 5 |
| 22 | `["argmax", "reduction", "component", 1]` | medium (0.8329) | 41260527 | DIFF 1, PASS 4 |
| 23 | `["unique_all", "generic", "component", 1]` | medium (0.716) | 633260523 | DIFF 1 |
| 24 | `["log10", "generic", "component", 1]` | high (0.8921) | 194260528 | DIFF 1, PASS 5 |
| 25 | `["xlogy", "generic", "component", 1]` | medium (0.8189) | 346260526 | DIFF 1, PASS 3 |
| 26 | `["unique_counts", "generic", "component", 1]` | medium (0.7037) | 634260529 | DIFF 1, PASS 6 |
| 27 | `["ceil", "generic", "component", 1]` | high (0.9074) | 67260529 | DIFF 1, PASS 6 |
| 28 | `["corrcoef", "generic", "component", 1]` | medium (0.8071) | 83260527 | DIFF 1, PASS 4 |
| 29 | `["unique", "generic", "component", 1]` | low (0.6758) | 331260524 | DIFF 1, PASS 1 |
| 30 | `["ndtri", "generic", "component", 1]` | medium (0.7568) | 419260527 | DIFF 1, PASS 4 |
| 31 | `["unique_inverse", "generic", "component", 1]` | medium (0.7079) | 635260528 | DIFF 1, PASS 5 |
| 32 | `["argwhere", "generic", "component", 1]` | high (0.8671) | 44260524 | DIFF 1, PASS 1 |
| 33 | `["log2", "generic", "component", 1]` | high (0.8909) | 196260526 | DIFF 1, PASS 3 |
| 34 | `["cond", "linalg", "component", 1]` | high (0.8986) | 388275511 | DIFF 1, PASS 14988 |
| 35 | `["gammaln", "generic", "component", 1]` | high (0.8505) | 412260527 | DIFF 1, PASS 4 |
| 36 | `["normalize", "generic", "component", 1]` | high (0.8955) | 564296018 | DIFF 1, PASS 35495 |
| 37 | `["unique_values", "generic", "component", 1]` | medium (0.7006) | 636260527 | DIFF 1, PASS 4 |


## Current Static DIFFs After Adapter Fixes

| # | Key | Chosen APIs |
| ---: | --- | --- |
| 1 | `["max", "reduction"]` | `chainer:chainer.functions.max; jax:jax.numpy.amax; keras:keras.ops.amax; mxnet:mxnet.ndarray.max; numpy:numpy.amax; paddle:paddle.amax; tensorflow:tensorflow.math.reduce_max; torch:torch.amax` |
| 2 | `["min", "reduction"]` | `chainer:chainer.functions.min; jax:jax.numpy.amin; keras:keras.ops.amin; mxnet:mxnet.ndarray.min; numpy:numpy.amin; paddle:paddle.amin; tensorflow:tensorflow.math.reduce_min; torch:torch.amin` |
| 3 | `["dot", "generic"]` | `jax:jax.numpy.dot; keras:keras.ops.dot; mxnet:mxnet.ndarray.dot; numpy:numpy.dot; paddle:paddle.dot; torch:torch.dot` |
| 4 | `["flatten", "generic"]` | `chainer:chainer.functions.flatten; mxnet:mxnet.ndarray.Flatten; paddle:paddle.flatten; torch:torch.flatten` |
| 5 | `["mean", "reduction"]` | `chainer:chainer.functions.mean; jax:jax.numpy.mean; keras:keras.ops.mean; mxnet:mxnet.ndarray.mean; numpy:numpy.mean; paddle:paddle.mean; tensorflow:tensorflow.math.reduce_mean; torch:torch.mean` |
| 6 | `["nansum", "generic"]` | `jax:jax.numpy.nansum; keras:keras.ops.nansum; mxnet:mxnet.ndarray.nansum; numpy:numpy.nansum; paddle:paddle.nansum; torch:torch.nansum` |
| 7 | `["norm", "linalg"]` | `jax:jax.numpy.linalg.norm; keras:keras.ops.norm; mxnet:mxnet.ndarray.norm; numpy:numpy.linalg.norm; paddle:paddle.linalg.norm; scipy:scipy.linalg.norm; tensorflow:tensorflow.linalg.norm; torch:torch.linalg.norm` |
| 8 | `["prod", "reduction"]` | `chainer:chainer.functions.prod; jax:jax.numpy.prod; keras:keras.ops.prod; mxnet:mxnet.ndarray.prod; numpy:numpy.prod; paddle:paddle.prod; tensorflow:tensorflow.math.reduce_prod; torch:torch.prod` |
| 9 | `["sequence_mask", "generic"]` | `mxnet:mxnet.ndarray.SequenceMask; tensorflow:tensorflow.sequence_mask` |
| 10 | `["hardsigmoid", "generic"]` | `chainer:chainer.functions.hard_sigmoid; keras:keras.ops.hard_sigmoid; mxnet:mxnet.ndarray.hard_sigmoid; tensorflow:tensorflow.keras.activations.hard_sigmoid` |
| 11 | `["l1_loss", "loss"]` | `chainer:chainer.functions.mean_absolute_error; keras:keras.losses.mean_absolute_error; tensorflow:tensorflow.keras.losses.MAE` |
| 12 | `["mse_loss", "loss"]` | `chainer:chainer.functions.mean_squared_error; keras:keras.losses.mean_squared_error; tensorflow:tensorflow.keras.losses.MSE` |
| 13 | `["nanprod", "generic"]` | `jax:jax.numpy.nanprod; keras:keras.ops.nanprod; mxnet:mxnet.ndarray.nanprod; numpy:numpy.nanprod` |
| 14 | `["partition", "generic"]` | `jax:jax.numpy.partition; mxnet:mxnet.numpy.partition; numpy:numpy.partition` |
| 15 | `["polydiv", "generic"]` | `jax:jax.numpy.polydiv; mxnet:mxnet.numpy.polydiv; numpy:numpy.polydiv` |
| 16 | `["roots", "generic"]` | `jax:jax.numpy.roots; mxnet:mxnet.numpy.roots; numpy:numpy.roots` |

## Original 60s DIFF Candidates

These are the bug candidates from the completed timed fuzz run. Keep them for triage even if some later reduce to tolerance or normalization issues.

| # | Key | Confidence | Seed | Chosen APIs |
| ---: | --- | --- | ---: | --- |
| 1 | `["max", "reduction"]` | medium (0.8354) | 28260520 | `chainer:chainer.functions.max; jax:jax.numpy.amax; keras:keras.ops.amax; mxnet:mxnet.ndarray.max; numpy:numpy.amax; paddle:paddle.amax; tensorflow:tensorflow.math.reduce_max; torch:torch.amax` |
| 2 | `["min", "reduction"]` | medium (0.8357) | 29260520 | `chainer:chainer.functions.min; jax:jax.numpy.amin; keras:keras.ops.amin; mxnet:mxnet.ndarray.min; numpy:numpy.amin; paddle:paddle.amin; tensorflow:tensorflow.math.reduce_min; torch:torch.amin` |
| 3 | `["angle", "generic"]` | high (0.8538) | 30260524 | `jax:jax.numpy.angle; keras:keras.ops.angle; numpy:numpy.angle; paddle:paddle.angle; tensorflow:tensorflow.math.angle; torch:torch.angle` |
| 4 | `["argmax", "reduction"]` | medium (0.8222) | 38260523 | `chainer:chainer.functions.argmax; jax:jax.numpy.argmax; keras:keras.ops.argmax; mxnet:mxnet.ndarray.argmax; numpy:numpy.argmax; paddle:paddle.argmax; tensorflow:tensorflow.math.argmax; torch:torch.argmax` |
| 5 | `["ceil", "generic"]` | high (0.8686) | 63260526 | `chainer:chainer.functions.ceil; jax:jax.numpy.ceil; keras:keras.ops.ceil; mxnet:mxnet.ndarray.ceil; paddle:paddle.ceil; tensorflow:tensorflow.math.ceil; torch:torch.ceil` |
| 6 | `["corrcoef", "generic"]` | medium (0.7824) | 75260521 | `jax:jax.numpy.corrcoef; keras:keras.ops.corrcoef; mxnet:mxnet.numpy.corrcoef; numpy:numpy.corrcoef; paddle:paddle.tensor.corrcoef; torch:torch.corrcoef` |
| 7 | `["count_nonzero", "reduction"]` | high (0.8789) | 78260525 | `jax:jax.numpy.count_nonzero; keras:keras.ops.count_nonzero; mxnet:mxnet.numpy.count_nonzero; numpy:numpy.count_nonzero; paddle:paddle.count_nonzero; tensorflow:tensorflow.math.count_nonzero; torch:torch.count_nonzero` |
| 8 | `["cov", "generic"]` | high (0.869) | 79264102 | `jax:jax.numpy.cov; mxnet:mxnet.numpy.cov; numpy:numpy.cov; paddle:paddle.tensor.cov; torch:torch.cov` |
| 9 | `["digamma", "generic"]` | high (0.8664) | 92260525 | `chainer:chainer.functions.digamma; jax:jax.scipy.special.digamma; paddle:paddle.digamma; tensorflow:tensorflow.math.digamma; torch:torch.special.digamma` |
| 10 | `["dot", "generic"]` | medium (0.7543) | 94260520 | `jax:jax.numpy.dot; keras:keras.ops.dot; mxnet:mxnet.ndarray.dot; numpy:numpy.dot; paddle:paddle.dot; torch:torch.dot` |
| 11 | `["flatten", "generic"]` | medium (0.7898) | 113260520 | `chainer:chainer.functions.flatten; mxnet:mxnet.ndarray.Flatten; paddle:paddle.flatten; torch:torch.flatten` |
| 12 | `["heaviside", "generic"]` | medium (0.7401) | 139260524 | `jax:jax.numpy.heaviside; keras:keras.ops.heaviside; paddle:paddle.heaviside; torch:torch.heaviside` |
| 13 | `["inv", "linalg"]` | medium (0.8359) | 150261326 | `chainer:chainer.functions.inv; jax:jax.numpy.linalg.inv; keras:keras.ops.inv; mxnet:mxnet.ndarray.linalg.inverse; numpy:numpy.linalg.inv; paddle:paddle.linalg.inv; scipy:scipy.linalg.inv; tensorflow:tensorflow.linalg.inv; torch:torch.linalg.inv` |
| 14 | `["layer_norm", "nn", 5]` | medium (0.7857) | 167260521 | `keras:keras.ops.layer_normalization; torch:torch.nn.functional.layer_norm` |
| 15 | `["lgamma", "generic"]` | high (0.8906) | 173260521 | `chainer:chainer.functions.lgamma; jax:jax.lax.lgamma; paddle:paddle.lgamma; tensorflow:tensorflow.math.lgamma; torch:torch.lgamma` |
| 16 | `["log10", "generic"]` | high (0.8608) | 177260524 | `chainer:chainer.functions.log10; jax:jax.numpy.log10; keras:keras.ops.log10; mxnet:mxnet.ndarray.log10; paddle:paddle.log10; torch:torch.log10` |
| 17 | `["log2", "generic"]` | high (0.8603) | 179260522 | `chainer:chainer.functions.log2; jax:jax.numpy.log2; keras:keras.ops.log2; mxnet:mxnet.ndarray.log2; paddle:paddle.log2; torch:torch.log2` |
| 18 | `["lstsq", "generic"]` | medium (0.8106) | 189261160 | `keras:keras.ops.lstsq; paddle:paddle.tensor.lstsq; torch:torch.lstsq` |
| 19 | `["lu_solve", "linalg"]` | medium (0.7777) | 191265571 | `jax:jax.scipy.linalg.lu_solve; paddle:paddle.linalg.lu_solve; scipy:scipy.linalg.lu_solve; tensorflow:tensorflow.linalg.lu_solve; torch:torch.linalg.lu_solve` |
| 20 | `["matrix_exp", "linalg"]` | medium (0.7735) | 194260700 | `paddle:paddle.linalg.matrix_exp; torch:torch.linalg.matrix_exp` |
| 21 | `["mean", "reduction"]` | medium (0.8396) | 198260520 | `chainer:chainer.functions.mean; jax:jax.numpy.mean; keras:keras.ops.mean; mxnet:mxnet.ndarray.mean; numpy:numpy.mean; paddle:paddle.mean; tensorflow:tensorflow.math.reduce_mean; torch:torch.mean` |
| 22 | `["nansum", "generic"]` | medium (0.8349) | 211260520 | `jax:jax.numpy.nansum; keras:keras.ops.nansum; mxnet:mxnet.ndarray.nansum; numpy:numpy.nansum; paddle:paddle.nansum; torch:torch.nansum` |
| 23 | `["nonzero", "generic"]` | medium (0.7902) | 215260521 | `jax:jax.numpy.nonzero; keras:keras.ops.nonzero; mxnet:mxnet.numpy.nonzero; numpy:numpy.nonzero; paddle:paddle.nonzero; torch:torch.nonzero` |
| 24 | `["norm", "linalg"]` | high (0.8723) | 216260520 | `jax:jax.numpy.linalg.norm; keras:keras.ops.norm; mxnet:mxnet.ndarray.norm; numpy:numpy.linalg.norm; paddle:paddle.linalg.norm; scipy:scipy.linalg.norm; tensorflow:tensorflow.linalg.norm; torch:torch.linalg.norm` |
| 25 | `["prod", "reduction"]` | medium (0.8307) | 234260520 | `chainer:chainer.functions.prod; jax:jax.numpy.prod; keras:keras.ops.prod; mxnet:mxnet.ndarray.prod; numpy:numpy.prod; paddle:paddle.prod; tensorflow:tensorflow.math.reduce_prod; torch:torch.prod` |
| 26 | `["rsqrt", "generic"]` | medium (0.8408) | 258260520 | `chainer:chainer.functions.rsqrt; jax:jax.lax.rsqrt; keras:keras.ops.rsqrt; mxnet:mxnet.ndarray.rsqrt; paddle:paddle.rsqrt; tensorflow:tensorflow.math.rsqrt; torch:torch.rsqrt` |
| 27 | `["sign", "generic"]` | high (0.8819) | 267260525 | `chainer:chainer.functions.sign; jax:jax.numpy.sign; keras:keras.ops.sign; mxnet:mxnet.ndarray.sign; paddle:paddle.sign; tensorflow:tensorflow.math.sign; torch:torch.sign` |
| 28 | `["solve", "linalg"]` | medium (0.7827) | 273261232 | `jax:jax.numpy.linalg.solve; keras:keras.ops.solve; mxnet:mxnet.numpy.linalg.solve; numpy:numpy.linalg.solve; paddle:paddle.linalg.solve; scipy:scipy.linalg.solve; tensorflow:tensorflow.linalg.solve; torch:torch.linalg.solve` |
| 29 | `["svd_lowrank", "linalg"]` | high (0.9781) | 285260551 | `paddle:paddle.linalg.svd_lowrank; torch:torch.svd_lowrank` |
| 30 | `["threshold", "generic"]` | medium (0.82) | 292260521 | `keras:keras.ops.threshold; tensorflow:tensorflow.keras.activations.threshold; torch:torch.threshold` |
| 31 | `["topk", "generic", 3]` | medium (0.7687) | 295260525 | `jax:jax.lax.top_k; keras:keras.ops.top_k` |
| 32 | `["cond", "linalg"]` | high (0.9019) | 356261016 | `jax:jax.numpy.linalg.cond; mxnet:mxnet.numpy.linalg.cond; numpy:numpy.linalg.cond; paddle:paddle.linalg.cond; torch:torch.linalg.cond` |
| 33 | `["lstsq", "linalg"]` | medium (0.7716) | 360260626 | `jax:jax.numpy.linalg.lstsq; mxnet:mxnet.numpy.linalg.lstsq; numpy:numpy.linalg.lstsq; paddle:paddle.linalg.lstsq; scipy:scipy.linalg.lstsq; tensorflow:tensorflow.linalg.lstsq; torch:torch.linalg.lstsq` |
| 34 | `["pinv", "linalg"]` | medium (0.8349) | 366261017 | `jax:jax.numpy.linalg.pinv; mxnet:mxnet.numpy.linalg.pinv; numpy:numpy.linalg.pinv; paddle:paddle.linalg.pinv; scipy:scipy.linalg.pinv; tensorflow:tensorflow.linalg.pinv; torch:torch.linalg.pinv` |
| 35 | `["gammaln", "generic"]` | high (0.8678) | 373260524 | `jax:jax.scipy.special.gammaln; mxnet:mxnet.ndarray.gammaln; paddle:paddle.gammaln; torch:torch.special.gammaln` |
| 36 | `["ndtri", "generic"]` | medium (0.7568) | 380260524 | `chainer:chainer.functions.ndtri; jax:jax.scipy.special.ndtri; tensorflow:tensorflow.math.ndtri; torch:torch.special.ndtri` |
| 37 | `["sequence_mask", "generic"]` | low (0.6357) | 405260520 | `mxnet:mxnet.ndarray.SequenceMask; tensorflow:tensorflow.sequence_mask` |
| 38 | `["erfcinv", "generic"]` | high (0.929) | 412260520 | `chainer:chainer.functions.erfcinv; tensorflow:tensorflow.math.erfcinv` |
| 39 | `["logm", "linalg"]` | high (0.8502) | 420260520 | `scipy:scipy.linalg.logm; tensorflow:tensorflow.linalg.logm` |
| 40 | `["gamma", "random"]` | medium (0.7348) | 428260520 | `jax:jax.random.gamma; keras:keras.random.gamma; mxnet:mxnet.ndarray.gamma; numpy:numpy.random.gamma; tensorflow:tensorflow.random.gamma` |
| 41 | `["hardsigmoid", "generic"]` | high (0.8812) | 438260520 | `chainer:chainer.functions.hard_sigmoid; keras:keras.ops.hard_sigmoid; mxnet:mxnet.ndarray.hard_sigmoid; tensorflow:tensorflow.keras.activations.hard_sigmoid` |
| 42 | `["l1_loss", "loss"]` | high (0.872) | 451260520 | `chainer:chainer.functions.mean_absolute_error; keras:keras.losses.mean_absolute_error; tensorflow:tensorflow.keras.losses.MAE` |
| 43 | `["mse_loss", "loss"]` | high (0.872) | 452260520 | `chainer:chainer.functions.mean_squared_error; keras:keras.losses.mean_squared_error; tensorflow:tensorflow.keras.losses.MSE` |
| 44 | `["nanprod", "generic"]` | high (0.8777) | 510260520 | `jax:jax.numpy.nanprod; keras:keras.ops.nanprod; mxnet:mxnet.ndarray.nanprod; numpy:numpy.nanprod` |
| 45 | `["normalize", "generic"]` | high (0.8955) | 514260527 | `chainer:chainer.functions.normalize; keras:keras.ops.normalize` |
| 46 | `["array_str", "generic"]` | high (1.0) | 528260520 | `jax:jax.numpy.array_str; numpy:numpy.array_str` |
| 47 | `["factorial", "generic"]` | high (0.8548) | 598276967 | `jax:jax.scipy.special.factorial; scipy:scipy.special.factorial` |
| 48 | `["rsf2csf", "linalg"]` | high (1.0) | 606260523 | `jax:jax.scipy.linalg.rsf2csf; scipy:scipy.linalg.rsf2csf` |
| 49 | `["detrend", "signal"]` | high (1.0) | 610261613 | `jax:jax.scipy.signal.detrend; scipy:scipy.signal.detrend` |
| 50 | `["fftconvolve", "signal"]` | high (1.0) | 611268517 | `jax:jax.scipy.signal.fftconvolve; scipy:scipy.signal.fftconvolve` |
