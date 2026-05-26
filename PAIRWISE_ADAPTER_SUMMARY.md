# Pairwise Adapter-Aware Expansion

> Note: this is a legacy 10-library snapshot kept for traceability. The current
> Xamt++ target set excludes NumPy and SciPy as standalone libraries; regenerate
> this summary before reporting DL-only counts.

Strategy: `pairwise-adapter-aware` clusters executable pair matches into connected functionality groups. It does not emit each pair as a separate group.
Full component JSONL: `/tmp/xamt_pairwise_adapter_components_static_10libs.jsonl`

## Coverage

- Groups: 650
- Unique APIs: 4372
- MindSpore APIs: 1314
- API memberships: 4372
- Libraries: 10 (chainer, jax, keras, mindspore, mxnet, numpy, paddle, scipy, tensorflow, torch)
- Collected APIs by library: {"chainer": 216, "jax": 729, "keras": 440, "mindspore": 2839, "mxnet": 642, "numpy": 387, "paddle": 1192, "scipy": 615, "tensorflow": 710, "torch": 1012}
- Base keys: 650

## Summary

| State | Groups | Unique APIs |
| --- | ---: | ---: |
| PASS | 604 | 3994 |
| DIFF | 46 | 378 |
| ERROR | 0 | 0 |
| SKIP | 0 | 0 |

## Confidence By Status

| State | High | Medium | Low |
| --- | ---: | ---: | ---: |
| PASS | 319 | 280 | 5 |
| DIFF | 17 | 26 | 3 |
| ERROR | 0 | 0 | 0 |
| SKIP | 0 | 0 | 0 |

## DIFF Base Keys

| Base Key | DIFF Groups |
| --- | ---: |
| `["max", "reduction"]` | 1 |
| `["min", "reduction"]` | 1 |
| `["aminmax", "generic"]` | 1 |
| `["bernoulli", "random"]` | 1 |
| `["cummax", "generic"]` | 1 |
| `["cummin", "generic"]` | 1 |
| `["dot", "generic"]` | 1 |
| `["eig", "linalg"]` | 1 |
| `["flatten", "generic"]` | 1 |
| `["inv", "linalg"]` | 1 |
| `["kaiser_window", "signal"]` | 1 |
| `["kl_divergence", "loss"]` | 1 |
| `["layer_norm", "nn"]` | 1 |
| `["mean", "reduction"]` | 1 |
| `["median", "generic"]` | 1 |
| `["nanmedian", "generic"]` | 1 |
| `["nansum", "generic"]` | 1 |
| `["norm", "linalg"]` | 1 |
| `["prod", "reduction"]` | 1 |
| `["promote_types", "generic"]` | 1 |
| `["range", "generic"]` | 1 |
| `["result_type", "generic"]` | 1 |
| `["std_mean", "reduction"]` | 1 |
| `["stft", "signal"]` | 1 |
| `["svd", "linalg"]` | 1 |
| `["triu_indices", "generic"]` | 1 |
| `["unique", "generic"]` | 1 |
| `["var_mean", "reduction"]` | 1 |
| `["vmap", "generic"]` | 1 |
| `["binary_cross_entropy", "nn"]` | 1 |
| `["linear", "nn"]` | 1 |
| `["bessel_y0", "generic"]` | 1 |
| `["bessel_y1", "generic"]` | 1 |
| `["fft", "fft"]` | 1 |
| `["ifft", "fft"]` | 1 |
| `["sequence_mask", "generic"]` | 1 |
| `["hardsigmoid", "generic"]` | 1 |
| `["l1_loss", "loss"]` | 1 |
| `["mse_loss", "loss"]` | 1 |
| `["binary_cross_entropy", "loss"]` | 1 |
| `["nanprod", "generic"]` | 1 |
| `["indices", "generic"]` | 1 |
| `["partition", "generic"]` | 1 |
| `["polydiv", "generic"]` | 1 |
| `["roots", "generic"]` | 1 |
| `["cho_factor", "linalg"]` | 1 |

## DIFF Group Candidates

| # | Key | APIs | Libraries | Confidence | Chosen APIs |
| ---: | --- | ---: | ---: | --- | --- |
| 10 | `["max", "reduction", "component", 1]` | 26 | 9 | high (0.8646) | `chainer:chainer.functions.max; jax:jax.numpy.amax; keras:keras.ops.amax; mindspore:mindspore.numpy.amax; mxnet:mxnet.ndarray.max; numpy:numpy.amax; paddle:paddle.amax; tensorflow:tensorflow.math.reduce_max; torch:torch.amax` |
| 11 | `["min", "reduction", "component", 1]` | 26 | 9 | high (0.8648) | `chainer:chainer.functions.min; jax:jax.numpy.amin; keras:keras.ops.amin; mindspore:mindspore.numpy.amin; mxnet:mxnet.ndarray.min; numpy:numpy.amin; paddle:paddle.amin; tensorflow:tensorflow.math.reduce_min; torch:torch.amin` |
| 12 | `["aminmax", "generic", "component", 1]` | 4 | 2 | medium (0.7725) | `mindspore:mindspore.ops.aminmax; torch:torch.aminmax` |
| 33 | `["bernoulli", "random", "component", 1]` | 6 | 4 | medium (0.7862) | `jax:jax.random.bernoulli; mindspore:mindspore.ops.bernoulli; paddle:paddle.bernoulli; torch:torch.bernoulli` |
| 71 | `["cummax", "generic", "component", 1]` | 2 | 2 | medium (0.7225) | `jax:jax.lax.cummax; paddle:paddle.cummax` |
| 72 | `["cummin", "generic", "component", 1]` | 2 | 2 | medium (0.7225) | `jax:jax.lax.cummin; paddle:paddle.cummin` |
| 86 | `["dot", "generic", "component", 1]` | 9 | 7 | medium (0.7523) | `jax:jax.numpy.dot; keras:keras.ops.dot; mindspore:mindspore.numpy.dot; mxnet:mxnet.ndarray.dot; numpy:numpy.dot; paddle:paddle.dot; torch:torch.dot` |
| 90 | `["eig", "linalg", "component", 1]` | 23 | 9 | high (0.8715) | `jax:jax.numpy.linalg.eig; keras:keras.ops.eig; mindspore:mindspore.ops.eig; mxnet:mxnet.numpy.linalg.eig; numpy:numpy.linalg.eig; paddle:paddle.linalg.eig; scipy:scipy.linalg.eig; tensorflow:tensorflow.linalg.eig; torch:torch.linalg.eig` |
| 103 | `["flatten", "generic", "component", 1]` | 9 | 5 | medium (0.7552) | `chainer:chainer.functions.flatten; mindspore:mindspore.ops.flatten; mxnet:mxnet.ndarray.Flatten; paddle:paddle.flatten; torch:torch.flatten` |
| 147 | `["inv", "linalg", "component", 1]` | 21 | 10 | high (0.8505) | `chainer:chainer.functions.inv; jax:jax.numpy.linalg.inv; keras:keras.ops.inv; mindspore:mindspore.ops.function.inv; mxnet:mxnet.ndarray.linalg.inverse; numpy:numpy.linalg.inv; paddle:paddle.linalg.inv; scipy:scipy.linalg.inv; tensorflow:tensorflow.linalg.inv; torch:torch.linalg.inv` |
| 161 | `["kaiser_window", "signal", "component", 1]` | 6 | 4 | medium (0.8426) | `mindspore:mindspore.ops.function.kaiser_window; paddle:paddle.kaiser_window; tensorflow:tensorflow.signal.kaiser_window; torch:torch.kaiser_window` |
| 162 | `["kl_divergence", "loss", "component", 1]` | 11 | 3 | medium (0.7672) | `keras:keras.losses.kl_divergence; mindspore:mindspore.ops.function.kl_div; tensorflow:tensorflow.keras.losses.KLD` |
| 165 | `["layer_norm", "nn", "component", 1]` | 7 | 5 | medium (0.7967) | `chainer:chainer.functions.layer_normalization; keras:keras.ops.layer_normalization; mxnet:mxnet.ndarray.LayerNorm; paddle:paddle.nn.functional.layer_norm; torch:torch.nn.functional.layer_norm` |
| 201 | `["mean", "reduction", "component", 1]` | 15 | 9 | high (0.8763) | `chainer:chainer.functions.mean; jax:jax.numpy.mean; keras:keras.ops.mean; mindspore:mindspore.numpy.mean; mxnet:mxnet.ndarray.mean; numpy:numpy.mean; paddle:paddle.mean; tensorflow:tensorflow.math.reduce_mean; torch:torch.mean` |
| 202 | `["median", "generic", "component", 1]` | 8 | 6 | high (0.8567) | `jax:jax.numpy.median; keras:keras.ops.median; mindspore:mindspore.ops.function.median; numpy:numpy.median; paddle:paddle.median; torch:torch.median` |
| 213 | `["nanmedian", "generic", "component", 1]` | 9 | 7 | high (0.8702) | `jax:jax.numpy.nanmedian; keras:keras.ops.nanmedian; mindspore:mindspore.ops.function.nanmedian; mxnet:mxnet.numpy.nanmedian; numpy:numpy.nanmedian; paddle:paddle.nanmedian; torch:torch.nanmedian` |
| 215 | `["nansum", "generic", "component", 1]` | 11 | 7 | medium (0.8359) | `jax:jax.numpy.nansum; keras:keras.ops.nansum; mindspore:mindspore.numpy.nansum; mxnet:mxnet.ndarray.nansum; numpy:numpy.nansum; paddle:paddle.nansum; torch:torch.nansum` |
| 221 | `["norm", "linalg", "component", 1]` | 16 | 9 | high (0.8933) | `jax:jax.numpy.linalg.norm; keras:keras.ops.norm; mindspore:mindspore.numpy.norm; mxnet:mxnet.ndarray.norm; numpy:numpy.linalg.norm; paddle:paddle.linalg.norm; scipy:scipy.linalg.norm; tensorflow:tensorflow.linalg.norm; torch:torch.linalg.norm` |
| 235 | `["prod", "reduction", "component", 1]` | 14 | 9 | high (0.8719) | `chainer:chainer.functions.prod; jax:jax.numpy.prod; keras:keras.ops.prod; mindspore:mindspore.ops.function.prod; mxnet:mxnet.ndarray.prod; numpy:numpy.prod; paddle:paddle.prod; tensorflow:tensorflow.math.reduce_prod; torch:torch.prod` |
| 236 | `["promote_types", "generic", "component", 1]` | 5 | 5 | medium (0.8173) | `jax:jax.numpy.promote_types; mindspore:mindspore.numpy.promote_types; mxnet:mxnet.numpy.promote_types; numpy:numpy.promote_types; torch:torch.promote_types` |
| 243 | `["range", "generic", "component", 1]` | 6 | 4 | medium (0.7616) | `mindspore:mindspore.ops.function.range; paddle:paddle.range; tensorflow:tensorflow.range; torch:torch.range` |
| 252 | `["result_type", "generic", "component", 1]` | 4 | 4 | high (0.9405) | `jax:jax.numpy.result_type; mindspore:mindspore.numpy.result_type; numpy:numpy.result_type; torch:torch.result_type` |
| 282 | `["std_mean", "reduction", "component", 1]` | 4 | 2 | medium (0.7658) | `mindspore:mindspore.ops.function.std_mean; torch:torch.std_mean` |
| 283 | `["stft", "signal", "component", 1]` | 4 | 4 | medium (0.7277) | `jax:jax.scipy.signal.stft; scipy:scipy.signal.stft; tensorflow:tensorflow.signal.stft; torch:torch.stft` |
| 286 | `["svd", "linalg", "component", 1]` | 18 | 9 | medium (0.7814) | `jax:jax.numpy.linalg.svd; keras:keras.ops.svd; mindspore:mindspore.ops.function.svd; mxnet:mxnet.numpy.linalg.svd; numpy:numpy.linalg.svd; paddle:paddle.linalg.svd; scipy:scipy.linalg.svd; tensorflow:tensorflow.linalg.svd; torch:torch.linalg.svd` |
| 307 | `["triu_indices", "generic", "component", 1]` | 4 | 4 | medium (0.834) | `jax:jax.numpy.triu_indices; mindspore:mindspore.numpy.triu_indices; numpy:numpy.triu_indices; paddle:paddle.triu_indices` |
| 311 | `["unique", "generic", "component", 1]` | 9 | 6 | low (0.6758) | `jax:jax.numpy.unique; mindspore:mindspore.numpy.unique; mxnet:mxnet.numpy.unique; numpy:numpy.unique; paddle:paddle.unique; torch:torch.unique` |
| 317 | `["var_mean", "reduction", "component", 1]` | 4 | 2 | medium (0.7642) | `mindspore:mindspore.ops.function.var_mean; torch:torch.var_mean` |
| 321 | `["vmap", "generic", "component", 1]` | 2 | 2 | low (0.69) | `mindspore:mindspore.ops.function.vmap; torch:torch.vmap` |
| 331 | `["binary_cross_entropy", "nn", "component", 1]` | 3 | 3 | medium (0.844) | `mindspore:mindspore.mint.nn.functional.binary_cross_entropy; paddle:paddle.nn.functional.binary_cross_entropy; torch:torch.nn.functional.binary_cross_entropy` |
| 349 | `["linear", "nn", "component", 1]` | 4 | 4 | medium (0.7529) | `chainer:chainer.functions.linear; keras:keras.activations.linear; paddle:paddle.nn.functional.linear; tensorflow:tensorflow.keras.activations.linear` |
| 386 | `["bessel_y0", "generic", "component", 1]` | 4 | 2 | medium (0.776) | `mindspore:mindspore.ops.bessel_y0; torch:torch.special.bessel_y0` |
| 387 | `["bessel_y1", "generic", "component", 1]` | 4 | 2 | medium (0.7771) | `mindspore:mindspore.ops.bessel_y1; torch:torch.special.bessel_y1` |
| 401 | `["fft", "fft", "component", 1]` | 10 | 8 | medium (0.8107) | `chainer:chainer.functions.fft; jax:jax.numpy.fft.fft; keras:keras.ops.fft; mindspore:mindspore.ops.fft; numpy:numpy.fft.fft; paddle:paddle.fft.fft; scipy:scipy.fft.fft; torch:torch.fft.fft` |
| 408 | `["ifft", "fft", "component", 1]` | 9 | 7 | medium (0.8359) | `chainer:chainer.functions.ifft; jax:jax.numpy.fft.ifft; mindspore:mindspore.ops.function.ifft; numpy:numpy.fft.ifft; paddle:paddle.fft.ifft; scipy:scipy.fft.ifft; torch:torch.fft.ifft` |
| 434 | `["sequence_mask", "generic", "component", 1]` | 2 | 2 | low (0.6357) | `mxnet:mxnet.ndarray.SequenceMask; tensorflow:tensorflow.sequence_mask` |
| 468 | `["hardsigmoid", "generic", "component", 1]` | 8 | 5 | high (0.9217) | `chainer:chainer.functions.hard_sigmoid; keras:keras.ops.hard_sigmoid; mindspore:mindspore.ops.function.hardsigmoid; mxnet:mxnet.ndarray.hard_sigmoid; tensorflow:tensorflow.keras.activations.hard_sigmoid` |
| 481 | `["l1_loss", "loss", "component", 1]` | 10 | 4 | medium (0.7388) | `chainer:chainer.functions.mean_absolute_error; keras:keras.losses.mean_absolute_error; mindspore:mindspore.ops.function.l1_loss; tensorflow:tensorflow.keras.losses.MAE` |
| 482 | `["mse_loss", "loss", "component", 1]` | 10 | 4 | medium (0.7388) | `chainer:chainer.functions.mean_squared_error; keras:keras.losses.mean_squared_error; mindspore:mindspore.ops.function.mse_loss; tensorflow:tensorflow.keras.losses.MSE` |
| 483 | `["binary_cross_entropy", "loss", "component", 1]` | 8 | 3 | medium (0.7391) | `keras:keras.ops.binary_crossentropy; mindspore:mindspore.ops.binary_cross_entropy; tensorflow:tensorflow.keras.losses.binary_crossentropy` |
| 540 | `["nanprod", "generic", "component", 1]` | 5 | 4 | high (0.8777) | `jax:jax.numpy.nanprod; keras:keras.ops.nanprod; mxnet:mxnet.ndarray.nanprod; numpy:numpy.nanprod` |
| 574 | `["indices", "generic", "component", 1]` | 4 | 4 | high (0.9407) | `jax:jax.numpy.indices; mindspore:mindspore.numpy.indices; mxnet:mxnet.numpy.indices; numpy:numpy.indices` |
| 588 | `["partition", "generic", "component", 1]` | 3 | 3 | high (0.8843) | `jax:jax.numpy.partition; mxnet:mxnet.numpy.partition; numpy:numpy.partition` |
| 595 | `["polydiv", "generic", "component", 1]` | 3 | 3 | high (0.9076) | `jax:jax.numpy.polydiv; mxnet:mxnet.numpy.polydiv; numpy:numpy.polydiv` |
| 605 | `["roots", "generic", "component", 1]` | 3 | 3 | high (0.8561) | `jax:jax.numpy.roots; mxnet:mxnet.numpy.roots; numpy:numpy.roots` |
| 633 | `["cho_factor", "linalg", "component", 1]` | 3 | 3 | high (1.0) | `jax:jax.scipy.linalg.cho_factor; mindspore:mindspore.scipy.linalg.cho_factor; scipy:scipy.linalg.cho_factor` |
