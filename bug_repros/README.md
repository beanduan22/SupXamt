# Verified XAMT Reproductions

> Note: these reproductions were audited for the previous generated snapshot.
> Regenerate this package before reporting DL-only Xamt++ counts.

Strict rule: only current live `status: DIFF` outputs are counted as real reproducible bugs. No stored fallback is counted.

- Candidate scripts audited: 188
- Verified live DIFF bugs: 177
- Not counted in current environment: 11

## Not Counted

| Script | Key | Current status | Reason |
| --- | --- | --- | --- |
| `bug_002_diagflat_generic_2.py` | `diagflat/generic/2` | `PASS` | current runner no longer produces a differential output |
| `bug_099_rsf2csf_linalg.py` | `rsf2csf/linalg` | `PASS` | current runner no longer produces a differential output |
| `bug_137_fft_fft.py` | `fft/fft` | `PASS` | current runner no longer produces a differential output |
| `bug_138_ifft_fft.py` | `ifft/fft` | `PASS` | current runner no longer produces a differential output |
| `bug_139_ifft2_fft.py` | `ifft2/fft` | `PASS` | current runner no longer produces a differential output |
| `bug_140_ifftn_fft.py` | `ifftn/fft` | `PASS` | current runner no longer produces a differential output |
| `bug_141_rfft_fft.py` | `rfft/fft` | `PASS` | current runner no longer produces a differential output |
| `bug_150_fft2_fft.py` | `fft2/fft` | `PASS` | current runner no longer produces a differential output |
| `bug_151_fftn_fft.py` | `fftn/fft` | `PASS` | current runner no longer produces a differential output |
| `bug_155_nanprod_generic.py` | `nanprod/generic` | `PASS` | current runner no longer produces a differential output |
| `bug_166_arctan2_generic_2.py` | `arctan2/generic/2` | `PASS` | current runner no longer produces a differential output |

## Verified Outputs

## bug_001 `clip/generic/3`

- script: `bug_001_clip_generic_3.py`
- count_status: `verified-live`

```text
bug_id: bug_001
key: clip/generic/3
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, jax, keras
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, 2.0, -1.0], [0.3003249168395996, -1.0, -0.0947190374135971]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[-1.0, 2.0, -1.0], [0.3003249168395996, -1.0, -0.0947190374135971]]}
```

## bug_002 `diagflat/generic/2`

- script: `bug_002_diagflat_generic_2.py`
- count_status: `not-counted`

```text
bug_id: bug_002
key: diagflat/generic/2
status: PASS
output_source: not_live
expected_source: majority
expected_libs: jax, keras, mindspore, mxnet, numpy
expected: {"dtype": "float32", "shape": [6, 6], "value": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, NaN, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -0.0]]}
wrong:
  none
```

## bug_003 `maximum/generic/2`

- script: `bug_003_maximum_generic_2.py`
- count_status: `verified-live`

```text
bug_id: bug_003
key: maximum/generic/2
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, keras, mxnet
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, Infinity, 1.2976676225662231], [-1.0, 1.0, 0.5523166656494141]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[1.2180202007293701, Infinity, 1.2976676225662231], [-1.0, 1.0, 0.5523166656494141]]}
```

## bug_004 `minimum/generic/2`

- script: `bug_004_minimum_generic_2.py`
- count_status: `verified-live`

```text
bug_id: bug_004
key: minimum/generic/2
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, keras, mxnet
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, 0.0, -Infinity], [-1.1046050786972046, -0.6005114316940308, -0.3333442807197571]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[-0.0, 0.0, -Infinity], [-1.1046050786972046, -0.6005114316940308, -0.3333442807197571]]}
```

## bug_005 `nancumsum/generic/3`

- script: `bug_005_nancumsum_generic_3.py`
- count_status: `verified-live`

```text
bug_id: bug_005
key: nancumsum/generic/3
status: DIFF
output_source: live
expected_source: reference:keras
expected_libs: keras
expected: {"dtype": "float32", "shape": [6], "value": [0.0, 3.4028234663852886e+38, 0.0, -4.2111430168151855, -17.07004737854004, -12.379446029663086]}
wrong:
  mindspore: {"dtype": "float32", "shape": [6], "value": [0.0, Infinity, NaN, NaN, NaN, NaN]}
```

## bug_006 `signbit/generic/1`

- script: `bug_006_signbit_generic_1.py`
- count_status: `verified-live`

```text
bug_id: bug_006
key: signbit/generic/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, keras
expected: {"dtype": "bool", "shape": [2, 3], "value": [[true, false, false], [true, false, false]]}
wrong:
  mindspore: {"dtype": "bool", "shape": [2, 3], "value": [[false, false, false], [true, false, false]]}
```

## bug_007 `tril/generic/2`

- script: `bug_007_tril_generic_2.py`
- count_status: `verified-live`

```text
bug_id: bug_007
key: tril/generic/2
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, keras, mxnet, numpy
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, 0.0, 0.0], [-1.0, 1.0, 0.0]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, NaN], [-1.0, 1.0, 0.0]]}
```

## bug_008 `hardshrink/generic/component/1`

- script: `bug_008_hardshrink_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_008
key: hardshrink/generic/component/1
status: DIFF
output_source: live
expected_source: reference:torch
expected_libs: torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, Infinity, -Infinity], [0.5860238075256348, 0.0, 0.0]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, Infinity, -Infinity], [0.5860238075256348, 0.0, 0.0]]}
```

## bug_009 `softmin/generic/component/1`

- script: `bug_009_softmin_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_009
key: softmin/generic/component/1
status: DIFF
output_source: live
expected_source: reference:mxnet
expected_libs: mxnet
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, NaN], [2.3552766492684896e-07, 0.9996299743652344, 0.00036978989373892546]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, 0.0, NaN], [2.3552674122129247e-07, 0.9996299743652344, 0.00036978887510485947]]}
```

## bug_010 `softshrink/nn/component/1`

- script: `bug_010_softshrink_nn_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_010
key: softshrink/nn/component/1
status: DIFF
output_source: live
expected_source: reference:torch
expected_libs: torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, Infinity, -Infinity], [-1.9228339195251465, 1.0421700477600098, 0.6015607118606567]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, Infinity, -Infinity], [-1.9228339195251465, 1.0421700477600098, 0.6015607118606567]]}
  paddle: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, NaN], [-1.9228339195251465, 1.0421700477600098, 0.6015607118606567]]}
```

## bug_011 `angle/generic/2`

- script: `bug_011_angle_generic_2.py`
- count_status: `verified-live`

```text
bug_id: bug_011
key: angle/generic/2
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, numpy, paddle
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, 0.0, 3.1415927410125732], [3.1415927410125732, 0.0, 0.0]]}
wrong:
  tensorflow: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, 0.0, 3.1415927410125732], [3.1415927410125732, 0.0, 0.0]]}
```

## bug_012 `count_nonzero/reduction/3`

- script: `bug_012_count_nonzero_reduction_3.py`
- count_status: `verified-live`

```text
bug_id: bug_012
key: count_nonzero/reduction/3
status: DIFF
output_source: live
expected_source: majority
expected_libs: mindspore, mxnet, numpy
expected: {"dtype": "int32", "shape": [], "value": 4}
wrong:
  jax: {"dtype": "int32", "shape": [], "value": 3}
```

## bug_013 `nancumprod/generic/component/1`

- script: `bug_013_nancumprod_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_013
key: nancumprod/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, mxnet, numpy
expected: {"dtype": "float32", "shape": [6], "value": [1.0, Infinity, -Infinity, Infinity, Infinity, -Infinity]}
wrong:
  keras: {"dtype": "float32", "shape": [6], "value": [1.0, 3.4028234663852886e+38, -Infinity, Infinity, Infinity, -Infinity]}
```

## bug_014 `nanstd/generic/component/1`

- script: `bug_014_nanstd_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_014
key: nanstd/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, keras, numpy
expected: {"dtype": "float32", "shape": [], "value": NaN}
wrong:
  mindspore: {"dtype": "float32", "shape": [], "value": 0.0}
```

## bug_015 `nanvar/generic/component/1`

- script: `bug_015_nanvar_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_015
key: nanvar/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, keras, numpy
expected: {"dtype": "float32", "shape": [], "value": NaN}
wrong:
  mindspore: {"dtype": "float32", "shape": [], "value": 0.0}
```

## bug_016 `unique_consecutive/generic/component/1`

- script: `bug_016_unique_consecutive_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_016
key: unique_consecutive/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: mindspore, paddle
expected: {"dtype": "float32", "shape": [6], "value": [NaN, NaN, Infinity, -Infinity, 0.0, 1.0]}
wrong:
  torch: {"dtype": "float32", "shape": [5], "value": [NaN, Infinity, -Infinity, 0.0, 1.0]}
```

## bug_017 `ihfft/fft/component/1`

- script: `bug_017_ihfft_fft_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_017
key: ihfft/fft/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: numpy, paddle, scipy, torch
expected: {"dtype": "complex64", "shape": [3], "value": [[NaN, -0.0], [NaN, Infinity], [NaN, -0.0]]}
wrong:
  jax: {"dtype": "complex64", "shape": [3], "value": [[NaN, NaN], [NaN, NaN], [NaN, NaN]]}
```

## bug_018 `sinc/generic/component/1`

- script: `bug_018_sinc_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_018
key: sinc/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, keras, mindspore, numpy, scipy, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, NaN], [0.17645473778247833, 0.9176711440086365, -0.03343509882688522]]}
wrong:
  paddle: {"dtype": "float32", "shape": [2, 3], "value": [[1.0, 1.0, 1.0], [0.17645473778247833, 0.9176711440086365, -0.03343509882688522]]}
```

## bug_019 `histc/generic/component/1`

- script: `bug_019_histc_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_019
key: histc/generic/component/1
status: DIFF
output_source: live
expected_source: reference:torch
expected_libs: torch
expected: {"dtype": "float32", "shape": [8], "value": [0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0]}
wrong:
  mindspore: {"dtype": "float32", "shape": [8], "value": [1.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0]}
```

## bug_020 `logcumsumexp/generic/component/1`

- script: `bug_020_logcumsumexp_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_020
key: logcumsumexp/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: mindspore, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, NaN], [-0.3630312979221344, 2.7054264545440674, 2.705869436264038]]}
wrong:
  paddle: {"dtype": "float32", "shape": [2, 3], "value": [[-3.4028234663852886e+38, Infinity, Infinity], [-0.3630312979221344, 2.7054264545440674, 2.705869436264038]]}
```

## bug_021 `mode/generic/component/1`

- script: `bug_021_mode_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_021
key: mode/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, paddle
expected: {"dtype": "float32", "shape": [2], "value": [-Infinity, 0.033594969660043716]}
wrong:
  torch: {"dtype": "float32", "shape": [2], "value": [NaN, 0.033594969660043716]}
```

## bug_022 `argmax/reduction/2`

- script: `bug_022_argmax_reduction_2.py`
- count_status: `verified-live`

```text
bug_id: bug_022
key: argmax/reduction/2
status: DIFF
output_source: live
expected_source: reference:chainer
expected_libs: chainer
expected: {"dtype": "int32", "shape": [2], "value": [2, 0]}
wrong:
  mindspore: {"dtype": "int64", "shape": [2], "value": [0, 0]}
```

## bug_023 `argmax/reduction/4`

- script: `bug_023_argmax_reduction_4.py`
- count_status: `verified-live`

```text
bug_id: bug_023
key: argmax/reduction/4
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, numpy
expected: {"dtype": "int32", "shape": [2], "value": [0, 0]}
wrong:
  tensorflow: {"dtype": "int64", "shape": [2], "value": [1, 0]}
```

## bug_024 `argmax/reduction/5`

- script: `bug_024_argmax_reduction_5.py`
- count_status: `verified-live`

```text
bug_id: bug_024
key: argmax/reduction/5
status: DIFF
output_source: live
expected_source: reference:paddle
expected_libs: paddle
expected: {"dtype": "int64", "shape": [2], "value": [1, 0]}
wrong:
  mxnet: {"dtype": "float32", "shape": [2], "value": [0.0, 0.0]}
```

## bug_025 `argmax/reduction/component/1`

- script: `bug_025_argmax_reduction_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_025
key: argmax/reduction/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, mindspore, mxnet, numpy, paddle, torch
expected: {"dtype": "int32", "shape": [2], "value": [2, 1]}
wrong:
  jax: {"dtype": "int32", "shape": [2], "value": [0, 1]}
  keras: {"dtype": "int32", "shape": [2], "value": [1, 1]}
  tensorflow: {"dtype": "int64", "shape": [2], "value": [0, 1]}
```

## bug_026 `argmin/reduction/3`

- script: `bug_026_argmin_reduction_3.py`
- count_status: `verified-live`

```text
bug_id: bug_026
key: argmin/reduction/3
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, mindspore, mxnet
expected: {"dtype": "int32", "shape": [2], "value": [0, 0]}
wrong:
  keras: {"dtype": "int32", "shape": [2], "value": [2, 0]}
```

## bug_027 `argmin/reduction/4`

- script: `bug_027_argmin_reduction_4.py`
- count_status: `verified-live`

```text
bug_id: bug_027
key: argmin/reduction/4
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, numpy
expected: {"dtype": "int32", "shape": [2], "value": [0, 2]}
wrong:
  tensorflow: {"dtype": "int64", "shape": [2], "value": [2, 2]}
```

## bug_028 `argmin/reduction/5`

- script: `bug_028_argmin_reduction_5.py`
- count_status: `verified-live`

```text
bug_id: bug_028
key: argmin/reduction/5
status: DIFF
output_source: live
expected_source: reference:paddle
expected_libs: paddle
expected: {"dtype": "int64", "shape": [2], "value": [2, 2]}
wrong:
  mxnet: {"dtype": "float32", "shape": [2], "value": [0.0, 2.0]}
```

## bug_029 `argmin/reduction/component/1`

- script: `bug_029_argmin_reduction_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_029
key: argmin/reduction/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, jax, mindspore, mxnet, numpy, torch
expected: {"dtype": "int32", "shape": [2], "value": [0, 0]}
wrong:
  keras: {"dtype": "int32", "shape": [2], "value": [2, 0]}
  paddle: {"dtype": "int64", "shape": [2], "value": [2, 0]}
  tensorflow: {"dtype": "int64", "shape": [2], "value": [2, 0]}
```

## bug_030 `argsort/generic/5`

- script: `bug_030_argsort_generic_5.py`
- count_status: `verified-live`

```text
bug_id: bug_030
key: argsort/generic/5
status: DIFF
output_source: live
expected_source: majority
expected_libs: numpy, paddle
expected: {"dtype": "int64", "shape": [2, 3], "value": [[2, 1, 0], [0, 1, 2]]}
wrong:
  tensorflow: {"dtype": "int32", "shape": [2, 3], "value": [[0, 2, 1], [0, 1, 2]]}
```

## bug_031 `argsort/generic/component/1`

- script: `bug_031_argsort_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_031
key: argsort/generic/component/1
status: DIFF
output_source: live
expected_source: reference:numpy
expected_libs: numpy
expected: {"dtype": "int64", "shape": [2, 3], "value": [[2, 1, 0], [2, 1, 0]]}
wrong:
  keras: {"dtype": "int32", "shape": [2, 3], "value": [[0, 2, 1], [2, 1, 0]]}
  mindspore: {"dtype": "int32", "shape": [2, 3], "value": [[0, 2, 1], [2, 1, 0]]}
  mxnet: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, 2.0, 1.0], [2.0, 1.0, 0.0]]}
  tensorflow: {"dtype": "int32", "shape": [2, 3], "value": [[0, 2, 1], [2, 1, 0]]}
```

## bug_032 `argwhere/generic/component/1`

- script: `bug_032_argwhere_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_032
key: argwhere/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: mxnet, numpy, paddle, torch
expected: {"dtype": "int64", "shape": [4, 2], "value": [[0, 2], [1, 0], [1, 1], [1, 2]]}
wrong:
  jax: {"dtype": "int32", "shape": [3, 2], "value": [[1, 0], [1, 1], [1, 2]]}
  mindspore: {"dtype": "int64", "shape": [3, 2], "value": [[1, 0], [1, 1], [1, 2]]}
```

## bug_033 `flatnonzero/generic/component/1`

- script: `bug_033_flatnonzero_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_033
key: flatnonzero/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: mxnet, numpy
expected: {"dtype": "int64", "shape": [4], "value": [2, 3, 4, 5]}
wrong:
  jax: {"dtype": "int32", "shape": [3], "value": [3, 4, 5]}
```

## bug_034 `msort/generic/1`

- script: `bug_034_msort_generic_1.py`
- count_status: `verified-live`

```text
bug_id: bug_034
key: msort/generic/1
status: DIFF
output_source: live
expected_source: reference:mxnet
expected_libs: mxnet
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-0.6830241084098816, -0.30688419938087463, -Infinity], [NaN, Infinity, 3.4525134563446045]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, -0.30688419938087463, -Infinity], [-0.6830241084098816, Infinity, 3.4525134563446045]]}
```

## bug_035 `msort/generic/component/1`

- script: `bug_035_msort_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_035
key: msort/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: mxnet, paddle, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-1.0333818197250366, 0.2164539247751236, -Infinity], [NaN, Infinity, 0.7698968052864075]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, 0.2164539247751236, -Infinity], [-1.0333818197250366, Infinity, 0.7698968052864075]]}
```

## bug_036 `nonzero/generic/component/1`

- script: `bug_036_nonzero_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_036
key: nonzero/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: mxnet, numpy, paddle, torch
expected: {"dtype": "int64", "shape": [4, 2], "value": [[0, 2], [1, 0], [1, 1], [1, 2]]}
wrong:
  jax: {"dtype": "int32", "shape": [3, 2], "value": [[1, 0], [1, 1], [1, 2]]}
  keras: {"dtype": "int32", "shape": [3, 2], "value": [[1, 0], [1, 1], [1, 2]]}
  mindspore: {"dtype": "int64", "shape": [3, 2], "value": [[1, 0], [1, 1], [1, 2]]}
```

## bug_037 `sort/generic/4`

- script: `bug_037_sort_generic_4.py`
- count_status: `verified-live`

```text
bug_id: bug_037
key: sort/generic/4
status: DIFF
output_source: live
expected_source: majority
expected_libs: mxnet, tensorflow
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, -Infinity, Infinity], [-7.590363502502441, 0.4991883635520935, 5.5815324783325195]]}
wrong:
  jax: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, Infinity, NaN], [-7.590363502502441, 0.4991883635520935, 5.5815324783325195]]}
errors: {"mindspore": "SKIP: mindspore.mint.sort CPU kernel is unavailable in this runner"}
```

## bug_038 `sort/generic/5`

- script: `bug_038_sort_generic_5.py`
- count_status: `verified-live`

```text
bug_id: bug_038
key: sort/generic/5
status: DIFF
output_source: live
expected_source: majority
expected_libs: numpy, paddle
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, Infinity, NaN], [-1.0, 9.999999974752427e-07, 1.0]]}
wrong:
  mxnet: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, -Infinity, Infinity], [-1.0, 9.999999974752427e-07, 1.0]]}
```

## bug_039 `sort/generic/component/1`

- script: `bug_039_sort_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_039
key: sort/generic/component/1
status: DIFF
output_source: live
expected_source: reference:numpy
expected_libs: numpy
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, Infinity, NaN], [-0.27603381872177124, -0.07597750425338745, 0.09934917837381363]]}
wrong:
  keras: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, -Infinity, Infinity], [-0.27603381872177124, -0.07597750425338745, 0.09934917837381363]]}
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, -Infinity, Infinity], [-0.27603381872177124, -0.07597750425338745, 0.09934917837381363]]}
  mxnet: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, -Infinity, Infinity], [-0.27603381872177124, -0.07597750425338745, 0.09934917837381363]]}
  tensorflow: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, -Infinity, Infinity], [-0.27603381872177124, -0.07597750425338745, 0.09934917837381363]]}
```

## bug_040 `gelu/nn/component/1`

- script: `bug_040_gelu_nn_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_040
key: gelu/nn/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, keras, mindspore, paddle, tensorflow
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, Infinity, NaN], [-0.01283420529216528, 0.06086074933409691, -0.02035370096564293]]}
wrong:
  torch: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, NaN], [-0.01283420529216528, 0.06086074933409691, -0.02035370096564293]]}
```

## bug_041 `hardswish/generic/component/1`

- script: `bug_041_hardswish_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_041
key: hardswish/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: keras, tensorflow
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, Infinity, NaN], [-0.3347257375717163, 0.903161883354187, -0.3685990869998932]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, Infinity, 0.0], [-0.3347257375717163, 0.9031618237495422, -0.3685990869998932]]}
```

## bug_042 `hardswish/nn/component/1`

- script: `bug_042_hardswish_nn_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_042
key: hardswish/nn/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, paddle, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, Infinity, NaN], [-0.2664388120174408, -0.31546351313591003, 0.16632360219955444]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, Infinity, 0.0], [-0.2664388120174408, -0.31546348333358765, 0.16632358729839325]]}
```

## bug_043 `log_softmax/nn/component/1`

- script: `bug_043_log_softmax_nn_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_043
key: log_softmax/nn/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, jax, keras, paddle, scipy, tensorflow, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, NaN], [-1.1771713495254517, -0.9787102341651917, -1.1518378257751465]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, -Infinity], [-1.1771693229675293, -0.9787082076072693, -1.1518357992172241]]}
  mxnet: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, -Infinity], [-1.1771693229675293, -0.9787082076072693, -1.1518357992172241]]}
```

## bug_044 `logsigmoid/nn/component/1`

- script: `bug_044_logsigmoid_nn_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_044
key: logsigmoid/nn/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, -0.0, -Infinity], [-0.35654228925704956, -0.5308101773262024, -0.7507658004760742]]}
wrong:
  paddle: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, -0.0, NaN], [-0.35654228925704956, -0.5308101773262024, -0.7507658004760742]]}
```

## bug_045 `mish/generic/component/1`

- script: `bug_045_mish_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_045
key: mish/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: keras, tensorflow
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, Infinity, NaN], [-0.28082358837127686, 0.5038935542106628, -0.2434835582971573]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, NaN], [-0.28082358837127686, 0.5038936138153076, -0.2434834986925125]]}
```

## bug_046 `relu/nn/component/1`

- script: `bug_046_relu_nn_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_046
key: relu/nn/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, keras, mxnet, tensorflow, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, Infinity, 0.0], [1.2985645532608032, 0.0, 0.0]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, Infinity, 0.0], [1.2985645532608032, 0.0, 0.0]]}
  paddle: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, Infinity, 0.0], [1.2985645532608032, 0.0, 0.0]]}
```

## bug_047 `relu6/generic/component/1`

- script: `bug_047_relu6_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_047
key: relu6/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, keras, tensorflow
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, 6.0, 0.0], [0.36694541573524475, 0.31849777698516846, 0.21835516393184662]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, 6.0, 0.0], [0.36694541573524475, 0.31849777698516846, 0.21835516393184662]]}
```

## bug_048 `softmax/nn/component/1`

- script: `bug_048_softmax_nn_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_048
key: softmax/nn/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, jax, paddle, scipy, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, NaN], [0.6757315993309021, 0.1351672261953354, 0.1891012042760849]]}
wrong:
  keras: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, 0.0], [0.6757315993309021, 0.1351672112941742, 0.1891011744737625]]}
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, 0.0], [0.6757317185401917, 0.13516715168952942, 0.18910114467144012]]}
  mxnet: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, 0.0], [0.6757315993309021, 0.1351672112941742, 0.1891011744737625]]}
  tensorflow: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, 0.0], [0.6757315993309021, 0.1351672112941742, 0.1891011744737625]]}
```

## bug_049 `softplus/nn/component/1`

- script: `bug_049_softplus_nn_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_049
key: softplus/nn/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, jax, keras, paddle, scipy, tensorflow, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, Infinity, 0.0], [0.6998663544654846, 0.596236526966095, 0.8535803556442261]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[Infinity, Infinity, 0.0], [0.6998661160469055, 0.596236526966095, 0.8535799384117126]]}
```

## bug_050 `threshold/generic/component/1`

- script: `bug_050_threshold_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_050
key: threshold/generic/component/1
status: DIFF
output_source: live
expected_source: reference:tensorflow
expected_libs: tensorflow
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-1.0, -1.0, -1.0], [-1.0, 1.0, 9.999999974752427e-07]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[-1.0, -1.0, 1.401298464324817e-45], [-1.0, 1.0, 9.999999974752427e-07]]}
  torch: {"dtype": "float32", "shape": [2, 3], "value": [[-1.0, -1.0, 1.401298464324817e-45], [-1.0, 1.0, 9.999999974752427e-07]]}
```

## bug_051 `argwhere/generic/1`

- script: `bug_051_argwhere_generic_1.py`
- count_status: `verified-live`

```text
bug_id: bug_051
key: argwhere/generic/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: mxnet, numpy, paddle
expected: {"dtype": "int64", "shape": [4, 2], "value": [[0, 2], [1, 0], [1, 1], [1, 2]]}
wrong:
  mindspore: {"dtype": "int64", "shape": [3, 2], "value": [[1, 0], [1, 1], [1, 2]]}
```

## bug_052 `ceil/generic/1`

- script: `bug_052_ceil_generic_1.py`
- count_status: `verified-live`

```text
bug_id: bug_052
key: ceil/generic/1
status: DIFF
output_source: live
expected_source: reference:jax
expected_libs: jax
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-0.0, 0.0, 0.0], [-1.0, 1.0, 1.0]]}
wrong:
  chainer: {"dtype": "float32", "shape": [2, 3], "value": [[-0.0, 0.0, 1.0], [-1.0, 1.0, 1.0]]}
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[-0.0, 0.0, 1.0], [-1.0, 1.0, 1.0]]}
```

## bug_053 `ceil/generic/2`

- script: `bug_053_ceil_generic_2.py`
- count_status: `verified-live`

```text
bug_id: bug_053
key: ceil/generic/2
status: DIFF
output_source: live
expected_source: majority
expected_libs: mindspore, mxnet
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-0.0, 0.0, 1.0], [-1.0, 1.0, 1.0]]}
wrong:
  tensorflow: {"dtype": "float32", "shape": [2, 3], "value": [[-0.0, 0.0, 0.0], [-1.0, 1.0, 1.0]]}
errors: {"paddle": "SKIP: paddle.tensor aliases include low-level wrappers; prefer stable paddle top-level APIs"}
```

## bug_054 `ceil/generic/component/1`

- script: `bug_054_ceil_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_054
key: ceil/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, mindspore, mxnet, paddle, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-0.0, 0.0, 1.0], [-1.0, 1.0, 1.0]]}
wrong:
  jax: {"dtype": "float32", "shape": [2, 3], "value": [[-0.0, 0.0, 0.0], [-1.0, 1.0, 1.0]]}
  keras: {"dtype": "float32", "shape": [2, 3], "value": [[-0.0, 0.0, 0.0], [-1.0, 1.0, 1.0]]}
  tensorflow: {"dtype": "float32", "shape": [2, 3], "value": [[-0.0, 0.0, 0.0], [-1.0, 1.0, 1.0]]}
```

## bug_055 `count_nonzero/reduction/component/1`

- script: `bug_055_count_nonzero_reduction_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_055
key: count_nonzero/reduction/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: mindspore, mxnet, numpy, paddle, torch
expected: {"dtype": "int32", "shape": [], "value": 4}
wrong:
  jax: {"dtype": "int32", "shape": [], "value": 3}
  keras: {"dtype": "int32", "shape": [], "value": 3}
  tensorflow: {"dtype": "int64", "shape": [], "value": 3}
```

## bug_056 `diagflat/generic/component/1`

- script: `bug_056_diagflat_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_056
key: diagflat/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, keras, mxnet, numpy, paddle, torch
expected: {"dtype": "float32", "shape": [6, 6], "value": [[NaN, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, Infinity, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -Infinity, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.258932113647461, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.5087960362434387, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -0.03927762433886528]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [6, 6], "value": [[NaN, NaN, NaN, 0.0, 0.0, -0.0], [NaN, Infinity, NaN, 0.0, 0.0, -0.0], [NaN, NaN, -Infinity, 0.0, 0.0, -0.0], [NaN, NaN, NaN, 1.258932113647461, 0.0, -0.0], [NaN, NaN, NaN, 0.0, 0.5087960362434387, -0.0], [NaN, NaN, NaN, 0.0, 0.0, -0.03927762433886528]]}
```

## bug_057 `frexp/generic/component/1`

- script: `bug_057_frexp_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_057
key: frexp/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: paddle, torch
expected: [{"dtype": "float32", "shape": [2, 3], "value": [[0.0, 0.0, 0.5], [-0.5, 0.5, 0.5242879986763]]}, {"dtype": "float32", "shape": [2, 3], "value": [[0.0, 0.0, -148.0], [1.0, 1.0, -19.0]]}]
wrong:
  jax: [{"dtype": "float32", "shape": [2, 3], "value": [[-0.0, 0.0, 0.5], [-0.5, 0.5, 0.5242879986763]]}, {"dtype": "int32", "shape": [2, 3], "value": [[0, 0, -149], [1, 1, -19]]}]
```

## bug_058 `hardswish/generic/1`

- script: `bug_058_hardswish_generic_1.py`
- count_status: `verified-live`

```text
bug_id: bug_058
key: hardswish/generic/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: keras, tensorflow
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, Infinity, NaN], [-0.06668020784854889, 0.45176005363464355, 0.29167962074279785]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, Infinity, 0.0], [-0.06668020784854889, 0.45176002383232117, 0.29167965054512024]]}
```

## bug_059 `hardswish/nn/1`

- script: `bug_059_hardswish_nn_1.py`
- count_status: `verified-live`

```text
bug_id: bug_059
key: hardswish/nn/1
status: DIFF
output_source: live
expected_source: reference:jax
expected_libs: jax
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, Infinity, NaN], [-0.2272556722164154, 0.16052395105361938, -0.06130940839648247]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, Infinity, 0.0], [-0.2272556573152542, 0.1605239361524582, -0.06130940839648247]]}
```

## bug_060 `heaviside/generic/3`

- script: `bug_060_heaviside_generic_3.py`
- count_status: `verified-live`

```text
bug_id: bug_060
key: heaviside/generic/3
status: DIFF
output_source: live
expected_source: reference:paddle
expected_libs: paddle
expected: {"dtype": "float32", "shape": [3], "value": [0.0, 1.0, 0.5]}
wrong:
  mindspore: {"dtype": "float32", "shape": [3], "value": [0.5, 1.0, 0.5]}
```

## bug_061 `heaviside/generic/component/1`

- script: `bug_061_heaviside_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_061
key: heaviside/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: mindspore, paddle, torch
expected: {"dtype": "float32", "shape": [3], "value": [0.5, 0.5, 1.0]}
wrong:
  jax: {"dtype": "float32", "shape": [3], "value": [0.5, 0.5, 0.5]}
  keras: {"dtype": "float32", "shape": [3], "value": [0.5, 0.5, 0.5]}
```

## bug_062 `isclose/generic/component/1`

- script: `bug_062_isclose_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_062
key: isclose/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, mindspore, mxnet, numpy, paddle, torch
expected: {"dtype": "bool", "shape": [2, 3], "value": [[false, true, true], [true, false, true]]}
wrong:
  keras: {"dtype": "bool", "shape": [2, 3], "value": [[false, false, false], [true, false, true]]}
```

## bug_063 `log_softmax/nn/2`

- script: `bug_063_log_softmax_nn_2.py`
- count_status: `verified-live`

```text
bug_id: bug_063
key: log_softmax/nn/2
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, jax, keras, scipy, tensorflow
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, NaN], [-0.892629861831665, -0.9274938702583313, -1.6353760957717896]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, -Infinity], [-0.8926306962966919, -0.9274947047233582, -1.6353769302368164]]}
```

## bug_064 `mish/generic/1`

- script: `bug_064_mish_generic_1.py`
- count_status: `verified-live`

```text
bug_id: bug_064
key: mish/generic/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: keras, tensorflow
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, Infinity, NaN], [-0.30340149998664856, 0.8650984168052673, 6.000003622830263e-07]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, NaN], [-0.30340102314949036, 0.865098237991333, 6.000003054396075e-07]]}
```

## bug_065 `multiply/generic/2`

- script: `bug_065_multiply_generic_2.py`
- count_status: `verified-live`

```text
bug_id: bug_065
key: multiply/generic/2
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, keras, mindspore
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, NaN], [-19.644245147705078, -0.033647242933511734, 6.5154495132446755e-06]]}
wrong:
  mxnet: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, -Infinity], [-19.644245147705078, -0.033647242933511734, 6.5154495132446755e-06]]}
```

## bug_066 `multiply/generic/3`

- script: `bug_066_multiply_generic_3.py`
- count_status: `verified-live`

```text
bug_id: bug_066
key: multiply/generic/3
status: DIFF
output_source: live
expected_source: majority
expected_libs: mindspore, tensorflow
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, NaN], [0.18958938121795654, 0.2641873359680176, -1.0282534645966734e-07]]}
wrong:
  mxnet: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, -Infinity], [0.18958938121795654, 0.2641873359680176, -1.0282534645966734e-07]]}
errors: {"paddle": "SKIP: in-place tensor variants mutate inputs"}
```

## bug_067 `multiply/generic/component/1`

- script: `bug_067_multiply_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_067
key: multiply/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, keras, mindspore, tensorflow
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, NaN], [0.35449573397636414, 0.04463260620832443, 1.1511584574463996e-07]]}
wrong:
  mxnet: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, -Infinity], [0.35449573397636414, 0.04463260620832443, 1.1511584574463996e-07]]}
  paddle: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, -Infinity], [0.35449573397636414, 0.04463260620832443, 1.1511584574463996e-07]]}
  torch: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, -Infinity], [0.35449573397636414, 0.04463260620832443, 1.1511584574463996e-07]]}
```

## bug_068 `nancumsum/generic/component/1`

- script: `bug_068_nancumsum_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_068
key: nancumsum/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, mindspore, mxnet, numpy
expected: {"dtype": "float32", "shape": [6], "value": [0.0, Infinity, NaN, NaN, NaN, NaN]}
wrong:
  keras: {"dtype": "float32", "shape": [6], "value": [0.0, 3.4028234663852886e+38, 0.0, -1.0, 0.0, 9.999999974752427e-07]}
```

## bug_069 `nonzero/generic/1`

- script: `bug_069_nonzero_generic_1.py`
- count_status: `verified-live`

```text
bug_id: bug_069
key: nonzero/generic/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: mxnet, numpy
expected: {"dtype": "int64", "shape": [4, 2], "value": [[0, 2], [1, 0], [1, 1], [1, 2]]}
wrong:
  keras: {"dtype": "int32", "shape": [3, 2], "value": [[1, 0], [1, 1], [1, 2]]}
```

## bug_070 `nonzero/generic/3`

- script: `bug_070_nonzero_generic_3.py`
- count_status: `verified-live`

```text
bug_id: bug_070
key: nonzero/generic/3
status: DIFF
output_source: live
expected_source: reference:jax
expected_libs: jax
expected: {"dtype": "int32", "shape": [3, 2], "value": [[1, 0], [1, 1], [1, 2]]}
wrong:
  paddle: {"dtype": "int64", "shape": [4, 2], "value": [[0, 2], [1, 0], [1, 1], [1, 2]]}
```

## bug_071 `relu/nn/2`

- script: `bug_071_relu_nn_2.py`
- count_status: `verified-live`

```text
bug_id: bug_071
key: relu/nn/2
status: DIFF
output_source: live
expected_source: reference:tensorflow
expected_libs: tensorflow
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, Infinity, 0.0], [0.0, 6.798453330993652, 0.0]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, Infinity, 0.0], [0.0, 6.798453330993652, 0.0]]}
  paddle: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, Infinity, 0.0], [0.0, 6.798453330993652, 0.0]]}
```

## bug_072 `relu6/generic/1`

- script: `bug_072_relu6_generic_1.py`
- count_status: `verified-live`

```text
bug_id: bug_072
key: relu6/generic/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, keras, tensorflow
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, 6.0, 0.0], [6.0, 0.07076296210289001, 3.5615999698638916]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, 6.0, 0.0], [6.0, 0.07076296210289001, 3.5615999698638916]]}
```

## bug_073 `sign/generic/1`

- script: `bug_073_sign_generic_1.py`
- count_status: `verified-live`

```text
bug_id: bug_073
key: sign/generic/1
status: DIFF
output_source: live
expected_source: reference:jax
expected_libs: jax
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-0.0, 0.0, 0.0], [-1.0, 1.0, 1.0]]}
wrong:
  chainer: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, 0.0, 1.0], [-1.0, 1.0, 1.0]]}
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, 0.0, 1.0], [-1.0, 1.0, 1.0]]}
```

## bug_074 `sign/generic/2`

- script: `bug_074_sign_generic_2.py`
- count_status: `verified-live`

```text
bug_id: bug_074
key: sign/generic/2
status: DIFF
output_source: live
expected_source: majority
expected_libs: mindspore, mxnet
expected: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, 0.0, 1.0], [-1.0, 1.0, 1.0]]}
wrong:
  tensorflow: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, 0.0, 0.0], [-1.0, 1.0, 1.0]]}
```

## bug_075 `sign/generic/3`

- script: `bug_075_sign_generic_3.py`
- count_status: `verified-live`

```text
bug_id: bug_075
key: sign/generic/3
status: DIFF
output_source: live
expected_source: reference:paddle
expected_libs: paddle
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, 1.0, -1.0], [1.0, -1.0, 1.0]]}
wrong:
  mxnet: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, 1.0, -1.0], [1.0, -1.0, 1.0]]}
```

## bug_076 `sign/generic/component/1`

- script: `bug_076_sign_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_076
key: sign/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, mindspore, mxnet, paddle, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, 0.0, 1.0], [-1.0, 1.0, 1.0]]}
wrong:
  jax: {"dtype": "float32", "shape": [2, 3], "value": [[-0.0, 0.0, 0.0], [-1.0, 1.0, 1.0]]}
  keras: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, 0.0, 0.0], [-1.0, 1.0, 1.0]]}
  tensorflow: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, 0.0, 0.0], [-1.0, 1.0, 1.0]]}
```

## bug_077 `signbit/generic/2`

- script: `bug_077_signbit_generic_2.py`
- count_status: `verified-live`

```text
bug_id: bug_077
key: signbit/generic/2
status: DIFF
output_source: live
expected_source: reference:paddle
expected_libs: paddle
expected: {"dtype": "bool", "shape": [2, 3], "value": [[true, false, false], [true, false, false]]}
wrong:
  mindspore: {"dtype": "bool", "shape": [2, 3], "value": [[false, false, false], [true, false, false]]}
```

## bug_078 `signbit/generic/component/1`

- script: `bug_078_signbit_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_078
key: signbit/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, keras, paddle, torch
expected: {"dtype": "bool", "shape": [2, 3], "value": [[true, false, false], [true, false, false]]}
wrong:
  mindspore: {"dtype": "bool", "shape": [2, 3], "value": [[false, false, false], [true, false, false]]}
```

## bug_079 `softmax/nn/2`

- script: `bug_079_softmax_nn_2.py`
- count_status: `verified-live`

```text
bug_id: bug_079
key: softmax/nn/2
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, jax, scipy
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, NaN], [3.5585912883107085e-06, 0.0015719423536211252, 0.9984245300292969]]}
wrong:
  keras: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, 0.0], [3.5585917430580594e-06, 0.0015719423536211252, 0.9984245300292969]]}
  tensorflow: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, 0.0], [3.5585917430580594e-06, 0.0015719423536211252, 0.9984245300292969]]}
```

## bug_080 `softmax/nn/3`

- script: `bug_080_softmax_nn_3.py`
- count_status: `verified-live`

```text
bug_id: bug_080
key: softmax/nn/3
status: DIFF
output_source: live
expected_source: majority
expected_libs: mindspore, tensorflow
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, 0.0], [0.23795250058174133, 0.33366093039512634, 0.42838653922080994]]}
wrong:
  jax: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, NaN], [0.23795250058174133, 0.3336610198020935, 0.4283864498138428]]}
```

## bug_081 `tril/generic/component/1`

- script: `bug_081_tril_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_081
key: tril/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, keras, mxnet, numpy, paddle, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, 0.0, 0.0], [-1.0, 1.0, 0.0]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, NaN], [-1.0, 1.0, 0.0]]}
```

## bug_082 `trim_zeros/generic/3`

- script: `bug_082_trim_zeros_generic_3.py`
- count_status: `verified-live`

```text
bug_id: bug_082
key: trim_zeros/generic/3
status: DIFF
output_source: live
expected_source: reference:numpy
expected_libs: numpy
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-0.0, 0.0, 1.401298464324817e-45], [-1.0, 1.0, 9.999999974752427e-07]]}
wrong:
  jax: {"dtype": "float32", "shape": [1, 3], "value": [[-1.0, 1.0, 9.999999974752427e-07]]}
```

## bug_083 `trim_zeros/generic/component/1`

- script: `bug_083_trim_zeros_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_083
key: trim_zeros/generic/component/1
status: DIFF
output_source: live
expected_source: reference:numpy
expected_libs: numpy
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-0.0, 0.0, 1.401298464324817e-45], [-1.0, 1.0, 9.999999974752427e-07]]}
wrong:
  jax: {"dtype": "float32", "shape": [1, 3], "value": [[-1.0, 1.0, 9.999999974752427e-07]]}
```

## bug_084 `triu_indices/generic/component/1`

- script: `bug_084_triu_indices_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_084
key: triu_indices/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, mindspore, numpy
expected: {"dtype": "int32", "shape": [2, 0], "value": [[], []]}
wrong:
  paddle: {"dtype": "int64", "shape": [2, 6], "value": [[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]}
```

## bug_085 `unique/generic/component/1`

- script: `bug_085_unique_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_085
key: unique/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, mindspore, numpy, paddle, torch
expected: {"dtype": "float32", "shape": [5], "value": [-Infinity, 0.0, 1.0, Infinity, NaN]}
wrong:
  mxnet: {"dtype": "float32", "shape": [1], "value": [NaN]}
```

## bug_086 `unique_all/generic/component/1`

- script: `bug_086_unique_all_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_086
key: unique_all/generic/component/1
status: DIFF
output_source: live
expected_source: reference:numpy
expected_libs: numpy
expected: [{"dtype": "float32", "shape": [5], "value": [-1.0, -0.0, 1.401298464324817e-45, 9.999999974752427e-07, 1.0]}, {"dtype": "int64", "shape": [5], "value": [3, 0, 2, 5, 4]}, {"dtype": "int64", "shape": [2, 3], "value": [[1, 1, 2], [0, 4, 3]]}, {"dtype": "int64", "shape": [5], "value": [1, 2, 1, 1, 1]}]
wrong:
  jax: [{"dtype": "float32", "shape": [4], "value": [-1.0, -0.0, 9.999999974752427e-07, 1.0]}, {"dtype": "int32", "shape": [4], "value": [3, 0, 5, 4]}, {"dtype": "int32", "shape": [2, 3], "value": [[1, 1, 1], [0, 3, 2]]}, {"dtype": "int32", "shape": [4], "value": [1, 3, 1, 1]}]
```

## bug_087 `unique_counts/generic/component/1`

- script: `bug_087_unique_counts_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_087
key: unique_counts/generic/component/1
status: DIFF
output_source: live
expected_source: reference:numpy
expected_libs: numpy
expected: [{"dtype": "float32", "shape": [5], "value": [-1.0, -0.0, 1.401298464324817e-45, 9.999999974752427e-07, 1.0]}, {"dtype": "int64", "shape": [5], "value": [1, 2, 1, 1, 1]}]
wrong:
  jax: [{"dtype": "float32", "shape": [4], "value": [-1.0, -0.0, 9.999999974752427e-07, 1.0]}, {"dtype": "int32", "shape": [4], "value": [1, 3, 1, 1]}]
```

## bug_088 `unique_inverse/generic/component/1`

- script: `bug_088_unique_inverse_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_088
key: unique_inverse/generic/component/1
status: DIFF
output_source: live
expected_source: reference:numpy
expected_libs: numpy
expected: [{"dtype": "float32", "shape": [5], "value": [-1.0, -0.0, 1.401298464324817e-45, 9.999999974752427e-07, 1.0]}, {"dtype": "int64", "shape": [2, 3], "value": [[1, 1, 2], [0, 4, 3]]}]
wrong:
  jax: [{"dtype": "float32", "shape": [4], "value": [-1.0, -0.0, 9.999999974752427e-07, 1.0]}, {"dtype": "int32", "shape": [2, 3], "value": [[1, 1, 1], [0, 3, 2]]}]
```

## bug_089 `unique_values/generic/component/1`

- script: `bug_089_unique_values_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_089
key: unique_values/generic/component/1
status: DIFF
output_source: live
expected_source: reference:numpy
expected_libs: numpy
expected: {"dtype": "float32", "shape": [5], "value": [-1.0, -0.0, 1.401298464324817e-45, 9.999999974752427e-07, 1.0]}
wrong:
  jax: {"dtype": "float32", "shape": [4], "value": [-1.0, -0.0, 9.999999974752427e-07, 1.0]}
```

## bug_090 `cholesky/linalg/4`

- script: `bug_090_cholesky_linalg_4.py`
- count_status: `verified-live`

```text
bug_id: bug_090
key: cholesky/linalg/4
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, scipy
expected: {"dtype": "float32", "shape": [2, 2], "value": [[2.246246099472046, 0.0], [-0.5446306467056274, 0.8483017086982727]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 2], "value": [[2.246246099472046, -0.5446306467056274], [0.0, 0.8483017086982727]]}
```

## bug_091 `logm/linalg/2`

- script: `bug_091_logm_linalg_2.py`
- count_status: `verified-live`

```text
bug_id: bug_091
key: logm/linalg/2
status: DIFF
output_source: live
expected_source: reference:scipy
expected_libs: scipy
expected: {"dtype": "complex128", "shape": [2, 2], "value": [[[1.1103631784640262, 1.9516190639709639], [-0.1988762739579102, 1.2463638567245867]], [[-0.2973211347022723, 1.863320891720253], [1.2318952214097851, 1.189973932819967]]]}
wrong:
  tensorflow: {"dtype": "complex64", "shape": [2, 2], "value": [[[1.1103628873825073, -1.9516187906265259], [-0.19887620210647583, -1.246363639831543]], [[-0.29732105135917664, -1.8633205890655518], [1.231894850730896, -1.1899737119674683]]]}
```

## bug_092 `lstsq/linalg/5`

- script: `bug_092_lstsq_linalg_5.py`
- count_status: `verified-live`

```text
bug_id: bug_092
key: lstsq/linalg/5
status: DIFF
output_source: live
expected_source: reference:tensorflow
expected_libs: tensorflow
expected: {"dtype": "float32", "shape": [2, 1], "value": [[-1099.765869140625], [285.66302490234375]]}
wrong:
  paddle: {"dtype": "float32", "shape": [2, 1], "value": [[-1134.48486328125], [294.6731872558594]]}
```

## bug_093 `clip_by_norm/linalg/4`

- script: `bug_093_clip_by_norm_linalg_4.py`
- count_status: `verified-live`

```text
bug_id: bug_093
key: clip_by_norm/linalg/4
status: DIFF
output_source: live
expected_source: reference:tensorflow
expected_libs: tensorflow
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, NaN], [NaN, NaN, NaN]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, Infinity, -Infinity], [-1.0, 1.0, 9.999999974752427e-07]]}
```

## bug_094 `geqrf/generic/component/1`

- script: `bug_094_geqrf_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_094
key: geqrf/generic/component/1
status: DIFF
output_source: live
expected_source: reference:torch
expected_libs: torch
expected: [{"dtype": "float32", "shape": [2, 3], "value": [[-1.0, 1.0, 9.999999974752427e-07], [-1.0, 0.0, 0.0]]}, {"dtype": "float32", "shape": [2], "value": [1.0, 0.0]}]
wrong:
  mindspore: [{"dtype": "float32", "shape": [2, 3], "value": [[1.0, -1.0, -9.999999974752427e-07], [1.0, 0.0, 0.0]]}, {"dtype": "float32", "shape": [2], "value": [1.0, 0.0]}]
```

## bug_095 `inv/linalg/component/1`

- script: `bug_095_inv_linalg_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_095
key: inv/linalg/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, keras, paddle, scipy, tensorflow, torch
expected: {"dtype": "float32", "shape": [2, 2], "value": [[6265.81298828125, 6306.576171875], [5852.66064453125, 5890.87646484375]]}
wrong:
  chainer: {"dtype": "float32", "shape": [2, 2], "value": [[6251.0224609375, 6291.689453125], [5838.845703125, 5876.9716796875]]}
  mxnet: {"dtype": "float32", "shape": [2, 2], "value": [[6257.04736328125, 6297.75390625], [5844.47314453125, 5882.63623046875]]}
  numpy: {"dtype": "float32", "shape": [2, 2], "value": [[6251.0224609375, 6291.689453125], [5838.845703125, 5876.9716796875]]}
errors: {"mindspore": "SKIP: mindspore.ops.function.inv is elementwise reciprocal, not matrix inverse"}
```

## bug_096 `logm/linalg/component/1`

- script: `bug_096_logm_linalg_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_096
key: logm/linalg/component/1
status: DIFF
output_source: live
expected_source: reference:scipy
expected_libs: scipy
expected: {"dtype": "complex128", "shape": [2, 2], "value": [[[0.25543883970325987, 2.6065553190417727], [-0.5619171914561192, 1.0972394837356936]], [[-0.6509101695253043, 1.2710135038180814], [1.3163022147083416, 0.5350380214693218]]]}
wrong:
  tensorflow: {"dtype": "complex64", "shape": [2, 2], "value": [[[0.25543904304504395, -2.6065549850463867], [-0.5619169473648071, -1.0972392559051514]], [[-0.650909960269928, -1.2710133790969849], [1.3163020610809326, -0.5350378751754761]]]}
```

## bug_097 `lstsq/generic/component/1`

- script: `bug_097_lstsq_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_097
key: lstsq/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: keras, mindspore
expected: {"dtype": "float32", "shape": [2, 1], "value": [[24279.0234375], [-17318.212890625]]}
wrong:
  torch: {"dtype": "float32", "shape": [2, 1], "value": [[24242.37109375], [-17292.064453125]]}
```

## bug_098 `lstsq/linalg/component/1`

- script: `bug_098_lstsq_linalg_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_098
key: lstsq/linalg/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, mindspore, mxnet, numpy, paddle, scipy, torch
expected: {"dtype": "float32", "shape": [2, 1], "value": [[-1341.9998779296875], [-431.5251770019531]]}
wrong:
  tensorflow: {"dtype": "float32", "shape": [2, 1], "value": [[-1328.3541259765625], [-427.1346130371094]]}
```

## bug_099 `rsf2csf/linalg`

- script: `bug_099_rsf2csf_linalg.py`
- count_status: `not-counted`

```text
bug_id: bug_099
key: rsf2csf/linalg
status: PASS
output_source: not_live
expected_source: majority
expected_libs: jax, scipy
expected: [{"dtype": "complex64", "shape": [2, 2], "value": [[[3.251811981201172, 1.6166623830795288], [-3.5150370597839355, 1.1920928955078125e-07]], [[0.0, 0.0], [3.2518110275268555, -1.6166623830795288]]]}, {"dtype": "complex64", "shape": [2, 2], "value": [[[0.260574072599411, 0.34882885217666626], [-0.8944792151451111, -0.10161858797073364]], [[0.8944792151451111, -0.10161858797073364], [0.260574072599411, -0.34882885217666626]]]}]
wrong:
  none
```

## bug_100 `csd/signal/component/1`

- script: `bug_100_csd_signal_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_100
key: csd/signal/component/1
status: DIFF
output_source: live
expected_source: reference:scipy
expected_libs: scipy
expected: [{"dtype": "float64", "shape": [3], "value": [0.0, 0.25, 0.5]}, {"dtype": "float32", "shape": [3], "value": [NaN, NaN, NaN]}]
wrong:
  jax: [{"dtype": "float32", "shape": [3], "value": [0.0, 0.25, 0.5]}, {"dtype": "complex64", "shape": [3], "value": [[NaN, NaN], [NaN, NaN], [NaN, NaN]]}]
```

## bug_101 `cbrt/generic/2`

- script: `bug_101_cbrt_generic_2.py`
- count_status: `verified-live`

```text
bug_id: bug_101
key: cbrt/generic/2
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, mxnet
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-0.0, 0.0, 8.881775197276188e-16], [-1.0, 1.0, 0.009999998845160007]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, 8.881775197276188e-16], [-1.0, 1.0, 0.009999998845160007]]}
```

## bug_102 `cbrt/generic/component/1`

- script: `bug_102_cbrt_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_102
key: cbrt/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, keras, mxnet
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-0.0, 0.0, 8.881775197276188e-16], [-1.0, 1.0, 0.009999998845160007]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, 8.881775197276188e-16], [-1.0, 1.0, 0.009999998845160007]]}
```

## bug_103 `digamma/generic/1`

- script: `bug_103_digamma_generic_1.py`
- count_status: `verified-live`

```text
bug_id: bug_103
key: digamma/generic/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, mindspore
expected: {"dtype": "float32", "shape": [2, 3], "value": [[Infinity, -Infinity, -Infinity], [NaN, -0.5772156715393066, -1000000.5625]]}
wrong:
  jax: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, NaN], [NaN, -0.5772153735160828, -1000000.5625]]}
```

## bug_104 `digamma/generic/2`

- script: `bug_104_digamma_generic_2.py`
- count_status: `verified-live`

```text
bug_id: bug_104
key: digamma/generic/2
status: DIFF
output_source: live
expected_source: reference:tensorflow
expected_libs: tensorflow
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, NaN], [NaN, -0.5772159099578857, -1000000.5625]]}
wrong:
  paddle: {"dtype": "float32", "shape": [2, 3], "value": [[Infinity, -Infinity, -Infinity], [NaN, -0.5772159099578857, -1000000.5625]]}
```

## bug_105 `digamma/generic/component/1`

- script: `bug_105_digamma_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_105
key: digamma/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, mindspore, paddle, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[Infinity, -Infinity, -Infinity], [NaN, -0.5772156715393066, -1000000.5625]]}
wrong:
  jax: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, NaN], [NaN, -0.5772153735160828, -1000000.5625]]}
  tensorflow: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, NaN, NaN], [NaN, -0.5772159099578857, -1000000.5625]]}
```

## bug_106 `erfcinv/generic/component/1`

- script: `bug_106_erfcinv_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_106
key: erfcinv/generic/component/1
status: DIFF
output_source: live
expected_source: reference:tensorflow
expected_libs: tensorflow
expected: {"dtype": "float32", "shape": [2, 3], "value": [[Infinity, Infinity, Infinity], [NaN, -0.0, 3.4589104652404785]]}
wrong:
  chainer: {"dtype": "float32", "shape": [2, 3], "value": [[Infinity, Infinity, 10.019834518432617], [NaN, -0.0, 3.4589107036590576]]}
```

## bug_107 `gammaln/generic/component/1`

- script: `bug_107_gammaln_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_107
key: gammaln/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: mxnet, paddle, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[Infinity, Infinity, 103.2789306640625], [Infinity, 0.0, 13.815509796142578]]}
wrong:
  jax: {"dtype": "float32", "shape": [2, 3], "value": [[Infinity, Infinity, Infinity], [Infinity, 4.76837158203125e-07, 13.815508842468262]]}
```

## bug_108 `lgamma/generic/1`

- script: `bug_108_lgamma_generic_1.py`
- count_status: `verified-live`

```text
bug_id: bug_108
key: lgamma/generic/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, mindspore
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, Infinity, Infinity], [1.389909267425537, 1.269892930984497, 2.4105467796325684]]}
wrong:
  chainer: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, Infinity, -Infinity], [1.3899093866348267, 1.2698928117752075, 2.4105470180511475]]}
```

## bug_109 `lgamma/generic/2`

- script: `bug_109_lgamma_generic_2.py`
- count_status: `verified-live`

```text
bug_id: bug_109
key: lgamma/generic/2
status: DIFF
output_source: live
expected_source: reference:tensorflow
expected_libs: tensorflow
expected: {"dtype": "float32", "shape": [2, 3], "value": [[Infinity, Infinity, 103.97207641601562], [Infinity, 0.0, 13.815509796142578]]}
wrong:
  paddle: {"dtype": "float32", "shape": [2, 3], "value": [[Infinity, Infinity, 103.2789306640625], [Infinity, 0.0, 13.815509796142578]]}
```

## bug_110 `lgamma/generic/component/1`

- script: `bug_110_lgamma_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_110
key: lgamma/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, mindspore, paddle, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[Infinity, Infinity, 103.2789306640625], [Infinity, 0.0, 13.815509796142578]]}
wrong:
  jax: {"dtype": "float32", "shape": [2, 3], "value": [[Infinity, Infinity, Infinity], [Infinity, 4.76837158203125e-07, 13.815508842468262]]}
  tensorflow: {"dtype": "float32", "shape": [2, 3], "value": [[Infinity, Infinity, 103.97207641601562], [Infinity, 0.0, 13.815509796142578]]}
```

## bug_111 `log10/generic/1`

- script: `bug_111_log10_generic_1.py`
- count_status: `verified-live`

```text
bug_id: bug_111
key: log10/generic/1
status: DIFF
output_source: live
expected_source: reference:jax
expected_libs: jax
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -Infinity], [NaN, 0.0, -6.0]]}
wrong:
  chainer: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -44.85346984863281], [NaN, 0.0, -6.000000476837158]]}
  keras: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -45.15449905395508], [NaN, 0.0, -6.0]]}
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -38.23080062866211], [NaN, 0.0, -6.0]]}
```

## bug_112 `log10/generic/2`

- script: `bug_112_log10_generic_2.py`
- count_status: `verified-live`

```text
bug_id: bug_112
key: log10/generic/2
status: DIFF
output_source: live
expected_source: reference:mxnet
expected_libs: mxnet
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -44.85346984863281], [NaN, 0.0, -6.0]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -38.23080062866211], [NaN, 0.0, -6.0]]}
errors: {"paddle": "SKIP: in-place tensor variants mutate inputs"}
```

## bug_113 `log10/generic/component/1`

- script: `bug_113_log10_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_113
key: log10/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, mxnet, paddle, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -44.85346984863281], [NaN, 0.0, -6.000000476837158]]}
wrong:
  jax: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -Infinity], [NaN, 0.0, -6.0]]}
  keras: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -45.15449905395508], [NaN, 0.0, -6.0]]}
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -38.23080062866211], [NaN, 0.0, -6.0]]}
```

## bug_114 `log2/generic/1`

- script: `bug_114_log2_generic_1.py`
- count_status: `verified-live`

```text
bug_id: bug_114
key: log2/generic/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, mindspore
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -Infinity], [NaN, 0.0, -19.931568145751953]]}
wrong:
  chainer: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -149.0], [NaN, 0.0, -19.931568145751953]]}
  keras: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -150.0], [NaN, 0.0, -19.931568145751953]]}
```

## bug_115 `log2/generic/2`

- script: `bug_115_log2_generic_2.py`
- count_status: `verified-live`

```text
bug_id: bug_115
key: log2/generic/2
status: DIFF
output_source: live
expected_source: reference:mxnet
expected_libs: mxnet
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -149.0], [NaN, 0.0, -19.931568145751953]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -Infinity], [NaN, 0.0, -19.931568145751953]]}
errors: {"paddle": "SKIP: in-place tensor variants mutate inputs"}
```

## bug_116 `log2/generic/component/1`

- script: `bug_116_log2_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_116
key: log2/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, mxnet, paddle, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -149.0], [NaN, 0.0, -19.931568145751953]]}
wrong:
  jax: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -Infinity], [NaN, 0.0, -19.931568145751953]]}
  keras: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -150.0], [NaN, 0.0, -19.931568145751953]]}
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -Infinity], [NaN, 0.0, -19.931568145751953]]}
```

## bug_117 `ndtri/generic/1`

- script: `bug_117_ndtri_generic_1.py`
- count_status: `verified-live`

```text
bug_id: bug_117
key: ndtri/generic/1
status: DIFF
output_source: live
expected_source: reference:jax
expected_libs: jax
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -Infinity], [NaN, Infinity, -4.753424167633057]]}
wrong:
  chainer: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -14.121426582336426], [NaN, Infinity, -4.753424167633057]]}
```

## bug_118 `ndtri/generic/component/1`

- script: `bug_118_ndtri_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_118
key: ndtri/generic/component/1
status: DIFF
output_source: live
expected_source: reference:jax
expected_libs: jax
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -Infinity], [NaN, Infinity, -4.753424167633057]]}
wrong:
  chainer: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -14.121426582336426], [NaN, Infinity, -4.753424167633057]]}
  torch: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -14.121426582336426], [NaN, Infinity, -4.753424167633057]]}
```

## bug_119 `rsqrt/generic/1`

- script: `bug_119_rsqrt_generic_1.py`
- count_status: `verified-live`

```text
bug_id: bug_119
key: rsqrt/generic/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, mindspore
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, Infinity, 2.671373844909537e+22], [NaN, 1.0, 999.9999389648438]]}
wrong:
  keras: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, Infinity, Infinity], [NaN, 1.0, 999.9999389648438]]}
```

## bug_120 `rsqrt/generic/component/1`

- script: `bug_120_rsqrt_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_120
key: rsqrt/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, mindspore, mxnet, paddle, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, Infinity, 2.671373844909537e+22], [NaN, 1.0, 999.9999389648438]]}
wrong:
  jax: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, Infinity, Infinity], [NaN, 1.0, 1000.0]]}
  keras: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, Infinity, Infinity], [NaN, 1.0, 999.9999389648438]]}
  tensorflow: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, Infinity, Infinity], [NaN, 1.0, 999.9999389648438]]}
```

## bug_121 `sgn/generic/component/1`

- script: `bug_121_sgn_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_121
key: sgn/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: mindspore, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]}
wrong:
  paddle: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, 1.0, -1.0], [0.0, 0.0, 1.0]]}
```

## bug_122 `xlogy/generic/component/1`

- script: `bug_122_xlogy_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_122
key: xlogy/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: tensorflow, torch
expected: {"dtype": "float32", "shape": [3], "value": [0.0, 0.0, 0.0]}
wrong:
  mindspore: {"dtype": "float32", "shape": [3], "value": [NaN, NaN, -1.401298464324817e-45]}
```

## bug_123 `angle/generic/component/1`

- script: `bug_123_angle_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_123
key: angle/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: keras, paddle, tensorflow, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, 0.0, 0.0], [3.1415927410125732, 0.0, 0.0]]}
wrong:
  jax: {"dtype": "float32", "shape": [2, 3], "value": [[3.1415927410125732, 0.0, 0.0], [3.1415927410125732, 0.0, 0.0]]}
  numpy: {"dtype": "float32", "shape": [2, 3], "value": [[3.1415927410125732, 0.0, 0.0], [3.1415927410125732, 0.0, 0.0]]}
```

## bug_124 `corrcoef/generic/component/1`

- script: `bug_124_corrcoef_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_124
key: corrcoef/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, keras, torch
expected: {"dtype": "float32", "shape": [2, 2], "value": [[NaN, NaN], [NaN, 1.0]]}
wrong:
  mxnet: {"dtype": "float64", "shape": [2, 2], "value": [[0.9999999999999998, 5.773502677158356e-07], [5.773502677158356e-07, 1.0]]}
  numpy: {"dtype": "float64", "shape": [2, 2], "value": [[0.9999999999999998, 5.773502677158356e-07], [5.773502677158356e-07, 1.0]]}
errors: {"mindspore": "SKIP: MindSpore NumPy corrcoef is unstable in this environment"}
```

## bug_125 `csd/signal/12`

- script: `bug_125_csd_signal_12.py`
- count_status: `verified-live`

```text
bug_id: bug_125
key: csd/signal/12
status: DIFF
output_source: live
expected_source: reference:scipy
expected_libs: scipy
expected: [{"dtype": "float64", "shape": [3], "value": [0.0, 0.25, 0.5]}, {"dtype": "float32", "shape": [3], "value": [NaN, NaN, NaN]}]
wrong:
  jax: [{"dtype": "float32", "shape": [3], "value": [0.0, 0.25, 0.5]}, {"dtype": "complex64", "shape": [3], "value": [[NaN, NaN], [NaN, NaN], [NaN, NaN]]}]
```

## bug_126 `isclose/generic/5`

- script: `bug_126_isclose_generic_5.py`
- count_status: `verified-live`

```text
bug_id: bug_126
key: isclose/generic/5
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, mindspore, mxnet, numpy
expected: {"dtype": "bool", "shape": [2, 3], "value": [[false, true, true], [true, true, true]]}
wrong:
  keras: {"dtype": "bool", "shape": [2, 3], "value": [[false, false, false], [true, true, true]]}
```

## bug_127 `percentile/generic/7`

- script: `bug_127_percentile_generic_7.py`
- count_status: `verified-live`

```text
bug_id: bug_127
key: percentile/generic/7
status: DIFF
output_source: live
expected_source: reference:jax
expected_libs: jax
expected: {"dtype": "float32", "shape": [], "value": NaN}
wrong:
  mxnet: {"dtype": "float32", "shape": [], "value": -0.6650597453117371}
```

## bug_128 `percentile/generic/component/1`

- script: `bug_128_percentile_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_128
key: percentile/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, numpy
expected: {"dtype": "float32", "shape": [], "value": NaN}
wrong:
  mxnet: {"dtype": "float32", "shape": [], "value": -0.7107062339782715}
```

## bug_129 `quantile/generic/7`

- script: `bug_129_quantile_generic_7.py`
- count_status: `verified-live`

```text
bug_id: bug_129
key: quantile/generic/7
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, paddle
expected: {"dtype": "float32", "shape": [], "value": NaN}
wrong:
  mxnet: {"dtype": "float32", "shape": [], "value": 0.23651906847953796}
```

## bug_130 `quantile/generic/component/1`

- script: `bug_130_quantile_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_130
key: quantile/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, keras, numpy, paddle, torch
expected: {"dtype": "float32", "shape": [], "value": NaN}
wrong:
  mxnet: {"dtype": "float32", "shape": [], "value": -0.025599300861358643}
```

## bug_131 `rankdata/generic/4`

- script: `bug_131_rankdata_generic_4.py`
- count_status: `verified-live`

```text
bug_id: bug_131
key: rankdata/generic/4
status: DIFF
output_source: live
expected_source: reference:scipy
expected_libs: scipy
expected: {"dtype": "float64", "shape": [6], "value": [NaN, NaN, NaN, NaN, NaN, NaN]}
wrong:
  jax: {"dtype": "float32", "shape": [6], "value": [6.0, 5.0, 1.0, 3.0, 4.0, 2.0]}
```

## bug_132 `rankdata/generic/component/1`

- script: `bug_132_rankdata_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_132
key: rankdata/generic/component/1
status: DIFF
output_source: live
expected_source: reference:scipy
expected_libs: scipy
expected: {"dtype": "float64", "shape": [6], "value": [2.5, 2.5, 4.0, 1.0, 6.0, 5.0]}
wrong:
  jax: {"dtype": "float32", "shape": [6], "value": [3.0, 3.0, 3.0, 1.0, 6.0, 5.0]}
```

## bug_133 `threshold/generic/3`

- script: `bug_133_threshold_generic_3.py`
- count_status: `verified-live`

```text
bug_id: bug_133
key: threshold/generic/3
status: DIFF
output_source: live
expected_source: majority
expected_libs: keras, tensorflow
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-1.0, -1.0, -1.0], [-1.0, 1.0, 9.999999974752427e-07]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[-1.0, -1.0, 1.401298464324817e-45], [-1.0, 1.0, 9.999999974752427e-07]]}
```

## bug_134 `argmax/reduction/3`

- script: `bug_134_argmax_reduction_3.py`
- count_status: `verified-live`

```text
bug_id: bug_134
key: argmax/reduction/3
status: DIFF
output_source: live
expected_source: reference:jax
expected_libs: jax
expected: {"dtype": "int32", "shape": [2], "value": [0, 1]}
wrong:
  keras: {"dtype": "int32", "shape": [2], "value": [1, 1]}
  mxnet: {"dtype": "int64", "shape": [2], "value": [2, 1]}
errors: {"mindspore": "SKIP: mindspore.mint.argmax CPU kernel is unavailable in this runner"}
```

## bug_135 `float_power/generic/2`

- script: `bug_135_float_power_generic_2.py`
- count_status: `verified-live`

```text
bug_id: bug_135
key: float_power/generic/2
status: DIFF
output_source: live
expected_source: reference:jax
expected_libs: jax
expected: {"dtype": "float32", "shape": [3], "value": [0.0, Infinity, Infinity]}
wrong:
  mindspore: {"dtype": "float64", "shape": [3], "value": [0.0, Infinity, 3.8938863925217126e+70]}
```

## bug_136 `tril_indices/generic`

- script: `bug_136_tril_indices_generic.py`
- count_status: `verified-live`

```text
bug_id: bug_136
key: tril_indices/generic
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, mindspore, numpy
expected: {"dtype": "int32", "shape": [2, 9], "value": [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]]}
wrong:
  paddle: {"dtype": "int64", "shape": [2, 6], "value": [[0, 1, 1, 2, 2, 2], [0, 0, 1, 0, 1, 2]]}
  torch: {"dtype": "int64", "shape": [2, 6], "value": [[0, 1, 1, 2, 2, 2], [0, 0, 1, 0, 1, 2]]}
```

## bug_137 `fft/fft`

- script: `bug_137_fft_fft.py`
- count_status: `not-counted`

```text
bug_id: bug_137
key: fft/fft
status: PASS
output_source: not_live
expected_source: majority
expected_libs: chainer, jax, keras, mindspore, numpy, paddle, scipy, tensorflow, torch
expected: {"dtype": "complex64", "shape": [4], "value": [[NaN, 0.0], [NaN, -Infinity], [NaN, 0.0], [NaN, Infinity]]}
wrong:
  none
```

## bug_138 `ifft/fft`

- script: `bug_138_ifft_fft.py`
- count_status: `not-counted`

```text
bug_id: bug_138
key: ifft/fft
status: PASS
output_source: not_live
expected_source: majority
expected_libs: chainer, jax, mindspore, numpy, paddle, scipy, tensorflow, torch
expected: {"dtype": "complex64", "shape": [4], "value": [[NaN, 0.0], [NaN, Infinity], [NaN, 0.0], [NaN, -Infinity]]}
wrong:
  none
```

## bug_139 `ifft2/fft`

- script: `bug_139_ifft2_fft.py`
- count_status: `not-counted`

```text
bug_id: bug_139
key: ifft2/fft
status: PASS
output_source: not_live
expected_source: majority
expected_libs: jax, numpy, paddle, scipy, torch
expected: {"dtype": "complex64", "shape": [2, 3], "value": [[[NaN, 0.0], [NaN, Infinity], [NaN, -Infinity]], [[NaN, 0.0], [NaN, Infinity], [NaN, -Infinity]]]}
wrong:
  none
```

## bug_140 `ifftn/fft`

- script: `bug_140_ifftn_fft.py`
- count_status: `not-counted`

```text
bug_id: bug_140
key: ifftn/fft
status: PASS
output_source: not_live
expected_source: majority
expected_libs: jax, numpy, paddle, scipy, torch
expected: {"dtype": "complex64", "shape": [2, 3], "value": [[[NaN, 0.0], [NaN, Infinity], [NaN, -Infinity]], [[NaN, 0.0], [NaN, Infinity], [NaN, -Infinity]]]}
wrong:
  none
```

## bug_141 `rfft/fft`

- script: `bug_141_rfft_fft.py`
- count_status: `not-counted`

```text
bug_id: bug_141
key: rfft/fft
status: PASS
output_source: not_live
expected_source: majority
expected_libs: jax, keras, mindspore, numpy, paddle, scipy, tensorflow, torch
expected: {"dtype": "complex64", "shape": [3], "value": [[NaN, 0.0], [NaN, -Infinity], [NaN, 0.0]]}
wrong:
  none
```

## bug_142 `argpartition/generic`

- script: `bug_142_argpartition_generic.py`
- count_status: `verified-live`

```text
bug_id: bug_142
key: argpartition/generic
status: DIFF
output_source: live
expected_source: majority
expected_libs: mxnet, numpy
expected: {"dtype": "int64", "shape": [2, 3], "value": [[2, 1, 0], [0, 1, 2]]}
wrong:
  jax: {"dtype": "int32", "shape": [2, 3], "value": [[2, 1, 0], [1, 0, 2]]}
errors: {"keras": "SKIP: keras.ops.argpartition is not value-compatible with numpy argpartition here"}
```

## bug_143 `divide/generic`

- script: `bug_143_divide_generic.py`
- count_status: `verified-live`

```text
bug_id: bug_143
key: divide/generic
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, keras, mxnet, paddle, tensorflow, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, Infinity, -Infinity], [0.0, -0.0, 0.25]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, Infinity, -Infinity], [0.0, -0.0, 0.25]]}
```

## bug_144 `group_norm/nn`

- script: `bug_144_group_norm_nn.py`
- count_status: `verified-live`

```text
bug_id: bug_144
key: group_norm/nn
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, paddle, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, 0.0, 0.0], [-4.484155085839415e-44, 0.0, 0.0]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[1.2247449159622192, -1.2247449159622192, 0.0], [-2.942726775082116e-44, 1.5414283107572988e-44, 1.5414283107572988e-44]]}
errors: {"mxnet": "SKIP: MXNet GroupNorm requires operator-attribute adapters"}
```

## bug_145 `log/generic`

- script: `bug_145_log_generic.py`
- count_status: `verified-live`

```text
bug_id: bug_145
key: log/generic
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, mxnet, paddle, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -103.2789306640625, 0.0], [Infinity, -69.07755279541016, 1.3862943649291992]]}
wrong:
  jax: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, 0.0], [Infinity, -69.07755279541016, 1.3862943649291992]]}
  keras: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -103.97207641601562, 0.0], [Infinity, -69.07755279541016, 1.3862943649291992]]}
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, 0.0], [Infinity, -69.07755279541016, 1.3862943649291992]]}
  tensorflow: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -103.97207641601562, 0.0], [Infinity, -69.07755279541016, 1.3862943649291992]]}
errors: {"scipy": "SKIP: scipy.stats distribution objects are not tensor math APIs"}
```

## bug_146 `logaddexp/generic`

- script: `bug_146_logaddexp_generic.py`
- count_status: `verified-live`

```text
bug_id: bug_146
key: logaddexp/generic
status: DIFF
output_source: live
expected_source: majority
expected_libs: keras, paddle, torch
expected: {"dtype": "float32", "shape": [3], "value": [9.999999680285692e+37, 0.0, 0.6931471824645996]}
wrong:
  mindspore: {"dtype": "float32", "shape": [3], "value": [Infinity, 0.0, 0.6931471824645996]}
```

## bug_147 `logaddexp2/generic`

- script: `bug_147_logaddexp2_generic.py`
- count_status: `verified-live`

```text
bug_id: bug_147
key: logaddexp2/generic
status: DIFF
output_source: live
expected_source: majority
expected_libs: keras, torch
expected: {"dtype": "float32", "shape": [3], "value": [9.999999680285692e+37, 0.0, 1.0]}
wrong:
  mindspore: {"dtype": "float32", "shape": [3], "value": [Infinity, 0.0, 1.0]}
```

## bug_148 `renorm/generic`

- script: `bug_148_renorm_generic.py`
- count_status: `verified-live`

```text
bug_id: bug_148
key: renorm/generic
status: DIFF
output_source: live
expected_source: majority
expected_libs: paddle, torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, -0.0, 0.0], [-1.401298464324817e-45, 0.0, -0.0]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[70.71067810058594, -70.71067810058594, 0.0], [-1.401298464324817e-45, 0.0, -0.0]]}
```

## bug_149 `bessel_y1/generic`

- script: `bug_149_bessel_y1_generic.py`
- count_status: `verified-live`

```text
bug_id: bug_149
key: bessel_y1/generic
status: DIFF
output_source: live
expected_source: reference:torch
expected_libs: torch
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -0.7812128067016602], [NaN, -6.366197321461539e+29, 0.39792561531066895]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[NaN, -Infinity, -0.7812128067016602], [NaN, -6.366197321461539e+29, 0.3979256749153137]]}
```

## bug_150 `fft2/fft`

- script: `bug_150_fft2_fft.py`
- count_status: `not-counted`

```text
bug_id: bug_150
key: fft2/fft
status: PASS
output_source: not_live
expected_source: majority
expected_libs: jax, numpy, paddle, scipy, torch
expected: {"dtype": "complex64", "shape": [2, 3], "value": [[[0.0, 0.0], [1.5000000027488779e+38, 8.660253226880323e+37], [1.5000000027488779e+38, -8.660253226880323e+37]], [[0.0, 0.0], [1.5000000027488779e+38, 8.660253226880323e+37], [1.5000000027488779e+38, -8.660253226880323e+37]]]}
wrong:
  none
```

## bug_151 `fftn/fft`

- script: `bug_151_fftn_fft.py`
- count_status: `not-counted`

```text
bug_id: bug_151
key: fftn/fft
status: PASS
output_source: not_live
expected_source: majority
expected_libs: jax, numpy, paddle, scipy, torch
expected: {"dtype": "complex64", "shape": [2, 3], "value": [[[0.0, 0.0], [1.5000000027488779e+38, 8.660253226880323e+37], [1.5000000027488779e+38, -8.660253226880323e+37]], [[0.0, 0.0], [1.5000000027488779e+38, 8.660253226880323e+37], [1.5000000027488779e+38, -8.660253226880323e+37]]]}
wrong:
  none
```

## bug_152 `hfft/fft`

- script: `bug_152_hfft_fft.py`
- count_status: `verified-live`

```text
bug_id: bug_152
key: hfft/fft
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, paddle, scipy
expected: {"dtype": "float32", "shape": [6], "value": [-1.0000000694406173e+38, 0.0, 2.0000001388812345e+38, 3.0000000054977558e+38, 2.0000001388812345e+38, 2.028240960365167e+31]}
wrong:
  numpy: {"dtype": "float32", "shape": [6], "value": [-9.999999680285692e+37, 0.0, 1.9999999360571385e+38, 3.0000000054977558e+38, 1.9999999360571385e+38, 0.0]}
  torch: {"dtype": "float32", "shape": [6], "value": [-9.999999680285692e+37, 0.0, 1.9999999360571385e+38, 3.0000000054977558e+38, 1.9999999360571385e+38, 0.0]}
```

## bug_153 `irfft/fft`

- script: `bug_153_irfft_fft.py`
- count_status: `verified-live`

```text
bug_id: bug_153
key: irfft/fft
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, paddle, scipy
expected: {"dtype": "float32", "shape": [6], "value": [-1.6666666978909888e+37, 0.0, 3.3333333957819775e+37, 4.999999840142846e+37, 3.3333333957819775e+37, 3.38040170135243e+30]}
wrong:
  numpy: {"dtype": "float32", "shape": [6], "value": [-1.6666666978909888e+37, 0.0, 3.3333333957819775e+37, 4.999999840142846e+37, 3.3333333957819775e+37, 0.0]}
  torch: {"dtype": "float32", "shape": [6], "value": [-1.6666666978909888e+37, 0.0, 3.3333333957819775e+37, 5.000000347203086e+37, 3.3333333957819775e+37, 0.0]}
```

## bug_154 `logsigmoid/generic`

- script: `bug_154_logsigmoid_generic.py`
- count_status: `verified-live`

```text
bug_id: bug_154
key: logsigmoid/generic
status: DIFF
output_source: live
expected_source: majority
expected_libs: keras, tensorflow
expected: {"dtype": "float32", "shape": [2, 3], "value": [[-0.0, -9.999999680285692e+37, -0.6931471824645996], [-0.6931471824645996, -0.6931471824645996, -0.6931471824645996]]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2, 3], "value": [[0.0, -Infinity, -0.6931470036506653], [-0.6931470036506653, -0.6931471824645996, -0.6931471824645996]]}
```

## bug_155 `nanprod/generic`

- script: `bug_155_nanprod_generic.py`
- count_status: `not-counted`

```text
bug_id: bug_155
key: nanprod/generic
status: PASS
output_source: not_live
expected_source: majority
expected_libs: jax, keras, mxnet, numpy
expected: {"dtype": "float32", "shape": [], "value": Infinity}
wrong:
  none
```

## bug_156 `convolve/signal`

- script: `bug_156_convolve_signal.py`
- count_status: `verified-live`

```text
bug_id: bug_156
key: convolve/signal
status: DIFF
output_source: live
expected_source: reference:numpy
expected_libs: numpy
expected: {"dtype": "float32", "shape": [5], "value": [NaN, NaN, NaN, NaN, 0.0]}
wrong:
  jax: {"dtype": "float32", "shape": [5], "value": [NaN, NaN, NaN, NaN, NaN]}
  mindspore: {"dtype": "float32", "shape": [5], "value": [NaN, NaN, NaN, NaN, NaN]}
```

## bug_157 `fftconvolve/signal`

- script: `bug_157_fftconvolve_signal.py`
- count_status: `verified-live`

```text
bug_id: bug_157
key: fftconvolve/signal
status: DIFF
output_source: live
expected_source: reference:scipy
expected_libs: scipy
expected: {"dtype": "float32", "shape": [5], "value": [0.9999999403953552, -1.9999998807907104, 1.0, 4.7683716530855236e-08, -1.9073486612342094e-07]}
wrong:
  jax: {"dtype": "float32", "shape": [5], "value": [0.0, 0.0, 0.0, 0.0, 0.0]}
```

## bug_158 `welch/signal`

- script: `bug_158_welch_signal.py`
- count_status: `verified-live`

```text
bug_id: bug_158
key: welch/signal
status: DIFF
output_source: live
expected_source: reference:scipy
expected_libs: scipy
expected: [{"dtype": "float64", "shape": [3], "value": [0.0, 0.25, 0.5]}, {"dtype": "float32", "shape": [3], "value": [Infinity, Infinity, Infinity]}]
wrong:
  jax: [{"dtype": "float32", "shape": [3], "value": [0.0, 0.25, 0.5]}, {"dtype": "float32", "shape": [3], "value": [Infinity, NaN, Infinity]}]
```

## bug_159 `aminmax/generic/component/1`

- script: `bug_159_aminmax_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_159
key: aminmax/generic/component/1
status: DIFF
output_source: live
expected_source: reference:torch
expected_libs: torch
expected: [{"dtype": "float32", "shape": [], "value": NaN}, {"dtype": "float32", "shape": [], "value": NaN}]
wrong:
  mindspore: [{"dtype": "float32", "shape": [], "value": -0.0}, {"dtype": "float32", "shape": [], "value": 2.0}]
```

## bug_160 `cov/generic`

- script: `bug_160_cov_generic.py`
- count_status: `verified-live`

```text
bug_id: bug_160
key: cov/generic
status: DIFF
output_source: live
expected_source: reference:numpy
expected_libs: numpy
expected: {"dtype": "float64", "shape": [2, 2], "value": [[9.999999360571395e+75, -7.006492097616501e-08], [-7.006492097616501e-08, 6.545457953730302e-91]]}
wrong:
  jax: {"dtype": "float32", "shape": [2, 2], "value": [[Infinity, 0.0], [0.0, 0.0]]}
  torch: {"dtype": "float32", "shape": [2, 2], "value": [[Infinity, -7.006492097616501e-08], [-7.006492097616501e-08, 0.0]]}
errors: {"paddle": "SKIP: paddle.tensor aliases include low-level wrappers; prefer stable paddle top-level APIs"}
```

## bug_161 `dot/generic`

- script: `bug_161_dot_generic.py`
- count_status: `verified-live`

```text
bug_id: bug_161
key: dot/generic
status: DIFF
output_source: live
expected_source: majority
expected_libs: mxnet, numpy, paddle, torch
expected: {"dtype": "float32", "shape": [1], "value": [1.999999761581421]}
wrong:
  jax: {"dtype": "float32", "shape": [], "value": 0.0}
  keras: {"dtype": "float32", "shape": [], "value": 0.0}
errors: {"tensorflow": "SKIP: Keras layer merge wrapper has layer defaults, not raw tensor op semantics"}
```

## bug_162 `factorial/generic/component/1`

- script: `bug_162_factorial_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_162
key: factorial/generic/component/1
status: DIFF
output_source: live
expected_source: reference:scipy
expected_libs: scipy
expected: {"dtype": "float64", "shape": [2, 3], "value": [[Infinity, 0.0, 1.0], [0.0, 1.0, 1.0]]}
wrong:
  jax: {"dtype": "float32", "shape": [2, 3], "value": [[Infinity, 0.0, 1.0000004768371582], [1.0000004768371582, 1.0000004768371582, 1.0000004768371582]]}
```

## bug_163 `inner/generic/component/1`

- script: `bug_163_inner_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_163
key: inner/generic/component/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: mindspore, mxnet, numpy, paddle, torch
expected: {"dtype": "float32", "shape": [], "value": 1.999999761581421}
wrong:
  jax: {"dtype": "float32", "shape": [], "value": 0.0}
  keras: {"dtype": "float32", "shape": [], "value": 0.0}
```

## bug_164 `std_mean/reduction/component/1`

- script: `bug_164_std_mean_reduction_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_164
key: std_mean/reduction/component/1
status: DIFF
output_source: live
expected_source: reference:torch
expected_libs: torch
expected: [{"dtype": "float32", "shape": [2], "value": [8.164965404383231e+37, 0.0]}, {"dtype": "float32", "shape": [2], "value": [0.0, -0.0]}]
wrong:
  mindspore: [{"dtype": "float32", "shape": [2], "value": [Infinity, 0.0]}, {"dtype": "float32", "shape": [2], "value": [0.0, -0.0]}]
```

## bug_165 `topk/generic/component/1`

- script: `bug_165_topk_generic_component_1.py`
- count_status: `verified-live`

```text
bug_id: bug_165
key: topk/generic/component/1
status: DIFF
output_source: live
expected_source: reference:jax
expected_libs: jax
expected: {"dtype": "float32", "shape": [2, 2], "value": [[NaN, 1.0], [2.0, 2.0]]}
wrong:
  keras: {"dtype": "float32", "shape": [2, 2], "value": [[1.0, 1.0], [2.0, 2.0]]}
  mindspore: {"dtype": "float32", "shape": [2, 2], "value": [[1.0, 1.0], [2.0, 2.0]]}
  tensorflow: {"dtype": "float32", "shape": [2, 2], "value": [[1.0, 1.0], [2.0, 2.0]]}
```

## bug_166 `arctan2/generic/2`

- script: `bug_166_arctan2_generic_2.py`
- count_status: `not-counted`

```text
bug_id: bug_166
key: arctan2/generic/2
status: PASS
output_source: not_live
expected_source: majority
expected_libs: chainer, jax, keras, mindspore
expected: {"dtype": "float32", "shape": [3], "value": [1.5707963705062866, -1.570796251296997, 1.5707963705062866]}
wrong:
  none
```

## bug_167 `cumprod/generic/5`

- script: `bug_167_cumprod_generic_5.py`
- count_status: `verified-live`

```text
bug_id: bug_167
key: cumprod/generic/5
status: DIFF
output_source: live
expected_source: majority
expected_libs: numpy, paddle
expected: {"dtype": "float32", "shape": [2, 3], "value": [[9.999999680285692e+37, -Infinity, -Infinity], [-1.401298464324817e-45, -0.0, 0.0]]}
wrong:
  tensorflow: {"dtype": "float32", "shape": [2, 3], "value": [[9.999999680285692e+37, -Infinity, NaN], [-0.0, -0.0, 0.0]]}
```

## bug_168 `floor/generic/2`

- script: `bug_168_floor_generic_2.py`
- count_status: `verified-live`

```text
bug_id: bug_168
key: floor/generic/2
status: DIFF
output_source: live
expected_source: majority
expected_libs: mindspore, mxnet
expected: {"dtype": "float32", "shape": [2, 3], "value": [[9.999999680285692e+37, -9.999999680285692e+37, 0.0], [-1.0, 0.0, -0.0]]}
wrong:
  tensorflow: {"dtype": "float32", "shape": [2, 3], "value": [[9.999999680285692e+37, -9.999999680285692e+37, 0.0], [-0.0, 0.0, -0.0]]}
errors: {"paddle": "SKIP: paddle.tensor aliases include low-level wrappers; prefer stable paddle top-level APIs"}
```

## bug_169 `max/reduction/3`

- script: `bug_169_max_reduction_3.py`
- count_status: `verified-live`

```text
bug_id: bug_169
key: max/reduction/3
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, keras
expected: {"dtype": "float32", "shape": [], "value": NaN}
wrong:
  mindspore: {"dtype": "float32", "shape": [], "value": 2.0}
```

## bug_170 `min/reduction/3`

- script: `bug_170_min_reduction_3.py`
- count_status: `verified-live`

```text
bug_id: bug_170
key: min/reduction/3
status: DIFF
output_source: live
expected_source: majority
expected_libs: chainer, keras
expected: {"dtype": "float32", "shape": [], "value": NaN}
wrong:
  mindspore: {"dtype": "float32", "shape": [], "value": -0.0}
```

## bug_171 `correlate/generic/3`

- script: `bug_171_correlate_generic_3.py`
- count_status: `verified-live`

```text
bug_id: bug_171
key: correlate/generic/3
status: DIFF
output_source: live
expected_source: majority
expected_libs: mindspore, mxnet, numpy
expected: {"dtype": "float32", "shape": [1], "value": [1.999999761581421]}
wrong:
  keras: {"dtype": "float32", "shape": [1], "value": [0.0]}
```

## bug_172 `diag/generic/2`

- script: `bug_172_diag_generic_2.py`
- count_status: `verified-live`

```text
bug_id: bug_172
key: diag/generic/2
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, keras, mxnet, numpy
expected: {"dtype": "float32", "shape": [2], "value": [1.0, 2.0]}
wrong:
  mindspore: {"dtype": "float32", "shape": [2], "value": [NaN, 2.0]}
```

## bug_173 `greater_equal/comparison/3`

- script: `bug_173_greater_equal_comparison_3.py`
- count_status: `verified-live`

```text
bug_id: bug_173
key: greater_equal/comparison/3
status: DIFF
output_source: live
expected_source: majority
expected_libs: mindspore, mxnet
expected: {"dtype": "bool", "shape": [2, 3], "value": [[true, false, true], [false, true, false]]}
wrong:
  tensorflow: {"dtype": "bool", "shape": [2, 3], "value": [[true, false, true], [false, true, true]]}
errors: {"paddle": "SKIP: in-place tensor variants mutate inputs"}
```

## bug_174 `hankel/linalg/2`

- script: `bug_174_hankel_linalg_2.py`
- count_status: `verified-live`

```text
bug_id: bug_174
key: hankel/linalg/2
status: DIFF
output_source: live
expected_source: reference:scipy
expected_libs: scipy
expected: {"dtype": "float32", "shape": [3, 3], "value": [[NaN, Infinity, -0.0], [Infinity, -0.0, 0.0], [-0.0, 0.0, 0.0]]}
wrong:
  jax: {"dtype": "float32", "shape": [3, 3], "value": [[NaN, NaN, NaN], [Infinity, NaN, NaN], [0.0, 0.0, 0.0]]}
```

## bug_175 `hypot/generic/2`

- script: `bug_175_hypot_generic_2.py`
- count_status: `verified-live`

```text
bug_id: bug_175
key: hypot/generic/2
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, mindspore
expected: {"dtype": "float32", "shape": [3], "value": [NaN, Infinity, 0.0]}
wrong:
  keras: {"dtype": "float32", "shape": [3], "value": [NaN, NaN, 0.0]}
```

## bug_176 `left_shift/generic/2`

- script: `bug_176_left_shift_generic_2.py`
- count_status: `verified-live`

```text
bug_id: bug_176
key: left_shift/generic/2
status: DIFF
output_source: live
expected_source: reference:jax
expected_libs: jax
expected: {"dtype": "int32", "shape": [3], "value": [0, -4, 0]}
wrong:
  keras: {"dtype": "int64", "shape": [3], "value": [0, -4, 2]}
  mindspore: {"dtype": "int64", "shape": [3], "value": [0, -4, -9223372036854775808]}
```

## bug_177 `less/comparison/3`

- script: `bug_177_less_comparison_3.py`
- count_status: `verified-live`

```text
bug_id: bug_177
key: less/comparison/3
status: DIFF
output_source: live
expected_source: majority
expected_libs: mindspore, mxnet
expected: {"dtype": "bool", "shape": [2, 3], "value": [[false, true, false], [true, false, true]]}
wrong:
  tensorflow: {"dtype": "bool", "shape": [2, 3], "value": [[false, true, false], [true, false, false]]}
errors: {"paddle": "SKIP: in-place tensor variants mutate inputs"}
```

## bug_178 `nextafter/generic/2`

- script: `bug_178_nextafter_generic_2.py`
- count_status: `verified-live`

```text
bug_id: bug_178
key: nextafter/generic/2
status: DIFF
output_source: live
expected_source: majority
expected_libs: keras, mindspore
expected: {"dtype": "float32", "shape": [3], "value": [NaN, Infinity, 0.0]}
wrong:
  jax: {"dtype": "float32", "shape": [3], "value": [NaN, 3.4028234663852886e+38, 0.0]}
```

## bug_179 `not_equal/comparison/3`

- script: `bug_179_not_equal_comparison_3.py`
- count_status: `verified-live`

```text
bug_id: bug_179
key: not_equal/comparison/3
status: DIFF
output_source: live
expected_source: majority
expected_libs: mindspore, mxnet
expected: {"dtype": "bool", "shape": [2, 3], "value": [[false, true, true], [true, true, true]]}
wrong:
  tensorflow: {"dtype": "bool", "shape": [2, 3], "value": [[false, true, true], [true, true, false]]}
errors: {"paddle": "SKIP: in-place tensor variants mutate inputs"}
```

## bug_180 `outer/generic/2`

- script: `bug_180_outer_generic_2.py`
- count_status: `verified-live`

```text
bug_id: bug_180
key: outer/generic/2
status: DIFF
output_source: live
expected_source: majority
expected_libs: mindspore, mxnet
expected: {"dtype": "float32", "shape": [3, 3], "value": [[0.9999998807907104, -0.9999998807907104, 0.0], [-0.9999998807907104, 0.9999998807907104, 0.0], [0.0, -0.0, 0.0]]}
wrong:
  keras: {"dtype": "float32", "shape": [3, 3], "value": [[0.0, -0.0, -0.0], [-0.0, 0.0, 0.0], [0.0, -0.0, -0.0]]}
```

## bug_181 `outer/linalg/2`

- script: `bug_181_outer_linalg_2.py`
- count_status: `verified-live`

```text
bug_id: bug_181
key: outer/linalg/2
status: DIFF
output_source: live
expected_source: reference:numpy
expected_libs: numpy
expected: {"dtype": "float32", "shape": [3, 3], "value": [[0.9999998807907104, -0.9999998807907104, -0.0], [-0.9999998807907104, 0.9999998807907104, 0.0], [0.0, -0.0, -0.0]]}
wrong:
  jax: {"dtype": "float32", "shape": [3, 3], "value": [[0.0, -0.0, -0.0], [-0.0, 0.0, 0.0], [0.0, -0.0, -0.0]]}
```

## bug_182 `poly/generic/1`

- script: `bug_182_poly_generic_1.py`
- count_status: `verified-live`

```text
bug_id: bug_182
key: poly/generic/1
status: DIFF
output_source: live
expected_source: majority
expected_libs: mxnet, numpy
expected: {"dtype": "float32", "shape": [4], "value": [1.0, -1.401298464324817e-45, -Infinity, Infinity]}
wrong:
  jax: {"dtype": "float32", "shape": [4], "value": [1.0, 0.0, -Infinity, NaN]}
```

## bug_183 `polyval/generic/3`

- script: `bug_183_polyval_generic_3.py`
- count_status: `verified-live`

```text
bug_id: bug_183
key: polyval/generic/3
status: DIFF
output_source: live
expected_source: reference:jax
expected_libs: jax
expected: {"dtype": "float32", "shape": [3], "value": [NaN, NaN, 1.0]}
wrong:
  tensorflow: {"dtype": "float32", "shape": [3], "value": [NaN, Infinity, 1.0]}
```

## bug_184 `ptp/generic/3`

- script: `bug_184_ptp_generic_3.py`
- count_status: `verified-live`

```text
bug_id: bug_184
key: ptp/generic/3
status: DIFF
output_source: live
expected_source: reference:keras
expected_libs: keras
expected: {"dtype": "float32", "shape": [], "value": NaN}
wrong:
  mindspore: {"dtype": "float32", "shape": [], "value": 2.0}
```

## bug_185 `reciprocal/generic/2`

- script: `bug_185_reciprocal_generic_2.py`
- count_status: `verified-live`

```text
bug_id: bug_185
key: reciprocal/generic/2
status: DIFF
output_source: live
expected_source: majority
expected_libs: mindspore, mxnet, paddle
expected: {"dtype": "float32", "shape": [2, 3], "value": [[Infinity, 1.0000000694406173e+38, 1.0], [1.0000000751754868e-38, 0.0625, 0.03999999910593033]]}
wrong:
  tensorflow: {"dtype": "float32", "shape": [2, 3], "value": [[Infinity, Infinity, 1.0], [0.0, 0.0625, 0.03999999910593033]]}
```

## bug_186 `toeplitz/linalg/2`

- script: `bug_186_toeplitz_linalg_2.py`
- count_status: `verified-live`

```text
bug_id: bug_186
key: toeplitz/linalg/2
status: DIFF
output_source: live
expected_source: reference:scipy
expected_libs: scipy
expected: {"dtype": "float32", "shape": [3, 3], "value": [[NaN, Infinity, -0.0], [Infinity, NaN, Infinity], [-0.0, Infinity, NaN]]}
wrong:
  jax: {"dtype": "float32", "shape": [3, 3], "value": [[NaN, NaN, NaN], [NaN, NaN, NaN], [NaN, NaN, NaN]]}
```

## bug_187 `vander/generic/3`

- script: `bug_187_vander_generic_3.py`
- count_status: `verified-live`

```text
bug_id: bug_187
key: vander/generic/3
status: DIFF
output_source: live
expected_source: majority
expected_libs: jax, keras, mindspore
expected: {"dtype": "float32", "shape": [3, 3], "value": [[Infinity, 9.999999680285692e+37, 1.0], [Infinity, -9.999999680285692e+37, 1.0], [0.0, 1.401298464324817e-45, 1.0]]}
wrong:
  mxnet: {"dtype": "float64", "shape": [3, 3], "value": [[9.999999360571395e+75, 9.999999680285692e+37, 1.0], [9.999999360571395e+75, -9.999999680285692e+37, 1.0], [1.9636373861190906e-90, 1.401298464324817e-45, 1.0]]}
  numpy: {"dtype": "float64", "shape": [3, 3], "value": [[9.999999360571395e+75, 9.999999680285692e+37, 1.0], [9.999999360571395e+75, -9.999999680285692e+37, 1.0], [1.9636373861190906e-90, 1.401298464324817e-45, 1.0]]}
```

## bug_188 `vdot/generic/2`

- script: `bug_188_vdot_generic_2.py`
- count_status: `verified-live`

```text
bug_id: bug_188
key: vdot/generic/2
status: DIFF
output_source: live
expected_source: reference:keras
expected_libs: keras
expected: {"dtype": "float32", "shape": [], "value": 0.0}
wrong:
  mxnet: {"dtype": "float32", "shape": [], "value": 1.999999761581421}
```
