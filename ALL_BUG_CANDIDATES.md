# All Xamt Bug Candidates

> Note: this candidate table comes from a pre-DL-only target snapshot.
> Regenerate it before reporting Xamt++ counts without standalone NumPy/SciPy.

Snapshot: `2026-05-25 14:29:54 AEST`
Count definition: unique executable DIFF group keys; current-code ERROR/SKIP/PASS rows are not counted.

Total current bug candidates: **194**

## Source Counts

| Source | Count |
| --- | ---: |
| alias arity short scan current run | 2 |
| alias category short scan current run | 2 |
| old pairwise DIFF replayed current -> DIFF | 5 |
| pairwise edge+nonfinite current run | 69 |
| post-fix ERROR replay -> DIFF | 2 |
| seed20260539 DIFF | 60 |
| seed20260541 live DIFF (current code) | 2 |
| strict edge-offset current recheck | 6 |
| custom boundary current recheck | 16 |
| old unseen current recheck | 7 |
| broad edge current recheck | 23 |

## Class Counts

| Class | Count |
| --- | ---: |
| NaN/order | 25 |
| activation/nn | 16 |
| adapter-normalization | 1 |
| edge-value | 62 |
| numeric/branch | 16 |
| numeric/linalg | 11 |
| signal/fft | 14 |
| special/edge | 32 |
| value | 17 |

## Selected Next Triage

- `hardshrink/generic/component/1`
- `histc/generic/component/1`
- `logcumsumexp/generic/component/1`
- `nanstd/generic/component/1`
- `nanvar/generic/component/1`
- `softshrink/nn/component/1`
- `unique_consecutive/generic/component/1`
- `ihfft/fft/component/1`
- `sinc/generic/component/1`
- `mode/generic/component/1`
- `softmin/generic/component/1`
- `nancumprod/generic/component/1`

## Candidates

| # | Priority | Key | Class | Confidence | Seed | Libraries | Source |
| ---: | --- | --- | --- | --- | ---: | --- | --- |
| 1 | P0 | `clip/generic/3` | edge-value | medium (0.8318) | 416260555 | chainer, jax, keras, mindspore | seed20260539 DIFF |
| 2 | P0 | `diagflat/generic/2` | edge-value | medium (0.819) | 431260557 | jax, keras, mindspore, mxnet, numpy | seed20260539 DIFF |
| 3 | P0 | `maximum/generic/2` | edge-value | medium (0.8008) | 503260539 | chainer, keras, mindspore, mxnet | seed20260539 DIFF |
| 4 | P0 | `minimum/generic/2` | edge-value | medium (0.8003) | 506260544 | chainer, keras, mindspore, mxnet | seed20260539 DIFF |
| 5 | P0 | `nancumsum/generic/3` | edge-value | high (1.0) | 511260560 | keras, mindspore | seed20260539 DIFF |
| 6 | P0 | `signbit/generic/1` | edge-value | high (1.0) | 541260545 | jax, keras, mindspore | seed20260539 DIFF |
| 7 | P0 | `tril/generic/2` | edge-value | medium (0.8074) | 564260541 | jax, keras, mindspore, mxnet, numpy | seed20260539 DIFF |
| 8 | P1 | `hardshrink/generic/component/1` | activation/nn | medium (0.76) | 150260610 | mindspore, torch | pairwise edge+nonfinite current run |
| 9 | P1 | `softmin/generic/component/1` | activation/nn | medium (0.8014) | 669260601 | mindspore, mxnet | pairwise edge+nonfinite current run |
| 10 | P1 | `softshrink/nn/component/1` | activation/nn | medium (0.7623) | 383260592 | mindspore, paddle, torch | pairwise edge+nonfinite current run |
| 11 | P1 | `angle/generic/2` | edge-value | high (0.9024) | 237260541 | jax, numpy, paddle, tensorflow | seed20260539 DIFF |
| 12 | P1 | `count_nonzero/reduction/3` | edge-value | high (1.0) | 601260541 | jax, mindspore, mxnet, numpy | seed20260539 DIFF |
| 13 | P1 | `nancumprod/generic/component/1` | edge-value | high (0.9902) | 556260612 | jax, keras, mxnet, numpy | pairwise edge+nonfinite current run |
| 14 | P1 | `nanstd/generic/component/1` | edge-value | high (0.8502) | 561260605 | jax, keras, mindspore, numpy | pairwise edge+nonfinite current run |
| 15 | P1 | `nanvar/generic/component/1` | edge-value | medium (0.8488) | 562260599 | jax, keras, mindspore, numpy | pairwise edge+nonfinite current run |
| 16 | P1 | `unique_consecutive/generic/component/1` | edge-value | medium (0.8095) | 332260599 | mindspore, paddle, torch | pairwise edge+nonfinite current run |
| 17 | P1 | `conv_transpose3d/nn/8` | numeric/branch | medium (0.8091) | 284260539 | mindspore, tensorflow | post-fix ERROR replay -> DIFF |
| 18 | P1 | `ihfft/fft/component/1` | signal/fft | high (0.8668) | 432260595 | jax, numpy, paddle, scipy, torch | pairwise edge+nonfinite current run |
| 19 | P1 | `sinc/generic/component/1` | special/edge | high (0.9043) | 290260598 | jax, keras, mindspore, numpy, paddle, scipy, torch | pairwise edge+nonfinite current run |
| 20 | P1 | `histc/generic/component/1` | value | medium (0.8284) | 152260598 | mindspore, torch | pairwise edge+nonfinite current run |
| 21 | P1 | `huber_loss/loss/4` | value | medium (0.8038) | 946260539 | chainer, mindspore | post-fix ERROR replay -> DIFF |
| 22 | P1 | `logcumsumexp/generic/component/1` | value | medium (0.7919) | 199260592 | mindspore, paddle, torch | pairwise edge+nonfinite current run |
| 23 | P1 | `mode/generic/component/1` | value | medium (0.8171) | 225260597 | jax, paddle, torch | pairwise edge+nonfinite current run |
| 24 |  | `argmax/reduction/2` | NaN/order | high (1.0) | 950260549 | chainer, mindspore | seed20260539 DIFF |
| 25 |  | `argmax/reduction/4` | NaN/order | high (0.8843) | 161260540 | jax, numpy, tensorflow | seed20260539 DIFF |
| 26 |  | `argmax/reduction/5` | NaN/order | high (0.925) | 838260553 | mxnet, paddle | seed20260539 DIFF |
| 27 |  | `argmax/reduction/component/1` | NaN/order | medium (0.8329) | 41260590 | chainer, jax, keras, mindspore, mxnet, numpy, paddle, tensorflow, torch | pairwise edge+nonfinite current run |
| 28 |  | `argmin/reduction/3` | NaN/order | high (0.8731) | 398260548 | jax, keras, mindspore, mxnet | seed20260539 DIFF |
| 29 |  | `argmin/reduction/4` | NaN/order | high (0.88) | 162260561 | jax, numpy, tensorflow | seed20260539 DIFF |
| 30 |  | `argmin/reduction/5` | NaN/order | high (0.925) | 839260547 | mxnet, paddle | seed20260539 DIFF |
| 31 |  | `argmin/reduction/component/1` | NaN/order | medium (0.8478) | 42260591 | chainer, jax, keras, mindspore, mxnet, numpy, paddle, tensorflow, torch | pairwise edge+nonfinite current run |
| 32 |  | `argsort/generic/5` | NaN/order | high (0.9146) | 163260555 | numpy, paddle, tensorflow | seed20260539 DIFF |
| 33 |  | `argsort/generic/component/1` | NaN/order | medium (0.8056) | 43260608 | jax, keras, mindspore, mxnet, numpy, paddle, tensorflow, torch | pairwise edge+nonfinite current run |
| 34 |  | `argwhere/generic/component/1` | NaN/order | high (0.8671) | 44260594 | jax, mindspore, mxnet, numpy, paddle, torch | pairwise edge+nonfinite current run |
| 35 |  | `flatnonzero/generic/component/1` | NaN/order | medium (0.8085) | 590260594 | jax, mxnet, numpy | pairwise edge+nonfinite current run |
| 36 |  | `msort/generic/1` | NaN/order | high (1.0) | 941260556 | mindspore, mxnet | seed20260539 DIFF |
| 37 |  | `msort/generic/component/1` | NaN/order | high (0.8864) | 227260608 | mindspore, mxnet, paddle, torch | pairwise edge+nonfinite current run |
| 38 |  | `nonzero/generic/component/1` | NaN/order | medium (0.7973) | 240260594 | jax, keras, mindspore, mxnet, numpy, paddle, torch | pairwise edge+nonfinite current run |
| 39 |  | `sort/generic/4` | NaN/order | medium (0.798) | 223260563 | jax, mindspore, mxnet, tensorflow | seed20260541 live DIFF (current code) |
| 40 |  | `sort/generic/5` | NaN/order | medium (0.8343) | 809260543 | mxnet, numpy, paddle | seed20260539 DIFF |
| 41 |  | `sort/generic/component/1` | NaN/order | medium (0.8008) | 295260591 | jax, keras, mindspore, mxnet, numpy, paddle, tensorflow, torch | pairwise edge+nonfinite current run |
| 42 |  | `gelu/nn/component/1` | activation/nn | high (0.9044) | 360260592 | jax, keras, mindspore, paddle, tensorflow, torch | pairwise edge+nonfinite current run |
| 43 |  | `hardswish/generic/component/1` | activation/nn | high (1.0) | 489260600 | keras, mindspore, tensorflow | pairwise edge+nonfinite current run |
| 44 |  | `hardswish/nn/component/1` | activation/nn | high (0.8882) | 364260591 | jax, mindspore, paddle, torch | pairwise edge+nonfinite current run |
| 45 |  | `log_softmax/nn/component/1` | activation/nn | medium (0.802) | 197260604 | chainer, jax, keras, mindspore, mxnet, paddle, scipy, tensorflow, torch | pairwise edge+nonfinite current run |
| 46 |  | `logsigmoid/nn/component/1` | activation/nn | medium (0.8296) | 370260601 | jax, paddle, torch | pairwise edge+nonfinite current run |
| 47 |  | `mish/generic/component/1` | activation/nn | high (1.0) | 491260611 | keras, mindspore, tensorflow | pairwise edge+nonfinite current run |
| 48 |  | `relu/nn/component/1` | activation/nn | medium (0.7786) | 267260598 | chainer, keras, mindspore, mxnet, paddle, tensorflow, torch | pairwise edge+nonfinite current run |
| 49 |  | `relu6/generic/component/1` | activation/nn | high (1.0) | 492260605 | chainer, keras, mindspore, tensorflow | pairwise edge+nonfinite current run |
| 50 |  | `softmax/nn/component/1` | activation/nn | medium (0.8138) | 293260603 | chainer, jax, keras, mindspore, mxnet, paddle, scipy, tensorflow, torch | pairwise edge+nonfinite current run |
| 51 |  | `softplus/nn/component/1` | activation/nn | medium (0.7624) | 382260598 | chainer, jax, keras, mindspore, paddle, scipy, tensorflow, torch | pairwise edge+nonfinite current run |
| 52 |  | `threshold/generic/component/1` | activation/nn | medium (0.7969) | 317260594 | keras, mindspore, tensorflow, torch | pairwise edge+nonfinite current run |
| 53 |  | `split/shape/4` | adapter-normalization | medium (0.8014) | 234260539 | mxnet, paddle, tensorflow | seed20260539 DIFF |
| 54 |  | `argwhere/generic/1` | edge-value | high (1.0) | 781260543 | mindspore, mxnet, numpy, paddle | seed20260539 DIFF |
| 55 |  | `ceil/generic/1` | edge-value | high (1.0) | 414260539 | chainer, jax, keras, mindspore | seed20260539 DIFF |
| 56 |  | `ceil/generic/2` | edge-value | high (0.9288) | 238260540 | mindspore, mxnet, tensorflow | seed20260539 DIFF |
| 57 |  | `ceil/generic/component/1` | edge-value | high (0.9074) | 67260592 | chainer, jax, keras, mindspore, mxnet, paddle, tensorflow, torch | pairwise edge+nonfinite current run |
| 58 |  | `count_nonzero/reduction/component/1` | edge-value | high (0.8847) | 87260593 | jax, keras, mindspore, mxnet, numpy, paddle, tensorflow, torch | pairwise edge+nonfinite current run |
| 59 |  | `diagflat/generic/component/1` | edge-value | medium (0.8054) | 100260611 | jax, keras, mindspore, mxnet, numpy, paddle, torch | pairwise edge+nonfinite current run |
| 60 |  | `frexp/generic/component/1` | edge-value | medium (0.822) | 133260596 | jax, paddle, torch | pairwise edge+nonfinite current run |
| 61 |  | `hardswish/generic/1` | edge-value | high (1.0) | 314260546 | keras, mindspore, tensorflow | seed20260539 DIFF |
| 62 |  | `hardswish/nn/1` | edge-value | high (1.0) | 722260559 | jax, mindspore | seed20260539 DIFF |
| 63 |  | `heaviside/generic/3` | edge-value | medium (0.7) | 866260546 | mindspore, paddle | seed20260539 DIFF |
| 64 |  | `heaviside/generic/component/1` | edge-value | medium (0.7838) | 151260596 | jax, keras, mindspore, paddle, torch | pairwise edge+nonfinite current run |
| 65 |  | `isclose/generic/component/1` | edge-value | high (0.9274) | 172260593 | jax, keras, mindspore, mxnet, numpy, paddle, torch | pairwise edge+nonfinite current run |
| 66 |  | `log_softmax/nn/2` | edge-value | high (0.8884) | 319260539 | chainer, jax, keras, mindspore, scipy, tensorflow | seed20260539 DIFF |
| 67 |  | `mish/generic/1` | edge-value | high (1.0) | 320260556 | keras, mindspore, tensorflow | seed20260539 DIFF |
| 68 |  | `multiply/generic/2` | edge-value | high (0.8859) | 509260618 | jax, keras, mindspore, mxnet | seed20260539 DIFF |
| 69 |  | `multiply/generic/3` | edge-value | medium (0.818) | 197260650 | mindspore, mxnet, tensorflow | seed20260539 DIFF |
| 70 |  | `multiply/generic/component/1` | edge-value | medium (0.8336) | 228260717 | jax, keras, mindspore, mxnet, paddle, tensorflow, torch | pairwise edge+nonfinite current run |
| 71 |  | `nancumsum/generic/component/1` | edge-value | high (0.9913) | 557260606 | jax, keras, mindspore, mxnet, numpy | pairwise edge+nonfinite current run |
| 72 |  | `nonzero/generic/1` | edge-value | high (1.0) | 517260541 | keras, mxnet, numpy | seed20260539 DIFF |
| 73 |  | `nonzero/generic/3` | edge-value | medium (0.779) | 654260544 | jax, paddle | seed20260539 DIFF |
| 74 |  | `relu/nn/2` | edge-value | medium (0.8296) | 142260543 | mindspore, paddle, tensorflow, torch | seed20260539 DIFF |
| 75 |  | `relu6/generic/1` | edge-value | high (1.0) | 322260567 | chainer, keras, mindspore, tensorflow | seed20260539 DIFF |
| 76 |  | `sign/generic/1` | edge-value | high (1.0) | 540260539 | chainer, jax, keras, mindspore | seed20260539 DIFF |
| 77 |  | `sign/generic/2` | edge-value | high (0.9144) | 219260541 | mindspore, mxnet, tensorflow | seed20260539 DIFF |
| 78 |  | `sign/generic/3` | edge-value | high (1.0) | 909260541 | mxnet, paddle | seed20260539 DIFF |
| 79 |  | `sign/generic/component/1` | edge-value | high (0.9066) | 287260593 | chainer, jax, keras, mindspore, mxnet, paddle, tensorflow, torch | pairwise edge+nonfinite current run |
| 80 |  | `signbit/generic/2` | edge-value | high (0.8518) | 910260540 | mindspore, paddle | seed20260539 DIFF |
| 81 |  | `signbit/generic/component/1` | edge-value | high (0.8811) | 288260595 | jax, keras, mindspore, paddle, torch | pairwise edge+nonfinite current run |
| 82 |  | `softmax/nn/2` | edge-value | high (1.0) | 327260560 | chainer, jax, keras, scipy, tensorflow | seed20260539 DIFF |
| 83 |  | `softmax/nn/3` | edge-value | medium (0.7987) | 261260542 | jax, mindspore, tensorflow | seed20260539 DIFF |
| 84 |  | `tril/generic/component/1` | edge-value | medium (0.7947) | 324260601 | jax, keras, mindspore, mxnet, numpy, paddle, torch | pairwise edge+nonfinite current run |
| 85 |  | `trim_zeros/generic/3` | edge-value | high (1.0) | 689260544 | jax, numpy | seed20260539 DIFF |
| 86 |  | `trim_zeros/generic/component/1` | edge-value | high (1.0) | 630260596 | jax, numpy | pairwise edge+nonfinite current run |
| 87 |  | `triu_indices/generic/component/1` | edge-value | medium (0.834) | 327260590 | jax, mindspore, numpy, paddle | pairwise edge+nonfinite current run |
| 88 |  | `unique/generic/component/1` | edge-value | low (0.6758) | 331260594 | jax, mindspore, mxnet, numpy, paddle, torch | pairwise edge+nonfinite current run |
| 89 |  | `unique_all/generic/component/1` | edge-value | medium (0.716) | 633260593 | jax, numpy | pairwise edge+nonfinite current run |
| 90 |  | `unique_counts/generic/component/1` | edge-value | medium (0.7037) | 634260592 | jax, numpy | pairwise edge+nonfinite current run |
| 91 |  | `unique_inverse/generic/component/1` | edge-value | medium (0.7079) | 635260591 | jax, numpy | pairwise edge+nonfinite current run |
| 92 |  | `unique_values/generic/component/1` | edge-value | medium (0.7006) | 636260590 | jax, numpy | pairwise edge+nonfinite current run |
| 93 |  | `cholesky/linalg/4` | numeric/branch | high (1.0) | 758260539 | jax, mindspore, scipy | seed20260539 DIFF |
| 94 |  | `logm/linalg/2` | numeric/branch | high (0.8502) | 272260768 | scipy, tensorflow | seed20260539 DIFF |
| 95 |  | `lstsq/linalg/5` | numeric/branch | medium (0.7642) | 273261222 | paddle, tensorflow | seed20260539 DIFF |
| 96 |  | `clip_by_norm/linalg/4` | numeric/linalg | medium (0.7) | 170260559 | mindspore, tensorflow | seed20260541 live DIFF (current code) |
| 97 |  | `cond/linalg/component/1` | numeric/linalg | high (0.8986) | 388275511 | jax, mxnet, numpy, paddle, torch | old pairwise DIFF replayed current -> DIFF |
| 98 |  | `geqrf/generic/component/1` | numeric/linalg | medium (0.7689) | 141260595 | mindspore, torch | pairwise edge+nonfinite current run |
| 99 |  | `inv/linalg/component/1` | numeric/linalg | high (0.8505) | 167281393 | chainer, jax, keras, mindspore, mxnet, numpy, paddle, scipy, tensorflow, torch | old pairwise DIFF replayed current -> DIFF |
| 100 |  | `logm/linalg/component/1` | numeric/linalg | high (0.8502) | 472260846 | scipy, tensorflow | pairwise edge+nonfinite current run |
| 101 |  | `lstsq/generic/component/1` | numeric/linalg | high (0.875) | 209288881 | keras, mindspore, torch | old pairwise DIFF replayed current -> DIFF |
| 102 |  | `lstsq/linalg/component/1` | numeric/linalg | medium (0.7847) | 392260837 | jax, mindspore, mxnet, numpy, paddle, scipy, tensorflow, torch | pairwise edge+nonfinite current run |
| 103 |  | `rsf2csf/linalg` | numeric/linalg | high (1.0) | 606260523 | jax, scipy | old pairwise DIFF replayed current -> DIFF |
| 104 |  | `csd/signal/component/1` | signal/fft | high (1.0) | 660260607 | jax, scipy | pairwise edge+nonfinite current run |
| 105 |  | `cbrt/generic/2` | special/edge | medium (0.8333) | 729260539 | jax, mindspore, mxnet | seed20260539 DIFF |
| 106 |  | `cbrt/generic/component/1` | special/edge | high (0.8536) | 543260592 | jax, keras, mindspore, mxnet | pairwise edge+nonfinite current run |
| 107 |  | `digamma/generic/1` | special/edge | high (1.0) | 730260545 | chainer, jax, mindspore | seed20260539 DIFF |
| 108 |  | `digamma/generic/2` | special/edge | high (1.0) | 241260544 | paddle, tensorflow | seed20260539 DIFF |
| 109 |  | `digamma/generic/component/1` | special/edge | high (0.8832) | 103260591 | chainer, jax, mindspore, paddle, tensorflow, torch | pairwise edge+nonfinite current run |
| 110 |  | `erfcinv/generic/component/1` | special/edge | high (0.929) | 464260594 | chainer, tensorflow | pairwise edge+nonfinite current run |
| 111 |  | `gammaln/generic/component/1` | special/edge | high (0.8505) | 412260590 | jax, mxnet, paddle, torch | pairwise edge+nonfinite current run |
| 112 |  | `lgamma/generic/1` | special/edge | high (1.0) | 736260539 | chainer, jax, mindspore | seed20260539 DIFF |
| 113 |  | `lgamma/generic/2` | special/edge | high (1.0) | 246260539 | paddle, tensorflow | seed20260539 DIFF |
| 114 |  | `lgamma/generic/component/1` | special/edge | high (0.9114) | 191260594 | chainer, jax, mindspore, paddle, tensorflow, torch | pairwise edge+nonfinite current run |
| 115 |  | `log10/generic/1` | special/edge | high (1.0) | 490260540 | chainer, jax, keras, mindspore | seed20260539 DIFF |
| 116 |  | `log10/generic/2` | special/edge | high (0.9117) | 881260541 | mindspore, mxnet | seed20260539 DIFF |
| 117 |  | `log10/generic/component/1` | special/edge | high (0.8921) | 194260591 | chainer, jax, keras, mindspore, mxnet, paddle, torch | pairwise edge+nonfinite current run |
| 118 |  | `log2/generic/1` | special/edge | high (1.0) | 492260545 | chainer, jax, keras, mindspore | seed20260539 DIFF |
| 119 |  | `log2/generic/2` | special/edge | high (0.911) | 884260545 | mindspore, mxnet | seed20260539 DIFF |
| 120 |  | `log2/generic/component/1` | special/edge | high (0.8909) | 196260596 | chainer, jax, keras, mindspore, mxnet, paddle, torch | pairwise edge+nonfinite current run |
| 121 |  | `ndtri/generic/1` | special/edge | medium (0.7) | 754260542 | chainer, jax | seed20260539 DIFF |
| 122 |  | `ndtri/generic/component/1` | special/edge | medium (0.7568) | 419260590 | chainer, jax, tensorflow, torch | pairwise edge+nonfinite current run |
| 123 |  | `rsqrt/generic/1` | special/edge | high (1.0) | 537260549 | chainer, keras, mindspore | seed20260539 DIFF |
| 124 |  | `rsqrt/generic/component/1` | special/edge | high (0.8872) | 277260592 | chainer, jax, keras, mindspore, mxnet, paddle, tensorflow, torch | pairwise edge+nonfinite current run |
| 125 |  | `sgn/generic/component/1` | special/edge | medium (0.8465) | 285260598 | mindspore, paddle, torch | pairwise edge+nonfinite current run |
| 126 |  | `xlogy/generic/component/1` | special/edge | medium (0.8189) | 346260590 | mindspore, tensorflow, torch | pairwise edge+nonfinite current run |
| 127 |  | `angle/generic/component/1` | value | medium (0.8452) | 33260591 | jax, keras, numpy, paddle, tensorflow, torch | pairwise edge+nonfinite current run |
| 128 |  | `corrcoef/generic/component/1` | value | medium (0.8071) | 83260590 | jax, keras, mindspore, mxnet, numpy, torch | pairwise edge+nonfinite current run |
| 129 |  | `csd/signal/12` | value | high (1.0) | 773260550 | jax, scipy | seed20260539 DIFF |
| 130 |  | `isclose/generic/5` | value | high (0.9656) | 473260558 | jax, keras, mindspore, mxnet, numpy | seed20260539 DIFF |
| 131 |  | `normalize/generic/component/1` | value | high (0.8955) | 564296018 | chainer, keras | old pairwise DIFF replayed current -> DIFF |
| 132 |  | `percentile/generic/7` | value | high (0.9374) | 658260552 | jax, mxnet | seed20260539 DIFF |
| 133 |  | `percentile/generic/component/1` | value | high (0.9325) | 609260593 | jax, mxnet, numpy | pairwise edge+nonfinite current run |
| 134 |  | `quantile/generic/7` | value | high (0.9231) | 671260543 | jax, mxnet, paddle | seed20260539 DIFF |
| 135 |  | `quantile/generic/component/1` | value | high (0.8588) | 258260606 | jax, keras, mxnet, numpy, paddle, torch | pairwise edge+nonfinite current run |
| 136 |  | `rankdata/generic/4` | value | high (1.0) | 778260539 | jax, scipy | seed20260539 DIFF |
| 137 |  | `rankdata/generic/component/1` | value | high (1.0) | 664260590 | jax, scipy | pairwise edge+nonfinite current run |
| 138 |  | `threshold/generic/3` | value | medium (0.817) | 336260540 | keras, mindspore, tensorflow | seed20260539 DIFF |
| 139 |  | `argmax/reduction/3` | NaN/order | high (0.8731) | 397260612 | jax, keras, mxnet | alias arity short scan current run |
| 140 |  | `float_power/generic/2` | value | high (0.85) | 612260618 | jax, mindspore | alias arity short scan current run |
| 141 |  | `conv_transpose2d/nn` | numeric/branch | medium (0.8169) | 105260613 | paddle, tensorflow, torch | alias category short scan current run |
| 142 |  | `tril_indices/generic` | edge-value | medium (0.7981) | 398260613 | jax, mindspore, numpy, paddle, torch | alias category short scan current run |

| 143 |  | `fft/fft` | signal/fft | medium (0.7831) | 2025378 | chainer, jax, keras, mindspore, numpy, paddle, scipy, tensorflow, torch | strict edge-offset current recheck |
| 144 |  | `ifft/fft` | signal/fft | medium (0.8119) | 2025378 | chainer, jax, mindspore, numpy, paddle, scipy, tensorflow, torch | strict edge-offset current recheck |
| 145 |  | `ifft2/fft` | signal/fft | high (0.8729) | 2020067 | jax, numpy, paddle, scipy, torch | strict edge-offset current recheck |
| 146 |  | `ifftn/fft` | signal/fft | high (0.8697) | 2020067 | jax, numpy, paddle, scipy, torch | strict edge-offset current recheck |
| 147 |  | `rfft/fft` | signal/fft | medium (0.8296) | 20733900 | jax, keras, mindspore, numpy, paddle, scipy, tensorflow, torch | strict edge-offset current recheck |
| 148 |  | `argpartition/generic` | NaN/order | high (0.9059) | 2020067 | jax, mxnet, numpy | strict edge-offset current recheck |

| 149 |  | `divide/generic` | edge-value | medium (0.7672) | custom0 | jax, keras, mindspore, mxnet, paddle, tensorflow, torch | custom boundary current recheck |
| 150 |  | `group_norm/nn` | activation/nn | medium (0.8384) | custom2 | chainer, mindspore, paddle, torch | custom boundary current recheck |
| 151 |  | `log/generic` | special/edge | high (0.8711) | custom0 | chainer, jax, keras, mindspore, mxnet, paddle, tensorflow, torch | custom boundary current recheck |
| 152 |  | `logaddexp/generic` | special/edge | medium (0.808) | custom2 | keras, mindspore, paddle, torch | custom boundary current recheck |
| 153 |  | `logaddexp2/generic` | special/edge | medium (0.7641) | custom2 | keras, mindspore, torch | custom boundary current recheck |
| 154 |  | `renorm/generic` | numeric/branch | medium (0.8432) | custom2 | mindspore, paddle, torch | custom boundary current recheck |
| 155 |  | `bessel_y1/generic` | special/edge | medium (0.7771) | custom0 | mindspore, torch | custom boundary current recheck |
| 156 |  | `fft2/fft` | signal/fft | high (0.8715) | custom2 | jax, numpy, paddle, scipy, torch | custom boundary current recheck |
| 157 |  | `fftn/fft` | signal/fft | high (0.8644) | custom2 | jax, numpy, paddle, scipy, torch | custom boundary current recheck |
| 158 |  | `hfft/fft` | signal/fft | high (0.8608) | custom2 | jax, numpy, paddle, scipy, torch | custom boundary current recheck |
| 159 |  | `irfft/fft` | signal/fft | high (0.8662) | custom2 | jax, numpy, paddle, scipy, torch | custom boundary current recheck |
| 160 |  | `logsigmoid/generic` | activation/nn | high (0.9783) | custom2 | keras, mindspore, tensorflow | custom boundary current recheck |
| 161 |  | `nanprod/generic` | edge-value | high (0.8777) | custom2 | jax, keras, mxnet, numpy | custom boundary current recheck |
| 162 |  | `convolve/signal` | signal/fft | medium (0.7589) | custom0 | jax, mindspore, numpy, scipy | custom boundary current recheck |
| 163 |  | `fftconvolve/signal` | signal/fft | high (1.0) | custom2 | jax, scipy | custom boundary current recheck |
| 164 |  | `welch/signal` | signal/fft | high (1.0) | custom2 | jax, scipy | custom boundary current recheck |

| 165 |  | `aminmax/generic/component/1` | NaN/order | medium (0.7725) | ties_int | mindspore, torch | old unseen current recheck |
| 166 |  | `cov/generic` | numeric/branch | high (0.869) | extreme_finite | jax, mxnet, numpy, torch | old unseen current recheck |
| 167 |  | `dot/generic` | numeric/branch | medium (0.7543) | extreme_finite | jax, keras, mxnet, numpy, paddle, torch | old unseen current recheck |
| 168 |  | `factorial/generic/component/1` | special/edge | high (0.8548) | extreme_finite | jax, scipy | old unseen current recheck |
| 169 |  | `inner/generic/component/1` | numeric/branch | medium (0.8117) | extreme_finite | jax, keras, mindspore, mxnet, numpy, paddle, torch | old unseen current recheck |
| 170 |  | `std_mean/reduction/component/1` | numeric/branch | medium (0.7658) | extreme_finite | mindspore, torch | old unseen current recheck |
| 171 |  | `topk/generic/component/1` | NaN/order | medium (0.8327) | ties_int | jax, keras, mindspore, paddle, tensorflow, torch | old unseen current recheck |

| 172 |  | `arctan2/generic/2` | special/edge | medium (0.785) | extreme_finite | chainer, jax, keras, mindspore | broad edge current recheck |
| 173 |  | `cumprod/generic/5` | edge-value | high (0.8662) | extreme_finite | numpy, paddle, tensorflow | broad edge current recheck |
| 174 |  | `floor/generic/2` | edge-value | high (0.9374) | extreme_finite | mindspore, mxnet, tensorflow | broad edge current recheck |
| 175 |  | `max/reduction/3` | NaN/order | high (1.0) | ties_zero | chainer, keras, mindspore | broad edge current recheck |
| 176 |  | `min/reduction/3` | NaN/order | high (1.0) | ties_zero | chainer, keras, mindspore | broad edge current recheck |
| 177 |  | `correlate/generic/3` | numeric/branch | high (0.8837) | extreme_finite | keras, mindspore, mxnet, numpy | broad edge current recheck |
| 178 |  | `diag/generic/2` | edge-value | high (0.8503) | ties_zero | jax, keras, mindspore, mxnet, numpy | broad edge current recheck |
| 179 |  | `greater_equal/comparison/3` | edge-value | medium (0.8095) | extreme_finite | mindspore, mxnet, tensorflow | broad edge current recheck |
| 180 |  | `hankel/linalg/2` | numeric/linalg | high (1.0) | nan_inf | jax, scipy | broad edge current recheck |
| 181 |  | `hypot/generic/2` | special/edge | medium (0.7603) | nan_inf | jax, keras, mindspore | broad edge current recheck |
| 182 |  | `left_shift/generic/2` | edge-value | high (1.0) | nonneg_shift | jax, keras, mindspore | broad edge current recheck |
| 183 |  | `less/comparison/3` | edge-value | medium (0.825) | extreme_finite | mindspore, mxnet, tensorflow | broad edge current recheck |
| 184 |  | `nextafter/generic/2` | special/edge | medium (0.8187) | nan_inf | jax, keras, mindspore | broad edge current recheck |
| 185 |  | `not_equal/comparison/3` | edge-value | medium (0.8081) | extreme_finite | mindspore, mxnet, tensorflow | broad edge current recheck |
| 186 |  | `outer/generic/2` | numeric/branch | medium (0.7882) | extreme_finite | keras, mindspore, mxnet | broad edge current recheck |
| 187 |  | `outer/linalg/2` | numeric/linalg | high (1.0) | extreme_finite | jax, numpy | broad edge current recheck |
| 188 |  | `poly/generic/1` | numeric/branch | high (1.0) | extreme_finite | jax, mxnet, numpy | broad edge current recheck |
| 189 |  | `polyval/generic/3` | numeric/branch | medium (0.7761) | nan_inf | jax, tensorflow | broad edge current recheck |
| 190 |  | `ptp/generic/3` | NaN/order | high (1.0) | ties_zero | keras, mindspore | broad edge current recheck |
| 191 |  | `reciprocal/generic/2` | special/edge | high (0.9405) | extreme_finite | mindspore, mxnet, paddle, tensorflow | broad edge current recheck |
| 192 |  | `toeplitz/linalg/2` | numeric/linalg | high (1.0) | nan_inf | jax, scipy | broad edge current recheck |
| 193 |  | `vander/generic/3` | numeric/branch | high (1.0) | extreme_finite | jax, keras, mindspore, mxnet, numpy | broad edge current recheck |
| 194 |  | `vdot/generic/2` | numeric/branch | medium (0.732) | extreme_finite | keras, mxnet | broad edge current recheck |

## Not Counted

- Pairwise old DIFF replay under current code: PASS 58, SKIP 2, ERROR 1 were not counted; DIFF 27 were counted.
- Pairwise edge+nonfinite current run: ERROR 3 and SKIP 3 were not counted; DIFF 69 were counted.
- Alias arity short scan current run: ERROR 2 and SKIP 165 were not counted; DIFF 63 duplicates were not added, DIFF 2 new keys were counted.
- Alias category short scan current run: ERROR 213 and SKIP 49 were not counted; DIFF 70 base-duplicate keys were not added, DIFF 2 new base keys were counted.
- `seed20260542` live run currently has DIFF 51; all are duplicate key/base against this table and were not counted.
- Strict edge-offset current recheck: DIFF 6 counted; adapter-normalization/layout or non-deterministic artifacts were rechecked and not counted (`adaptive_max_pool1d`, `bilinear`, `conv_transpose1d`, `fft2/generic`, `ifft2/generic`, `randperm`, `scaled_dot_product_attention`, `scan`).
- Custom boundary current recheck: DIFF 16 counted from crafted nonfinite/subnormal/extreme finite states; output/layout artifacts from the previous round remain excluded.
- Old unseen current recheck: DIFF 7 counted after current-code replay; Keras layer/shape/scalar-return artifacts were excluded (`average/generic`, `concatenate/shape`, `equal/comparison`, `flatten/generic`).
- Broad edge current recheck: DIFF 23 counted from remaining non-random/non-shape groups; `right_shift/generic` was excluded because it only reproduced with a negative shift count.
