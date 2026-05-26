# Xamt++: Adapter-Aware Cross-Framework API Differential Testing

Xamt++ is the release artifact for the current cross-framework API matching and
differential testing pipeline. It extends the original XAMT prototype from a
small set of framework shims to an adapter-aware matcher and runner covering
8 deep-learning libraries:

`chainer`, `jax`, `keras`, `mindspore`, `mxnet`, `paddle`, `tensorflow`,
and `torch`.

NumPy and SciPy are not standalone target libraries in the Xamt++ method. They
may still appear as runtime helpers or as framework-owned namespaces such as
`jax.numpy`, `mxnet.numpy`, or `mindspore.scipy`.

The artifact contains the code used to mine APIs, form cross-library
equivalence groups, validate those groups through adapters, fuzz the executable
groups, and replay the current differential bug reproductions.

## Artifact Contents

```text
Xamt/
|- tools/
|  |- api_match_common.py              # namespaces, aliases, categories, role rules
|  |- compare_api_matchers.py          # API mining, grouping, confidence scoring
|  |- diff_static_candidate_groups.py  # adapters, execution, output normalization
|  |- timed_group_fuzz.py              # timed fuzzing over matched groups
|  |- artifact_check.py                # metadata and packaging sanity checks
|  `- build_artifact.py                # reproducible source-archive builder
|- bug_repros/
|  |- metadata.json                    # 188 curated reproduction records
|  |- bug_001_*.py ... bug_188_*.py    # one-command live replayers
|  `- README.md                        # current live replay audit
|- *_CANDIDATES.md, *_AUDIT.md         # result summaries and triage notes
|- ARTIFACT.md                         # step-by-step reproduction guide
|- RELEASE_MANIFEST.md                 # publication file manifest
`- README.md
```

The older `functions/`, `inputs/`, `run_tasks/`, `tests/`, and `utilities/`
directories are retained for compatibility with the original XAMT artifact.
The current Xamt++ pipeline is driven by `tools/` and `bug_repros/`.

## Current Coverage and Result Snapshot

Current artifact snapshot: `2026-05-25`.

| Metric | Count |
| --- | --- |
| Target DL libraries | 8 |
| Pairwise adapter-aware groups | regenerate for DL-only target set |
| Unique APIs in executable groups | regenerate for DL-only target set |
| Raw mined APIs in target libraries | 7780 |
| Static pairwise DIFF groups | regenerate for DL-only target set |
| Current unique DIFF candidates | regenerate for DL-only target set |
| Recommended likely real bugs | regenerate for DL-only target set |
| Strict reportable bugs | regenerate for DL-only target set |
| Current live reproducible DIFF scripts | regenerate for DL-only target set |

The count definitions and evidence are recorded in
`PAIRWISE_ADAPTER_SUMMARY.md`, `ALL_BUG_CANDIDATES.md`,
`REAL_BUG_AUDIT.md`, and `bug_repros/README.md`.

## Environment Model

The full 8-library DL run uses one main Python process plus optional external
Python workers for libraries that conflict with the main environment.

Main target libraries:

- `torch`
- `tensorflow`
- `keras`
- `jax`

Helper dependencies, not target libraries:

- `numpy`
- `scipy`

External workers:

- `XAMT_PADDLE_PY` for Paddle
- `XAMT_MINDSPORE_PY` for MindSpore
- `XAMT_CHAINER_PY` for Chainer
- `XAMT_MXNET_PY` for MXNet

If an external worker is not configured, the runner skips that library during
API collection or reports it as unavailable during execution. This lets partial
runs work in a single environment while keeping the full artifact reproducible
on a machine with all worker environments installed.

Observed local main environment (target libraries plus helpers):

```text
Python 3.13.5
numpy 2.2.6
scipy 1.16.3
torch 2.11.0+cu128
tensorflow 2.21.0
keras 3.14.0
jax 0.9.2
```

For a new machine, start with `requirements-main.txt`, then create external
worker environments for Paddle, MindSpore, Chainer, and MXNet as described in
`ARTIFACT.md`.

## Quick Start

Run the artifact sanity check. This does not import heavyweight DL libraries.

```bash
cd Xamt
python -B tools/artifact_check.py
```

Inspect matched API coverage after regenerating the summary for the DL-only target set.

```bash
sed -n '1,35p' PAIRWISE_ADAPTER_SUMMARY.md
```

Replay one curated bug. The exact result depends on which libraries are
installed and which external worker variables are set.

```bash
python -B bug_repros/bug_001_clip_generic_3.py
```

Run the deterministic adapter-aware validation over mined APIs.

```bash
python -B -m tools.diff_static_candidate_groups \
  --strategy pairwise-adapter-aware \
  --details 20
```

Run a timed fuzzing pass over already matched groups.

```bash
python -B -m tools.timed_group_fuzz \
  --strategy pairwise-adapter-aware \
  --seconds-per-group 60 \
  --include-edge-values \
  --include-nonfinite \
  --stop-on-diff \
  --jsonl results/pairwise_edge_nonfinite.jsonl
```

Build a clean source artifact archive.

```bash
python -B tools/build_artifact.py --out dist/xamtplusplus-artifact.tar.gz
```

## How the Pipeline Works

1. `api_match_common.py` defines the target namespaces, alias rules, category
   terms, and parameter role normalization.
2. `compare_api_matchers.py` mines callables, records signatures/docs, groups
   APIs by canonical name/category/arity/roles, and assigns confidence scores.
3. `diff_static_candidate_groups.py` maps a group to executable inputs through
   per-library adapters, normalizes outputs, and labels the group as `PASS`,
   `DIFF`, `ERROR`, or `SKIP`.
4. The `pairwise-adapter-aware` strategy validates executable pair matches and
   unions passing pairs into connected API components.
5. `timed_group_fuzz.py` refreshes canonical inputs with random, edge-value,
   and nonfinite states, then repeatedly executes each group within a time
   budget.
6. `bug_repros/` replays curated candidates against the current runner and
   counts only live `status: DIFF` outputs as reproducible bugs.

## Reporting Counts

Use the following default language when reporting this artifact:

> Xamt++ targets 8 deep-learning libraries: Chainer, JAX, Keras, MindSpore,
> MXNet, Paddle, TensorFlow, and PyTorch. NumPy and SciPy are helper/reference
> packages only and are not counted as target libraries. Regenerate the
> pairwise summary and bug-candidate tables after changing the target set before
> reporting final DL-only counts.

## Citation

```bibtex
@INPROCEEDINGS{Xamt,
  author={Duan, Bin and Dong, Ruican and Dong, Naipeng and Kim, Dan Dongseong and Yang, Guowei},
  booktitle={2025 IEEE 36th International Symposium on Software Reliability Engineering (ISSRE)},
  title={XAMT: Cross-Framework API Matching for Testing Deep Learning Libraries},
  year={2025},
  pages={191-202},
  keywords={Fuzzing;Software reliability;Testing;Deep Learning Libraries},
  doi={10.1109/ISSRE66568.2025.00030}}
```
