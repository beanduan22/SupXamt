# Xamt: Cross-Framework API Differential Testing

Xamt is a cross-framework API matching and differential testing pipeline for
deep-learning libraries. The target libraries are Chainer, JAX, Keras,
MindSpore, MXNet, PaddlePaddle, TensorFlow, and PyTorch.

NumPy and SciPy are not standalone target libraries in the Xamt method. They
may appear as runtime helpers or as framework-owned namespaces such as
`jax.numpy`, `mxnet.numpy`, or `mindspore.scipy`.

The artifact contains the code used to mine APIs, form cross-library
equivalence groups, validate those groups through adapters, and fuzz the
executable groups.

## Artifact Contents

```text
Xamt/
|- tools/
|  |- api_match_common.py
|  |- compare_api_matchers.py
|  |- diff_static_candidate_groups.py
|  |- timed_group_fuzz.py
|  |- artifact_check.py
|  `- build_artifact.py
|- ARTIFACT.md
`- README.md
```

`api_match_common.py` defines the target namespaces, API name aliases,
operation category terms, and parameter-role normalization rules.
`compare_api_matchers.py` mines callables, records signatures and
documentation, groups APIs by canonical name and category, and assigns
confidence scores. `diff_static_candidate_groups.py` maps a group to
executable inputs through per-library adapters, normalizes outputs, and labels
each execution. `timed_group_fuzz.py` runs timed fuzzing over
adapter-validated groups.

The older `functions/`, `inputs/`, `run_tasks/`, `tests/`, and `utilities/`
directories are retained for compatibility with the original XAMT artifact.
The current Xamt pipeline is driven by `tools/`.

## Environment Model

The pipeline uses one main Python process plus optional external Python
workers for libraries that conflict with the main environment.

Main-process target libraries:

- `torch`
- `tensorflow`
- `keras`
- `jax`

Helper dependencies, not target libraries:

- `numpy`
- `scipy`

External workers:

- `XAMT_PADDLE_PY` for PaddlePaddle
- `XAMT_MINDSPORE_PY` for MindSpore
- `XAMT_CHAINER_PY` for Chainer
- `XAMT_MXNET_PY` for MXNet

If an external worker is not configured, the runner skips that library during
API collection or reports it as unavailable during execution. This lets
partial runs work in a single environment while keeping the full pipeline
reproducible on a machine with all worker environments installed.

For a new machine, start with `requirements-main.txt`, then create external
worker environments for PaddlePaddle, MindSpore, Chainer, and MXNet as
described in `ARTIFACT.md`.

## Quick Start

Run the artifact sanity check. This does not import heavyweight DL libraries.

```bash
cd Xamt
python -B tools/artifact_check.py
```

Run the deterministic adapter-aware validation over mined APIs.

```bash
python -B -m tools.diff_static_candidate_groups \
  --strategy pairwise-adapter-aware
```

Run a timed fuzzing pass over matched groups.

```bash
python -B -m tools.timed_group_fuzz \
  --strategy pairwise-adapter-aware \
  --include-edge-values \
  --include-nonfinite
```

Build a clean source artifact archive.

```bash
python -B tools/build_artifact.py
```

## How the Pipeline Works

1. `api_match_common.py` defines the target namespaces, alias rules, category
   terms, and parameter-role normalization.
2. `compare_api_matchers.py` mines callables, records signatures and docs,
   groups APIs by canonical name and category, and assigns confidence scores.
3. `diff_static_candidate_groups.py` maps a group to executable inputs through
   per-library adapters, normalizes outputs, and labels the execution as
   `PASS`, `DIFF`, `ERROR`, or `SKIP`.
4. The `pairwise-adapter-aware` strategy validates executable pair matches and
   unions passing pairs into connected API components, which are re-validated
   at the group level.
5. `timed_group_fuzz.py` refreshes canonical inputs with random, edge-value,
   and nonfinite states, then repeatedly executes each group within a
   per-group time budget.

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
