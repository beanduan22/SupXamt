# Artifact Guide

This document describes how to evaluate and package the Xamt++ artifact. It is
written for reviewers or maintainers who start from a clean checkout.

## Claims Covered by This Artifact

The artifact is configured for these claims:

- Xamt++ mines APIs from 8 target deep-learning libraries: Chainer, JAX,
  Keras, MindSpore, MXNet, Paddle, TensorFlow, and PyTorch.
- NumPy and SciPy are not target libraries in the Xamt++ method. They may only
  appear as helper/runtime dependencies or inside framework-owned namespaces
  such as `jax.numpy`, `mxnet.numpy`, and `mindspore.scipy`.
- Pairwise group counts, DIFF counts, and replay counts must be regenerated
  after changing the target set before they are reported as DL-only results.

The supporting files are:

- `PAIRWISE_ADAPTER_SUMMARY.md`
- `ALL_BUG_CANDIDATES.md`
- `REAL_BUG_AUDIT.md`
- `bug_repros/README.md`
- `bug_repros/metadata.json`

## Directory Map

```text
tools/api_match_common.py
    Library namespace list, alias map, category terms, parameter role rules.

tools/compare_api_matchers.py
    API mining, callable metadata extraction, grouping, confidence scoring.

tools/diff_static_candidate_groups.py
    Adapter-aware execution, per-library argument construction, external
    workers, output encoding/normalization, PASS/DIFF/ERROR/SKIP decision.

tools/timed_group_fuzz.py
    Timed fuzzing over adapter-validated groups with random, edge-value, and
    nonfinite input states.

bug_repros/
    Curated one-command replayers and metadata for audited DIFF candidates.
```

## Environment Setup

The full experiment is intentionally multi-environment because some libraries
have incompatible Python and dependency constraints.

### Main Environment

The main process needs helper packages plus the target DL libraries that can
coexist in the same Python environment:

```bash
python -m venv .venv-main
. .venv-main/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-main.txt
```

The observed local main environment used for the current snapshot is listed in
`requirements-main.txt`.

### External Worker Environments

Create separate environments for libraries that conflict with the main process.
Then point Xamt++ at each worker executable:

```bash
export XAMT_PADDLE_PY=/path/to/paddle-env/bin/python
export XAMT_MINDSPORE_PY=/path/to/mindspore-env/bin/python
export XAMT_CHAINER_PY=/path/to/chainer-env/bin/python
export XAMT_MXNET_PY=/path/to/mxnet-env/bin/python
```

If these variables are not set, `diff_static_candidate_groups.py` falls back to
these conventional paths when present:

```text
/tmp/xamt_py312/bin/python  # paddle, mindspore
/tmp/xamt_py39/bin/python   # chainer, mxnet
```

External worker availability changes coverage. A single-environment run is
useful for debugging but should not be used for the full 8-library DL count.

## Sanity Check

Run the metadata check first. It only uses the Python standard library.

```bash
cd Xamt
python -B tools/artifact_check.py
```

Expected high-level output confirms the configured target library count and
checks that generated replay metadata is internally consistent.

## Reproduce Static Matching Coverage

The fastest way to inspect recorded coverage is to read the summary after it
has been regenerated for the current DL-only target set:

```bash
sed -n '1,35p' PAIRWISE_ADAPTER_SUMMARY.md
```

To recompute groups from installed APIs:

```bash
python -B -m tools.diff_static_candidate_groups \
  --strategy pairwise-adapter-aware \
  --details 20
```

This command imports all available target libraries and may take a long time
because the `pairwise-adapter-aware` strategy validates executable pairs before
forming connected components.

## Reproduce Timed Fuzzing

A full 60-second-per-group run is expensive. Use sharding for complete runs:

```bash
python -B -m tools.timed_group_fuzz \
  --strategy pairwise-adapter-aware \
  --seconds-per-group 60 \
  --include-edge-values \
  --include-nonfinite \
  --stop-on-diff \
  --shard-index 0 \
  --shard-count 4 \
  --jsonl results/shard0.jsonl
```

Repeat with `--shard-index 1`, `2`, and `3`, then combine JSONL files with a
stable line concatenation command. The runner emits one JSON object per group
with `final_state`, `counts`, `first_bad`, chosen APIs, and normalized outputs.

For a quick smoke run:

```bash
python -B -m tools.timed_group_fuzz \
  --strategy pairwise-adapter-aware \
  --seconds-per-group 1 \
  --max-groups 5 \
  --include-edge-values \
  --include-nonfinite \
  --stop-on-diff
```

## Replay Curated Bugs

Each bug script is intentionally tiny:

```bash
python -B bug_repros/bug_001_clip_generic_3.py
```

To audit all scripts, run them from `bug_repros/` in a configured environment
and count only live `status: DIFF` outputs. The current audited count is in
`bug_repros/README.md`.

## Build a Publishable Archive

Use the builder to create an archive that excludes caches, temporary files,
local result directories, and generated backend dumps:

```bash
python -B tools/build_artifact.py --out dist/xamtplusplus-artifact.tar.gz
```

The builder runs `tools/artifact_check.py` before creating the archive unless
`--skip-check` is supplied.

## Known Limitations

- The result JSONL files from some historical `/tmp` runs are summarized in
  markdown but not all raw shard files are kept in this source artifact.
- Full 8-library DL reproduction requires external worker environments.
- Some DIFF rows are edge-value or numerical-branch behavior and require
  manual documentation review before filing upstream bugs.
- `ERROR` rows in triage summaries are mostly adapter/input-plan work or
  backend limitations; they are not counted as bugs.
