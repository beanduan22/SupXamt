# Artifact Guide

This document describes how to set up and run the Xamt++ pipeline. It is
written for reviewers or maintainers who start from a clean checkout.

## Directory Map

```text
tools/api_match_common.py
    Library namespace list, alias map, category terms, parameter-role rules.

tools/compare_api_matchers.py
    API mining, callable metadata extraction, grouping, confidence scoring.

tools/diff_static_candidate_groups.py
    Adapter-aware execution, per-library argument construction, external
    workers, output encoding/normalization, PASS/DIFF/ERROR/SKIP decision.

tools/timed_group_fuzz.py
    Timed fuzzing over adapter-validated groups with random, edge-value, and
    nonfinite input states.
```

## Environment Setup

The full pipeline is intentionally multi-environment because some libraries
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

### External Worker Environments

Create separate environments for libraries that conflict with the main
process. Then point Xamt++ at each worker executable:

```bash
export XAMT_PADDLE_PY=/path/to/paddle-env/bin/python
export XAMT_MINDSPORE_PY=/path/to/mindspore-env/bin/python
export XAMT_CHAINER_PY=/path/to/chainer-env/bin/python
export XAMT_MXNET_PY=/path/to/mxnet-env/bin/python
```

If these variables are not set, `diff_static_candidate_groups.py` falls back
to conventional local worker paths when present. External worker availability
changes coverage: a single-environment run is useful for debugging but does
not exercise the full target set.

## Sanity Check

Run the metadata check first. It only uses the Python standard library.

```bash
cd Xamt
python -B tools/artifact_check.py
```

## Run Static Matching and Validation

To compute candidate groups from installed APIs and validate them through
execution:

```bash
python -B -m tools.diff_static_candidate_groups \
  --strategy pairwise-adapter-aware
```

This command imports all available target libraries and may take a long time
because the `pairwise-adapter-aware` strategy validates executable pairs
before forming connected components.

## Run Timed Fuzzing

Use sharding for complete runs:

```bash
python -B -m tools.timed_group_fuzz \
  --strategy pairwise-adapter-aware \
  --include-edge-values \
  --include-nonfinite \
  --shard-index 0 \
  --shard-count 4 \
  --jsonl results/shard0.jsonl
```

Repeat with the remaining shard indices, then combine the JSONL files with a
stable line concatenation command. The runner emits one JSON object per group
with the final state, per-state counts, the first divergent execution, chosen
APIs, and normalized outputs.

For a quick smoke run, lower `--seconds-per-group` and set `--max-groups` to a
small value.

## Build a Publishable Archive

Use the builder to create an archive that excludes caches, temporary files,
local result directories, and generated backend dumps:

```bash
python -B tools/build_artifact.py
```

The builder runs `tools/artifact_check.py` before creating the archive unless
`--skip-check` is supplied.

## Known Limitations

- Full multi-library reproduction requires external worker environments.
- Execution validation is input-dependent and does not prove semantic
  equivalence over the complete input domain.
