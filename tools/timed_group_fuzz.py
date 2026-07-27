
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from typing import Any, Optional

import numpy as np

try:
    from .compare_api_matchers import collect_apis, confidence_band, group_confidence
    from . import diff_static_candidate_groups as runner
except ImportError:
    from compare_api_matchers import collect_apis, confidence_band, group_confidence
    import diff_static_candidate_groups as runner


def softmax_rows(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32)


def inject_edges(value: np.ndarray, seed: int, *, include_edge_values: bool = False, include_nonfinite: bool = False) -> np.ndarray:
    result = value.copy()
    if result.size == 0:
        return result
    if include_edge_values and seed % 7 == 0:
        edges = np.array([-0.0, 0.0, np.float32(1.401298464e-45), -1.0, 1.0, np.float32(1e-6)], dtype=np.float32)
        flat = result.reshape(-1)
        flat[: min(flat.size, edges.size)] = edges[: min(flat.size, edges.size)]
    if include_nonfinite and seed % 23 == 0 and result.size >= 4:
        flat = result.reshape(-1)
        flat[0] = np.nan
        flat[1] = np.inf
        flat[2] = -np.inf
    return result


def install_fuzz_state(seed: int, *, include_edge_values: bool = False, include_nonfinite: bool = False) -> None:
    rng = np.random.default_rng(seed)
    scale = float(10 ** rng.uniform(-1.0, 1.0))

    x = (rng.normal(0.0, scale, size=(2, 3))).astype(np.float32)
    y = (rng.normal(0.0, scale, size=(2, 3))).astype(np.float32)
    pos = (rng.lognormal(mean=0.0, sigma=1.0, size=(2, 3)) + 1e-4).astype(np.float32)
    vec = (rng.normal(0.0, scale, size=4)).astype(np.float32)
    vec_a = (rng.normal(0.0, scale, size=3)).astype(np.float32)
    vec_b = (rng.normal(0.0, scale, size=3)).astype(np.float32)
    even = (rng.normal(0.0, scale, size=(2, 4))).astype(np.float32)

    mat = (rng.normal(0.0, scale, size=(2, 2)) + np.eye(2) * 2.0).astype(np.float32)
    mat_b = (rng.normal(0.0, scale, size=(2, 2)) + np.eye(2)).astype(np.float32)
    a = rng.normal(0.0, 1.0, size=(2, 2)).astype(np.float32)
    spd = (a.T @ a + np.eye(2, dtype=np.float32) * 0.5).astype(np.float32)
    rhs = rng.normal(0.0, scale, size=(2, 1)).astype(np.float32)

    y_true = rng.integers(0, 2, size=(2, 3)).astype(np.float32)
    y_pred = rng.uniform(0.05, 0.95, size=(2, 3)).astype(np.float32)
    labels = rng.integers(0, 3, size=2)
    cat_true = np.eye(3, dtype=np.float32)[labels]
    cat_pred = softmax_rows(rng.normal(size=(2, 3)).astype(np.float32))

    state = {
        "X": inject_edges(x, seed, include_edge_values=include_edge_values, include_nonfinite=include_nonfinite),
        "Y": inject_edges(y, seed + 1, include_edge_values=include_edge_values, include_nonfinite=include_nonfinite),
        "POS": pos,
        "VEC": inject_edges(vec, seed + 2, include_edge_values=include_edge_values, include_nonfinite=include_nonfinite),
        "INT_VEC": rng.integers(0, 4, size=6, dtype=np.int64),
        "BOOL": rng.random(size=(2, 3)) > 0.5,
        "MAT": mat,
        "MAT_B": mat_b,
        "SPD": spd,
        "RHS": rhs,
        "VEC_A": inject_edges(vec_a, seed + 3, include_edge_values=include_edge_values, include_nonfinite=include_nonfinite),
        "VEC_B": inject_edges(vec_b, seed + 4, include_edge_values=include_edge_values, include_nonfinite=include_nonfinite),
        "INT_A": rng.integers(1, 6, size=3, dtype=np.int64),
        "INT_B": rng.integers(1, 4, size=3, dtype=np.int64),
        "EVEN": inject_edges(even, seed + 5, include_edge_values=include_edge_values, include_nonfinite=include_nonfinite),
        "Y_TRUE": y_true,
        "Y_PRED": y_pred,
        "CAT_TRUE": cat_true,
        "CAT_PRED": cat_pred,
        "SPARSE_TRUE": labels.astype(np.int64),
    }
    for name, value in state.items():
        setattr(runner, name, value)


def compact_outputs(data: dict[str, Any]) -> dict[str, Any]:
    chosen = data.get("chosen", {})
    outputs = data.get("outputs", {})
    return {
        "chosen": chosen,
        "outputs": outputs,
        "skipped": data.get("skipped", {}),
    }


def fuzz_group(
    key: tuple[Any, ...],
    group: list[Any],
    *,
    seconds: float,
    seed: int,
    stop_on_diff: bool,
    include_nonfinite: bool,
    include_edge_values: bool,
) -> dict[str, Any]:
    deadline = time.monotonic() + seconds
    iterations = 0
    counts: Counter[str] = Counter()
    first_bad: Optional[dict[str, Any]] = None
    last_state = "UNKNOWN"

    while True:
        now = time.monotonic()
        if iterations > 0 and now >= deadline:
            break
        current_seed = seed + iterations
        install_fuzz_state(current_seed, include_edge_values=include_edge_values, include_nonfinite=include_nonfinite)
        state, data, errors = runner.run_group(key, group)
        counts[state] += 1
        last_state = state
        iterations += 1
        if state in {"DIFF", "ERROR"} and first_bad is None:
            first_bad = {
                "state": state,
                "seed": current_seed,
                "errors": errors,
                "data": compact_outputs(data),
            }
            if stop_on_diff:
                break

    if iterations == 0:
        install_fuzz_state(seed, include_edge_values=include_edge_values, include_nonfinite=include_nonfinite)
        state, data, errors = runner.run_group(key, group)
        counts[state] += 1
        last_state = state
        iterations = 1
        if state in {"DIFF", "ERROR"}:
            first_bad = {
                "state": state,
                "seed": seed,
                "errors": errors,
                "data": compact_outputs(data),
            }

    return {
        "key": list(key),
        "iterations": iterations,
        "counts": dict(sorted(counts.items())),
        "final_state": "DIFF" if counts.get("DIFF") else "ERROR" if counts.get("ERROR") else last_state,
        "confidence": confidence_band(group_confidence(group)),
        "confidence_score": round(group_confidence(group), 4),
        "apis": [api.qualified_name for api in group],
        "first_bad": first_bad,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="adapter-aware", choices=("alias-category-arity", "alias-category", "role-aware", "role-adapter-aware", "adapter-aware", "pairwise-adapter-aware"))
    parser.add_argument("--seconds-per-group", type=float, default=60.0)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--start-group", type=int, default=1, help="1-based local group index after sharding")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index to run")
    parser.add_argument("--shard-count", type=int, default=1, help="total number of shards")
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--stop-on-diff", action="store_true")
    parser.add_argument("--include-nonfinite", action="store_true")
    parser.add_argument("--include-edge-values", action="store_true", help="inject signed zero/subnormal/tie values during timed fuzzing")
    parser.add_argument("--jsonl", default="")
    parser.add_argument("--groups-jsonl", default="", help="precomputed group list from deterministic validation")
    parser.add_argument("--summary-every", type=int, default=25)
    args = parser.parse_args()

    apis = collect_apis()
    if args.groups_jsonl:
        by_qname = {api.qualified_name: api for api in apis}
        loaded_groups = []
        with open(args.groups_jsonl, encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                group = [by_qname[qname] for qname in row["apis"] if qname in by_qname]
                if len({api.library for api in group}) >= 2:
                    loaded_groups.append((tuple(row["key"]), group))
        groups = list(enumerate(loaded_groups, start=1))
    else:
        groups = list(enumerate(runner.build_candidate_groups(apis, args.strategy), start=1))
    if args.shard_count < 1:
        raise SystemExit("--shard-count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index must satisfy 0 <= index < shard-count")
    if args.shard_count > 1:
        groups = [item for item in groups if (item[0] - 1) % args.shard_count == args.shard_index]
    if args.start_group < 1:
        raise SystemExit("--start-group must be >= 1")
    if args.start_group > 1:
        groups = groups[args.start_group - 1:]
    if args.max_groups > 0:
        groups = groups[: args.max_groups]

    totals: Counter[str] = Counter()
    iteration_total = 0
    bad_groups: list[dict[str, Any]] = []
    output = open(args.jsonl, "w", encoding="utf-8") if args.jsonl else None
    started = time.monotonic()
    try:
        for idx, (original_index, (key, group)) in enumerate(groups, start=1):
            result = fuzz_group(
                key,
                group,
                seconds=args.seconds_per_group,
                seed=args.seed + original_index * 1_000_000,
                stop_on_diff=args.stop_on_diff,
                include_nonfinite=args.include_nonfinite,
                include_edge_values=args.include_edge_values,
            )
            iteration_total += result["iterations"]
            final_state = result["final_state"]
            totals[final_state] += 1
            if result["first_bad"] is not None:
                bad_groups.append(result)
            if output is not None:
                output.write(json.dumps(result, sort_keys=True) + "\n")
                output.flush()
            if args.summary_every and (idx == 1 or idx % args.summary_every == 0 or idx == len(groups)):
                elapsed = time.monotonic() - started
                print(
                    f"progress {idx}/{len(groups)} elapsed={elapsed:.1f}s "
                    f"totals={dict(sorted(totals.items()))} iterations={iteration_total}",
                    flush=True,
                )
    finally:
        if output is not None:
            output.close()
        runner.stop_external_workers()

    print("strategy:", args.strategy)
    print("shard_index:", args.shard_index)
    print("shard_count:", args.shard_count)
    print("start_group:", args.start_group)
    print("groups:", len(groups))
    print("seconds_per_group:", args.seconds_per_group)
    print("include_nonfinite:", args.include_nonfinite)
    print("include_edge_values:", args.include_edge_values)
    print("iterations:", iteration_total)
    print("summary:", json.dumps(dict(sorted(totals.items())), sort_keys=True))
    print("bad_groups:", len(bad_groups))
    for item in bad_groups[:20]:
        first_bad = item["first_bad"] or {}
        print(f"BAD {item['final_state']} seed={first_bad.get('seed')} key={tuple(item['key'])}")
        if first_bad.get("errors"):
            print("  errors:", json.dumps(first_bad["errors"], sort_keys=True)[:1200])
        chosen = (first_bad.get("data") or {}).get("chosen", {})
        if chosen:
            print("  chosen:", " | ".join(f"{lib}:{api}" for lib, api in sorted(chosen.items())))


if __name__ == "__main__":
    main()
