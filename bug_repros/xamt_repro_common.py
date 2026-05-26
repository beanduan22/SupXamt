import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

import diff_static_candidate_groups as runner
from compare_api_matchers import Api, collect_apis
from timed_group_fuzz import install_fuzz_state

REFERENCE_ORDER = ["jax", "tensorflow", "torch", "paddle", "keras", "chainer", "mxnet", "mindspore"]
API_BY_NAME = None


def metadata():
    with open(Path(__file__).with_name("metadata.json"), encoding="utf-8") as source:
        return json.load(source)


def record_by_id(bug_id):
    for record in metadata()["records"]:
        if record["id"] == bug_id:
            return record
    raise SystemExit(f"unknown bug id: {bug_id}")


def restore_state(snapshot):
    for name, value in snapshot.items():
        setattr(runner, name, value.copy() if hasattr(value, "copy") else value)


def set_state(token):
    if token is None:
        return
    if isinstance(token, int) or (isinstance(token, str) and token.isdigit()):
        install_fuzz_state(int(token), include_edge_values=True, include_nonfinite=True)
        return
    tiny = np.float32(1.401298464e-45)
    states = {
        "nan_inf": {
            "X": np.array([[np.nan, np.inf, -np.inf], [0.0, -0.0, 1.0]], dtype=np.float32),
            "Y": np.array([[1.0, -1.0, np.nan], [np.inf, -np.inf, 0.0]], dtype=np.float32),
            "POS": np.array([[0.0, tiny, 1.0], [np.inf, np.float32(1e-30), 4.0]], dtype=np.float32),
            "VEC": np.array([np.nan, np.inf, -np.inf, -0.0], dtype=np.float32),
            "VEC_A": np.array([np.nan, np.inf, -0.0], dtype=np.float32),
            "VEC_B": np.array([1.0, -np.inf, 0.0], dtype=np.float32),
            "EVEN": np.array([[np.nan, np.inf, -np.inf, -0.0], [0.0, 1.0, -1.0, tiny]], dtype=np.float32),
        },
        "extreme_finite": {
            "X": np.array([[np.float32(1e38), np.float32(-1e38), tiny], [-tiny, 0.0, -0.0]], dtype=np.float32),
            "Y": np.array([[np.float32(1e38), np.float32(1e-38), -1.0], [1.0, np.float32(-1e38), tiny]], dtype=np.float32),
            "POS": np.array([[tiny, np.float32(1e-38), 1.0], [np.float32(1e38), 16.0, 25.0]], dtype=np.float32),
            "VEC": np.array([np.float32(1e38), np.float32(-1e38), tiny, -0.0], dtype=np.float32),
            "VEC_A": np.array([np.float32(1e38), np.float32(-1e38), tiny], dtype=np.float32),
            "VEC_B": np.array([np.float32(1e-38), np.float32(-1e-38), -0.0], dtype=np.float32),
            "EVEN": np.array([[np.float32(1e38), np.float32(-1e38), tiny, -0.0], [0.0, 1.0, -1.0, np.float32(1e-38)]], dtype=np.float32),
        },
        "ties_zero": {
            "X": np.array([[1.0, 1.0, np.nan], [2.0, 2.0, -0.0]], dtype=np.float32),
            "Y": np.array([[1.0, -1.0, 0.0], [2.0, -2.0, np.nan]], dtype=np.float32),
            "VEC": np.array([1.0, 1.0, np.nan, -0.0], dtype=np.float32),
            "VEC_A": np.array([1.0, 1.0, np.nan], dtype=np.float32),
            "VEC_B": np.array([1.0, -1.0, 0.0], dtype=np.float32),
            "INT_VEC": np.array([2, 2, 1, 1, 0, 0], dtype=np.int64),
            "INT_A": np.array([0, -1, 2], dtype=np.int64),
            "INT_B": np.array([0, 2, -2], dtype=np.int64),
        },
        "ties_int": {
            "X": np.array([[1.0, 1.0, np.nan], [2.0, 2.0, -0.0]], dtype=np.float32),
            "Y": np.array([[1.0, -1.0, 0.0], [2.0, -2.0, np.nan]], dtype=np.float32),
            "VEC": np.array([1.0, 1.0, np.nan, -0.0], dtype=np.float32),
            "VEC_A": np.array([1.0, 1.0, np.nan], dtype=np.float32),
            "VEC_B": np.array([1.0, -1.0, 0.0], dtype=np.float32),
            "INT_VEC": np.array([2, 2, 1, 1, 0, 0], dtype=np.int64),
            "INT_A": np.array([0, -1, 2], dtype=np.int64),
            "INT_B": np.array([0, 2, -2], dtype=np.int64),
        },
        "nonneg_shift": {
            "INT_A": np.array([0, -1, 2], dtype=np.int64),
            "INT_B": np.array([0, 2, 62], dtype=np.int64),
        },
        "custom0": {
            "X": np.array([[np.nan, np.inf, -np.inf], [-0.0, 0.0, 1.0]], dtype=np.float32),
            "Y": np.array([[1.0, -1.0, np.nan], [np.inf, -np.inf, 0.0]], dtype=np.float32),
            "POS": np.array([[np.inf, 1.0, 1.0], [1.0, 1.0, np.inf]], dtype=np.float32),
            "VEC": np.array([np.nan, np.inf, -np.inf, -0.0], dtype=np.float32),
            "VEC_A": np.array([np.nan, np.inf, -0.0], dtype=np.float32),
            "VEC_B": np.array([1.0, -np.inf, 0.0], dtype=np.float32),
            "EVEN": np.array([[np.nan, np.inf, -np.inf, -0.0], [0.0, 1.0, -1.0, tiny]], dtype=np.float32),
        },
        "custom2": {
            "X": np.array([[np.float32(1e38), np.float32(-1e38), tiny], [-tiny, 0.0, -0.0]], dtype=np.float32),
            "Y": np.array([[np.float32(1e38), np.float32(1e-38), -1.0], [1.0, np.float32(-1e38), tiny]], dtype=np.float32),
            "POS": np.array([[tiny, np.float32(1e-38), 1.0], [np.float32(1e38), 16.0, 25.0]], dtype=np.float32),
            "VEC": np.array([np.float32(1e38), np.float32(-1e38), tiny, -0.0], dtype=np.float32),
            "VEC_A": np.array([np.float32(1e38), np.float32(-1e38), tiny], dtype=np.float32),
            "VEC_B": np.array([np.float32(1e-38), np.float32(-1e-38), -0.0], dtype=np.float32),
            "EVEN": np.array([[np.float32(1e38), np.float32(-1e38), tiny, -0.0], [0.0, 1.0, -1.0, np.float32(1e-38)]], dtype=np.float32),
        },
    }
    delta = states.get(str(token))
    if delta is None:
        return
    for name, value in delta.items():
        setattr(runner, name, value.copy() if hasattr(value, "copy") else value)


def api_by_name():
    global API_BY_NAME
    if API_BY_NAME is None:
        API_BY_NAME = {api.qualified_name: api for api in collect_apis()}
    return API_BY_NAME


def live_outputs(record):
    by_name = api_by_name()
    group = [by_name[name] for name in record["apis"] if name in by_name]
    snapshot = {name: getattr(runner, name) for name in runner.RUNTIME_ARRAY_NAMES}
    try:
        set_state(record.get("state_token"))
        state, data, errors = runner.run_group(tuple(record["key"]), group)
        source = "live" if state == "DIFF" else "not_live"
        return source, state, data.get("outputs", {}), data.get("chosen", {}), errors or data.get("skipped", {})
    except Exception as exc:
        return "not_live", "ERROR", {}, {}, {"runner": f"{type(exc).__name__}: {exc}"}
    finally:
        restore_state(snapshot)

def compact(value, limit=900):
    text = json.dumps(value, sort_keys=True, allow_nan=True)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def equivalent(a, b):
    try:
        return runner.equal_encoded(a, b)
    except Exception:
        return a == b


def choose_expected(outputs):
    clusters = []
    for lib, value in outputs.items():
        for cluster in clusters:
            if equivalent(value, cluster["value"]):
                cluster["libs"].append(lib)
                break
        else:
            clusters.append({"libs": [lib], "value": value})
    clusters.sort(key=lambda c: (-len(c["libs"]), min((REFERENCE_ORDER.index(lib) if lib in REFERENCE_ORDER else 999 for lib in c["libs"]))))
    if not clusters:
        return "undetermined", [], None
    if len(clusters[0]["libs"]) >= 2 and (len(clusters) == 1 or len(clusters[0]["libs"]) > len(clusters[1]["libs"])):
        return "majority", sorted(clusters[0]["libs"]), clusters[0]["value"]
    for preferred in REFERENCE_ORDER:
        if preferred in outputs:
            return f"reference:{preferred}", [preferred], outputs[preferred]
    return "single-reference", [clusters[0]["libs"][0]], clusters[0]["value"]


def render(record):
    source, status, outputs, chosen, errors = live_outputs(record)
    expected_source, expected_libs, expected = choose_expected(outputs)
    wrong = {lib: value for lib, value in outputs.items() if expected is None or not equivalent(value, expected)}
    lines = []
    lines.append(f"bug_id: {record['name']}")
    lines.append(f"key: {record['key_text']}")
    lines.append(f"status: {status}")
    lines.append(f"output_source: {source}")
    lines.append(f"expected_source: {expected_source}")
    lines.append(f"expected_libs: {', '.join(expected_libs) if expected_libs else 'none'}")
    lines.append(f"expected: {compact(expected)}")
    lines.append("wrong:")
    if wrong:
        for lib in sorted(wrong):
            lines.append(f"  {lib}: {compact(wrong[lib])}")
    else:
        lines.append("  none")
    if errors:
        lines.append(f"errors: {compact(errors)}")
    return "\n".join(lines)


def run_bug(bug_id):
    print(render(record_by_id(bug_id)))
