
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from typing import Any

import numpy as np


PY312 = os.environ.get("XAMT_PY312", "/tmp/xamt_py312/bin/python")


def main_runner_code() -> str:
    return r'''
import json, math, os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import torch
import tensorflow as tf
import keras
import jax
import jax.numpy as jnp

torch.set_grad_enabled(False)

paper_argsort = np.array(
    [-0.0, np.float32(1.401298464324817e-45), np.float32(1.100000023841858),
     -0.0, np.float32(5.960464477539063e-08), np.float32(-2.0000000135803223),
     np.float32(1000000.0), np.float32(722801.375), 0.0, np.float32(-1.100000023841858)],
    dtype=np.float32,
)
sort_nan = np.array([np.nan, 1.0, -1.0, np.nan, 0.0], dtype=np.float32)
x2 = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]], dtype=np.float32)
softmax_inf = np.array([[np.inf, 1.0, -np.inf], [1000.0, 1000.0, 1000.0]], dtype=np.float32)
zero = np.array([0.0, -0.0, 1.0, -1.0], dtype=np.float32)
extreme = np.array([-1000.0, 0.0, 1000.0], dtype=np.float32)
logsumexp_nan = np.array([[np.nan, 1.0], [-np.inf, -np.inf]], dtype=np.float32)

def encode(v):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    elif hasattr(v, "numpy"):
        v = v.numpy()
    elif hasattr(v, "tolist"):
        v = np.asarray(v)
    if isinstance(v, tuple):
        return [encode(x) for x in v]
    arr = np.asarray(v)
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "value": arr.tolist()}

def run_one(fn):
    try:
        return {"ok": True, "result": encode(fn())}
    except Exception as e:
        return {"ok": False, "error": type(e).__name__ + ": " + str(e)}

out = {}
out["torch"] = {
    "argsort_signed_zero": run_one(lambda: torch.argsort(torch.tensor(paper_argsort), dim=0)),
    "angle_complex_nan": run_one(lambda: torch.angle(torch.tensor(complex(float("nan"), float("nan"))))),
    "softmax_inf": run_one(lambda: torch.nn.functional.softmax(torch.tensor(softmax_inf), dim=1)),
    "logsumexp_nan": run_one(lambda: torch.logsumexp(torch.tensor(logsumexp_nan), dim=1)),
    "std_default": run_one(lambda: torch.std(torch.tensor(x2))),
    "sort_nan": run_one(lambda: torch.sort(torch.tensor(sort_nan)).values),
    "reciprocal_zero": run_one(lambda: torch.reciprocal(torch.tensor(zero))),
    "sigmoid_extreme": run_one(lambda: torch.sigmoid(torch.tensor(extreme))),
}
out["tensorflow"] = {
    "argsort_signed_zero": run_one(lambda: tf.argsort(tf.constant(paper_argsort), axis=0)),
    "angle_complex_nan": run_one(lambda: tf.math.angle(tf.constant(complex(float("nan"), float("nan")), dtype=tf.complex64))),
    "softmax_inf": run_one(lambda: tf.nn.softmax(tf.constant(softmax_inf), axis=1)),
    "logsumexp_nan": run_one(lambda: tf.reduce_logsumexp(tf.constant(logsumexp_nan), axis=1)),
    "std_default": run_one(lambda: tf.math.reduce_std(tf.constant(x2))),
    "sort_nan": run_one(lambda: tf.sort(tf.constant(sort_nan), axis=0)),
    "reciprocal_zero": run_one(lambda: tf.math.reciprocal(tf.constant(zero))),
    "sigmoid_extreme": run_one(lambda: tf.math.sigmoid(tf.constant(extreme))),
}
out["keras"] = {
    "argsort_signed_zero": run_one(lambda: keras.ops.argsort(paper_argsort, axis=0)),
    "angle_complex_nan": run_one(lambda: keras.ops.angle(np.array(complex(float("nan"), float("nan")), dtype=np.complex64))),
    "softmax_inf": run_one(lambda: keras.activations.softmax(softmax_inf, axis=1)),
    "logsumexp_nan": run_one(lambda: keras.ops.logsumexp(logsumexp_nan, axis=1)),
    "std_default": run_one(lambda: keras.ops.std(x2)),
    "sort_nan": run_one(lambda: keras.ops.sort(sort_nan, axis=0)),
    "reciprocal_zero": run_one(lambda: keras.ops.reciprocal(zero)),
    "sigmoid_extreme": run_one(lambda: keras.activations.sigmoid(extreme)),
}
out["jax"] = {
    "argsort_signed_zero": run_one(lambda: jnp.argsort(jnp.array(paper_argsort), axis=0)),
    "angle_complex_nan": run_one(lambda: jnp.angle(jnp.array(complex(float("nan"), float("nan")), dtype=jnp.complex64))),
    "softmax_inf": run_one(lambda: jax.nn.softmax(jnp.array(softmax_inf), axis=1)),
    "logsumexp_nan": run_one(lambda: jax.nn.logsumexp(jnp.array(logsumexp_nan), axis=1)),
    "std_default": run_one(lambda: jnp.std(jnp.array(x2))),
    "sort_nan": run_one(lambda: jnp.sort(jnp.array(sort_nan), axis=0)),
    "reciprocal_zero": run_one(lambda: jnp.reciprocal(jnp.array(zero))),
    "sigmoid_extreme": run_one(lambda: jax.nn.sigmoid(jnp.array(extreme))),
}
print(json.dumps(out))
'''


def paddle_runner_code() -> str:
    return r'''
import json
import numpy as np
import paddle

paper_argsort = np.array([-0.0, np.float32(1.401298464324817e-45), np.float32(1.100000023841858), -0.0, np.float32(5.960464477539063e-08), np.float32(-2.0000000135803223), np.float32(1000000.0), np.float32(722801.375), 0.0, np.float32(-1.100000023841858)], dtype=np.float32)
sort_nan = np.array([np.nan, 1.0, -1.0, np.nan, 0.0], dtype=np.float32)
x2 = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]], dtype=np.float32)
softmax_inf = np.array([[np.inf, 1.0, -np.inf], [1000.0, 1000.0, 1000.0]], dtype=np.float32)
zero = np.array([0.0, -0.0, 1.0, -1.0], dtype=np.float32)
extreme = np.array([-1000.0, 0.0, 1000.0], dtype=np.float32)
logsumexp_nan = np.array([[np.nan, 1.0], [-np.inf, -np.inf]], dtype=np.float32)

def encode(v):
    if hasattr(v, "numpy"):
        v = v.numpy()
    arr = np.asarray(v)
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "value": arr.tolist()}
def run_one(fn):
    try:
        return {"ok": True, "result": encode(fn())}
    except Exception as e:
        return {"ok": False, "error": type(e).__name__ + ": " + str(e)}

out = {"paddle": {
    "argsort_signed_zero": run_one(lambda: paddle.argsort(paddle.to_tensor(paper_argsort), axis=0)),
    "angle_complex_nan": run_one(lambda: paddle.angle(paddle.to_tensor(np.array(complex(float("nan"), float("nan")), dtype=np.complex64)))),
    "softmax_inf": run_one(lambda: paddle.nn.functional.softmax(paddle.to_tensor(softmax_inf), axis=1)),
    "logsumexp_nan": run_one(lambda: paddle.logsumexp(paddle.to_tensor(logsumexp_nan), axis=1)),
    "std_default": run_one(lambda: paddle.std(paddle.to_tensor(x2))),
    "sort_nan": run_one(lambda: paddle.sort(paddle.to_tensor(sort_nan), axis=0)),
    "reciprocal_zero": run_one(lambda: paddle.reciprocal(paddle.to_tensor(zero))),
    "sigmoid_extreme": run_one(lambda: paddle.nn.functional.sigmoid(paddle.to_tensor(extreme))),
}}
print(json.dumps(out))
'''


def mindspore_runner_code() -> str:
    return r'''
import json
import numpy as np
import mindspore as ms
import mindspore.ops as ops

paper_argsort = np.array([-0.0, np.float32(1.401298464324817e-45), np.float32(1.100000023841858), -0.0, np.float32(5.960464477539063e-08), np.float32(-2.0000000135803223), np.float32(1000000.0), np.float32(722801.375), 0.0, np.float32(-1.100000023841858)], dtype=np.float32)
sort_nan = np.array([np.nan, 1.0, -1.0, np.nan, 0.0], dtype=np.float32)
x2 = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]], dtype=np.float32)
softmax_inf = np.array([[np.inf, 1.0, -np.inf], [1000.0, 1000.0, 1000.0]], dtype=np.float32)
zero = np.array([0.0, -0.0, 1.0, -1.0], dtype=np.float32)
extreme = np.array([-1000.0, 0.0, 1000.0], dtype=np.float32)
logsumexp_nan = np.array([[np.nan, 1.0], [-np.inf, -np.inf]], dtype=np.float32)

def encode(v):
    if isinstance(v, tuple):
        v = v[0]
    if hasattr(v, "asnumpy"):
        v = v.asnumpy()
    arr = np.asarray(v)
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "value": arr.tolist()}
def run_one(fn):
    try:
        return {"ok": True, "result": encode(fn())}
    except Exception as e:
        return {"ok": False, "error": type(e).__name__ + ": " + str(e)}

out = {"mindspore": {
    "argsort_signed_zero": run_one(lambda: ops.argsort(ms.Tensor(paper_argsort), axis=0)),
    "angle_complex_nan": run_one(lambda: ops.angle(ms.Tensor(np.array(complex(float("nan"), float("nan")), dtype=np.complex64)))),
    "softmax_inf": run_one(lambda: ops.softmax(ms.Tensor(softmax_inf), axis=1)),
    "logsumexp_nan": run_one(lambda: ops.logsumexp(ms.Tensor(logsumexp_nan), axis=1)),
    "std_default": run_one(lambda: ops.std(ms.Tensor(x2))),
    "sort_nan": run_one(lambda: ops.sort(ms.Tensor(sort_nan), axis=0)),
    "reciprocal_zero": run_one(lambda: ops.reciprocal(ms.Tensor(zero))),
    "sigmoid_extreme": run_one(lambda: ops.sigmoid(ms.Tensor(extreme))),
}}
print(json.dumps(out))
'''


def run_python(executable: str, code: str) -> dict[str, Any]:
    raw = subprocess.check_output(
        [executable, "-B", "-c", code],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    return json.loads(raw.strip().splitlines()[-1])


def to_array(result: dict[str, Any]) -> np.ndarray | None:
    if not result.get("ok"):
        return None
    return np.asarray(result["result"]["value"])


def equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.allclose(a, b, atol=1e-5, rtol=1e-5, equal_nan=True)


def main() -> None:
    merged: dict[str, Any] = {}
    merged.update(run_python(sys.executable, main_runner_code()))
    merged.update(run_python(PY312, paddle_runner_code()))
    merged.update(run_python(PY312, mindspore_runner_code()))

    cases = sorted(next(iter(merged.values())).keys())
    print("libraries:", ", ".join(sorted(merged)))
    print()
    for case in cases:
        arrays = {lib: to_array(results[case]) for lib, results in merged.items()}
        errors = {lib: results[case]["error"] for lib, results in merged.items() if not results[case].get("ok")}
        ok_arrays = {lib: arr for lib, arr in arrays.items() if arr is not None}
        base_lib, base = next(iter(ok_arrays.items()))
        same = {lib: equal(base, arr) for lib, arr in ok_arrays.items()}
        passed = all(same.values()) and not errors
        print(f"{case}: {'PASS' if passed else 'DIFF'}")
        if errors:
            print("  errors:", errors)
        if not passed:
            for lib, arr in ok_arrays.items():
                print(f"  {lib:10s} {arr.tolist()}")


if __name__ == "__main__":
    main()
