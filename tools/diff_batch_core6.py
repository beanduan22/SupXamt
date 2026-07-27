
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

import numpy as np


PY312 = os.environ.get("XAMT_PY312", "/tmp/xamt_py312/bin/python")


DATA_CODE = r'''
import json
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np

x = np.array([[1.0, -2.0, 3.5], [4.0, 0.5, -6.0]], dtype=np.float32)
pos = np.array([[0.25, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=np.float32)
a = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]], dtype=np.float32)
b = np.array([[2.0, 3.0, -1.0], [0.5, -4.0, 2.0]], dtype=np.float32)
mat1 = np.array([[1.0, 2.0, -1.0], [3.0, 0.5, 4.0]], dtype=np.float32)
mat2 = np.array([[2.0, -3.0], [1.5, 0.0], [-2.0, 5.0]], dtype=np.float32)
vec3 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
cond = np.array([[True, False, True], [False, True, False]])
round_values = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32)
softmax_inf = np.array([[np.inf, 1.0, -np.inf], [1000.0, 1000.0, 1000.0]], dtype=np.float32)
logsumexp_nan = np.array([[np.nan, 1.0], [-np.inf, -np.inf]], dtype=np.float32)
sort_nan = np.array([np.nan, 1.0, -1.0, np.nan, 0.0], dtype=np.float32)
paper_argsort = np.array(
    [-0.0, np.float32(1.401298464324817e-45), np.float32(1.100000023841858),
     -0.0, np.float32(5.960464477539063e-08), np.float32(-2.0000000135803223),
     np.float32(1000000.0), np.float32(722801.375), 0.0, np.float32(-1.100000023841858)],
    dtype=np.float32,
)

def encode(v):
    if isinstance(v, tuple):
        v = v[0]
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    elif hasattr(v, "asnumpy"):
        v = v.asnumpy()
    elif hasattr(v, "numpy"):
        v = v.numpy()
    arr = np.asarray(v)
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "value": arr.tolist()}

def run_one(fn):
    try:
        return {"ok": True, "result": encode(fn())}
    except Exception as e:
        return {"ok": False, "error": type(e).__name__ + ": " + str(e)}
'''


MAIN_RUNNER = DATA_CODE + r'''
import torch
import tensorflow as tf
import keras
import jax
import jax.numpy as jnp

torch.set_grad_enabled(False)

out = {}
out["torch"] = {
    "abs": run_one(lambda: torch.abs(torch.tensor(x))),
    "negative": run_one(lambda: torch.negative(torch.tensor(x))),
    "exp": run_one(lambda: torch.exp(torch.tensor(vec3))),
    "log_positive": run_one(lambda: torch.log(torch.tensor(pos))),
    "sqrt_positive": run_one(lambda: torch.sqrt(torch.tensor(pos))),
    "sin": run_one(lambda: torch.sin(torch.tensor(x))),
    "cos": run_one(lambda: torch.cos(torch.tensor(x))),
    "tanh": run_one(lambda: torch.tanh(torch.tensor(x))),
    "sigmoid": run_one(lambda: torch.sigmoid(torch.tensor(x))),
    "relu": run_one(lambda: torch.relu(torch.tensor(x))),
    "softplus": run_one(lambda: torch.nn.functional.softplus(torch.tensor(x))),
    "floor": run_one(lambda: torch.floor(torch.tensor(x))),
    "ceil": run_one(lambda: torch.ceil(torch.tensor(x))),
    "round_halves": run_one(lambda: torch.round(torch.tensor(round_values))),
    "add": run_one(lambda: torch.add(torch.tensor(a), torch.tensor(b))),
    "subtract": run_one(lambda: torch.subtract(torch.tensor(a), torch.tensor(b))),
    "multiply": run_one(lambda: torch.multiply(torch.tensor(a), torch.tensor(b))),
    "divide": run_one(lambda: torch.divide(torch.tensor(a), torch.tensor(b))),
    "maximum": run_one(lambda: torch.maximum(torch.tensor(a), torch.tensor(b))),
    "minimum": run_one(lambda: torch.minimum(torch.tensor(a), torch.tensor(b))),
    "power": run_one(lambda: torch.pow(torch.tensor(pos), 1.5)),
    "sum_axis1": run_one(lambda: torch.sum(torch.tensor(x), dim=1)),
    "mean_axis1": run_one(lambda: torch.mean(torch.tensor(x), dim=1)),
    "max_axis1": run_one(lambda: torch.max(torch.tensor(x), dim=1).values),
    "min_axis1": run_one(lambda: torch.min(torch.tensor(x), dim=1).values),
    "prod_axis1": run_one(lambda: torch.prod(torch.tensor(x), dim=1)),
    "argmax_axis1": run_one(lambda: torch.argmax(torch.tensor(x), dim=1)),
    "argmin_axis1": run_one(lambda: torch.argmin(torch.tensor(x), dim=1)),
    "matmul": run_one(lambda: torch.matmul(torch.tensor(mat1), torch.tensor(mat2))),
    "reshape_3x2": run_one(lambda: torch.reshape(torch.tensor(x), (3, 2))),
    "transpose": run_one(lambda: torch.transpose(torch.tensor(x), 0, 1)),
    "squeeze": run_one(lambda: torch.squeeze(torch.tensor(x).reshape(1, 2, 3, 1))),
    "expand_dims": run_one(lambda: torch.unsqueeze(torch.tensor(x), 0)),
    "concat_axis0": run_one(lambda: torch.cat([torch.tensor(x), torch.tensor(x)], dim=0)),
    "stack_axis0": run_one(lambda: torch.stack([torch.tensor(x), torch.tensor(x)], dim=0)),
    "clip": run_one(lambda: torch.clamp(torch.tensor(x), -1.0, 2.0)),
    "where": run_one(lambda: torch.where(torch.tensor(cond), torch.tensor(a), torch.tensor(b))),
    "softmax_axis1": run_one(lambda: torch.nn.functional.softmax(torch.tensor(x), dim=1)),
    "log_softmax_axis1": run_one(lambda: torch.nn.functional.log_softmax(torch.tensor(x), dim=1)),
    "std_population": run_one(lambda: torch.std(torch.tensor(x), correction=0)),
    "std_default": run_one(lambda: torch.std(torch.tensor(x))),
    "argsort_signed_zero": run_one(lambda: torch.argsort(torch.tensor(paper_argsort), dim=0)),
    "sort_nan": run_one(lambda: torch.sort(torch.tensor(sort_nan)).values),
    "softmax_inf": run_one(lambda: torch.nn.functional.softmax(torch.tensor(softmax_inf), dim=1)),
    "logsumexp_nan": run_one(lambda: torch.logsumexp(torch.tensor(logsumexp_nan), dim=1)),
    "angle_complex_nan": run_one(lambda: torch.angle(torch.tensor(complex(float("nan"), float("nan"))))),
}
out["tensorflow"] = {
    "abs": run_one(lambda: tf.abs(tf.constant(x))),
    "negative": run_one(lambda: tf.negative(tf.constant(x))),
    "exp": run_one(lambda: tf.exp(tf.constant(vec3))),
    "log_positive": run_one(lambda: tf.math.log(tf.constant(pos))),
    "sqrt_positive": run_one(lambda: tf.sqrt(tf.constant(pos))),
    "sin": run_one(lambda: tf.sin(tf.constant(x))),
    "cos": run_one(lambda: tf.cos(tf.constant(x))),
    "tanh": run_one(lambda: tf.tanh(tf.constant(x))),
    "sigmoid": run_one(lambda: tf.math.sigmoid(tf.constant(x))),
    "relu": run_one(lambda: tf.nn.relu(tf.constant(x))),
    "softplus": run_one(lambda: tf.nn.softplus(tf.constant(x))),
    "floor": run_one(lambda: tf.floor(tf.constant(x))),
    "ceil": run_one(lambda: tf.math.ceil(tf.constant(x))),
    "round_halves": run_one(lambda: tf.round(tf.constant(round_values))),
    "add": run_one(lambda: tf.add(tf.constant(a), tf.constant(b))),
    "subtract": run_one(lambda: tf.subtract(tf.constant(a), tf.constant(b))),
    "multiply": run_one(lambda: tf.multiply(tf.constant(a), tf.constant(b))),
    "divide": run_one(lambda: tf.divide(tf.constant(a), tf.constant(b))),
    "maximum": run_one(lambda: tf.maximum(tf.constant(a), tf.constant(b))),
    "minimum": run_one(lambda: tf.minimum(tf.constant(a), tf.constant(b))),
    "power": run_one(lambda: tf.pow(tf.constant(pos), 1.5)),
    "sum_axis1": run_one(lambda: tf.reduce_sum(tf.constant(x), axis=1)),
    "mean_axis1": run_one(lambda: tf.reduce_mean(tf.constant(x), axis=1)),
    "max_axis1": run_one(lambda: tf.reduce_max(tf.constant(x), axis=1)),
    "min_axis1": run_one(lambda: tf.reduce_min(tf.constant(x), axis=1)),
    "prod_axis1": run_one(lambda: tf.reduce_prod(tf.constant(x), axis=1)),
    "argmax_axis1": run_one(lambda: tf.argmax(tf.constant(x), axis=1)),
    "argmin_axis1": run_one(lambda: tf.argmin(tf.constant(x), axis=1)),
    "matmul": run_one(lambda: tf.matmul(tf.constant(mat1), tf.constant(mat2))),
    "reshape_3x2": run_one(lambda: tf.reshape(tf.constant(x), (3, 2))),
    "transpose": run_one(lambda: tf.transpose(tf.constant(x), perm=(1, 0))),
    "squeeze": run_one(lambda: tf.squeeze(tf.reshape(tf.constant(x), (1, 2, 3, 1)))),
    "expand_dims": run_one(lambda: tf.expand_dims(tf.constant(x), axis=0)),
    "concat_axis0": run_one(lambda: tf.concat([tf.constant(x), tf.constant(x)], axis=0)),
    "stack_axis0": run_one(lambda: tf.stack([tf.constant(x), tf.constant(x)], axis=0)),
    "clip": run_one(lambda: tf.clip_by_value(tf.constant(x), -1.0, 2.0)),
    "where": run_one(lambda: tf.where(tf.constant(cond), tf.constant(a), tf.constant(b))),
    "softmax_axis1": run_one(lambda: tf.nn.softmax(tf.constant(x), axis=1)),
    "log_softmax_axis1": run_one(lambda: tf.nn.log_softmax(tf.constant(x), axis=1)),
    "std_population": run_one(lambda: tf.math.reduce_std(tf.constant(x))),
    "std_default": run_one(lambda: tf.math.reduce_std(tf.constant(x))),
    "argsort_signed_zero": run_one(lambda: tf.argsort(tf.constant(paper_argsort), axis=0)),
    "sort_nan": run_one(lambda: tf.sort(tf.constant(sort_nan), axis=0)),
    "softmax_inf": run_one(lambda: tf.nn.softmax(tf.constant(softmax_inf), axis=1)),
    "logsumexp_nan": run_one(lambda: tf.reduce_logsumexp(tf.constant(logsumexp_nan), axis=1)),
    "angle_complex_nan": run_one(lambda: tf.math.angle(tf.constant(complex(float("nan"), float("nan")), dtype=tf.complex64))),
}
out["keras"] = {
    "abs": run_one(lambda: keras.ops.abs(x)),
    "negative": run_one(lambda: keras.ops.negative(x)),
    "exp": run_one(lambda: keras.ops.exp(vec3)),
    "log_positive": run_one(lambda: keras.ops.log(pos)),
    "sqrt_positive": run_one(lambda: keras.ops.sqrt(pos)),
    "sin": run_one(lambda: keras.ops.sin(x)),
    "cos": run_one(lambda: keras.ops.cos(x)),
    "tanh": run_one(lambda: keras.ops.tanh(x)),
    "sigmoid": run_one(lambda: keras.ops.sigmoid(x)),
    "relu": run_one(lambda: keras.ops.relu(x)),
    "softplus": run_one(lambda: keras.ops.softplus(x)),
    "floor": run_one(lambda: keras.ops.floor(x)),
    "ceil": run_one(lambda: keras.ops.ceil(x)),
    "round_halves": run_one(lambda: keras.ops.round(round_values)),
    "add": run_one(lambda: keras.ops.add(a, b)),
    "subtract": run_one(lambda: keras.ops.subtract(a, b)),
    "multiply": run_one(lambda: keras.ops.multiply(a, b)),
    "divide": run_one(lambda: keras.ops.divide(a, b)),
    "maximum": run_one(lambda: keras.ops.maximum(a, b)),
    "minimum": run_one(lambda: keras.ops.minimum(a, b)),
    "power": run_one(lambda: keras.ops.power(pos, 1.5)),
    "sum_axis1": run_one(lambda: keras.ops.sum(x, axis=1)),
    "mean_axis1": run_one(lambda: keras.ops.mean(x, axis=1)),
    "max_axis1": run_one(lambda: keras.ops.max(x, axis=1)),
    "min_axis1": run_one(lambda: keras.ops.min(x, axis=1)),
    "prod_axis1": run_one(lambda: keras.ops.prod(x, axis=1)),
    "argmax_axis1": run_one(lambda: keras.ops.argmax(x, axis=1)),
    "argmin_axis1": run_one(lambda: keras.ops.argmin(x, axis=1)),
    "matmul": run_one(lambda: keras.ops.matmul(mat1, mat2)),
    "reshape_3x2": run_one(lambda: keras.ops.reshape(x, (3, 2))),
    "transpose": run_one(lambda: keras.ops.transpose(x, (1, 0))),
    "squeeze": run_one(lambda: keras.ops.squeeze(np.reshape(x, (1, 2, 3, 1)))),
    "expand_dims": run_one(lambda: keras.ops.expand_dims(x, axis=0)),
    "concat_axis0": run_one(lambda: keras.ops.concatenate([x, x], axis=0)),
    "stack_axis0": run_one(lambda: keras.ops.stack([x, x], axis=0)),
    "clip": run_one(lambda: keras.ops.clip(x, -1.0, 2.0)),
    "where": run_one(lambda: keras.ops.where(cond, a, b)),
    "softmax_axis1": run_one(lambda: keras.ops.softmax(x, axis=1)),
    "log_softmax_axis1": run_one(lambda: keras.ops.log_softmax(x, axis=1)),
    "std_population": run_one(lambda: keras.ops.std(x)),
    "std_default": run_one(lambda: keras.ops.std(x)),
    "argsort_signed_zero": run_one(lambda: keras.ops.argsort(paper_argsort, axis=0)),
    "sort_nan": run_one(lambda: keras.ops.sort(sort_nan, axis=0)),
    "softmax_inf": run_one(lambda: keras.ops.softmax(softmax_inf, axis=1)),
    "logsumexp_nan": run_one(lambda: keras.ops.logsumexp(logsumexp_nan, axis=1)),
    "angle_complex_nan": run_one(lambda: keras.ops.angle(np.array(complex(float("nan"), float("nan")), dtype=np.complex64))),
}
out["jax"] = {
    "abs": run_one(lambda: jnp.abs(jnp.array(x))),
    "negative": run_one(lambda: jnp.negative(jnp.array(x))),
    "exp": run_one(lambda: jnp.exp(jnp.array(vec3))),
    "log_positive": run_one(lambda: jnp.log(jnp.array(pos))),
    "sqrt_positive": run_one(lambda: jnp.sqrt(jnp.array(pos))),
    "sin": run_one(lambda: jnp.sin(jnp.array(x))),
    "cos": run_one(lambda: jnp.cos(jnp.array(x))),
    "tanh": run_one(lambda: jnp.tanh(jnp.array(x))),
    "sigmoid": run_one(lambda: jax.nn.sigmoid(jnp.array(x))),
    "relu": run_one(lambda: jax.nn.relu(jnp.array(x))),
    "softplus": run_one(lambda: jax.nn.softplus(jnp.array(x))),
    "floor": run_one(lambda: jnp.floor(jnp.array(x))),
    "ceil": run_one(lambda: jnp.ceil(jnp.array(x))),
    "round_halves": run_one(lambda: jnp.round(jnp.array(round_values))),
    "add": run_one(lambda: jnp.add(jnp.array(a), jnp.array(b))),
    "subtract": run_one(lambda: jnp.subtract(jnp.array(a), jnp.array(b))),
    "multiply": run_one(lambda: jnp.multiply(jnp.array(a), jnp.array(b))),
    "divide": run_one(lambda: jnp.divide(jnp.array(a), jnp.array(b))),
    "maximum": run_one(lambda: jnp.maximum(jnp.array(a), jnp.array(b))),
    "minimum": run_one(lambda: jnp.minimum(jnp.array(a), jnp.array(b))),
    "power": run_one(lambda: jnp.power(jnp.array(pos), 1.5)),
    "sum_axis1": run_one(lambda: jnp.sum(jnp.array(x), axis=1)),
    "mean_axis1": run_one(lambda: jnp.mean(jnp.array(x), axis=1)),
    "max_axis1": run_one(lambda: jnp.max(jnp.array(x), axis=1)),
    "min_axis1": run_one(lambda: jnp.min(jnp.array(x), axis=1)),
    "prod_axis1": run_one(lambda: jnp.prod(jnp.array(x), axis=1)),
    "argmax_axis1": run_one(lambda: jnp.argmax(jnp.array(x), axis=1)),
    "argmin_axis1": run_one(lambda: jnp.argmin(jnp.array(x), axis=1)),
    "matmul": run_one(lambda: jnp.matmul(jnp.array(mat1), jnp.array(mat2))),
    "reshape_3x2": run_one(lambda: jnp.reshape(jnp.array(x), (3, 2))),
    "transpose": run_one(lambda: jnp.transpose(jnp.array(x), (1, 0))),
    "squeeze": run_one(lambda: jnp.squeeze(jnp.reshape(jnp.array(x), (1, 2, 3, 1)))),
    "expand_dims": run_one(lambda: jnp.expand_dims(jnp.array(x), axis=0)),
    "concat_axis0": run_one(lambda: jnp.concatenate([jnp.array(x), jnp.array(x)], axis=0)),
    "stack_axis0": run_one(lambda: jnp.stack([jnp.array(x), jnp.array(x)], axis=0)),
    "clip": run_one(lambda: jnp.clip(jnp.array(x), -1.0, 2.0)),
    "where": run_one(lambda: jnp.where(jnp.array(cond), jnp.array(a), jnp.array(b))),
    "softmax_axis1": run_one(lambda: jax.nn.softmax(jnp.array(x), axis=1)),
    "log_softmax_axis1": run_one(lambda: jax.nn.log_softmax(jnp.array(x), axis=1)),
    "std_population": run_one(lambda: jnp.std(jnp.array(x), correction=0)),
    "std_default": run_one(lambda: jnp.std(jnp.array(x))),
    "argsort_signed_zero": run_one(lambda: jnp.argsort(jnp.array(paper_argsort), axis=0)),
    "sort_nan": run_one(lambda: jnp.sort(jnp.array(sort_nan), axis=0)),
    "softmax_inf": run_one(lambda: jax.nn.softmax(jnp.array(softmax_inf), axis=1)),
    "logsumexp_nan": run_one(lambda: jax.nn.logsumexp(jnp.array(logsumexp_nan), axis=1)),
    "angle_complex_nan": run_one(lambda: jnp.angle(jnp.array(complex(float("nan"), float("nan")), dtype=jnp.complex64))),
}
print(json.dumps(out))
'''


PADDLE_RUNNER = DATA_CODE + r'''
import paddle
import paddle.nn.functional as F

out = {"paddle": {
    "abs": run_one(lambda: paddle.abs(paddle.to_tensor(x))),
    "negative": run_one(lambda: paddle.negative(paddle.to_tensor(x))),
    "exp": run_one(lambda: paddle.exp(paddle.to_tensor(vec3))),
    "log_positive": run_one(lambda: paddle.log(paddle.to_tensor(pos))),
    "sqrt_positive": run_one(lambda: paddle.sqrt(paddle.to_tensor(pos))),
    "sin": run_one(lambda: paddle.sin(paddle.to_tensor(x))),
    "cos": run_one(lambda: paddle.cos(paddle.to_tensor(x))),
    "tanh": run_one(lambda: paddle.tanh(paddle.to_tensor(x))),
    "sigmoid": run_one(lambda: F.sigmoid(paddle.to_tensor(x))),
    "relu": run_one(lambda: F.relu(paddle.to_tensor(x))),
    "softplus": run_one(lambda: F.softplus(paddle.to_tensor(x))),
    "floor": run_one(lambda: paddle.floor(paddle.to_tensor(x))),
    "ceil": run_one(lambda: paddle.ceil(paddle.to_tensor(x))),
    "round_halves": run_one(lambda: paddle.round(paddle.to_tensor(round_values))),
    "add": run_one(lambda: paddle.add(paddle.to_tensor(a), paddle.to_tensor(b))),
    "subtract": run_one(lambda: paddle.subtract(paddle.to_tensor(a), paddle.to_tensor(b))),
    "multiply": run_one(lambda: paddle.multiply(paddle.to_tensor(a), paddle.to_tensor(b))),
    "divide": run_one(lambda: paddle.divide(paddle.to_tensor(a), paddle.to_tensor(b))),
    "maximum": run_one(lambda: paddle.maximum(paddle.to_tensor(a), paddle.to_tensor(b))),
    "minimum": run_one(lambda: paddle.minimum(paddle.to_tensor(a), paddle.to_tensor(b))),
    "power": run_one(lambda: paddle.pow(paddle.to_tensor(pos), 1.5)),
    "sum_axis1": run_one(lambda: paddle.sum(paddle.to_tensor(x), axis=1)),
    "mean_axis1": run_one(lambda: paddle.mean(paddle.to_tensor(x), axis=1)),
    "max_axis1": run_one(lambda: paddle.max(paddle.to_tensor(x), axis=1)),
    "min_axis1": run_one(lambda: paddle.min(paddle.to_tensor(x), axis=1)),
    "prod_axis1": run_one(lambda: paddle.prod(paddle.to_tensor(x), axis=1)),
    "argmax_axis1": run_one(lambda: paddle.argmax(paddle.to_tensor(x), axis=1)),
    "argmin_axis1": run_one(lambda: paddle.argmin(paddle.to_tensor(x), axis=1)),
    "matmul": run_one(lambda: paddle.matmul(paddle.to_tensor(mat1), paddle.to_tensor(mat2))),
    "reshape_3x2": run_one(lambda: paddle.reshape(paddle.to_tensor(x), (3, 2))),
    "transpose": run_one(lambda: paddle.transpose(paddle.to_tensor(x), perm=(1, 0))),
    "squeeze": run_one(lambda: paddle.squeeze(paddle.reshape(paddle.to_tensor(x), (1, 2, 3, 1)))),
    "expand_dims": run_one(lambda: paddle.unsqueeze(paddle.to_tensor(x), axis=0)),
    "concat_axis0": run_one(lambda: paddle.concat([paddle.to_tensor(x), paddle.to_tensor(x)], axis=0)),
    "stack_axis0": run_one(lambda: paddle.stack([paddle.to_tensor(x), paddle.to_tensor(x)], axis=0)),
    "clip": run_one(lambda: paddle.clip(paddle.to_tensor(x), min=-1.0, max=2.0)),
    "where": run_one(lambda: paddle.where(paddle.to_tensor(cond), paddle.to_tensor(a), paddle.to_tensor(b))),
    "softmax_axis1": run_one(lambda: F.softmax(paddle.to_tensor(x), axis=1)),
    "log_softmax_axis1": run_one(lambda: F.log_softmax(paddle.to_tensor(x), axis=1)),
    "std_population": run_one(lambda: paddle.std(paddle.to_tensor(x), unbiased=False)),
    "std_default": run_one(lambda: paddle.std(paddle.to_tensor(x))),
    "argsort_signed_zero": run_one(lambda: paddle.argsort(paddle.to_tensor(paper_argsort), axis=0)),
    "sort_nan": run_one(lambda: paddle.sort(paddle.to_tensor(sort_nan), axis=0)),
    "softmax_inf": run_one(lambda: F.softmax(paddle.to_tensor(softmax_inf), axis=1)),
    "logsumexp_nan": run_one(lambda: paddle.logsumexp(paddle.to_tensor(logsumexp_nan), axis=1)),
    "angle_complex_nan": run_one(lambda: paddle.angle(paddle.to_tensor(np.array(complex(float("nan"), float("nan")), dtype=np.complex64)))),
}}
print(json.dumps(out))
'''


MINDSPORE_RUNNER = DATA_CODE + r'''
import mindspore as ms
import mindspore.ops as ops

out = {"mindspore": {
    "abs": run_one(lambda: ops.abs(ms.Tensor(x))),
    "negative": run_one(lambda: ops.negative(ms.Tensor(x))),
    "exp": run_one(lambda: ops.exp(ms.Tensor(vec3))),
    "log_positive": run_one(lambda: ops.log(ms.Tensor(pos))),
    "sqrt_positive": run_one(lambda: ops.sqrt(ms.Tensor(pos))),
    "sin": run_one(lambda: ops.sin(ms.Tensor(x))),
    "cos": run_one(lambda: ops.cos(ms.Tensor(x))),
    "tanh": run_one(lambda: ops.tanh(ms.Tensor(x))),
    "sigmoid": run_one(lambda: ops.sigmoid(ms.Tensor(x))),
    "relu": run_one(lambda: ops.relu(ms.Tensor(x))),
    "softplus": run_one(lambda: ops.softplus(ms.Tensor(x))),
    "floor": run_one(lambda: ops.floor(ms.Tensor(x))),
    "ceil": run_one(lambda: ops.ceil(ms.Tensor(x))),
    "round_halves": run_one(lambda: ops.round(ms.Tensor(round_values))),
    "add": run_one(lambda: ops.add(ms.Tensor(a), ms.Tensor(b))),
    "subtract": run_one(lambda: ops.sub(ms.Tensor(a), ms.Tensor(b))),
    "multiply": run_one(lambda: ops.mul(ms.Tensor(a), ms.Tensor(b))),
    "divide": run_one(lambda: ops.div(ms.Tensor(a), ms.Tensor(b))),
    "maximum": run_one(lambda: ops.maximum(ms.Tensor(a), ms.Tensor(b))),
    "minimum": run_one(lambda: ops.minimum(ms.Tensor(a), ms.Tensor(b))),
    "power": run_one(lambda: ops.pow(ms.Tensor(pos), ms.Tensor(np.array(1.5, dtype=np.float32)))),
    "sum_axis1": run_one(lambda: ops.sum(ms.Tensor(x), dim=1)),
    "mean_axis1": run_one(lambda: ops.mean(ms.Tensor(x), axis=1)),
    "max_axis1": run_one(lambda: ops.max(ms.Tensor(x), axis=1)),
    "min_axis1": run_one(lambda: ops.min(ms.Tensor(x), axis=1)),
    "prod_axis1": run_one(lambda: ops.prod(ms.Tensor(x), axis=1)),
    "argmax_axis1": run_one(lambda: ops.argmax(ms.Tensor(x), dim=1)),
    "argmin_axis1": run_one(lambda: ops.argmin(ms.Tensor(x), axis=1)),
    "matmul": run_one(lambda: ops.matmul(ms.Tensor(mat1), ms.Tensor(mat2))),
    "reshape_3x2": run_one(lambda: ops.reshape(ms.Tensor(x), (3, 2))),
    "transpose": run_one(lambda: ops.transpose(ms.Tensor(x), (1, 0))),
    "squeeze": run_one(lambda: ops.squeeze(ops.reshape(ms.Tensor(x), (1, 2, 3, 1)))),
    "expand_dims": run_one(lambda: ops.expand_dims(ms.Tensor(x), 0)),
    "concat_axis0": run_one(lambda: ops.concat((ms.Tensor(x), ms.Tensor(x)), axis=0)),
    "stack_axis0": run_one(lambda: ops.stack((ms.Tensor(x), ms.Tensor(x)), axis=0)),
    "clip": run_one(lambda: ops.clip(ms.Tensor(x), -1.0, 2.0)),
    "where": run_one(lambda: ops.where(ms.Tensor(cond), ms.Tensor(a), ms.Tensor(b))),
    "softmax_axis1": run_one(lambda: ops.softmax(ms.Tensor(x), axis=1)),
    "log_softmax_axis1": run_one(lambda: ops.log_softmax(ms.Tensor(x), axis=1)),
    "std_population": run_one(lambda: ops.std(ms.Tensor(x), ddof=0)),
    "std_default": run_one(lambda: ops.std(ms.Tensor(x))),
    "argsort_signed_zero": run_one(lambda: ops.argsort(ms.Tensor(paper_argsort), axis=0)),
    "sort_nan": run_one(lambda: ops.sort(ms.Tensor(sort_nan), axis=0)),
    "softmax_inf": run_one(lambda: ops.softmax(ms.Tensor(softmax_inf), axis=1)),
    "logsumexp_nan": run_one(lambda: ops.logsumexp(ms.Tensor(logsumexp_nan), 1)),
    "angle_complex_nan": run_one(lambda: ops.angle(ms.Tensor(np.array(complex(float("nan"), float("nan")), dtype=np.complex64)))),
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
    if a.shape != b.shape:
        return False
    return np.allclose(a, b, atol=1e-5, rtol=1e-5, equal_nan=True)


def main() -> None:
    merged: dict[str, Any] = {}
    merged.update(run_python(sys.executable, MAIN_RUNNER))
    merged.update(run_python(PY312, PADDLE_RUNNER))
    merged.update(run_python(PY312, MINDSPORE_RUNNER))

    cases = sorted(next(iter(merged.values())).keys())
    statuses: dict[str, str] = {}
    details: list[tuple[str, dict[str, str], dict[str, np.ndarray]]] = []

    for case in cases:
        errors = {
            lib: results[case]["error"]
            for lib, results in merged.items()
            if not results[case].get("ok")
        }
        arrays = {
            lib: to_array(results[case])
            for lib, results in merged.items()
            if results[case].get("ok")
        }
        base = next(iter(arrays.values())) if arrays else None
        same = bool(base is not None and all(equal(base, arr) for arr in arrays.values()))
        if errors:
            statuses[case] = "ERROR"
        elif same:
            statuses[case] = "PASS"
        else:
            statuses[case] = "DIFF"
        if statuses[case] != "PASS":
            details.append((case, errors, arrays))

    print("libraries:", ", ".join(sorted(merged)))
    print("cases:", len(cases))
    print(
        "summary:",
        json.dumps(
            {status: list(statuses.values()).count(status) for status in ["PASS", "DIFF", "ERROR"]},
            sort_keys=True,
        ),
    )
    print()
    for case, errors, arrays in details:
        print(f"{case}: {statuses[case]}")
        if errors:
            print("  errors:", errors)
        if statuses[case] == "DIFF":
            for lib, arr in arrays.items():
                print(f"  {lib:10s} {arr.tolist()}")


if __name__ == "__main__":
    main()
