
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from typing import Any

import numpy as np

PY312 = os.environ.get("XAMT_PY312", "/tmp/xamt_py312/bin/python")
LIBS = {
    "torch": sys.executable,
    "tensorflow": sys.executable,
    "keras": sys.executable,
    "jax": sys.executable,
    "paddle": PY312,
    "mindspore": PY312,
}

RUNNER_CODE = r'''
from __future__ import annotations
import json, os
import numpy as np
LIB = os.environ["XAMT_LIB"]
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
if LIB == "torch":
    import torch
    import torch.nn.functional as F
    torch.set_grad_enabled(False)
elif LIB == "tensorflow":
    import tensorflow as tf
elif LIB == "keras":
    import keras
elif LIB == "jax":
    import jax
    import jax.numpy as jnp
elif LIB == "paddle":
    import paddle
    import paddle.nn.functional as F
elif LIB == "mindspore":
    import mindspore as ms
    import mindspore.ops as ops
    import mindspore.scipy.linalg as msl
else:
    raise RuntimeError(LIB)

x = np.array([[1.0, -2.0, 3.5], [4.0, 0.5, -6.0]], dtype=np.float32)
pos = np.array([[0.25, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=np.float32)
small_pos = np.array([1e-7, 1e-4, 1.0, 10.0], dtype=np.float32)
a = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]], dtype=np.float32)
b = np.array([[2.0, 3.0, -1.0], [0.5, -4.0, 2.0]], dtype=np.float32)
mat1 = np.array([[1.0, 2.0, -1.0], [3.0, 0.5, 4.0]], dtype=np.float32)
mat2 = np.array([[2.0, -3.0], [1.5, 0.0], [-2.0, 5.0]], dtype=np.float32)
square = np.array([[4.0, 1.0], [2.0, 3.0]], dtype=np.float32)
pdmat = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float32)
solve_rhs = np.array([[1.0], [2.0]], dtype=np.float32)
vec4 = np.array([1.0, -2.0, 3.0, 4.0], dtype=np.float32)
cond = np.array([[True, False, True], [False, True, False]])
bool_a = np.array([[True, False, True], [False, True, False]])
bool_b = np.array([[False, False, True], [True, True, False]])
round_values = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32)
zero = np.array([0.0, -0.0, 1.0, -1.0], dtype=np.float32)
div_zero_denom = np.array([0.0, -0.0, 2.0, -2.0], dtype=np.float32)
log_edge = np.array([0.0, -0.0, -1.0, 1.0], dtype=np.float32)
sqrt_edge = np.array([4.0, 0.0, -0.0, -1.0], dtype=np.float32)
softmax_inf = np.array([[np.inf, 1.0, -np.inf], [1000.0, 1000.0, 1000.0]], dtype=np.float32)
logsumexp_nan = np.array([[np.nan, 1.0], [-np.inf, -np.inf]], dtype=np.float32)
sort_nan = np.array([np.nan, 1.0, -1.0, np.nan, 0.0], dtype=np.float32)
minmax_nan = np.array([[np.nan, 1.0, -1.0], [np.nan, np.nan, 2.0]], dtype=np.float32)
paper_argsort = np.array([-0.0, np.float32(1.401298464324817e-45), np.float32(1.100000023841858), -0.0, np.float32(5.960464477539063e-08), np.float32(-2.0000000135803223), np.float32(1000000.0), np.float32(722801.375), 0.0, np.float32(-1.100000023841858)], dtype=np.float32)

def encode(v):
    if isinstance(v, (list, tuple)):
        return [encode(i) for i in v]
    if LIB == "torch" and hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    elif LIB == "mindspore" and hasattr(v, "asnumpy"):
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

class B:
    @staticmethod
    def t(v, dtype=None):
        if LIB == "torch":
            if dtype == "bool": return torch.tensor(v, dtype=torch.bool)
            if dtype == "int64": return torch.tensor(v, dtype=torch.int64)
            return torch.tensor(v)
        if LIB == "tensorflow":
            if dtype == "bool": return tf.constant(v, dtype=tf.bool)
            if dtype == "int64": return tf.constant(v, dtype=tf.int64)
            return tf.constant(v)
        if LIB == "keras": return keras.ops.convert_to_tensor(v)
        if LIB == "jax":
            if dtype == "bool": return jnp.array(v, dtype=jnp.bool_)
            if dtype == "int64": return jnp.array(v, dtype=jnp.int64)
            return jnp.array(v)
        if LIB == "paddle":
            if dtype == "bool": return paddle.to_tensor(v, dtype="bool")
            if dtype == "int64": return paddle.to_tensor(v, dtype="int64")
            return paddle.to_tensor(v)
        if dtype == "bool": return ms.Tensor(v, ms.bool_)
        if dtype == "int64": return ms.Tensor(v, ms.int64)
        return ms.Tensor(v)

    @staticmethod
    def unary(name, v):
        z = B.t(v)
        if name == "gelu":
            if LIB == "torch": return F.gelu(z, approximate="none")
            if LIB == "tensorflow": return tf.nn.gelu(z, approximate=False)
            if LIB == "keras": return keras.ops.gelu(z, approximate=False)
            if LIB == "jax": return jax.nn.gelu(z, approximate=False)
            if LIB == "paddle": return F.gelu(z, approximate=False)
            return ops.gelu(z)
        if LIB == "torch":
            m = {"abs": torch.abs, "negative": torch.negative, "exp": torch.exp, "expm1": torch.expm1, "log": torch.log, "log1p": torch.log1p, "sqrt": torch.sqrt, "square": torch.square, "sin": torch.sin, "cos": torch.cos, "tan": torch.tan, "tanh": torch.tanh, "sinh": torch.sinh, "cosh": torch.cosh, "floor": torch.floor, "ceil": torch.ceil, "round": torch.round, "sign": torch.sign, "reciprocal": torch.reciprocal, "sigmoid": torch.sigmoid, "relu": torch.relu, "softplus": F.softplus, "gelu": F.gelu, "elu": F.elu, "selu": F.selu}
        elif LIB == "tensorflow":
            m = {"abs": tf.abs, "negative": tf.negative, "exp": tf.exp, "expm1": tf.math.expm1, "log": tf.math.log, "log1p": tf.math.log1p, "sqrt": tf.sqrt, "square": tf.square, "sin": tf.sin, "cos": tf.cos, "tan": tf.tan, "tanh": tf.tanh, "sinh": tf.sinh, "cosh": tf.cosh, "floor": tf.floor, "ceil": tf.math.ceil, "round": tf.round, "sign": tf.sign, "reciprocal": tf.math.reciprocal, "sigmoid": tf.math.sigmoid, "relu": tf.nn.relu, "softplus": tf.nn.softplus, "gelu": tf.nn.gelu, "elu": tf.nn.elu, "selu": tf.nn.selu}
        elif LIB == "keras":
            m = {"abs": keras.ops.abs, "negative": keras.ops.negative, "exp": keras.ops.exp, "expm1": keras.ops.expm1, "log": keras.ops.log, "log1p": keras.ops.log1p, "sqrt": keras.ops.sqrt, "square": keras.ops.square, "sin": keras.ops.sin, "cos": keras.ops.cos, "tan": keras.ops.tan, "tanh": keras.ops.tanh, "sinh": keras.ops.sinh, "cosh": keras.ops.cosh, "floor": keras.ops.floor, "ceil": keras.ops.ceil, "round": keras.ops.round, "sign": keras.ops.sign, "reciprocal": keras.ops.reciprocal, "sigmoid": keras.ops.sigmoid, "relu": keras.ops.relu, "softplus": keras.ops.softplus, "gelu": keras.ops.gelu, "elu": keras.ops.elu, "selu": keras.ops.selu}
        elif LIB == "jax":
            m = {"abs": jnp.abs, "negative": jnp.negative, "exp": jnp.exp, "expm1": jnp.expm1, "log": jnp.log, "log1p": jnp.log1p, "sqrt": jnp.sqrt, "square": jnp.square, "sin": jnp.sin, "cos": jnp.cos, "tan": jnp.tan, "tanh": jnp.tanh, "sinh": jnp.sinh, "cosh": jnp.cosh, "floor": jnp.floor, "ceil": jnp.ceil, "round": jnp.round, "sign": jnp.sign, "reciprocal": jnp.reciprocal, "sigmoid": jax.nn.sigmoid, "relu": jax.nn.relu, "softplus": jax.nn.softplus, "gelu": jax.nn.gelu, "elu": jax.nn.elu, "selu": jax.nn.selu}
        elif LIB == "paddle":
            m = {"abs": paddle.abs, "negative": paddle.negative, "exp": paddle.exp, "expm1": paddle.expm1, "log": paddle.log, "log1p": paddle.log1p, "sqrt": paddle.sqrt, "square": paddle.square, "sin": paddle.sin, "cos": paddle.cos, "tan": paddle.tan, "tanh": paddle.tanh, "sinh": paddle.sinh, "cosh": paddle.cosh, "floor": paddle.floor, "ceil": paddle.ceil, "round": paddle.round, "sign": paddle.sign, "reciprocal": paddle.reciprocal, "sigmoid": F.sigmoid, "relu": F.relu, "softplus": F.softplus, "gelu": F.gelu, "elu": F.elu, "selu": F.selu}
        else:
            m = {"abs": ops.abs, "negative": ops.negative, "exp": ops.exp, "expm1": ops.expm1, "log": ops.log, "log1p": ops.log1p, "sqrt": ops.sqrt, "square": ops.square, "sin": ops.sin, "cos": ops.cos, "tan": ops.tan, "tanh": ops.tanh, "sinh": ops.sinh, "cosh": ops.cosh, "floor": ops.floor, "ceil": ops.ceil, "round": ops.round, "sign": ops.sign, "reciprocal": ops.reciprocal, "sigmoid": ops.sigmoid, "relu": ops.relu, "softplus": ops.softplus, "gelu": ops.gelu, "elu": ops.elu, "selu": ops.selu}
        return m[name](z)

    @staticmethod
    def binary(name, lhs, rhs):
        x0, y0 = B.t(lhs), B.t(rhs)
        if LIB == "torch": m = {"add": torch.add, "subtract": torch.subtract, "multiply": torch.multiply, "divide": torch.divide, "maximum": torch.maximum, "minimum": torch.minimum, "power": torch.pow, "equal": torch.eq, "not_equal": torch.ne, "greater": torch.gt, "greater_equal": torch.ge, "less": torch.lt, "less_equal": torch.le}
        elif LIB == "tensorflow": m = {"add": tf.add, "subtract": tf.subtract, "multiply": tf.multiply, "divide": tf.divide, "maximum": tf.maximum, "minimum": tf.minimum, "power": tf.pow, "equal": tf.equal, "not_equal": tf.not_equal, "greater": tf.greater, "greater_equal": tf.greater_equal, "less": tf.less, "less_equal": tf.less_equal}
        elif LIB == "keras": m = {"add": keras.ops.add, "subtract": keras.ops.subtract, "multiply": keras.ops.multiply, "divide": keras.ops.divide, "maximum": keras.ops.maximum, "minimum": keras.ops.minimum, "power": keras.ops.power, "equal": keras.ops.equal, "not_equal": keras.ops.not_equal, "greater": keras.ops.greater, "greater_equal": keras.ops.greater_equal, "less": keras.ops.less, "less_equal": keras.ops.less_equal}
        elif LIB == "jax": m = {"add": jnp.add, "subtract": jnp.subtract, "multiply": jnp.multiply, "divide": jnp.divide, "maximum": jnp.maximum, "minimum": jnp.minimum, "power": jnp.power, "equal": jnp.equal, "not_equal": jnp.not_equal, "greater": jnp.greater, "greater_equal": jnp.greater_equal, "less": jnp.less, "less_equal": jnp.less_equal}
        elif LIB == "paddle": m = {"add": paddle.add, "subtract": paddle.subtract, "multiply": paddle.multiply, "divide": paddle.divide, "maximum": paddle.maximum, "minimum": paddle.minimum, "power": paddle.pow, "equal": paddle.equal, "not_equal": paddle.not_equal, "greater": paddle.greater_than, "greater_equal": paddle.greater_equal, "less": paddle.less_than, "less_equal": paddle.less_equal}
        else: m = {"add": ops.add, "subtract": ops.sub, "multiply": ops.mul, "divide": ops.div, "maximum": ops.maximum, "minimum": ops.minimum, "power": ops.pow, "equal": ops.equal, "not_equal": ops.not_equal, "greater": ops.greater, "greater_equal": ops.greater_equal, "less": ops.less, "less_equal": ops.less_equal}
        return m[name](x0, y0)

    @staticmethod
    def logical(name, lhs, rhs=None):
        x0 = B.t(lhs, "bool"); y0 = B.t(rhs, "bool") if rhs is not None else None
        if LIB == "torch": m = {"and": torch.logical_and, "or": torch.logical_or, "not": torch.logical_not}
        elif LIB == "tensorflow": m = {"and": tf.logical_and, "or": tf.logical_or, "not": tf.logical_not}
        elif LIB == "keras": m = {"and": keras.ops.logical_and, "or": keras.ops.logical_or, "not": keras.ops.logical_not}
        elif LIB == "jax": m = {"and": jnp.logical_and, "or": jnp.logical_or, "not": jnp.logical_not}
        elif LIB == "paddle": m = {"and": paddle.logical_and, "or": paddle.logical_or, "not": paddle.logical_not}
        else: m = {"and": ops.logical_and, "or": ops.logical_or, "not": ops.logical_not}
        return m[name](x0) if rhs is None else m[name](x0, y0)

    @staticmethod
    def reduce(name, v, axis=1):
        z = B.t(v)
        if LIB == "torch":
            if name == "sum": return torch.sum(z, dim=axis)
            if name == "mean": return torch.mean(z, dim=axis)
            if name == "max": return torch.max(z, dim=axis).values
            if name == "min": return torch.min(z, dim=axis).values
            if name == "prod": return torch.prod(z, dim=axis)
            if name == "argmax": return torch.argmax(z, dim=axis)
            if name == "argmin": return torch.argmin(z, dim=axis)
            if name == "std": return torch.std(z, dim=axis, correction=0)
            if name == "var": return torch.var(z, dim=axis, correction=0)
            if name == "cumsum": return torch.cumsum(z, dim=axis)
            if name == "cumprod": return torch.cumprod(z, dim=axis)
        elif LIB == "tensorflow":
            if name == "sum": return tf.reduce_sum(z, axis=axis)
            if name == "mean": return tf.reduce_mean(z, axis=axis)
            if name == "max": return tf.reduce_max(z, axis=axis)
            if name == "min": return tf.reduce_min(z, axis=axis)
            if name == "prod": return tf.reduce_prod(z, axis=axis)
            if name == "argmax": return tf.argmax(z, axis=axis)
            if name == "argmin": return tf.argmin(z, axis=axis)
            if name == "std": return tf.math.reduce_std(z, axis=axis)
            if name == "var": return tf.math.reduce_variance(z, axis=axis)
            if name == "cumsum": return tf.cumsum(z, axis=axis)
            if name == "cumprod": return tf.math.cumprod(z, axis=axis)
        elif LIB == "keras":
            if name == "sum": return keras.ops.sum(z, axis=axis)
            if name == "mean": return keras.ops.mean(z, axis=axis)
            if name == "max": return keras.ops.max(z, axis=axis)
            if name == "min": return keras.ops.min(z, axis=axis)
            if name == "prod": return keras.ops.prod(z, axis=axis)
            if name == "argmax": return keras.ops.argmax(z, axis=axis)
            if name == "argmin": return keras.ops.argmin(z, axis=axis)
            if name == "std": return keras.ops.std(z, axis=axis)
            if name == "var": return keras.ops.var(z, axis=axis)
            if name == "cumsum": return keras.ops.cumsum(z, axis=axis)
            if name == "cumprod": return keras.ops.cumprod(z, axis=axis)
        elif LIB == "jax":
            if name == "sum": return jnp.sum(z, axis=axis)
            if name == "mean": return jnp.mean(z, axis=axis)
            if name == "max": return jnp.max(z, axis=axis)
            if name == "min": return jnp.min(z, axis=axis)
            if name == "prod": return jnp.prod(z, axis=axis)
            if name == "argmax": return jnp.argmax(z, axis=axis)
            if name == "argmin": return jnp.argmin(z, axis=axis)
            if name == "std": return jnp.std(z, axis=axis, correction=0)
            if name == "var": return jnp.var(z, axis=axis)
            if name == "cumsum": return jnp.cumsum(z, axis=axis)
            if name == "cumprod": return jnp.cumprod(z, axis=axis)
        elif LIB == "paddle":
            if name == "sum": return paddle.sum(z, axis=axis)
            if name == "mean": return paddle.mean(z, axis=axis)
            if name == "max": return paddle.max(z, axis=axis)
            if name == "min": return paddle.min(z, axis=axis)
            if name == "prod": return paddle.prod(z, axis=axis)
            if name == "argmax": return paddle.argmax(z, axis=axis)
            if name == "argmin": return paddle.argmin(z, axis=axis)
            if name == "std": return paddle.std(z, axis=axis, unbiased=False)
            if name == "var": return paddle.var(z, axis=axis, unbiased=False)
            if name == "cumsum": return paddle.cumsum(z, axis=axis)
            if name == "cumprod": return paddle.cumprod(z, dim=axis)
        else:
            if name == "sum": return ops.sum(z, dim=axis)
            if name == "mean": return ops.mean(z, axis=axis)
            if name == "max": return ops.max(z, axis=axis)[0]
            if name == "min": return ops.min(z, axis=axis)[0]
            if name == "prod": return ops.prod(z, axis=axis)
            if name == "argmax": return ops.argmax(z, dim=axis)
            if name == "argmin": return ops.argmin(z, axis=axis)
            if name == "std": return ops.std(z, axis=axis, ddof=0)
            if name == "var": return ops.var(z, axis=axis, ddof=0)
            if name == "cumsum": return ops.cumsum(z, axis=axis)
            if name == "cumprod": return ops.cumprod(z, dim=axis)
        raise NotImplementedError(name)

    @staticmethod
    def bool_reduce(name, v, axis=1):
        z = B.t(v, "bool")
        if LIB == "torch": return (torch.all if name == "all" else torch.any)(z, dim=axis)
        if LIB == "tensorflow": return (tf.reduce_all if name == "all" else tf.reduce_any)(z, axis=axis)
        if LIB == "keras": return (keras.ops.all if name == "all" else keras.ops.any)(z, axis=axis)
        if LIB == "jax": return (jnp.all if name == "all" else jnp.any)(z, axis=axis)
        if LIB == "paddle": return (paddle.all if name == "all" else paddle.any)(z, axis=axis)
        return (ops.all if name == "all" else ops.any)(z, axis=axis)

    @staticmethod
    def matmul():
        if LIB == "torch": return torch.matmul(B.t(mat1), B.t(mat2))
        if LIB == "tensorflow": return tf.matmul(B.t(mat1), B.t(mat2))
        if LIB == "keras": return keras.ops.matmul(B.t(mat1), B.t(mat2))
        if LIB == "jax": return jnp.matmul(B.t(mat1), B.t(mat2))
        if LIB == "paddle": return paddle.matmul(B.t(mat1), B.t(mat2))
        return ops.matmul(B.t(mat1), B.t(mat2))

    @staticmethod
    def shape(name):
        z = B.t(x)
        if LIB == "torch":
            if name == "reshape": return torch.reshape(z, (3, 2))
            if name == "transpose": return torch.transpose(z, 0, 1)
            if name == "squeeze": return torch.squeeze(torch.reshape(z, (1, 2, 3, 1)))
            if name == "expand_dims": return torch.unsqueeze(z, 0)
            if name == "concat": return torch.cat([z, z], dim=0)
            if name == "stack": return torch.stack([z, z], dim=0)
            if name == "tile": return torch.tile(z, (2, 1))
            if name == "broadcast_to": return torch.broadcast_to(torch.tensor([1.0, 2.0, 3.0]), (2, 3))
        if LIB == "tensorflow":
            if name == "reshape": return tf.reshape(z, (3, 2))
            if name == "transpose": return tf.transpose(z, perm=(1, 0))
            if name == "squeeze": return tf.squeeze(tf.reshape(z, (1, 2, 3, 1)))
            if name == "expand_dims": return tf.expand_dims(z, axis=0)
            if name == "concat": return tf.concat([z, z], axis=0)
            if name == "stack": return tf.stack([z, z], axis=0)
            if name == "tile": return tf.tile(z, (2, 1))
            if name == "broadcast_to": return tf.broadcast_to(tf.constant([1.0, 2.0, 3.0], dtype=tf.float32), (2, 3))
        if LIB == "keras":
            if name == "reshape": return keras.ops.reshape(z, (3, 2))
            if name == "transpose": return keras.ops.transpose(z, (1, 0))
            if name == "squeeze": return keras.ops.squeeze(keras.ops.reshape(z, (1, 2, 3, 1)))
            if name == "expand_dims": return keras.ops.expand_dims(z, axis=0)
            if name == "concat": return keras.ops.concatenate([z, z], axis=0)
            if name == "stack": return keras.ops.stack([z, z], axis=0)
            if name == "tile": return keras.ops.tile(z, (2, 1))
            if name == "broadcast_to": return keras.ops.broadcast_to(keras.ops.convert_to_tensor([1.0, 2.0, 3.0]), (2, 3))
        if LIB == "jax":
            if name == "reshape": return jnp.reshape(z, (3, 2))
            if name == "transpose": return jnp.transpose(z, (1, 0))
            if name == "squeeze": return jnp.squeeze(jnp.reshape(z, (1, 2, 3, 1)))
            if name == "expand_dims": return jnp.expand_dims(z, axis=0)
            if name == "concat": return jnp.concatenate([z, z], axis=0)
            if name == "stack": return jnp.stack([z, z], axis=0)
            if name == "tile": return jnp.tile(z, (2, 1))
            if name == "broadcast_to": return jnp.broadcast_to(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32), (2, 3))
        if LIB == "paddle":
            if name == "reshape": return paddle.reshape(z, (3, 2))
            if name == "transpose": return paddle.transpose(z, perm=(1, 0))
            if name == "squeeze": return paddle.squeeze(paddle.reshape(z, (1, 2, 3, 1)))
            if name == "expand_dims": return paddle.unsqueeze(z, axis=0)
            if name == "concat": return paddle.concat([z, z], axis=0)
            if name == "stack": return paddle.stack([z, z], axis=0)
            if name == "tile": return paddle.tile(z, (2, 1))
            if name == "broadcast_to": return paddle.broadcast_to(paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32"), (2, 3))
        if name == "reshape": return ops.reshape(z, (3, 2))
        if name == "transpose": return ops.transpose(z, (1, 0))
        if name == "squeeze": return ops.squeeze(ops.reshape(z, (1, 2, 3, 1)))
        if name == "expand_dims": return ops.expand_dims(z, 0)
        if name == "concat": return ops.concat((z, z), axis=0)
        if name == "stack": return ops.stack((z, z), axis=0)
        if name == "tile": return ops.tile(z, (2, 1))
        if name == "broadcast_to": return ops.broadcast_to(ms.Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32)), (2, 3))
        raise NotImplementedError(name)

    @staticmethod
    def misc(name):
        if name == "clip":
            if LIB == "torch": return torch.clamp(B.t(x), -1.0, 2.0)
            if LIB == "tensorflow": return tf.clip_by_value(B.t(x), -1.0, 2.0)
            if LIB == "keras": return keras.ops.clip(B.t(x), -1.0, 2.0)
            if LIB == "jax": return jnp.clip(B.t(x), -1.0, 2.0)
            if LIB == "paddle": return paddle.clip(B.t(x), min=-1.0, max=2.0)
            return ops.clip(B.t(x), -1.0, 2.0)
        if name == "where":
            if LIB == "torch": return torch.where(B.t(cond, "bool"), B.t(a), B.t(b))
            if LIB == "tensorflow": return tf.where(B.t(cond, "bool"), B.t(a), B.t(b))
            if LIB == "keras": return keras.ops.where(B.t(cond, "bool"), B.t(a), B.t(b))
            if LIB == "jax": return jnp.where(B.t(cond, "bool"), B.t(a), B.t(b))
            if LIB == "paddle": return paddle.where(B.t(cond, "bool"), B.t(a), B.t(b))
            return ops.where(B.t(cond, "bool"), B.t(a), B.t(b))
        if name == "softmax":
            if LIB == "torch": return F.softmax(B.t(x), dim=1)
            if LIB == "tensorflow": return tf.nn.softmax(B.t(x), axis=1)
            if LIB == "keras": return keras.ops.softmax(B.t(x), axis=1)
            if LIB == "jax": return jax.nn.softmax(B.t(x), axis=1)
            if LIB == "paddle": return F.softmax(B.t(x), axis=1)
            return ops.softmax(B.t(x), axis=1)
        if name == "log_softmax":
            if LIB == "torch": return F.log_softmax(B.t(x), dim=1)
            if LIB == "tensorflow": return tf.nn.log_softmax(B.t(x), axis=1)
            if LIB == "keras": return keras.ops.log_softmax(B.t(x), axis=1)
            if LIB == "jax": return jax.nn.log_softmax(B.t(x), axis=1)
            if LIB == "paddle": return F.log_softmax(B.t(x), axis=1)
            return ops.log_softmax(B.t(x), axis=1)
        if name == "logsumexp":
            if LIB == "torch": return torch.logsumexp(B.t(x), dim=1)
            if LIB == "tensorflow": return tf.reduce_logsumexp(B.t(x), axis=1)
            if LIB == "keras": return keras.ops.logsumexp(B.t(x), axis=1)
            if LIB == "jax": return jax.nn.logsumexp(B.t(x), axis=1)
            if LIB == "paddle": return paddle.logsumexp(B.t(x), axis=1)
            return ops.logsumexp(B.t(x), 1)
        if name == "sort":
            if LIB == "torch": return torch.sort(B.t(vec4), dim=0).values
            if LIB == "tensorflow": return tf.sort(B.t(vec4), axis=0)
            if LIB == "keras": return keras.ops.sort(B.t(vec4), axis=0)
            if LIB == "jax": return jnp.sort(B.t(vec4), axis=0)
            if LIB == "paddle": return paddle.sort(B.t(vec4), axis=0)
            return ops.sort(B.t(vec4), axis=0)[0]
        if name == "argsort":
            if LIB == "torch": return torch.argsort(B.t(vec4), dim=0)
            if LIB == "tensorflow": return tf.argsort(B.t(vec4), axis=0)
            if LIB == "keras": return keras.ops.argsort(B.t(vec4), axis=0)
            if LIB == "jax": return jnp.argsort(B.t(vec4), axis=0)
            if LIB == "paddle": return paddle.argsort(B.t(vec4), axis=0)
            return ops.argsort(B.t(vec4), axis=0)
        if name == "angle_complex_nan":
            c = np.array(complex(float("nan"), float("nan")), dtype=np.complex64)
            if LIB == "torch": return torch.angle(torch.tensor(c))
            if LIB == "tensorflow": return tf.math.angle(tf.constant(c, dtype=tf.complex64))
            if LIB == "keras": return keras.ops.angle(keras.ops.convert_to_tensor(c))
            if LIB == "jax": return jnp.angle(jnp.array(c, dtype=jnp.complex64))
            if LIB == "paddle": return paddle.angle(paddle.to_tensor(c))
            return ops.angle(ms.Tensor(c))
        raise NotImplementedError(name)

    @staticmethod
    def linalg(name):
        if LIB == "torch":
            if name == "det": return torch.linalg.det(B.t(square))
            if name == "inv": return torch.linalg.inv(B.t(square))
            if name == "solve": return torch.linalg.solve(B.t(square), B.t(solve_rhs))
            if name == "cholesky": return torch.linalg.cholesky(B.t(pdmat))
            if name == "eigh": return torch.linalg.eigh(B.t(pdmat))[0]
            if name == "svd_s": return torch.linalg.svd(B.t(mat1), full_matrices=False)[1]
            if name == "qr_r": return torch.linalg.qr(B.t(mat1), mode="reduced")[1]
            if name == "norm": return torch.linalg.norm(B.t(mat1))
        if LIB == "tensorflow":
            if name == "det": return tf.linalg.det(B.t(square))
            if name == "inv": return tf.linalg.inv(B.t(square))
            if name == "solve": return tf.linalg.solve(B.t(square), B.t(solve_rhs))
            if name == "cholesky": return tf.linalg.cholesky(B.t(pdmat))
            if name == "eigh": return tf.linalg.eigh(B.t(pdmat))[0]
            if name == "svd_s": return tf.linalg.svd(B.t(mat1), full_matrices=False, compute_uv=False)
            if name == "qr_r": return tf.linalg.qr(B.t(mat1), full_matrices=False)[1]
            if name == "norm": return tf.linalg.norm(B.t(mat1))
        if LIB == "keras":
            if name == "det": return keras.ops.det(B.t(square))
            if name == "inv": return keras.ops.inv(B.t(square))
            if name == "solve": return keras.ops.solve(B.t(square), B.t(solve_rhs))
            if name == "cholesky": return keras.ops.cholesky(B.t(pdmat))
            if name == "eigh": return keras.ops.eigh(B.t(pdmat))[0]
            if name == "svd_s": return keras.ops.svd(B.t(mat1), full_matrices=False, compute_uv=False)
            if name == "qr_r": return keras.ops.qr(B.t(mat1), mode="reduced")[1]
            if name == "norm": return keras.ops.norm(B.t(mat1))
        if LIB == "jax":
            if name == "det": return jnp.linalg.det(B.t(square))
            if name == "inv": return jnp.linalg.inv(B.t(square))
            if name == "solve": return jnp.linalg.solve(B.t(square), B.t(solve_rhs))
            if name == "cholesky": return jnp.linalg.cholesky(B.t(pdmat))
            if name == "eigh": return jnp.linalg.eigh(B.t(pdmat))[0]
            if name == "svd_s": return jnp.linalg.svd(B.t(mat1), full_matrices=False, compute_uv=False)
            if name == "qr_r": return jnp.linalg.qr(B.t(mat1), mode="reduced")[1]
            if name == "norm": return jnp.linalg.norm(B.t(mat1))
        if LIB == "paddle":
            if name == "det": return paddle.linalg.det(B.t(square))
            if name == "inv": return paddle.linalg.inv(B.t(square))
            if name == "solve": return paddle.linalg.solve(B.t(square), B.t(solve_rhs))
            if name == "cholesky": return paddle.linalg.cholesky(B.t(pdmat))
            if name == "eigh": return paddle.linalg.eigh(B.t(pdmat))[0]
            if name == "svd_s": return paddle.linalg.svd(B.t(mat1), full_matrices=False)[1]
            if name == "qr_r": return paddle.linalg.qr(B.t(mat1), mode="reduced")[1]
            if name == "norm": return paddle.linalg.norm(B.t(mat1))
        if name == "det": return ops.det(B.t(square))
        if name == "inv": return ops.inverse(B.t(square))
        if name == "solve": return ops.matrix_solve(B.t(square), B.t(solve_rhs))
        if name == "cholesky": return ops.cholesky(B.t(pdmat))
        if name == "eigh": return msl.eigh(B.t(pdmat))[0]
        if name == "svd_s": return ops.svd(B.t(mat1), full_matrices=False, compute_uv=False)
        if name == "qr_r": return ops.qr(B.t(mat1), mode="reduced")[1]
        if name == "norm": return ops.norm(B.t(mat1))
        raise NotImplementedError(name)

    @staticmethod
    def edge(name):
        if name == "reciprocal_zero": return B.unary("reciprocal", zero)
        if name == "divide_zero": return B.binary("divide", np.array([1.0, -1.0, 0.0, -0.0], dtype=np.float32), div_zero_denom)
        if name == "log_edge": return B.unary("log", log_edge)
        if name == "sqrt_edge": return B.unary("sqrt", sqrt_edge)
        if name == "softmax_inf":
            if LIB == "torch": return F.softmax(B.t(softmax_inf), dim=1)
            if LIB == "tensorflow": return tf.nn.softmax(B.t(softmax_inf), axis=1)
            if LIB == "keras": return keras.ops.softmax(B.t(softmax_inf), axis=1)
            if LIB == "jax": return jax.nn.softmax(B.t(softmax_inf), axis=1)
            if LIB == "paddle": return F.softmax(B.t(softmax_inf), axis=1)
            return ops.softmax(B.t(softmax_inf), axis=1)
        if name == "logsumexp_nan":
            if LIB == "torch": return torch.logsumexp(B.t(logsumexp_nan), dim=1)
            if LIB == "tensorflow": return tf.reduce_logsumexp(B.t(logsumexp_nan), axis=1)
            if LIB == "keras": return keras.ops.logsumexp(B.t(logsumexp_nan), axis=1)
            if LIB == "jax": return jax.nn.logsumexp(B.t(logsumexp_nan), axis=1)
            if LIB == "paddle": return paddle.logsumexp(B.t(logsumexp_nan), axis=1)
            return ops.logsumexp(B.t(logsumexp_nan), 1)
        if name == "sort_nan":
            if LIB == "torch": return torch.sort(B.t(sort_nan), dim=0).values
            if LIB == "tensorflow": return tf.sort(B.t(sort_nan), axis=0)
            if LIB == "keras": return keras.ops.sort(B.t(sort_nan), axis=0)
            if LIB == "jax": return jnp.sort(B.t(sort_nan), axis=0)
            if LIB == "paddle": return paddle.sort(B.t(sort_nan), axis=0)
            return ops.sort(B.t(sort_nan), axis=0)[0]
        if name == "argsort_signed_zero":
            if LIB == "torch": return torch.argsort(B.t(paper_argsort), dim=0)
            if LIB == "tensorflow": return tf.argsort(B.t(paper_argsort), axis=0)
            if LIB == "keras": return keras.ops.argsort(B.t(paper_argsort), axis=0)
            if LIB == "jax": return jnp.argsort(B.t(paper_argsort), axis=0)
            if LIB == "paddle": return paddle.argsort(B.t(paper_argsort), axis=0)
            return ops.argsort(B.t(paper_argsort), axis=0)
        if name == "min_nan": return B.reduce("min", minmax_nan, axis=1)
        if name == "max_nan": return B.reduce("max", minmax_nan, axis=1)
        if name == "sum_nan": return B.reduce("sum", minmax_nan, axis=1)
        if name == "mean_nan": return B.reduce("mean", minmax_nan, axis=1)
        raise NotImplementedError(name)

def build_cases():
    cases = {}
    for name in ["abs", "negative", "exp", "expm1", "log", "log1p", "sqrt", "square", "sin", "cos", "tan", "tanh", "sinh", "cosh", "floor", "ceil", "round", "sign", "reciprocal", "sigmoid", "relu", "softplus", "gelu", "elu", "selu"]:
        arr = pos if name in {"log", "sqrt", "reciprocal"} else small_pos if name == "log1p" else x
        if name == "round": arr = round_values
        cases[f"unary_{name}"] = (lambda n=name, v=arr: B.unary(n, v))
    for name in ["add", "subtract", "multiply", "divide", "maximum", "minimum", "power"]:
        lhs, rhs = (pos, np.full_like(pos, 1.5)) if name == "power" else (a, b)
        cases[f"binary_{name}"] = (lambda n=name, l=lhs, r=rhs: B.binary(n, l, r))
    for name in ["equal", "not_equal", "greater", "greater_equal", "less", "less_equal"]:
        cases[f"compare_{name}"] = (lambda n=name: B.binary(n, a, b))
    cases["logical_and"] = lambda: B.logical("and", bool_a, bool_b)
    cases["logical_or"] = lambda: B.logical("or", bool_a, bool_b)
    cases["logical_not"] = lambda: B.logical("not", bool_a)
    for name in ["sum", "mean", "max", "min", "prod", "argmax", "argmin", "std", "var", "cumsum", "cumprod"]:
        cases[f"reduce_{name}_axis1"] = (lambda n=name: B.reduce(n, x, axis=1))
    cases["bool_all_axis1"] = lambda: B.bool_reduce("all", bool_a, axis=1)
    cases["bool_any_axis1"] = lambda: B.bool_reduce("any", bool_a, axis=1)
    for name in ["reshape", "transpose", "squeeze", "expand_dims", "concat", "stack", "tile", "broadcast_to"]:
        cases[f"shape_{name}"] = (lambda n=name: B.shape(n))
    for name in ["clip", "where", "softmax", "log_softmax", "logsumexp", "sort", "argsort", "angle_complex_nan"]:
        cases[f"misc_{name}"] = (lambda n=name: B.misc(n))
    cases["matmul"] = B.matmul
    for name in ["det", "inv", "solve", "cholesky", "eigh", "svd_s", "qr_r", "norm"]:
        cases[f"linalg_{name}"] = (lambda n=name: B.linalg(n))
    for name in ["reciprocal_zero", "divide_zero", "log_edge", "sqrt_edge", "softmax_inf", "logsumexp_nan", "sort_nan", "argsort_signed_zero", "min_nan", "max_nan", "sum_nan", "mean_nan"]:
        cases[f"edge_{name}"] = (lambda n=name: B.edge(n))
    return cases

print(json.dumps({LIB: {name: run_one(fn) for name, fn in build_cases().items()}}, allow_nan=True))
'''

def run_library(lib: str, executable: str) -> dict[str, Any]:
    env = os.environ.copy()
    env["XAMT_LIB"] = lib
    raw = subprocess.check_output([executable, "-B", "-c", RUNNER_CODE], text=True, stderr=subprocess.DEVNULL, env=env, timeout=180)
    return json.loads(raw.strip().splitlines()[-1])[lib]

def leaf_to_array(obj: dict[str, Any]) -> np.ndarray:
    return np.asarray(obj["value"])

def encoded_equal(a: Any, b: Any) -> bool:
    if isinstance(a, list) or isinstance(b, list):
        return isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and all(encoded_equal(x, y) for x, y in zip(a, b))
    aa, bb = leaf_to_array(a), leaf_to_array(b)
    if aa.shape != bb.shape:
        return False
    if aa.dtype.kind in "bui" and bb.dtype.kind in "bui":
        return np.array_equal(aa, bb)
    return bool(np.allclose(aa, bb, atol=1e-5, rtol=1e-5, equal_nan=True))

def summarize_value(obj: Any) -> Any:
    if isinstance(obj, list):
        return [summarize_value(x) for x in obj]
    return obj["value"]

def main() -> None:
    by_lib = {lib: run_library(lib, exe) for lib, exe in LIBS.items()}
    all_cases = sorted(set().union(*(set(results) for results in by_lib.values())))
    status: dict[str, str] = {}
    details: list[tuple[str, str, dict[str, Any]]] = []
    for case in all_cases:
        results = {lib: by_lib[lib].get(case, {"ok": False, "error": "missing case"}) for lib in LIBS}
        errors = {lib: r["error"] for lib, r in results.items() if not r.get("ok")}
        if errors:
            status[case] = "ERROR"
            details.append((case, "ERROR", errors))
            continue
        encoded = {lib: r["result"] for lib, r in results.items()}
        base = next(iter(encoded.values()))
        if all(encoded_equal(base, r) for r in encoded.values()):
            status[case] = "PASS"
        else:
            status[case] = "DIFF"
            details.append((case, "DIFF", encoded))
    counts = Counter(status.values())
    print("libraries:", ", ".join(LIBS))
    print("total_cases:", len(all_cases))
    print("summary:", json.dumps({k: counts.get(k, 0) for k in ["PASS", "DIFF", "ERROR"]}, sort_keys=True))
    print("checked_all_6:", counts.get("PASS", 0) + counts.get("DIFF", 0))
    print()
    print("DIFF_DETAILS")
    for case, kind, data in details:
        if kind == "DIFF":
            print(f"{case}: DIFF")
            for lib, result in data.items():
                print(f"  {lib:10s} {summarize_value(result)}")
    print()
    print("ERROR_DETAILS")
    for case, kind, data in details:
        if kind == "ERROR":
            print(f"{case}: ERROR {data}")

if __name__ == "__main__":
    main()
