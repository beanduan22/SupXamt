
from __future__ import annotations

import argparse
import atexit
import importlib
import json
import os
import select
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    from .compare_api_matchers import Api, collect_apis, confidence_band, cross_library_groups, group_apis, group_confidence, role_aware_groups
except ImportError:
    from compare_api_matchers import Api, collect_apis, confidence_band, cross_library_groups, group_apis, group_confidence, role_aware_groups


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def optional_external_python(*env_names: str, default: str = "") -> str:
    for env_name in env_names:
        path = os.environ.get(env_name)
        if path:
            return path
    if default and os.path.exists(default):
        return default
    return ""


EXTERNAL_LIBRARY_PYTHONS = {
    "paddle": optional_external_python("XAMT_PADDLE_PY", "XAMT_PY312", default="/tmp/xamt_py312/bin/python"),
    "mindspore": optional_external_python("XAMT_MINDSPORE_PY", "XAMT_PY312", default="/tmp/xamt_py312/bin/python"),
    "chainer": optional_external_python("XAMT_CHAINER_PY", "XAMT_PY39", default="/tmp/xamt_py39/bin/python"),
    "mxnet": optional_external_python("XAMT_MXNET_PY", "XAMT_PY39", default="/tmp/xamt_py39/bin/python"),
}
EXTERNAL_RUNNER_FLAG = "XAMT_EXTERNAL_RUNNER"
EXTERNAL_WORKERS: dict[str, subprocess.Popen[str]] = {}
EXTERNAL_RESPONSE_TIMEOUT_SECONDS = float(os.environ.get("XAMT_EXTERNAL_RESPONSE_TIMEOUT_SECONDS", "30"))


X = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]], dtype=np.float32)
Y = np.array([[2.0, 3.0, -1.0], [0.5, -4.0, 2.0]], dtype=np.float32)
POS = np.array([[0.25, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=np.float32)
VEC = np.array([1.0, -2.0, 3.0, 4.0], dtype=np.float32)
INT_VEC = np.array([0, 1, 1, 2, 2, 2], dtype=np.int64)
BOOL = np.array([[True, False, True], [False, True, False]])
MAT = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
MAT_B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
SPD = np.array([[2.0, 0.3], [0.3, 1.0]], dtype=np.float32)
RHS = np.array([[1.0], [2.0]], dtype=np.float32)
VEC_A = np.array([1.0, -2.0, 3.0], dtype=np.float32)
VEC_B = np.array([4.0, 0.5, -1.0], dtype=np.float32)
INT_A = np.array([1, 2, 4], dtype=np.int64)
INT_B = np.array([1, 1, 1], dtype=np.int64)
EVEN = np.array([[1.0, -2.0, 3.0, -4.0], [4.0, 0.5, -6.0, 2.0]], dtype=np.float32)
Y_TRUE = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
Y_PRED = np.array([[0.9, 0.2, 0.8], [0.1, 0.7, 0.3]], dtype=np.float32)
CAT_TRUE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
CAT_PRED = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]], dtype=np.float32)
SPARSE_TRUE = np.array([0, 1], dtype=np.int64)
CURRENT_QNAME = ""


RUNTIME_ARRAY_NAMES = (
    "X",
    "Y",
    "POS",
    "VEC",
    "INT_VEC",
    "BOOL",
    "MAT",
    "MAT_B",
    "SPD",
    "RHS",
    "VEC_A",
    "VEC_B",
    "INT_A",
    "INT_B",
    "EVEN",
    "Y_TRUE",
    "Y_PRED",
    "CAT_TRUE",
    "CAT_PRED",
    "SPARSE_TRUE",
)


def export_runtime_state() -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    for name in RUNTIME_ARRAY_NAMES:
        value = np.asarray(globals()[name])
        state[name] = {"dtype": str(value.dtype), "value": value.tolist()}
    return state


def apply_runtime_state(state: Optional[dict[str, dict[str, Any]]]) -> None:
    if not state:
        return
    for name in RUNTIME_ARRAY_NAMES:
        if name not in state:
            continue
        payload = state[name]
        globals()[name] = np.asarray(payload["value"], dtype=np.dtype(payload["dtype"]))


NON_COMPARABLE = {
    "array_str",
    "vmap",
    "empty",
    "empty_like",
    "gumbel_softmax",
    "manual_seed",
    "print",
}

CONFIG_CATEGORIES = {
    "activation_config",
    "layer_config",
    "loss_config",
    "metric_config",
}


@dataclass(frozen=True)
class Execution:
    api: Api
    value: Any


class UnsupportedCall(Exception):
    pass


class SkipCall(Exception):
    pass


def resolve(api: Api) -> Any:
    module_name, _, attr = api.qualified_name.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def candidate_priority(api: Api) -> tuple[int, str]:
    qname = api.qualified_name
    if api.library == "mindspore":
        cpu_preferred = (
            "conv",
            "conv1d",
            "conv2d",
            "conv3d",
            "avg_pool1d",
            "avg_pool2d",
            "avg_pool3d",
            "max_pool1d",
            "max_pool2d",
            "max_pool3d",
        )
        if api.name in cpu_preferred and qname.startswith(("mindspore.ops.", "mindspore.ops.function.", "mindspore.ops.functional.")):
            return (-1, qname)
        if api.name in cpu_preferred and qname.startswith("mindspore.mint.nn.functional."):
            return (4, qname)
    preferred_segments = (
        ".ops.",
        ".math.",
        ".linalg.",
        ".nn.",
        ".functions.",
        ".ndarray.",
        ".numpy.",
        ".special.",
        ".signal.",
        ".fft.",
        ".image.",
        ".random.",
    )
    if any(segment in qname for segment in preferred_segments):
        return (0, qname)
    if ".keras.activations." in qname or ".keras.losses." in qname or ".keras.metrics." in qname:
        return (1, qname)
    if ".layers." in qname:
        return (3, qname)
    return (2, qname)


def tensor(value: Any, library: str) -> Any:
    if isinstance(value, np.ndarray):
        if library == "torch":
            import torch

            return torch.tensor(value)
        if library == "tensorflow":
            import tensorflow as tf

            return tf.constant(value)
        if library == "jax":
            import jax.numpy as jnp

            return jnp.array(value)
        if library == "paddle":
            import paddle

            return paddle.to_tensor(value)
        if library == "mindspore":
            import mindspore as ms

            return ms.Tensor(value)
        if library == "mxnet":
            import mxnet as mx

            if CURRENT_QNAME.startswith("mxnet.numpy."):
                mx.npx.set_np()
                return mx.np.array(value)
            return mx.nd.array(value)
        if library == "chainer":
            return value
        if library == "keras":
            import keras

            return keras.ops.convert_to_tensor(value)
    return value


def dtype_arg(library: str, name: str) -> Any:
    if library == "torch":
        import torch

        return getattr(torch, name)
    if library == "tensorflow":
        import tensorflow as tf

        return getattr(tf, name)
    if library == "jax":
        import jax.numpy as jnp

        return getattr(jnp, name)
    if library == "paddle":
        import paddle

        return getattr(paddle, name)
    if library == "mindspore":
        import mindspore as ms

        return getattr(ms, name)
    if library == "mxnet":
        return name
    return getattr(np, name)


def args_for(api: Api, canonical: str, category: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
    global CURRENT_QNAME
    lib = api.library
    qname = api.qualified_name
    CURRENT_QNAME = qname

    if qname.startswith("chainer.links."):
        raise SkipCall("Chainer Links are stateful layer objects, not direct tensor functions")
    if qname.startswith("paddle.tensor."):
        raise SkipCall("paddle.tensor aliases include low-level wrappers; prefer stable paddle top-level APIs")
    if qname.rpartition(".")[2].endswith("_"):
        raise SkipCall("in-place tensor variants mutate inputs")
    if qname.startswith(("mxnet.symbol.", "mxnet.gluon.")):
        raise SkipCall("MXNet symbolic/gluon APIs need graph or layer adapters")
    if qname.startswith("mxnet.ndarray.contrib."):
        raise SkipCall("MXNet contrib operators are not part of the stable imperative API set")
    if canonical in NON_COMPARABLE:
        raise SkipCall(f"{canonical} is non-deterministic or not value-comparable")
    if canonical in {"deserialize", "serialize", "get"} and category not in CONFIG_CATEGORIES:
        raise SkipCall(f"{canonical} needs framework-specific object/function adapters")
    if canonical in {"from_dlpack", "to_dlpack", "load", "save", "savez"}:
        raise SkipCall(f"{canonical} needs external buffers or files")
    if canonical in {
        "assign",
        "get_device_module",
        "set_default_device",
        "set_default_dtype",
        "initial_seed",
        "get_rng_state",
        "set_rng_state",
        "get_autocast_cpu_dtype",
        "get_autocast_gpu_dtype",
        "get_autocast_dtype",
        "is_autocast_enabled",
    }:
        raise SkipCall(f"{canonical} mutates or returns framework runtime state")
    if canonical == "kl_divergence" and (qname.startswith("jax.scipy.special.") or qname == "torch.kl_div"):
        raise SkipCall(f"{qname} is an elementwise KL helper, not the reduced loss API")
    if canonical == "kl_divergence" and lib == "mindspore":
        raise SkipCall("MindSpore kl_div expects log-probability inputs and has KLDivLoss semantics")
    if canonical == "inv" and category == "linalg" and qname == "mindspore.ops.function.inv":
        raise SkipCall("mindspore.ops.function.inv is elementwise reciprocal, not matrix inverse")
    if canonical in {"batch_norm_elemt", "batch_norm_gather_stats_with_counts", "batch_norm_stats", "householder_product", "lstm", "lu_unpack", "orgqr", "ormqr", "split_with_sizes"}:
        raise SkipCall(f"{canonical} is a low-level/internal API that needs framework-specific state adapters")
    if canonical == "linear" and qname.endswith(".activations.linear"):
        raise SkipCall("activation linear is an identity activation, not an affine layer")
    if canonical == "hardsigmoid" and lib in {"chainer", "mxnet"}:
        raise SkipCall("hard_sigmoid uses a different historical slope/offset than keras/tensorflow/mindspore")
    if canonical in {"max", "min"} and qname in {"mindspore.numpy.max", "mindspore.numpy.amax", "mindspore.numpy.min", "mindspore.numpy.amin"}:
        raise SkipCall("MindSpore NumPy min/max reductions are unstable in this environment")
    if canonical == "corrcoef" and qname == "mindspore.numpy.corrcoef":
        raise SkipCall("MindSpore NumPy corrcoef is unstable in this environment")
    if canonical == "flatten" and qname in {"mindspore.ops.flatten", "mxnet.ndarray.Flatten", "mxnet.ndarray.flatten"}:
        raise SkipCall("this flatten variant preserves the batch dimension instead of flattening all elements")
    if canonical == "stft" and qname == "tensorflow.signal.stft":
        raise SkipCall("tensorflow.signal.stft uses frame layout/padding semantics that need a separate adapter")
    if canonical == "sequence_mask" and lib == "mxnet":
        raise SkipCall("MXNet SequenceMask masks data values, not TensorFlow-style boolean mask construction")
    if canonical == "batch_norm" and lib == "chainer":
        raise SkipCall("Chainer batch_normalization uses training-stat semantics that need a separate adapter")
    if canonical == "bilinear" and lib in {"chainer", "paddle"}:
        raise SkipCall("bilinear needs framework-specific weight/bias layout normalization")
    if canonical == "conv" and lib == "mxnet":
        raise SkipCall("MXNet Convolution requires operator-attribute adapters")
    if canonical == "conv_transpose" and lib == "jax":
        raise SkipCall("JAX lax conv_transpose needs explicit dimension-number adapters")
    if canonical == "ctc_loss" and lib in {"mindspore", "mxnet"}:
        raise SkipCall(f"{lib} ctc_loss needs framework-specific sparse-label adapters")
    if canonical == "cond" and lib == "mindspore":
        raise SkipCall("MindSpore cond captures this runner's matrix input state incorrectly")
    if canonical == "group_norm" and lib == "mxnet":
        raise SkipCall("MXNet GroupNorm requires operator-attribute adapters")
    if canonical == "instance_norm" and lib == "mxnet":
        raise SkipCall("MXNet InstanceNorm requires running-stat attribute adapters")
    if canonical == "linear" and lib == "mxnet":
        raise SkipCall("MXNet FullyConnected requires num_hidden and layout-specific adapters")
    if canonical == "identity" and qname == "chainer.functions.identity":
        raise SkipCall("chainer identity is a tensor passthrough, not an identity-matrix constructor")
    if canonical == "select" and lib == "torch":
        raise SkipCall("torch.select indexes one dimension, unlike numpy select")
    if canonical == "slice" and lib == "mxnet":
        raise SkipCall("MXNet slice needs begin/end operator attributes")
    if canonical in {"fractional_max_pool2d", "fractional_max_pool3d"}:
        raise SkipCall("fractional max-pool needs deterministic random-sample adapters")
    if canonical == "scaled_dot_product_attention":
        raise SkipCall("scaled_dot_product_attention uses different query/key/value layout conventions")
    if canonical == "topk" and qname.endswith("in_top_k"):
        raise SkipCall("in_top_k returns membership booleans, not top-k values")
    if canonical == "topk" and lib == "mxnet":
        raise SkipCall("MXNet topk needs a separate axis/return-type adapter")
    if canonical == "stft" and lib == "keras":
        raise SkipCall("keras stft tuple layout differs from scipy/torch stft in this runner")
    if canonical in {"approx_max_k", "approx_min_k"} and lib == "tensorflow":
        raise SkipCall("TensorFlow ApproxTopK kernels are unavailable on this CPU runner")
    if canonical in {"fftshift", "ifftshift"} and lib == "paddle":
        raise SkipCall("paddle fftshift/ifftshift fails in the installed backend roll kernel")
    if canonical == "exponential" and qname == "paddle.tensor.exponential_":
        raise SkipCall("paddle.tensor.exponential_ samples an in-place exponential distribution")
    if canonical == "result_type" and lib == "mxnet":
        raise SkipCall("mxnet.numpy.result_type returns a numpy dtype through a wrapper that cannot encode it as mx.np.ndarray")
    if canonical in {"deg2rad", "rad2deg"} and lib == "mxnet":
        raise SkipCall("mxnet.numpy deg/rad helpers are declared but unavailable in the installed MXNet backend")
    if canonical == "dsplit" and lib == "mxnet":
        raise SkipCall("mxnet.numpy.dsplit hangs in the installed MXNet backend under this runner")
    if canonical == "histogram_bin_edges" and lib == "mxnet":
        raise SkipCall("mxnet.numpy.histogram_bin_edges hangs in the installed MXNet backend under this runner")
    if canonical in {"conj", "real", "imag", "view_as_real"} and lib == "mxnet":
        raise SkipCall("the installed MXNet NumPy backend does not support complex-valued arrays")
    if canonical == "copy" and lib == "chainer":
        raise SkipCall("chainer.functions.copy transfers arrays between devices, not NumPy-style value copying")
    if canonical == "min_scalar_type" and lib == "mxnet":
        raise SkipCall("mxnet.numpy.min_scalar_type cannot encode numpy dtype inputs in this runner")
    if category == "random" and canonical in {"categorical", "fold_in", "log_uniform_candidate_sampler", "rand_like", "shuffle", "split"}:
        raise SkipCall(f"{canonical} random API is not value-comparable in this runner")
    if category == "random" and qname.rpartition(".")[2].endswith("_"):
        raise SkipCall("in-place random fill variants mutate an existing tensor")
    if canonical in {"dropout", "dropout1d", "dropout2d", "dropout3d", "alpha_dropout", "feature_alpha_dropout", "rrelu"} and qname.rpartition(".")[2].endswith("_"):
        raise SkipCall("in-place neural-network variants mutate an existing tensor")
    if canonical == "resize" and qname in {"jax.numpy.resize", "mxnet.numpy.resize", "numpy.resize"}:
        raise SkipCall(f"{qname} repeats array data; it is not image resize")
    if lib == "mxnet" and canonical in {"crop", "depth_to_space", "space_to_depth"}:
        raise SkipCall(f"MXNet {canonical} needs a layout/operator-specific adapter")
    if qname == "mxnet.ndarray.random_exponential":
        raise SkipCall("mxnet.ndarray.random_exponential is a sampler, not elementwise exponential")
    if lib == "mindspore" and canonical in {"max_unpool1d", "max_unpool2d", "max_unpool3d", "quantile"}:
        raise SkipCall(f"MindSpore {canonical} is unavailable on the current CPU backend")
    if lib == "mindspore" and canonical == "space_to_batch_nd":
        raise SkipCall("MindSpore space_to_batch_nd needs a layout-specific adapter")
    if lib == "paddle" and canonical == "rms_norm":
        raise SkipCall("Paddle rms_norm kernel is unavailable in the current runner")
    if canonical == "argpartition" and qname == "keras.ops.argpartition":
        raise SkipCall("keras.ops.argpartition is not value-compatible with numpy argpartition here")
    if canonical == "repeat" and qname == "paddle.tensor.repeat":
        raise SkipCall("paddle.tensor.repeat tiles data; repeat_interleave has numpy repeat semantics")
    if canonical == "binomial" and category != "random":
        raise SkipCall("torch/paddle binomial APIs are sampling helpers with different semantics")
    if canonical == "bernoulli" and qname.endswith("special.bernoulli"):
        raise SkipCall("special.bernoulli returns Bernoulli numbers, not random samples")
    if category == "random" and lib == "mindspore" and canonical in {"gamma", "random_integer", "uniform", "uniform_candidate_sampler"}:
        raise SkipCall(f"MindSpore {canonical} random API needs a framework-specific adapter")
    if canonical == "uniform_candidate_sampler":
        raise SkipCall("uniform_candidate_sampler needs categorical sampler-specific inputs")
    if canonical == "geterr":
        raise SkipCall("geterr returns library-specific floating-point error policy")
    if canonical in {"set_printoptions", "seed"}:
        raise SkipCall(f"{canonical} mutates or returns library-specific runtime state")

    if qname.startswith("scipy.stats.") and canonical not in {"rankdata", "sem"}:
        raise SkipCall("scipy.stats distribution objects are not tensor math APIs")
    if qname.startswith("scipy.ndimage."):
        raise SkipCall("scipy.ndimage APIs use image/label semantics, not plain tensor semantics")
    if canonical == "identity" and qname == "tensorflow.identity":
        raise SkipCall("tensorflow.identity is a passthrough op, not an identity matrix constructor")
    if canonical == "layer_norm" and lib == "mindspore":
        raise SkipCall("MindSpore layer_norm CPU kernel is unavailable in this runner")
    if lib == "mindspore" and canonical in {"accumulate_n", "atleast_1d", "atleast_2d", "atleast_3d", "angle", "batch_norm", "clone", "conv2d", "conv_transpose2d", "dropout2d", "eigh_tridiagonal", "embedding", "einsum", "fmin", "fold", "index_select", "jvp", "kthvalue", "l1_loss", "matrix_exp", "mse_loss", "nanquantile", "normalize", "one_hot", "rms_norm", "scatter_add", "soft_margin_loss", "stft", "triplet_margin_loss", "upsample"}:
        raise SkipCall(f"MindSpore {canonical} is unavailable or unstable on the current CPU backend")
    if lib == "mindspore" and category == "nn" and canonical in {"adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d", "adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d", "cross_entropy", "elu", "hardtanh", "logsigmoid", "mish", "pad", "relu6", "selu", "threshold", "unfold"}:
        raise SkipCall(f"MindSpore {canonical} nn kernel is unavailable on the current CPU backend")
    if lib == "mindspore" and category == "random" and canonical in {"normal", "poisson"}:
        raise SkipCall(f"MindSpore {canonical} random API needs a framework-specific adapter")
    if qname in {"mindspore.mint.argmax", "mindspore.mint.argsort", "mindspore.mint.count_nonzero", "mindspore.mint.logsumexp", "mindspore.mint.sort", "mindspore.mint.special.log_softmax", "mindspore.mint.unique"}:
        raise SkipCall(f"{qname} CPU kernel is unavailable in this runner")
    if qname == "torch.solve":
        raise SkipCall("torch.solve was removed; use torch.linalg.solve instead")
    if canonical in {"put", "randint_like", "randperm", "reduce", "rrelu", "scatter_nd_add"}:
        raise SkipCall(f"{canonical} needs a mutation/random/low-level adapter")
    if qname == "mxnet.ndarray.ravel_multi_index":
        raise SkipCall("mxnet.ndarray.ravel_multi_index expects a single NDArray coordinate encoding")
    if qname == "mxnet.ndarray.gamma":
        raise SkipCall("mxnet.ndarray.gamma uses ndarray shape/data semantics that need a separate adapter")
    if canonical == "equal" and qname == "torch.equal":
        raise SkipCall("torch.equal returns scalar all-elements equality, not elementwise equality")
    if qname.startswith("jax.lax.scatter") or qname == "jax.lax.gather":
        raise SkipCall("JAX lax gather/scatter needs low-level dimension-number adapters")
    if qname == "keras.ops.scatter":
        raise SkipCall("keras.ops.scatter creates a sparse tensor from indices, not torch/paddle scatter-update")
    if qname == "chainer.functions.scatter_add":
        raise SkipCall("chainer scatter_add uses slice-update semantics that need a separate adapter")
    if canonical == "einsum" and qname.endswith("einsum_path"):
        raise SkipCall("einsum_path returns an execution plan, not the einsum result")
    if (qname.startswith("keras.layers.") or ".keras.layers." in qname) and canonical in {"add", "average", "concatenate", "dot", "maximum", "minimum", "multiply", "subtract"}:
        raise SkipCall("Keras layer merge wrapper has layer defaults, not raw tensor op semantics")

    if canonical == "arange":
        return (0, 5), {}
    if canonical == "range":
        if lib in {"paddle", "torch"}:
            return (0, 2, 1), {}
        return (0, 3, 1), {}
    if canonical in {"asarray", "array", "convert_to_tensor"}:
        return (tensor(X, lib),), {}
    if canonical == "from_numpy":
        return (X.copy(),), {}
    if canonical == "aminmax":
        if lib == "mindspore":
            return (tensor(X, lib),), {"axis": None}
        return (tensor(X, lib),), {}
    if canonical == "identity":
        return (3,), {}
    if canonical in {"atleast_1d", "atleast_2d", "atleast_3d"}:
        return (tensor(np.array([1.0, 2.0], dtype=np.float32), lib),), {}
    if canonical == "bincount":
        return (tensor(INT_VEC, lib),), {}
    if canonical == "broadcast_shapes":
        return ((2, 1), (1, 3)), {}
    if canonical == "broadcast_to":
        return (tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), lib), (2, 3)), {}
    if canonical == "broadcast_tensors":
        left = np.array([[1.0], [2.0]], dtype=np.float32)
        right = np.array([3.0, 4.0], dtype=np.float32)
        return (tensor(left, lib), tensor(right, lib)), {}
    if canonical == "expand":
        shape = np.array([2, 3], dtype=np.int64)
        if lib == "mindspore":
            return (tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), lib), tensor(shape, lib)), {}
        return (tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), lib), (2, 3)), {}
    if canonical == "expand_as":
        other = np.zeros((2, 3), dtype=np.float32)
        return (tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), lib), tensor(other, lib)), {}
    if canonical == "fill":
        if lib == "torch":
            return (tensor(np.zeros((2, 3), dtype=np.float32), lib), 5.0), {}
        if lib == "mindspore":
            return (dtype_arg(lib, "float32"), (2, 3), 5.0), {}
        return ((2, 3), 5.0), {}
    if canonical == "sequence_mask":
        lengths = np.array([1, 3, 2], dtype=np.int32)
        return (tensor(lengths, lib), 4), {}
    if canonical == "can_cast":
        return (dtype_arg(lib, "float32"), dtype_arg(lib, "float64")), {}
    if canonical == "promote_types":
        return (dtype_arg(lib, "float32"), dtype_arg(lib, "float64")), {}
    if canonical == "result_type":
        return (tensor(np.array([1.0], dtype=np.float32), lib), tensor(np.array([2.0], dtype=np.float32), lib)), {}
    if canonical == "is_tensor":
        return (tensor(X, lib),), {}
    if canonical == "typename":
        raise SkipCall("typename returns library-specific type names")
    if canonical == "shape":
        return (tensor(X, lib),), {}
    if canonical in {"dtype", "is_tensor"}:
        return (tensor(X, lib),), {}
    if canonical in {"geterr", "get_default_dtype", "get_default_device", "is_grad_enabled"}:
        return (), {}
    if canonical == "array_repr":
        return (tensor(np.array([1.0, 2.0], dtype=np.float32), lib),), {}
    if canonical in {"accumulate_n", "add_n"}:
        if lib == "mxnet":
            return (tensor(X, lib), tensor(Y, lib)), {}
        return ([tensor(X, lib), tensor(Y, lib)],), {}
    if category in CONFIG_CATEGORIES and canonical in {"get", "serialize", "deserialize"}:
        module = importlib.import_module(api.namespace)
        if category == "activation_config":
            if canonical == "serialize":
                return (getattr(module, "relu"),), {}
            return ("relu",), {}
        if category == "layer_config":
            layer = module.Dense(1, name="dense")
            config = module.serialize(layer)
            if canonical == "serialize":
                return (layer,), {}
            return (config,), {}
        object_name = "MeanSquaredError"
        if category == "loss_config":
            obj = module.MeanSquaredError(name="mean_squared_error")
            identifier = "mean_squared_error"
        else:
            obj = module.MeanSquaredError(name="mean_squared_error")
            identifier = "mean_squared_error"
        config = module.serialize(obj)
        if canonical == "get":
            return (identifier,), {}
        if canonical == "serialize":
            return (obj,), {}
        return (config,), {}
    if canonical == "zeros":
        return ((2, 3),), {}
    if canonical == "ones":
        return ((2, 3),), {}
    if canonical == "full":
        return ((2, 3), 2.0), {}
    if canonical == "eye":
        return (3,), {}
    if canonical in {"zeros_like", "ones_like"}:
        return (tensor(X, lib),), {}
    if canonical == "full_like":
        return (tensor(X, lib), 2.0), {}
    if canonical == "where":
        return (tensor(BOOL, lib), tensor(X, lib), tensor(Y, lib)), {}
    if canonical in {"allclose", "isclose"}:
        return (tensor(X, lib), tensor(X + 1e-6, lib)), {}
    if canonical == "unique":
        return (tensor(INT_VEC, lib),), {}
    if canonical == "append":
        return (tensor(VEC_A, lib), tensor(VEC_B, lib)), {}
    if canonical == "searchsorted":
        base = np.array([1.0, 3.0, 5.0], dtype=np.float32)
        values = np.array([0.0, 3.0, 6.0], dtype=np.float32)
        return (tensor(base, lib), tensor(values, lib)), {}
    if canonical == "unravel_index":
        return (tensor(np.array([1, 3], dtype=np.int64), lib), (2, 3)), {}
    if canonical == "polyval":
        coeffs = [2.0, 1.0] if lib == "tensorflow" else tensor(np.array([2.0, 1.0], dtype=np.float32), lib)
        return (coeffs, tensor(VEC_A, lib)), {}
    if canonical == "zeta":
        return (tensor(np.array([2.0, 3.0], dtype=np.float32), lib), tensor(np.array([1.0, 2.0], dtype=np.float32), lib)), {}
    if canonical == "polygamma":
        value = tensor(np.array([1.0, 2.0], dtype=np.float32), lib)
        if lib == "chainer":
            return (tensor(np.array([1, 1], dtype=np.int32), lib), value), {}
        if lib == "paddle":
            return (value, 1), {}
        if lib == "mindspore":
            return (tensor(np.array(1, dtype=np.int32), lib), value), {}
        return (1, value), {}
    if canonical == "multigammaln":
        return (tensor(np.array([3.0, 4.0], dtype=np.float32), lib), 2), {}
    if canonical == "frombuffer":
        buf = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
        return (buf,), {"dtype": dtype_arg(lib, "float32")}
    if category == "random":
        if canonical == "normal":
            if lib == "jax":
                import jax

                return (jax.random.PRNGKey(0), (0,)), {}
            if lib == "mxnet":
                if "randn" in qname:
                    return (0,), {}
                if qname.startswith("mxnet.ndarray."):
                    return (), {"loc": 0.0, "scale": 1.0, "shape": (0,)}
                return (), {"loc": 0.0, "scale": 1.0, "size": (0,)}
            if qname == "numpy.random.randn":
                return (0,), {}
            if qname == "numpy.random.standard_normal":
                return ((0,),), {}
            if lib == "numpy":
                return (0.0, 1.0, (0,)), {}
            if lib == "keras":
                return ((0,),), {"seed": 0}
            if lib == "tensorflow":
                return ((0,),), {"seed": 0}
            if lib == "paddle":
                if "randn" in qname or "standard_normal" in qname:
                    return ((0,),), {}
                return (0.0, 1.0, (0,)), {}
            if lib == "torch":
                if qname == "torch.randn":
                    return (0,), {}
                return (), {"mean": 0.0, "std": 1.0, "size": (0,)}
        if canonical == "lognormal":
            if lib == "jax":
                import jax

                return (jax.random.PRNGKey(0),), {"shape": (0,)}
            if lib == "numpy":
                return (0.0, 1.0, (0,)), {}
            if lib == "mxnet":
                size_kw = "shape" if qname.startswith("mxnet.ndarray.") else "size"
                return (0.0, 1.0), {size_kw: (0,)}
        if canonical == "uniform":
            if lib == "jax":
                import jax

                return (jax.random.PRNGKey(0), (0,)), {}
            if lib == "mxnet":
                if qname.endswith(".rand"):
                    return (0,), {}
                if qname.startswith("mxnet.ndarray."):
                    return (), {"low": 0.0, "high": 1.0, "shape": (0,)}
                return (), {"low": 0.0, "high": 1.0, "size": (0,)}
            if qname == "numpy.random.rand":
                return (0,), {}
            if lib == "numpy":
                return (0.0, 1.0, (0,)), {}
            if lib == "keras":
                return ((0,),), {"seed": 0}
            if lib == "tensorflow":
                return ((0,),), {"seed": 0}
            return ((0,),), {}
        if canonical == "random_integer":
            if lib == "jax":
                import jax

                return (jax.random.PRNGKey(0), (2,), 1, 2), {}
            if lib == "mxnet":
                shape_kw = "shape" if qname.startswith("mxnet.ndarray.") else "size"
                return (1, 2), {shape_kw: (2,)}
            if lib == "keras":
                return ((2,), 1, 2), {"seed": 0}
            if lib == "numpy":
                return (1, 2), {"size": (2,)}
            if lib == "paddle":
                return (1, 2, (2,)), {}
            if lib == "torch":
                return (1, 2, (2,)), {}
        if canonical == "bernoulli":
            if lib == "jax":
                import jax

                return (jax.random.PRNGKey(0), 0.0, (2,)), {}
            if lib == "mindspore":
                return (tensor(np.zeros((2,), dtype=np.float32), lib),), {"p": 0.0}
            return (tensor(np.zeros((2,), dtype=np.float32), lib),), {}
        if canonical == "poisson":
            if lib == "jax":
                import jax

                return (jax.random.PRNGKey(0), 0.0), {"shape": (2,)}
            if lib == "numpy":
                return (0.0,), {"size": (2,)}
            if lib == "tensorflow":
                return ((2,), 0.0), {"seed": 0}
            return (tensor(np.zeros((2,), dtype=np.float32), lib),), {}
        if canonical == "binomial":
            if lib == "jax":
                import jax

                return (jax.random.PRNGKey(0), 4, 0.0), {"shape": (2,)}
            if lib == "keras":
                return ((2,), 4, 0.0), {"seed": 0}
            if lib == "numpy":
                return (4, 0.0), {"size": (2,)}
        if canonical in {"beta", "gamma", "truncated_normal", "laplace", "logistic", "exponential"}:
            if lib == "jax":
                import jax

                key = jax.random.PRNGKey(0)
                if canonical == "beta":
                    return (key, 1.0, 1.0), {"shape": (0,)}
                if canonical == "gamma":
                    return (key, 1.0), {"shape": (0,)}
                if canonical == "truncated_normal":
                    return (key, -1.0, 1.0), {"shape": (0,)}
                return (key, (0,)), {}
            if lib == "keras":
                if canonical == "beta":
                    return ((0,), 1.0, 1.0), {"seed": 0}
                if canonical == "gamma":
                    return ((0,), 1.0), {"seed": 0}
                if canonical == "truncated_normal":
                    return ((0,),), {"seed": 0}
            if lib == "numpy":
                if canonical == "beta":
                    return (1.0, 1.0), {"size": (0,)}
                if canonical == "gamma":
                    return (1.0,), {"size": (0,)}
                if canonical == "laplace":
                    return (0.0, 1.0, (0,)), {}
                if canonical == "logistic":
                    return (0.0, 1.0, (0,)), {}
                if canonical == "exponential":
                    return (1.0,), {"size": (0,)}
            if lib == "tensorflow" and canonical in {"gamma", "truncated_normal"}:
                if canonical == "gamma":
                    return ((0,), 1.0), {"seed": 0}
                return ((0,),), {"seed": 0}
            if lib == "mxnet":
                size_kw = "shape" if qname.startswith("mxnet.ndarray.") else "size"
                if canonical == "beta":
                    return (1.0, 1.0), {size_kw: (0,)}
                if canonical == "gamma":
                    return (1.0,), {size_kw: (0,)}
                if canonical in {"laplace", "logistic"}:
                    return (0.0, 1.0), {size_kw: (0,)}
                if canonical == "exponential":
                    return (1.0,), {size_kw: (0,)}
        if canonical == "power":
            if lib in {"numpy", "mxnet"}:
                size_kw = "shape" if qname.startswith("mxnet.ndarray.") else "size"
                return (1.5,), {size_kw: (0,)}
        if canonical in {"chisquare", "gumbel", "pareto", "rayleigh"}:
            if lib == "jax":
                import jax

                key = jax.random.PRNGKey(0)
                if canonical == "chisquare":
                    return (key, 1.0), {"shape": (0,)}
                if canonical == "gumbel":
                    return (key,), {"shape": (0,)}
                if canonical == "pareto":
                    return (key, 1.5), {"shape": (0,)}
                return (key, 1.0), {"shape": (0,)}
            if lib == "mxnet":
                if canonical == "chisquare":
                    return (1.0,), {"size": (0,)}
                if canonical == "gumbel":
                    return (0.0, 1.0), {"size": (0,)}
                if canonical == "pareto":
                    return (1.5,), {"size": (0,)}
                return (1.0,), {"size": (0,)}
        if canonical == "multinomial":
            if lib in {"numpy", "mxnet"}:
                pvals = np.array([0.25, 0.75], dtype=np.float32)
                return (4, tensor(pvals, lib) if lib == "mxnet" else pvals), {"size": (0,)}
        if canonical == "multivariate_normal":
            if lib in {"numpy", "mxnet"}:
                mean = np.array([0.0, 1.0], dtype=np.float32)
                cov = np.eye(2, dtype=np.float32)
                if lib == "mxnet":
                    return (tensor(mean, lib), tensor(cov, lib)), {"size": (0,)}
                return (mean, cov), {"size": (0,)}
        if canonical == "standard_gamma":
            if lib == "numpy":
                return (1.0,), {"size": (0,)}
            if lib == "paddle":
                return (tensor(np.ones((0,), dtype=np.float32), lib),), {}
        if canonical == "shuffle":
            value = np.array([1.0], dtype=np.float32)
            if lib in {"keras", "tensorflow"}:
                return (tensor(value, lib),), {"seed": 0}
            if lib == "mxnet":
                return (tensor(value, lib),), {}
            return (value.copy(),), {}
    if canonical == "broadcast_arrays":
        return (tensor(np.array([[1.0], [2.0]], dtype=np.float32), lib), tensor(np.array([3.0, 4.0], dtype=np.float32), lib)), {}
    if canonical == "slice":
        if lib == "paddle":
            return (tensor(X, lib), [0, 1], [0, 0], [2, 2]), {}
        if lib in {"jax", "tensorflow", "keras", "mindspore"}:
            return (tensor(X, lib), (0, 0), (2, 2)), {}
    if canonical == "strided_slice":
        if lib == "tensorflow":
            return (tensor(X, lib), [0, 0], [2, 3], [1, 1]), {}
        if lib == "mindspore":
            return (tensor(X, lib), (0, 0), (2, 3), (1, 1)), {}
        return (tensor(X, lib), [0, 1], [0, 0], [2, 3], [1, 1]), {}
    if canonical == "index_put":
        base = np.zeros((3,), dtype=np.float32)
        indices = (tensor(np.array([1], dtype=np.int64), lib),)
        value = tensor(np.array([5.0], dtype=np.float32), lib)
        return (tensor(base, lib), indices, value), {"accumulate": False}
    if canonical in {"index_add", "index_fill", "index_select"}:
        indices = np.array([0, 1], dtype=np.int64)
        if canonical == "index_select":
            if lib == "torch" or (lib == "mindspore" and ".mint." in qname):
                return (tensor(X, lib), 0, tensor(indices, lib)), {}
            return (tensor(X, lib), tensor(indices, lib)), {"axis": 0}
        if canonical == "index_add":
            value = np.array([[0.5, 1.0, -1.5], [2.0, -0.5, 1.0]], dtype=np.float32)
            if lib == "torch":
                return (tensor(X, lib), 0, tensor(indices, lib), tensor(value, lib)), {"alpha": 1.0}
            if lib == "mindspore":
                return (tensor(X, lib), tensor(indices.astype(np.int32), lib), tensor(value, lib)), {"axis": 0}
            return (tensor(X, lib), tensor(indices, lib), 0, tensor(value, lib)), {"alpha": 1.0}
        fill_value = np.array(9.0, dtype=np.float32)
        if lib == "mindspore":
            return (tensor(X, lib), 0, tensor(indices.astype(np.int32), lib), float(fill_value)), {}
        if lib == "torch":
            return (tensor(X, lib), 0, tensor(indices, lib), float(fill_value)), {}
        return (tensor(X, lib), tensor(indices, lib), 0, tensor(fill_value, lib)), {}
    if canonical in {"masked_fill", "masked_scatter", "masked_select"}:
        mask = np.array([[True, False, False], [False, True, False]])
        if canonical == "masked_select":
            return (tensor(X, lib), tensor(mask, lib)), {}
        if canonical == "masked_fill":
            return (tensor(X, lib), tensor(mask, lib), -9.0), {}
        value = np.array([7.0, 8.0], dtype=np.float32)
        return (tensor(X, lib), tensor(mask, lib), tensor(value, lib)), {}
    if canonical == "scatter_update":
        base = np.zeros((3,), dtype=np.float32)
        if lib == "mindspore":
            indices = np.array([1, 2], dtype=np.int32)
        else:
            indices = np.array([[1], [2]], dtype=np.int64)
        updates = np.array([5.0, 6.0], dtype=np.float32)
        return (tensor(base, lib), tensor(indices, lib), tensor(updates, lib)), {}
    if canonical in {"scatter", "scatter_add", "scatter_reduce"}:
        indices = np.array([[0, 2, 1], [1, 0, 2]], dtype=np.int64)
        updates = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
        if canonical == "scatter":
            return (tensor(np.zeros_like(X), lib), 1, tensor(indices, lib), tensor(updates, lib)), {}
        if canonical == "scatter_add":
            return (tensor(np.zeros_like(X), lib), 1, tensor(indices, lib), tensor(updates, lib)), {}
        return (tensor(np.ones_like(X), lib), 1, tensor(indices, lib), tensor(updates, lib), "sum"), {"include_self": True}
    if canonical in {"select_scatter", "slice_scatter"}:
        if canonical == "select_scatter":
            values = np.array([9.0, 8.0], dtype=np.float32)
            if lib == "torch":
                return (tensor(X, lib), tensor(values, lib), 1, 1), {}
            return (tensor(X, lib), tensor(values, lib), 1, 1), {}
        values = np.array([[9.0], [8.0]], dtype=np.float32)
        if lib in {"torch", "mindspore"}:
            return (tensor(X, lib), tensor(values, lib), 1, 1, 2, 1), {}
        return (tensor(X, lib), tensor(values, lib), [1], [1], [2], [1]), {}
    if canonical in {"array_split", "hsplit", "vsplit", "dsplit"}:
        value = np.reshape(np.arange(24, dtype=np.float32), (2, 4, 3)) if canonical == "dsplit" else EVEN
        axis = {"array_split": 0, "hsplit": 1, "vsplit": 0, "dsplit": 2}[canonical]
        sections = 3 if canonical == "dsplit" else 2
        if canonical in {"hsplit", "vsplit", "dsplit"}:
            return (tensor(value, lib), sections), {}
        return (tensor(value, lib), sections), {"axis": axis}
    if canonical in {"dstack", "hstack", "vstack", "column_stack", "row_stack"}:
        return ([tensor(X, lib), tensor(X, lib)],), {}
    if canonical == "meshgrid":
        return (tensor(np.array([1.0, 2.0], dtype=np.float32), lib), tensor(np.array([3.0, 4.0], dtype=np.float32), lib)), {"indexing": "ij"}

    if canonical in {"concatenate", "stack"}:
        if "keras.layers" in qname:
            return ([tensor(X, lib), tensor(X, lib)],), {}
        if lib == "mxnet" and qname.startswith("mxnet.ndarray."):
            kwargs = {"dim": 0} if canonical == "concatenate" else {"axis": 0}
            return (tensor(X, lib), tensor(X, lib)), kwargs
        axis_kw = {"dim": 0} if lib == "torch" and "torch." in qname else {"axis": 0}
        return ([tensor(X, lib), tensor(X, lib)],), axis_kw
    if canonical == "split":
        if "unstack" in qname:
            if lib == "tensorflow":
                return (tensor(X, lib),), {"axis": 0}
            return (tensor(X, lib),), {"axis": 0}
        if qname == "jax.lax.split":
            raise SkipCall("jax.lax.split has low-level split-size semantics")
        if lib == "torch":
            return (tensor(X, lib), 1), {"dim": 0}
        return (tensor(X, lib), 2), {"axis": 0}
    if canonical == "reshape":
        return (tensor(X, lib), (3, 2)), {}
    if canonical == "squeeze":
        return (tensor(np.reshape(X, (1, 2, 3, 1)), lib),), {}
    if canonical == "expand_dims":
        if lib == "torch":
            return (tensor(X, lib), 0), {}
        return (tensor(X, lib),), {"axis": 0}
    if canonical == "transpose":
        if "matrix_transpose" in qname:
            return (tensor(X, lib),), {}
        if lib == "torch" or qname == "mindspore.mint.transpose":
            return (tensor(X, lib), 0, 1), {}
        if lib == "tensorflow":
            return (tensor(X, lib),), {"perm": (1, 0)}
        return (tensor(X, lib), (1, 0)), {}
    if canonical == "tile":
        return (tensor(X, lib), (2, 1)), {}

    if canonical in {"dot", "vdot", "inner", "outer", "vecdot"}:
        if "keras.layers" in qname and canonical == "dot":
            raise SkipCall("Keras layer dot is a batched layer wrapper")
        return (tensor(VEC_A, lib), tensor(VEC_B, lib)), {}
    if canonical in {"cummax", "cummin"}:
        if lib == "jax":
            return (tensor(X, lib), 1), {}
        if lib == "torch":
            return (tensor(X, lib),), {"dim": 1}
        if lib == "paddle":
            return (tensor(X, lib),), {"axis": 1}
        return (tensor(X, lib),), {"axis": 1}
    if canonical == "block_diag":
        if lib == "paddle":
            return ([tensor(MAT, lib), tensor(MAT_B, lib)],), {}
        return (tensor(MAT, lib), tensor(MAT_B, lib)), {}
    if canonical == "cholesky_inverse":
        chol = np.linalg.cholesky(SPD).astype(np.float32)
        return (tensor(chol, lib),), {}

    if canonical == "matmul":
        if qname == "jax.lax.dot_general":
            return (tensor(MAT, lib), tensor(MAT_B, lib), (((1,), (0,)), ((), ()))), {}
        if "keras.layers" in qname:
            return ([tensor(MAT, lib), tensor(MAT_B, lib)],), {}
        return (tensor(MAT, lib), tensor(MAT_B, lib)), {}
    if canonical == "matmul_vector":
        return (tensor(MAT, lib), tensor(np.array([1.0, 2.0], dtype=np.float32), lib)), {}
    if canonical == "batch_matmul":
        lhs = np.stack([MAT, MAT + 1.0]).astype(np.float32)
        rhs = np.stack([MAT_B, MAT_B + 1.0]).astype(np.float32)
        return (tensor(lhs, lib), tensor(rhs, lib)), {}
    if canonical == "addmm":
        input_value = np.array([[0.5, -1.0], [2.0, 0.25]], dtype=np.float32)
        mat1 = np.array([[1.0, 2.0, -1.0], [0.5, -0.5, 3.0]], dtype=np.float32)
        mat2 = np.array([[2.0, -1.0], [0.0, 1.5], [1.0, 0.5]], dtype=np.float32)
        return (tensor(input_value, lib), tensor(mat1, lib), tensor(mat2, lib)), {"beta": 1.0, "alpha": 1.0}
    if canonical == "baddbmm":
        input_value = np.ones((2, 2, 2), dtype=np.float32)
        batch1 = np.reshape(np.arange(12, dtype=np.float32), (2, 2, 3)) / 5.0
        batch2 = np.reshape(np.arange(12, dtype=np.float32), (2, 3, 2)) / 7.0
        return (tensor(input_value, lib), tensor(batch1, lib), tensor(batch2, lib)), {"beta": 1.0, "alpha": 1.0}
    if canonical == "addbmm":
        input_value = np.ones((2, 2), dtype=np.float32)
        batch1 = np.reshape(np.arange(12, dtype=np.float32), (2, 2, 3)) / 5.0
        batch2 = np.reshape(np.arange(12, dtype=np.float32), (2, 3, 2)) / 7.0
        return (tensor(input_value, lib), tensor(batch1, lib), tensor(batch2, lib)), {"beta": 1.0, "alpha": 1.0}
    if canonical in {"addcmul", "addcdiv"}:
        left = np.array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]], dtype=np.float32)
        right = POS if canonical == "addcdiv" else Y
        return (tensor(X, lib), tensor(left, lib), tensor(right, lib)), {"value": 1.0}
    if canonical == "addmv":
        input_value = np.array([0.5, -1.0], dtype=np.float32)
        mat = np.array([[1.0, 2.0, -1.0], [0.5, -0.5, 3.0]], dtype=np.float32)
        vec = np.array([2.0, -1.0, 0.5], dtype=np.float32)
        return (tensor(input_value, lib), tensor(mat, lib), tensor(vec, lib)), {"beta": 1.0, "alpha": 1.0}
    if canonical == "addr":
        input_value = np.zeros((2, 3), dtype=np.float32)
        vec1 = np.array([1.0, -2.0], dtype=np.float32)
        vec2 = np.array([0.5, 1.5, -1.0], dtype=np.float32)
        return (tensor(input_value, lib), tensor(vec1, lib), tensor(vec2, lib)), {"beta": 1.0, "alpha": 1.0}
    if canonical == "tensordot":
        if lib == "torch":
            return (tensor(MAT, lib), tensor(MAT_B, lib)), {"dims": 1}
        return (tensor(MAT, lib), tensor(MAT_B, lib)), {"axes": 1}
    if canonical == "einsum":
        return ("ij,jk->ik", tensor(MAT, lib), tensor(MAT_B, lib)), {}

    if canonical == "sqrtm" and lib == "jax":
        raise SkipCall("jax sqrtm is unsupported on the current backend")
    if canonical in {"matrix_exp", "expm"}:
        return (tensor(MAT * 0.1, lib),), {}
    if canonical == "expm_frechet":
        direction = np.eye(2, dtype=np.float32) * 0.1
        return (tensor(MAT * 0.1, lib), tensor(direction, lib)), {}
    if canonical == "eigh_tridiagonal":
        diag = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        offdiag = np.array([0.25, 0.5], dtype=np.float32)
        return (tensor(diag, lib), tensor(offdiag, lib)), {"eigvals_only": True}
    if canonical == "sqrtm":
        return (tensor(SPD, lib),), {}
    if canonical in {"det", "inv", "pinv"}:
        return (tensor(MAT, lib),), {}
    if canonical == "slogdet":
        return (tensor(SPD, lib),), {}
    if canonical == "logm":
        value = MAT.astype(np.complex64) if lib == "tensorflow" else MAT
        return (tensor(value, lib),), {}
    if canonical == "cholesky":
        kwargs = {"lower": True} if qname.startswith(("scipy.linalg.", "jax.scipy.linalg.")) else {}
        return (tensor(SPD, lib),), kwargs
    if canonical == "cho_factor":
        kwargs = {"lower": True} if qname.startswith(("scipy.linalg.", "jax.scipy.linalg.")) else {}
        return (tensor(SPD, lib),), kwargs
    if canonical in {"eigh", "eig", "eigvals", "qr", "svd", "matrix_rank"}:
        return (tensor(SPD, lib),), {}
    if canonical == "matrix_power":
        return (tensor(MAT, lib), 2), {}
    if canonical in {"tensorinv", "tensorsolve"}:
        arr = np.eye(4, dtype=np.float32).reshape(2, 2, 2, 2)
        if canonical == "tensorsolve":
            return (tensor(arr, lib), tensor(np.ones((2, 2), dtype=np.float32), lib)), {}
        return (tensor(arr, lib),), {}
    if canonical in {"solve", "lstsq"}:
        return (tensor(MAT, lib), tensor(RHS, lib)), {}
    if canonical == "solve_triangular":
        upper = np.triu(MAT).astype(np.float32)
        if lib == "tensorflow":
            return (tensor(upper, lib), tensor(RHS, lib)), {"lower": False}
        if lib == "torch":
            return (tensor(upper, lib), tensor(RHS, lib)), {"upper": True}
        if lib == "paddle":
            return (tensor(upper, lib), tensor(RHS, lib)), {"upper": True}
        return (tensor(upper, lib), tensor(RHS, lib)), {"lower": False}
    if canonical == "lu":
        return (tensor(MAT, lib),), {}
    if canonical == "lu_solve":
        if lib == "tensorflow":
            return (tensor(MAT, lib), tensor(RHS, lib)), {}
        if lib == "mindspore":
            import scipy.linalg

            lu, pivots = scipy.linalg.lu_factor(MAT)
            return ((tensor(lu, lib), tensor(pivots, lib)), tensor(RHS, lib)), {}
        if lib == "torch":
            import torch

            lu, pivots = torch.linalg.lu_factor(tensor(MAT, lib))
            return (lu, pivots, tensor(RHS, lib)), {}
        if lib == "paddle":
            import paddle

            lu, pivots = paddle.linalg.lu(tensor(MAT, lib))
            return (tensor(RHS, lib), lu, pivots), {}
        import scipy.linalg

        lu, pivots = scipy.linalg.lu_factor(MAT)
        if lib == "jax":
            return ((tensor(lu, lib), tensor(pivots, lib)), tensor(RHS, lib)), {}
        return ((lu, pivots), RHS), {}
    if canonical == "cho_solve":
        chol = np.linalg.cholesky(SPD).astype(np.float32)
        if lib == "tensorflow":
            return (tensor(chol, lib), tensor(RHS, lib)), {}
        if qname.startswith("mindspore.scipy.linalg."):
            return ((tensor(chol, lib), True), tensor(RHS, lib)), {}
        if lib in {"torch", "paddle", "mindspore"}:
            return (tensor(RHS, lib), tensor(chol, lib)), {}
        if lib == "jax":
            return ((tensor(chol, lib), True), tensor(RHS, lib)), {}
        return ((chol, True), RHS), {}
    if canonical in {"norm", "normalize"}:
        return (tensor(MAT, lib),), {}
    if canonical == "cross":
        a = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        return (tensor(a, lib), tensor(b, lib)), {}

    if canonical in {"hankel", "toeplitz"}:
        return (tensor(VEC_A, lib),), {}

    if canonical in {"fft", "ifft", "rfft", "irfft", "fft2", "ifft2", "rfft2", "irfft2", "fftn", "ifftn", "rfftn", "irfftn", "hfft", "hfft2", "ihfft", "hfftn", "ihfftn", "ihfft2"}:
        data = X if canonical.endswith("2") or canonical.endswith("n") else VEC
        if lib == "tensorflow" and qname.startswith("tensorflow.signal.") and canonical in {"fft", "ifft", "fft2", "ifft2", "fftn", "ifftn"}:
            return (tensor(data.astype(np.complex64), lib),), {}
        if canonical == "irfft" and lib == "keras":
            return ((tensor(data, lib), tensor(np.zeros_like(data), lib)), 6), {}
        if canonical in {"fft", "ifft", "fft2", "ifft2"} and lib == "keras":
            return ((tensor(data, lib), tensor(np.zeros_like(data), lib)),), {}
        if canonical in {"fft", "ifft"} and lib == "chainer":
            return ((data, np.zeros_like(data)),), {}
        return (tensor(data, lib),), {}
    if canonical == "stft":
        if lib == "torch":
            import torch

            window = torch.hann_window(4, periodic=True)
            return (tensor(VEC, lib), 4), {"hop_length": 2, "win_length": 4, "window": window, "center": True, "pad_mode": "constant", "return_complex": True}
        if qname == "tensorflow.signal.stft":
            return (tensor(VEC, lib), 4, 2), {}
        if qname == "keras.ops.stft":
            return (tensor(VEC, lib), 4, 2, 4), {}
        return (tensor(VEC, lib),), {"nperseg": 4, "noverlap": 2}
    if canonical == "istft":
        spectrum = np.zeros((3, 3), dtype=np.complex64)
        if lib == "keras":
            parts = (np.zeros((3, 3), dtype=np.float32), np.zeros((3, 3), dtype=np.float32))
            return (parts, 4, 2, 4), {"length": 4}
        if lib == "torch":
            import torch

            window = torch.hann_window(4, periodic=True)
            return (tensor(spectrum, lib), 4), {"hop_length": 2, "win_length": 4, "window": window, "center": True, "length": 4}
        if lib == "paddle":
            import paddle

            window = paddle.hann_window(4, periodic=True)
            return (tensor(spectrum, lib), 4), {"hop_length": 2, "win_length": 4, "window": window, "center": True, "length": 4}
        return (tensor(spectrum, lib),), {"nperseg": 4, "noverlap": 2}

    if canonical == "leaky_relu":
        if lib == "chainer":
            return (tensor(X, lib),), {"slope": 0.2}
        if lib == "mxnet":
            return (tensor(X, lib),), {"act_type": "leaky", "slope": 0.2}
        if lib == "mindspore":
            return (tensor(X, lib), 0.2), {}
        kwargs = {"alpha": 0.2} if qname == "tensorflow.nn.leaky_relu" else {"negative_slope": 0.2}
        return (tensor(X, lib),), kwargs
    if canonical == "gelu":
        if lib in {"torch", "mindspore"}:
            return (tensor(X, lib),), {"approximate": "none"}
        if lib in {"chainer", "mxnet"}:
            return (tensor(X, lib),), {}
        return (tensor(X, lib),), {"approximate": False}
    if canonical in {"dropout", "dropout1d", "dropout2d", "dropout3d", "alpha_dropout", "feature_alpha_dropout"}:
        if canonical == "dropout3d":
            value = np.reshape(np.arange(120, dtype=np.float32), (1, 2, 3, 4, 5))
        elif canonical == "dropout2d":
            value = np.reshape(np.arange(24, dtype=np.float32), (1, 2, 3, 4))
        else:
            value = X
        if lib == "tensorflow":
            return (tensor(value, lib),), {"rate": 0.0}
        if lib == "keras":
            return (tensor(value, lib), 0.0), {"seed": 0}
        if lib == "chainer":
            return (tensor(value, lib),), {"ratio": 0.0}
        if lib == "mxnet":
            return (tensor(value, lib),), {"p": 0.0}
        return (tensor(value, lib),), {"p": 0.0, "training": True}
    if canonical == "rrelu":
        return (tensor(X, lib),), {"lower": 0.2, "upper": 0.2, "training": True}
    if canonical == "randn_like":
        return (tensor(np.empty((0,), dtype=np.float32), lib),), {}

    if canonical in {"softmax", "log_softmax"}:
        if lib == "torch" or (lib == "mindspore" and ".mint." in qname):
            return (tensor(X, lib),), {"dim": 1}
        return (tensor(X, lib),), {"axis": 1}
    if canonical == "logsumexp":
        if lib == "torch" or (lib == "mindspore" and ".mint." in qname):
            return (tensor(X, lib),), {"dim": 1}
        return (tensor(X, lib),), {"axis": 1}
    if canonical == "clip":
        if qname == "jax.lax.clamp":
            return (-1.0, tensor(X, lib), 2.0), {}
        if lib == "tensorflow":
            return (tensor(X, lib), -1.0, 2.0), {}
        return (tensor(X, lib), -1.0, 2.0), {}
    if canonical == "topk":
        if lib == "mxnet":
            return (tensor(X, lib), 2), {"axis": 1}
        return (tensor(X, lib), 2), {}
    if canonical == "histc":
        return (tensor(X, lib),), {"bins": 8, "min": -4.0, "max": 4.0}
    if canonical == "kthvalue":
        if lib == "torch" or (lib == "mindspore" and ".mint." in qname):
            return (tensor(X, lib), 2), {"dim": 1}
        return (tensor(X, lib), 2), {"axis": 1}
    if canonical == "round":
        values = np.array([[-2.4, -1.2, -0.2], [0.2, 1.2, 2.4]], dtype=np.float32)
        return (tensor(values, lib),), {}

    if canonical == "argpartition":
        return (tensor(X, lib), 1), {"axis": 1}
    if canonical in {"argmax", "argmin"}:
        if qname.startswith("jax.lax."):
            import jax.numpy as jnp

            return (tensor(X, lib), 1, jnp.int32), {}
        if lib == "torch" or (lib == "mindspore" and ".mint." in qname):
            return (tensor(X, lib),), {"dim": 1}
        return (tensor(X, lib),), {"axis": 1}
    if canonical in {"sort", "argsort"}:
        return (tensor(X, lib),), {}
    if canonical in {"median", "nanmedian"}:
        return (tensor(np.array([1.0, -2.0, 3.0], dtype=np.float32), lib),), {}
    if canonical == "mode":
        if lib == "torch":
            return (tensor(X, lib),), {"dim": 1}
        return (tensor(X, lib),), {"axis": 1}
    if canonical in {"sum", "std", "var"}:
        if qname == "jax.lax.reduce_sum":
            return (tensor(X, lib), (1,)), {}
        if lib == "torch" or (lib == "mindspore" and ".mint." in qname):
            kwargs = {"dim": 1}
            if canonical in {"std", "var"}:
                kwargs["correction"] = 0
            return (tensor(X, lib),), kwargs
        kwargs = {"axis": 1}
        if lib == "paddle" and canonical in {"std", "var"}:
            kwargs["unbiased"] = False
        return (tensor(X, lib),), kwargs
    if canonical in {"mean", "max", "min", "prod", "all", "any", "count_nonzero"}:
        value = BOOL if canonical in {"all", "any"} else X
        return (tensor(value, lib),), {}
    if canonical in {"std_mean", "var_mean"}:
        if lib == "torch":
            return (tensor(X, lib),), {"dim": 1, "correction": 0}
        if lib == "mindspore":
            return (tensor(X, lib),), {"axis": 1, "ddof": 0}
        return (tensor(X, lib),), {"axis": 1}
    if canonical in {"cumsum", "cumprod"}:
        if lib == "torch" or (lib == "paddle" and canonical == "cumprod"):
            return (tensor(X, lib),), {"dim": 1}
        return (tensor(X, lib),), {"axis": 1}
    if canonical == "logcumsumexp":
        if lib == "torch":
            return (tensor(X, lib), 1), {}
        return (tensor(X, lib),), {"axis": 1}
    if canonical == "threshold":
        return (tensor(X, lib), 0.0, -1.0), {}
    if canonical == "fold":
        patches = np.ones((1, 4, 9), dtype=np.float32)
        return (tensor(patches, lib), (4, 4), (2, 2)), {"stride": 1, "padding": 0, "dilation": 1}
    if canonical == "upsample":
        image = np.reshape(np.arange(16, dtype=np.float32), (1, 1, 4, 4))
        if lib == "tensorflow":
            image = np.transpose(image, (0, 2, 3, 1))
            return (tensor(image, lib), (8, 8)), {"method": "nearest"}
        return (tensor(image, lib),), {"size": (8, 8), "mode": "nearest"}
    if canonical == "glu":
        return (tensor(EVEN, lib),), {}
    if canonical == "batch_norm":
        mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        var = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        if lib == "keras":
            return (tensor(X, lib), tensor(mean, lib), tensor(var, lib), -1), {"offset": tensor(offset, lib), "scale": tensor(scale, lib), "epsilon": 1e-3}
        if lib == "torch":
            return (tensor(X, lib), tensor(mean, lib), tensor(var, lib)), {"weight": tensor(scale, lib), "bias": tensor(offset, lib), "training": False, "eps": 1e-3}
        if lib == "paddle":
            return (tensor(X, lib), tensor(mean, lib), tensor(var, lib)), {"weight": tensor(scale, lib), "bias": tensor(offset, lib), "training": False, "epsilon": 1e-3}
        if lib == "chainer":
            return (tensor(X, lib), tensor(scale, lib), tensor(offset, lib), tensor(mean, lib), tensor(var, lib)), {"eps": 1e-3}
        if lib == "mxnet":
            return (tensor(X, lib), tensor(scale, lib), tensor(offset, lib), tensor(mean, lib), tensor(var, lib)), {"eps": 1e-3}
        return (tensor(X, lib), tensor(mean, lib), tensor(var, lib), tensor(offset, lib), tensor(scale, lib), 1e-3), {}
    if canonical == "instance_norm":
        value = np.reshape(np.arange(12, dtype=np.float32), (1, 3, 2, 2))
        scale = np.ones((3,), dtype=np.float32)
        offset = np.zeros((3,), dtype=np.float32)
        mean = np.zeros((3,), dtype=np.float32)
        var = np.ones((3,), dtype=np.float32)
        if lib in {"torch", "paddle"}:
            return (tensor(value, lib),), {"weight": tensor(scale, lib), "bias": tensor(offset, lib), "use_input_stats": True, "eps": 1e-3}
    if canonical == "group_norm":
        scale = np.ones((3,), dtype=np.float32)
        offset = np.zeros((3,), dtype=np.float32)
        if lib == "torch":
            return (tensor(X, lib), 1), {"weight": tensor(scale, lib), "bias": tensor(offset, lib), "eps": 1e-3}
        if lib == "paddle":
            return (tensor(X, lib), 1), {"weight": tensor(scale, lib), "bias": tensor(offset, lib), "epsilon": 1e-3}
        if lib == "mindspore":
            return (tensor(X, lib), 1, tensor(scale, lib), tensor(offset, lib)), {"eps": 1e-3}
        if lib == "chainer":
            return (tensor(X, lib), 1, tensor(scale, lib), tensor(offset, lib)), {"eps": 1e-3}
    if canonical == "layer_norm":
        if lib == "torch":
            return (tensor(X, lib), (3,)), {"eps": 1e-3}
        if lib == "mindspore":
            return (tensor(X, lib), (3,)), {"eps": 1e-3}
        if lib == "paddle":
            return (tensor(X, lib), (3,)), {"epsilon": 1e-3}
        if lib == "chainer":
            gamma = np.ones((3,), dtype=np.float32)
            beta = np.zeros((3,), dtype=np.float32)
            return (tensor(X, lib), tensor(gamma, lib), tensor(beta, lib)), {"eps": 1e-3}
        if lib == "mxnet":
            gamma = np.ones((3,), dtype=np.float32)
            beta = np.zeros((3,), dtype=np.float32)
            return (tensor(X, lib), tensor(gamma, lib), tensor(beta, lib)), {"axis": -1, "eps": 1e-3}
        return (tensor(X, lib),), {"axis": -1, "epsilon": 1e-3} if lib == "keras" else {"axis": -1}
    if canonical == "prelu":
        weight = np.full((3,), 0.25, dtype=np.float32)
        return (tensor(X, lib), tensor(weight, lib)), {}
    if canonical == "linear":
        x = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]], dtype=np.float32)
        weight_out_in = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.5]], dtype=np.float32)
        bias = np.array([0.25, -0.75], dtype=np.float32)
        if lib == "paddle":
            return (tensor(x, lib), tensor(weight_out_in.T, lib)), {"bias": tensor(bias, lib)}
        if lib in {"torch", "mindspore", "chainer"}:
            return (tensor(x, lib), tensor(weight_out_in, lib), tensor(bias, lib)), {}
    if canonical == "embedding":
        indices = np.array([[0, 2], [1, 3]], dtype=np.int64)
        weight = np.array([[0.0, 0.5], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5]], dtype=np.float32)
        if lib in {"torch", "paddle", "mindspore"}:
            return (tensor(indices, lib), tensor(weight, lib)), {}
        if lib == "mxnet":
            return (tensor(indices, lib),), {"input_dim": 4, "output_dim": 2, "weight": tensor(weight, lib)}
    if canonical == "conv":
        if qname == "torch.convolution":
            value = np.ones((1, 1, 5), dtype=np.float32)
            weight = np.ones((1, 1, 3), dtype=np.float32)
            return (tensor(value, lib), tensor(weight, lib), None, [1], [0], [1], False, [0], 1), {}
        if lib == "jax":
            lhs = np.ones((1, 1, 5), dtype=np.float32)
            rhs = np.ones((1, 1, 3), dtype=np.float32)
            return (tensor(lhs, lib), tensor(rhs, lib), (1,), "VALID"), {}
        lhs = np.ones((1, 5, 1), dtype=np.float32)
        rhs = np.ones((3, 1, 1), dtype=np.float32)
        if lib == "tensorflow":
            return (tensor(lhs, lib), tensor(rhs, lib)), {"strides": [1], "padding": "VALID", "data_format": "NWC"}
        return (tensor(lhs, lib), tensor(rhs, lib)), {"strides": 1, "padding": "valid", "data_format": "channels_last"}
    if canonical in {"conv1d", "conv2d", "conv3d"}:
        dim = int(canonical[-2])
        spatial = (5,) if dim == 1 else (4, 4) if dim == 2 else (3, 3, 3)
        kernel = (3,) if dim == 1 else (2, 2) if dim == 2 else (2, 2, 2)
        if lib == "tensorflow":
            value = np.ones((1, *spatial, 1), dtype=np.float32)
            weight = np.ones((*kernel, 1, 1), dtype=np.float32)
            strides = [1, 1, 1, 1, 1] if dim == 3 else 1
            return (tensor(value, lib), tensor(weight, lib), strides, "VALID"), {}
        value = np.ones((1, 1, *spatial), dtype=np.float32)
        weight = np.ones((1, 1, *kernel), dtype=np.float32)
        kwargs = {"stride": 1, "padding": 0} if lib == "torch" else {"stride": 1, "padding": 0}
        return (tensor(value, lib), tensor(weight, lib)), kwargs
    if canonical == "conv_transpose":
        value = np.ones((1, 3, 3, 1), dtype=np.float32)
        weight = np.ones((2, 2, 1, 1), dtype=np.float32)
        if lib == "tensorflow":
            return (tensor(value, lib), tensor(weight, lib), (1, 4, 4, 1), 1, "VALID"), {}
        return (tensor(value, lib), tensor(weight, lib)), {"strides": 1, "padding": "valid", "data_format": "channels_last"}
    if canonical in {"conv_transpose1d", "conv_transpose2d", "conv_transpose3d"}:
        dim = int(canonical[-2])
        spatial = (3,) if dim == 1 else (3, 3) if dim == 2 else (3, 3, 3)
        kernel = (2,) if dim == 1 else (2, 2) if dim == 2 else (2, 2, 2)
        if lib == "tensorflow":
            value = np.ones((1, *spatial, 1), dtype=np.float32)
            weight = np.ones((*kernel, 1, 1), dtype=np.float32)
            output_shape = (1, *tuple(s + 1 for s in spatial), 1)
            strides = [1, 1, 1, 1, 1] if dim == 3 else 1
            return (tensor(value, lib), tensor(weight, lib), output_shape, strides, "VALID"), {}
        value = np.ones((1, 1, *spatial), dtype=np.float32)
        weight = np.ones((1, 1, *kernel), dtype=np.float32)
        return (tensor(value, lib), tensor(weight, lib)), {"stride": 1, "padding": 0}
    if canonical in {"avg_pool", "max_pool"}:
        values = np.reshape(np.arange(16, dtype=np.float32), (1, 4, 4, 1))
        if lib == "keras":
            return (tensor(values, lib), (2, 2)), {"strides": 1, "padding": "valid", "data_format": "channels_last"}
        return (tensor(values, lib), 2, 1, "VALID"), {}
    if canonical == "interpolate":
        image = np.reshape(np.arange(16, dtype=np.float32), (1, 1, 4, 4))
        return (tensor(image, lib),), {"size": (2, 2), "mode": "nearest"}
    if canonical == "local_response_norm":
        image = np.reshape(np.arange(16, dtype=np.float32), (1, 1, 4, 4))
        return (tensor(image, lib), 3), {}
    if canonical in {"lp_pool1d", "lp_pool2d"}:
        dim = 1 if canonical.endswith("1d") else 2
        shape = (1, 1, 5) if dim == 1 else (1, 1, 4, 4)
        value = np.reshape(np.arange(__import__('numpy').prod(shape), dtype=np.float32), shape)
        return (tensor(value, lib), 2.0, 2), {"stride": 1}
    if canonical in {"max_pool1d", "max_pool2d", "max_pool3d", "avg_pool1d", "avg_pool2d", "avg_pool3d"}:
        dim = int(canonical[-2])
        spatial = (5,) if dim == 1 else (4, 4) if dim == 2 else (3, 3, 3)
        values = np.reshape(np.arange(np.prod(spatial), dtype=np.float32), (1, *spatial, 1))
        if lib == "tensorflow":
            return (tensor(values, lib), 2, 1, "VALID"), {}
        channel_first = np.transpose(values, (0, len(spatial) + 1, *range(1, len(spatial) + 1)))
        return (tensor(channel_first, lib), 2), {"stride": 1, "padding": 0}
    if canonical in {"adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d", "adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d"}:
        dim = int(canonical[-2])
        spatial = (5,) if dim == 1 else (3, 3) if dim == 2 else (3, 3, 3)
        value = np.reshape(np.arange(np.prod(spatial), dtype=np.float32), (1, 1, *spatial))
        output_size = 1 if dim == 1 else (1,) * dim
        return (tensor(value, lib), output_size), {}
    if canonical == "grid_sample":
        value = np.reshape(np.arange(4, dtype=np.float32), (1, 1, 2, 2))
        grid = np.array([[[[-1.0, -1.0], [1.0, -1.0]], [[-1.0, 1.0], [1.0, 1.0]]]], dtype=np.float32)
        return (tensor(value, lib), tensor(grid, lib)), {"mode": "bilinear", "padding_mode": "zeros", "align_corners": True}
    if canonical in {"max_unpool1d", "max_unpool2d", "max_unpool3d"}:
        dim = int(canonical[-2])
        if dim == 1:
            pooled = np.array([[[2.0, 4.0]]], dtype=np.float32)
            indices = np.array([[[1, 3]]], dtype=np.int64)
            return (tensor(pooled, lib), tensor(indices, lib), 2), {"stride": 2, "padding": 0, "output_size": (1, 1, 4)}
        if dim == 2:
            pooled = np.array([[[[4.0]]]], dtype=np.float32)
            indices = np.array([[[[3]]]], dtype=np.int64)
            return (tensor(pooled, lib), tensor(indices, lib), 2), {"stride": 2, "padding": 0, "output_size": (1, 1, 2, 2)}
        pooled = np.array([[[[[8.0]]]]], dtype=np.float32)
        indices = np.array([[[[[7]]]]], dtype=np.int64)
        return (tensor(pooled, lib), tensor(indices, lib), 2), {"stride": 2, "padding": 0, "output_size": (1, 1, 2, 2, 2)}
    if canonical == "crop_and_resize":
        image = np.reshape(np.arange(16, dtype=np.float32), (1, 4, 4, 1))
        boxes = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        box_indices = np.array([0], dtype=np.int32)
        crop_size = np.array([2, 2], dtype=np.int32)
        return (tensor(image, lib), tensor(boxes, lib), tensor(box_indices, lib), tuple(crop_size.tolist())), {}
    if canonical == "crop":
        image = np.reshape(np.arange(24, dtype=np.float32), (2, 3, 4))
        if lib == "tensorflow":
            return (tensor(image, lib), image.shape), {"seed": 0}
        if lib == "paddle":
            return (tensor(image, lib), image.shape, [0, 0, 0]), {}
    if canonical in {"rot90", "flip", "fliplr", "flipud", "resize"}:
        image = np.reshape(np.arange(12, dtype=np.float32), (2, 3, 2))
        if canonical == "resize":
            if lib == "numpy":
                return (image, (3, 4)), {}
            if lib == "tensorflow":
                return (tensor(image, lib), (3, 4)), {}
            return (tensor(image, lib), (3, 4, 2)), {}
        if canonical == "flip":
            axis = 0
            if lib == "torch":
                return (tensor(image, lib),), {"dims": [axis]}
            if lib == "paddle":
                return (tensor(image, lib), [axis]), {}
            if lib == "chainer":
                return (tensor(image, lib), axis), {}
            return (tensor(image, lib),), {"axis": axis}
        return (tensor(image, lib),), {}
    if canonical in {"bartlett", "blackman", "hamming", "hanning"}:
        return (5,), {}
    if canonical in {"hann_window", "hamming_window", "bartlett_window", "blackman_window"}:
        length = np.array(5, dtype=np.int32)
        if lib == "mindspore" and canonical in {"bartlett_window", "blackman_window"}:
            return (tensor(length, lib),), {"periodic": False}
        return (5,), {"periodic": False}
    if canonical == "kaiser_window":
        if lib in {"torch", "paddle", "mindspore"}:
            return (5,), {"periodic": False, "beta": 1.5}
        return (5,), {"beta": 1.5}
    if canonical == "kaiser":
        return (5, 1.5), {}
    if "keras.layers" in qname and canonical in {"add", "average", "maximum", "minimum", "multiply", "subtract"}:
        return ([tensor(X, lib), tensor(Y, lib)],), {}

    if canonical == "betainc":
        a = np.array([0.5, 1.5, 2.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        x = np.array([0.25, 0.5, 0.75], dtype=np.float32)
        return (tensor(a, lib), tensor(b, lib), tensor(x, lib)), {}
    if canonical == "divide_no_nan":
        denom = np.array([[1.0, 0.0, 2.0], [0.0, -4.0, 8.0]], dtype=np.float32)
        return (tensor(X, lib), tensor(denom, lib)), {}
    if canonical == "bucketize":
        values = np.array([-2.0, -0.5, 0.0, 1.5, 3.0], dtype=np.float32)
        boundaries = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
        if lib == "mindspore":
            return (tensor(values, lib), boundaries.tolist()), {}
        return (tensor(values, lib), tensor(boundaries, lib)), {}
    if canonical == "affine_grid":
        theta = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32)
        return (tensor(theta, lib), (1, 1, 2, 2)), {"align_corners": False}
    if canonical in {"approx_max_k", "approx_min_k"}:
        return (tensor(X, lib), 2), {}
    if canonical == "as_strided":
        return (tensor(np.arange(6, dtype=np.float32), lib), (2, 3), (3, 1)), {}
    if canonical == "bilinear":
        input1 = np.array([[1.0, 2.0, -1.0], [0.5, -0.5, 3.0]], dtype=np.float32)
        input2 = np.array([[2.0, -1.0], [1.5, 0.25]], dtype=np.float32)
        weight = np.reshape(np.arange(12, dtype=np.float32), (2, 3, 2)) / 10.0
        if lib == "paddle":
            return (tensor(input1, lib), tensor(input2, lib), tensor(weight, lib)), {}
        bias = np.array([0.25, -0.75], dtype=np.float32)
        return (tensor(input1, lib), tensor(input2, lib), tensor(weight, lib), tensor(bias, lib)), {}
    if canonical == "channel_shuffle":
        value = np.reshape(np.arange(16, dtype=np.float32), (1, 4, 2, 2))
        return (tensor(value, lib), 2), {}
    if canonical == "chunk":
        if lib == "torch" or (lib == "mindspore" and ".mint." in qname):
            return (tensor(EVEN, lib), 2), {"dim": 1}
        return (tensor(EVEN, lib), 2), {"axis": 1}
    if canonical == "combinations":
        return (tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), lib),), {"r": 2}
    if canonical == "diagonal_scatter":
        base = np.zeros((3, 3), dtype=np.float32)
        src = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        return (tensor(base, lib), tensor(src, lib)), {}
    if canonical == "ger":
        return (tensor(np.array([1.0, -2.0], dtype=np.float32), lib), tensor(np.array([0.5, 1.5, -1.0], dtype=np.float32), lib)), {}
    if canonical == "view_as":
        other = np.zeros((3, 2), dtype=np.float32)
        return (tensor(np.arange(6, dtype=np.float32), lib), tensor(other, lib)), {}
    if canonical == "rms_norm":
        weight = np.ones((3,), dtype=np.float32)
        if lib == "mindspore":
            return (tensor(X, lib), tensor(weight, lib)), {}
        return (tensor(X, lib), (3,)), {"weight": tensor(weight, lib), "eps": 1e-6}
    if canonical in {"unsorted_segment_max", "unsorted_segment_min", "unsorted_segment_prod", "unsorted_segment_sum"}:
        data = np.array([1.0, -2.0, 3.0, 4.0], dtype=np.float32)
        segment_ids = np.array([0, 1, 0, 2], dtype=np.int32)
        return (tensor(data, lib), tensor(segment_ids, lib), 4), {}
    if canonical in {"xdivy", "xlog1py"}:
        left = np.array([0.0, 1.0, -2.0], dtype=np.float32)
        right = np.array([0.0, 2.0, 3.0], dtype=np.float32)
        return (tensor(left, lib), tensor(right, lib)), {}
    if canonical == "in1d":
        return (tensor(np.array([1, 2, 3, 2], dtype=np.int64), lib), tensor(np.array([2, 4], dtype=np.int64), lib)), {}
    if canonical in {"pixel_shuffle", "pixel_unshuffle"}:
        factor = 2
        shape = (1, 4, 2, 2) if canonical == "pixel_shuffle" else (1, 1, 4, 4)
        value = np.reshape(np.arange(np.prod(shape), dtype=np.float32), shape)
        return (tensor(value, lib), factor), {}
    if canonical in {"convolve2d", "correlate2d"}:
        image = np.array([[1.0, 2.0, 0.0], [0.5, -1.0, 3.0], [2.0, 1.0, -0.5]], dtype=np.float32)
        kernel = np.array([[1.0, 0.5], [-0.25, 2.0]], dtype=np.float32)
        return (tensor(image, lib), tensor(kernel, lib)), {"mode": "same", "boundary": "fill", "fillvalue": 0.0}
    if canonical == "vectorized_map":
        return (lambda z: z + 1, tensor(VEC_A, lib)), {}
    if canonical == "scan":
        xs = tensor(VEC_A, lib)
        if lib == "tensorflow":
            return (lambda acc, x: acc + x, xs), {"initializer": tensor(np.array(0.0, dtype=np.float32), lib)}
        return (lambda carry, x: (carry + x, carry + x), tensor(np.array(0.0, dtype=np.float32), lib), xs), {}
    if canonical == "map":
        return (lambda z: z + 1, tensor(VEC_A, lib)), {}
    if canonical == "fori_loop":
        return (0, 3, lambda i, val: val + i, tensor(np.array(0.0, dtype=np.float32), lib)), {}
    if canonical == "switch":
        return (1, [lambda z: z - 1, lambda z: z + 1], tensor(VEC_A, lib)), {}
    if canonical in {"segment_max", "segment_sum"}:
        data = np.array([1.0, -2.0, 3.0, 4.0], dtype=np.float32)
        segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
        return (tensor(data, lib), tensor(segment_ids, lib)), {}
    if canonical == "space_to_batch_nd":
        value = np.reshape(np.arange(4, dtype=np.float32), (1, 2, 2, 1))
        paddings = np.array([[0, 0], [0, 0]], dtype=np.int32)
        if lib == "mindspore":
            return (tensor(value, lib), 2, tuple(map(tuple, paddings.tolist()))), {}
        return (tensor(value, lib), [2, 2], tensor(paddings, lib)), {}
    if canonical in {"space_to_depth", "depth_to_space"}:
        value = np.reshape(np.arange(16, dtype=np.float32), (1, 2, 2, 4)) if canonical == "depth_to_space" else np.reshape(np.arange(16, dtype=np.float32), (1, 4, 4, 1))
        return (tensor(value, lib), 2), {"data_format": "channels_last"} if lib == "keras" else {}
    if canonical == "moments":
        return (tensor(X, lib), (0,)), {}
    if canonical == "saturate_cast":
        return (tensor(X, lib), dtype_arg(lib, "float32")), {}
    if canonical == "reverse_sequence":
        value = np.reshape(np.arange(6, dtype=np.float32), (2, 3))
        seq_lengths = np.array([3, 2], dtype=np.int32)
        if lib == "tensorflow":
            return (tensor(value, lib), tensor(seq_lengths, lib), 1), {"batch_axis": 0}
        return (tensor(value, lib), tensor(seq_lengths, lib), 1, 0), {}
    if canonical == "is_nonzero":
        return (tensor(np.array(1.0, dtype=np.float32), lib),), {}
    if canonical == "movedim":
        return (tensor(X, lib), 0, 1), {}
    if canonical == "mvlgamma":
        return (tensor(np.array([1.5, 2.0, 3.0], dtype=np.float32), lib), 2), {}
    if canonical == "randperm":
        return (5,), {}
    if canonical == "squared_difference":
        return (tensor(X, lib), tensor(Y, lib)), {}

    if canonical in {
        "add",
        "subtract",
        "multiply",
        "divide",
        "maximum",
        "minimum",
        "power",
        "equal",
        "not_equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "left_shift",
        "right_shift",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "floor_divide",
        "floor_mod",
        "divmod",
        "fmod",
        "copysign",
        "fmax",
        "fmin",
        "gammainc",
        "gammaincc",
        "igamma",
        "igammac",
        "logaddexp",
        "logaddexp2",
        "float_power",
        "xlogy",
        "gcd",
        "heaviside",
        "hypot",
        "arctan2",
        "atan2",
        "kron",
        "correlate",
        "lcm",
        "ldexp",
        "mod",
        "remainder",
        "nextafter",
        "array_equal",
        "array_equiv",
        "logical_and",
        "logical_or",
        "logical_xor",
    }:
        if canonical in {"left_shift", "right_shift", "bitwise_and", "bitwise_or", "bitwise_xor", "gcd"}:
            return (tensor(INT_A, lib), tensor(INT_B, lib)), {}
        if canonical in {"logical_and", "logical_or", "logical_xor"}:
            return (tensor(BOOL, lib), tensor(~BOOL, lib)), {}
        if canonical in {"floor_divide", "floor_mod", "divmod"}:
            lhs = np.array([-5, -1, 5], dtype=np.int64)
            rhs = np.array([2, 2, 3], dtype=np.int64)
            return (tensor(lhs, lib), tensor(rhs, lib)), {}
        if canonical == "fmod":
            return (tensor(VEC_A, lib), tensor(np.array([2.0, 2.0, 2.0], dtype=np.float32), lib)), {}
        if canonical in {"copysign", "remainder"}:
            return (tensor(VEC_A, lib), tensor(np.array([2.0, -2.0, 2.0], dtype=np.float32), lib)), {}
        if canonical in {"fmax", "fmin", "logaddexp", "logaddexp2", "float_power", "xlogy"}:
            return (tensor(VEC_A, lib), tensor(VEC_B, lib)), {}
        if canonical in {"gammainc", "gammaincc", "igamma", "igammac"}:
            return (tensor(np.array([1.0, 2.0], dtype=np.float32), lib), tensor(np.array([0.5, 1.5], dtype=np.float32), lib)), {}
        if canonical == "heaviside":
            return (tensor(VEC_A, lib), tensor(np.array([0.5, 0.5, 0.5], dtype=np.float32), lib)), {}
        if canonical in {"hypot", "arctan2", "atan2"}:
            return (tensor(VEC_A, lib), tensor(VEC_B, lib)), {}
        if canonical == "kron":
            return (tensor(MAT, lib), tensor(MAT_B, lib)), {}
        if canonical == "correlate":
            return (tensor(VEC_A, lib), tensor(VEC_B, lib)), {}
        if canonical in {"lcm", "mod"}:
            return (tensor(INT_A, lib), tensor(np.array([2, 3, 5], dtype=np.int64), lib)), {}
        if canonical == "ldexp":
            exp_dtype = np.float32 if lib == "mxnet" else np.int64
            return (tensor(VEC_A, lib), tensor(np.array([1, 2, 3], dtype=exp_dtype), lib)), {}
        if canonical == "nextafter":
            return (tensor(VEC_A, lib), tensor(VEC_B, lib)), {}
        if canonical in {"array_equal", "array_equiv"}:
            return (tensor(X, lib), tensor(X.copy(), lib)), {}
        if canonical == "power":
            if lib == "mxnet" and qname.startswith("mxnet.ndarray."):
                return (tensor(POS, lib), tensor(np.full_like(POS, 1.5), lib)), {}
            return (tensor(POS, lib), 1.5), {}
        rhs = POS if canonical == "divide" else Y
        return (tensor(X, lib), tensor(rhs, lib)), {}
    if canonical == "population_count":
        return (tensor(np.array([1, 2, 3], dtype=np.int64), lib),), {}
    if canonical in {"bitwise_not", "logical_not"}:
        value = BOOL if canonical == "logical_not" else np.array([1, 2, 3], dtype=np.int64)
        return (tensor(value, lib),), {}
    if canonical in {"lerp", "dist"}:
        if canonical == "lerp":
            return (tensor(X, lib), tensor(Y, lib), 0.25), {}
        return (tensor(X, lib), tensor(Y, lib), 2), {}
    if canonical == "renorm":
        return (tensor(X, lib), 2, 0, 100.0), {}
    if canonical == "scale":
        if lib == "chainer":
            scale = np.array([2.0, 2.0, 2.0], dtype=np.float32)
            return (tensor(X, lib), tensor(scale, lib), 1), {}
        return (tensor(X, lib),), {"scale": 2.0, "bias": 0.0, "bias_after_scale": True}

    if canonical == "associative_scan":
        return (lambda a, b: a + b, tensor(VEC_A, lib)), {}
    if canonical in {"apply_along_axis", "apply_over_axes"} and lib == "mindspore":
        raise SkipCall(f"MindSpore {canonical} needs a framework-native callable adapter")
    if canonical == "apply_along_axis":
        return (lambda row: np.sum(row), 1, tensor(X, lib)), {}
    if canonical == "apply_over_axes":
        return (lambda a, axis: np.sum(a, axis=axis), tensor(X, lib), [0]), {}
    if canonical == "cartesian_prod":
        return (tensor(np.array([1, 2], dtype=np.int64), lib), tensor(np.array([3, 4], dtype=np.int64), lib)), {}
    if canonical == "cdist":
        a = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        return (tensor(a, lib), tensor(b, lib)), {}
    if canonical == "jvp":
        primal = tensor(VEC_A, lib)
        tangent = tensor(np.ones_like(VEC_A), lib)
        return (lambda z: z * z, (primal,), (tangent,)), {}
    if canonical == "clip_by_norm":
        return (tensor(X, lib), 1.0), {}
    if canonical == "clip_by_global_norm":
        values = [tensor(VEC_A, lib), tensor(VEC_B, lib)]
        return (values, 1.0), {}
    if canonical in {"geomspace", "linspace", "logspace"}:
        return (1.0, 10.0, 5), {}
    if canonical in {"move_axis", "swap_axes", "permute_dims", "rollaxis"}:
        if canonical == "move_axis":
            return (tensor(X, lib), 0, 1), {}
        if canonical == "rollaxis":
            return (tensor(X, lib), 1), {}
        if canonical == "permute_dims":
            return (tensor(X, lib), (1, 0)), {}
        return (tensor(X, lib), 0, 1), {}
    if canonical == "repeat":
        return (tensor(VEC_A, lib), 2), {}
    if canonical == "repeat_interleave":
        return (tensor(INT_A, lib), 2), {}
    if canonical == "roll":
        if lib == "tensorflow":
            return (tensor(VEC_A, lib), 1, 0), {}
        return (tensor(VEC_A, lib), 1), {"axis": 0} if lib == "paddle" else {}
    if canonical == "reverse":
        if lib == "tensorflow":
            return (tensor(X, lib),), {"axis": [1]}
        if lib == "torch":
            return (tensor(X, lib),), {"dims": [1]}
        return (tensor(X, lib),), {"axis": [1]}
    if canonical == "select":
        return ([tensor(VEC_A > 0, lib)], [tensor(VEC_A, lib)],), {"default": tensor(np.zeros_like(VEC_A), lib)}
    if canonical == "ravel_multi_index":
        coords = (tensor(np.array([0, 1], dtype=np.int64), lib), tensor(np.array([1, 2], dtype=np.int64), lib))
        return (coords, (2, 3)), {}
    if canonical == "take_along_axis":
        indices = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64)
        return (tensor(X, lib), tensor(indices, lib), 1), {}
    if canonical == "take_along_dim":
        indices = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64)
        if lib == "torch":
            return (tensor(X, lib), tensor(indices, lib)), {"dim": 1}
        return (tensor(X, lib), tensor(indices, lib)), {"axis": 1}
    if canonical == "put_along_axis":
        base = np.zeros((2, 3), dtype=np.float32)
        indices = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int64)
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        kwargs = {"inplace": False} if lib == "jax" else {}
        return (tensor(base, lib), tensor(indices, lib), tensor(values, lib), 1), kwargs
    if canonical == "gather":
        index_1d = np.array([0, 2], dtype=np.int64)
        if lib == "torch":
            index_2d = np.tile(index_1d, (2, 1))
            return (tensor(X, lib), 1, tensor(index_2d, lib)), {}
        return (tensor(X, lib), tensor(index_1d, lib)), {"axis": 1}
    if canonical == "gather_nd":
        indices = np.array([[0, 1], [1, 2]], dtype=np.int64)
        if lib == "mxnet":
            return (tensor(X, lib), tensor(indices.T, lib)), {}
        return (tensor(X, lib), tensor(indices, lib)), {}
    if canonical == "take":
        return (tensor(VEC, lib), tensor(np.array([0, 2], dtype=np.int64), lib)), {}
    if canonical == "digitize":
        bins = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
        return (tensor(VEC_A, lib), tensor(bins, lib)), {}
    if canonical == "compress":
        return (tensor(np.array([True, False, True], dtype=bool), lib), tensor(VEC_A, lib)), {}
    if canonical == "delete":
        return (tensor(VEC, lib), 1), {}
    if canonical == "extract":
        return (tensor(np.array([True, False, True], dtype=bool), lib), tensor(VEC_A, lib)), {}
    if canonical in {"intersect1d", "setdiff1d", "setxor1d", "union1d"}:
        return (tensor(np.array([1, 2, 3], dtype=np.int64), lib), tensor(np.array([2, 3, 4], dtype=np.int64), lib)), {}
    if canonical == "packbits":
        return (tensor(np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8), lib),), {}
    if canonical == "unpackbits":
        return (tensor(np.array([178], dtype=np.uint8), lib),), {}
    if canonical == "partition":
        return (tensor(VEC, lib), 1), {}
    if canonical == "tensor_split":
        if lib == "torch":
            return (tensor(EVEN, lib), 2), {"dim": 1}
        return (tensor(EVEN, lib), 2), {"axis": 1}
    if canonical == "unflatten":
        if lib == "torch":
            return (tensor(VEC, lib), 0, (2, 2)), {}
        return (tensor(VEC, lib), 0, (2, 2)), {}
    if canonical == "narrow":
        return (tensor(X, lib), 1, 1, 2), {}
    if canonical == "multinomial" and category != "random":
        probs = np.array([0.0, 1.0], dtype=np.float32)
        return (tensor(probs, lib), 1), {"replacement": True}
    if canonical == "percentile":
        return (tensor(X, lib), 50.0), {}
    if canonical in {"polydiv", "polymul"}:
        return (tensor(np.array([1.0, -3.0, 2.0], dtype=np.float32), lib), tensor(np.array([1.0, -1.0], dtype=np.float32), lib)), {}
    if canonical == "roots":
        return (tensor(np.array([1.0, -3.0, 2.0], dtype=np.float32), lib),), {}
    if canonical == "one_hot":
        indices = np.array([0, 1, 2], dtype=np.int64)
        return (tensor(indices, lib), 3), {}
    if canonical == "scatter_nd":
        indices = np.array([[0], [2]], dtype=np.int64)
        updates = np.array([5.0, 7.0], dtype=np.float32)
        if lib == "mxnet":
            return (tensor(updates, lib), tensor(indices.T, lib), (3,)), {}
        return (tensor(indices, lib), tensor(updates, lib), (3,)), {}
    if canonical == "get_item":
        return (tensor(X, lib), (slice(None), 1)), {}
    if canonical == "tri":
        return (3, 4), {}
    if canonical == "vander":
        return (tensor(VEC_A, lib),), {}
    if canonical in {"astype", "cast"}:
        return (tensor(X, lib), dtype_arg(lib, "float64")), {}
    if canonical == "choose":
        choices = [tensor(VEC_A, lib), tensor(VEC_B, lib)]
        return (tensor(np.array([0, 1, 0], dtype=np.int64), lib), choices), {}
    if canonical == "indices":
        return ((3, 3),), {}
    if canonical == "diag_indices":
        return (3,), {}
    if canonical in {"tril_indices", "triu_indices"}:
        if lib in {"jax", "keras", "mxnet", "numpy"} or qname.startswith("mindspore.numpy."):
            return (3,), {"m": 3}
        return (3, 3), {}
    if canonical == "fill_diagonal":
        value = np.zeros((3, 3), dtype=np.float32)
        kwargs = {"inplace": False} if lib == "jax" else {}
        return (tensor(value, lib), 7.0), kwargs
    if canonical == "diag_indices_from":
        return (tensor(np.eye(3, dtype=np.float32), lib),), {}
    if canonical in {"histogram2d"}:
        x = np.array([-2.5, -0.5, 0.5, 2.5], dtype=np.float32)
        y = np.array([-1.5, 0.5, 1.5, 3.5], dtype=np.float32)
        return (tensor(x, lib), tensor(y, lib)), {"bins": 4, "range": ((-4.0, 4.0), (-2.0, 4.0))}
    if canonical == "histogramdd":
        sample = np.array([[-2.0, -1.0], [-0.5, 0.5], [0.5, 1.5], [2.0, 3.0]], dtype=np.float32)
        if lib == "torch":
            return (tensor(sample, lib),), {"bins": [2, 2], "range": [-4.0, 4.0, -2.0, 4.0]}
        return (tensor(sample, lib),), {"bins": (2, 2), "range": ((-4.0, 4.0), (-2.0, 4.0))}
    if canonical in {"histogram", "histogram_bin_edges"}:
        data = np.array([-2.5, -0.5, 0.5, 2.5], dtype=np.float32)
        if lib == "paddle":
            return (tensor(data, lib),), {"bins": 4, "min": -4.0, "max": 4.0}
        return (tensor(data, lib),), {"bins": 4, "range": (-4.0, 4.0)}
    if canonical in {"quantile", "nanquantile"}:
        return (tensor(X, lib), 0.5), {}
    if canonical in {"percentile", "nanpercentile"}:
        return (tensor(X, lib), 50.0), {}
    if canonical == "insert":
        return (tensor(VEC_A, lib), 1, tensor(np.array([9.0], dtype=np.float32), lib)), {}
    if canonical == "interp":
        xp = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        fp = np.array([0.0, 1.0, 4.0], dtype=np.float32)
        return (tensor(np.array([0.5, 1.5], dtype=np.float32), lib), tensor(xp, lib), tensor(fp, lib)), {}
    if canonical == "isdtype":
        return (dtype_arg(lib, "float32"), "real floating"), {}
    if canonical == "isin":
        return (tensor(INT_A, lib), tensor(np.array([1, 4], dtype=np.int64), lib)), {}
    if canonical == "issubdtype":
        return (dtype_arg(lib, "float32"), dtype_arg(lib, "float64")), {}
    if canonical == "ix":
        return (tensor(np.array([0, 1], dtype=np.int64), lib), tensor(np.array([1, 2], dtype=np.int64), lib)), {}
    if canonical == "pad":
        if lib == "chainer":
            return (tensor(X, lib), ((1, 1), (1, 1)), "constant"), {}
        if lib == "paddle" and category == "nn":
            return (tensor(X, lib), [1, 1, 1, 1]), {"mode": "constant", "value": 0.0}
        if lib in {"mindspore", "torch"}:
            return (tensor(X, lib), (1, 1, 1, 1)), {}
        return (tensor(X, lib), ((1, 1), (1, 1))), {}
    if canonical in {"poly", "polyadd", "polysub", "polyder", "polyint"}:
        if canonical == "poly":
            return (tensor(VEC_A, lib),), {}
        if canonical in {"polyadd", "polysub"}:
            return (tensor(VEC_A, lib), tensor(VEC_B, lib)), {}
        return (tensor(VEC_A, lib),), {}
    if canonical == "polyfit":
        return (tensor(np.array([0.0, 1.0, 2.0], dtype=np.float32), lib), tensor(VEC_A, lib), 1), {}
    if canonical == "piecewise":
        return (tensor(VEC_A, lib), [tensor(VEC_A < 0, lib), tensor(VEC_A >= 0, lib)], [lambda z: -z, lambda z: z]), {}
    if canonical in {"fftfreq", "rfftfreq"}:
        return (8,), {}
    if canonical in {"fftshift", "ifftshift"}:
        return (tensor(VEC, lib),), {}
    if canonical == "unfold":
        if category != "nn":
            raise SkipCall("1D tensor unfold and image patch unfold have different semantics")
        image = np.reshape(np.arange(16, dtype=np.float32), (1, 1, 4, 4))
        if lib == "torch":
            return (tensor(image, lib), (2, 2)), {"stride": 1, "padding": 0, "dilation": 1}
        if lib == "mindspore":
            return (tensor(image, lib), (2, 2)), {"stride": 1, "padding": 0, "dilation": 1}
        if lib == "paddle":
            return (tensor(image, lib), (2, 2)), {"strides": 1, "paddings": 0, "dilations": 1}
    if canonical == "cho_solve":
        import scipy.linalg

        return (scipy.linalg.cho_factor(SPD), RHS), {}
    if canonical == "multi_dot":
        return ([tensor(MAT, lib), tensor(MAT_B, lib)],), {}
    if canonical == "funm":
        if lib == "jax":
            raise SkipCall("jax funm is unsupported on the current backend")
        return (tensor(MAT, lib), lambda z: z), {}
    if canonical in {"hessenberg", "sqrtm"} and lib == "jax":
        raise SkipCall(f"jax {canonical} is unsupported on the current backend")
    if canonical == "rsf2csf":
        import scipy.linalg

        t, z = scipy.linalg.schur(MAT)
        return (t, z), {}
    if canonical == "hilbert":
        return (4,), {}
    if canonical in {"convolve", "fftconvolve"}:
        return (tensor(VEC_A, lib), tensor(VEC_B, lib)), {}
    if canonical == "detrend":
        clean_vec = np.array([1.0, -2.0, 3.0, 4.0], dtype=np.float32)
        return (tensor(clean_vec, lib),), {}
    if canonical == "csd":
        return (tensor(VEC, lib), tensor(VEC, lib)), {"nperseg": 4}
    if canonical == "welch":
        return (tensor(VEC, lib),), {"nperseg": 4}
    if canonical in {"geterr", "get_printoptions"}:
        return (), {}

    if canonical in {"ctc", "ctc_loss"}:
        probs_time = np.array([[[0.6, 0.3, 0.1]], [[0.6, 0.3, 0.1]]], dtype=np.float32)
        log_probs_time = np.log(probs_time).astype(np.float32)
        labels_dense = np.array([[1]], dtype=np.int32)
        labels_flat = np.array([1], dtype=np.int64)
        input_lengths = np.array([2], dtype=np.int64)
        label_lengths = np.array([1], dtype=np.int64)
        if canonical == "ctc":
            return (tensor(labels_dense, lib), tensor(np.transpose(probs_time, (1, 0, 2)), lib)), {}
        if qname == "keras.ops.ctc_loss":
            return (tensor(labels_dense, lib), tensor(np.transpose(log_probs_time, (1, 0, 2)), lib), tensor(label_lengths, lib), tensor(input_lengths, lib)), {"mask_index": 0}
        if lib == "tensorflow":
            return (tensor(labels_dense, lib), tensor(log_probs_time, lib), tensor(label_lengths, lib), tensor(input_lengths, lib)), {"logits_time_major": True, "blank_index": 0}
        if lib == "paddle":
            return (tensor(log_probs_time, lib), tensor(labels_dense, lib), tensor(input_lengths, lib), tensor(label_lengths, lib)), {"blank": 0, "reduction": "none", "zero_infinity": True}
        if lib == "torch":
            return (tensor(log_probs_time, lib), tensor(labels_flat, lib), tensor(input_lengths, lib), tensor(label_lengths, lib)), {"blank": 0, "reduction": "none", "zero_infinity": True}

    if canonical == "smooth_l1_loss":
        return (tensor(Y_PRED, lib), tensor(Y_TRUE, lib)), {}

    if canonical == "binary_cross_entropy_with_logits":
        logits = np.array([[2.0, -1.0, 0.5], [-0.5, 1.5, -2.0]], dtype=np.float32)
        labels = Y_TRUE.astype(np.float32)
        if lib == "tensorflow":
            return (), {"labels": tensor(labels, lib), "logits": tensor(logits, lib)}
        return (tensor(logits, lib), tensor(labels, lib)), {"reduction": "none"}

    if canonical == "softmax_cross_entropy":
        logits = np.array([[2.0, 0.5, -1.0]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)
        if lib == "tensorflow":
            return (tensor(np.eye(3, dtype=np.float32)[labels], lib), tensor(logits, lib)), {}
        if lib == "chainer":
            return (tensor(logits, lib), tensor(labels.astype(np.int32), lib)), {"reduce": "no"}
        if lib == "mxnet":
            return (tensor(logits, lib), tensor(labels, lib)), {}

    if canonical == "cross_entropy":
        logits = np.array([[2.0, 0.5, -1.0], [-0.25, 1.25, 0.75]], dtype=np.float32)
        labels = np.array([0, 2], dtype=np.int64)
        return (tensor(logits, lib), tensor(labels, lib)), {"reduction": "none"}

    if canonical == "cosine_embedding_loss":
        input1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        input2 = np.array([[0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
        target = np.array([1.0, -1.0], dtype=np.float32)
        return (tensor(input1, lib), tensor(input2, lib), tensor(target, lib)), {"reduction": "none"}

    if canonical in {"hinge_embedding_loss", "soft_margin_loss"}:
        values = np.array([0.5, -1.0, 2.0], dtype=np.float32)
        target = np.array([1.0, -1.0, 1.0], dtype=np.float32)
        return (tensor(values, lib), tensor(target, lib)), {"reduction": "none"}

    if canonical == "nll_loss":
        log_probs = np.log(np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=np.float32))
        target = np.array([0, 2], dtype=np.int64)
        return (tensor(log_probs, lib), tensor(target, lib)), {"reduction": "none"}

    if canonical == "gaussian_nll_loss":
        pred = np.array([0.5, -1.0, 2.0], dtype=np.float32)
        target = np.array([0.0, -0.5, 1.5], dtype=np.float32)
        var = np.array([1.0, 2.0, 0.5], dtype=np.float32)
        return (tensor(pred, lib), tensor(target, lib), tensor(var, lib)), {"reduction": "none"}

    if canonical == "poisson_nll_loss":
        pred = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        target = np.array([0.0, 1.0, 3.0], dtype=np.float32)
        return (tensor(pred, lib), tensor(target, lib)), {"log_input": True, "reduction": "none"}

    if canonical == "pairwise_distance":
        input1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        input2 = np.array([[0.5, 0.5], [1.0, 0.0]], dtype=np.float32)
        return (tensor(input1, lib), tensor(input2, lib)), {}

    if canonical == "scaled_dot_product_attention":
        q = np.reshape(np.arange(8, dtype=np.float32), (1, 1, 2, 4)) / 10.0
        k = q.copy()
        v = np.flip(q, axis=-1).copy()
        return (tensor(q, lib), tensor(k, lib), tensor(v, lib)), {"dropout_p": 0.0}

    if canonical == "triplet_margin_with_distance_loss":
        anchor = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        positive = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
        negative = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        return (tensor(anchor, lib), tensor(positive, lib), tensor(negative, lib)), {"reduction": "none"}

    if canonical == "multi_margin_loss":
        logits = np.array([[0.2, 1.5, -0.5], [2.0, -1.0, 0.25]], dtype=np.float32)
        target = np.array([1, 0], dtype=np.int64)
        return (tensor(logits, lib), tensor(target, lib)), {"reduction": "none"}

    if canonical == "margin_ranking_loss":
        input1 = np.array([1.0, 2.0, -0.5], dtype=np.float32)
        input2 = np.array([0.5, 1.5, 0.25], dtype=np.float32)
        target = np.array([1.0, -1.0, 1.0], dtype=np.float32)
        reduction = 0 if qname == "torch.margin_ranking_loss" else "none"
        return (tensor(input1, lib), tensor(input2, lib), tensor(target, lib)), {"reduction": reduction}

    if canonical == "triplet_margin_loss":
        anchor = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        positive = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
        negative = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        reduction = 0 if qname == "torch.triplet_margin_loss" else "none"
        return (tensor(anchor, lib), tensor(positive, lib), tensor(negative, lib)), {"reduction": reduction}

    if canonical == "add_n":
        return ([tensor(X, lib), tensor(Y, lib)],), {}

    if category == "loss" or canonical in {
        "kl_divergence",
        "l1_loss",
        "mse_loss",
        "smooth_l1_loss",
        "binary_cross_entropy",
        "binary_focal_crossentropy",
        "categorical_cross_entropy",
        "categorical_focal_crossentropy",
        "categorical_generalized_cross_entropy",
        "categorical_hinge",
        "circle",
        "cosine_similarity",
        "dice",
        "hinge",
        "huber",
        "poisson",
        "sparse_categorical_cross_entropy",
        "squared_hinge",
        "tversky",
        "binary_accuracy",
        "categorical_accuracy",
        "sparse_categorical_accuracy",
        "top_k_categorical_accuracy",
        "sparse_top_k_categorical_accuracy",
        "concordance_correlation",
        "pearson_correlation",
    }:
        if canonical in {"categorical_cross_entropy", "categorical_focal_crossentropy", "categorical_hinge", "top_k_categorical_accuracy", "categorical_accuracy"}:
            return (tensor(CAT_TRUE, lib), tensor(CAT_PRED, lib)), {}
        if canonical in {"sparse_categorical_cross_entropy", "sparse_categorical_accuracy", "sparse_top_k_categorical_accuracy"}:
            return (tensor(SPARSE_TRUE, lib), tensor(CAT_PRED, lib)), {}
        if canonical == "kl_divergence" and qname.startswith("jax.scipy.special."):
            return (tensor(CAT_TRUE + 0.1, lib), tensor(CAT_PRED + 0.1, lib)), {}
        if canonical == "categorical_generalized_cross_entropy":
            return (tensor(SPARSE_TRUE, lib), tensor(CAT_PRED, lib), 0.7), {}
        if canonical == "circle":
            labels = np.array([0, 1], dtype=np.int64)
            embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            return (tensor(labels, lib), tensor(embeddings, lib)), {}
        if canonical == "binary_cross_entropy" and lib in {"mindspore", "paddle", "torch"}:
            return (tensor(Y_PRED, lib), tensor(Y_TRUE, lib)), {"reduction": "none"}
        if canonical == "huber_loss" and lib == "chainer":
            return (tensor(Y_PRED, lib), tensor(Y_TRUE, lib), 1.0), {}
        return (tensor(Y_TRUE, lib), tensor(Y_PRED, lib)), {}

    if canonical == "input":
        raise SkipCall("Keras Input creates symbolic graph placeholders")
    if "keras.layers" in qname and canonical in {"add", "average", "maximum", "minimum", "multiply", "subtract"}:
        return ([tensor(X, lib), tensor(Y, lib)],), {}

    if canonical == "complex":
        return (tensor(VEC_A, lib), tensor(VEC_B, lib)), {}
    if canonical == "polar":
        return (tensor(np.abs(VEC_A).astype(np.float32), lib), tensor(np.array([0.0, 0.5, 1.0], dtype=np.float32), lib)), {}
    if canonical in {"conj", "real", "imag", "view_as_real"}:
        value = np.array([1.0 + 2.0j, -3.0 + 0.5j], dtype=np.complex64)
        return (tensor(value, lib),), {}
    if canonical == "view_as_complex":
        value = np.array([[1.0, 2.0], [-3.0, 0.5]], dtype=np.float32)
        return (tensor(value, lib),), {}

    if canonical == "erfinv":
        values = np.array([[-0.75, -0.25, 0.0], [0.25, 0.5, 0.75]], dtype=np.float32)
        return (tensor(values, lib),), {}

    if canonical == "silu" and lib == "chainer":
        return (tensor(X, lib), tensor(np.array(1.0, dtype=np.float32), lib)), {}

    if canonical in {"bessel_y0", "bessel_y1"}:
        return (tensor(POS, lib),), {}

    if category in {"nn"} or canonical in {
        "abs",
        "negative",
        "exp",
        "expm1",
        "log",
        "log1p",
        "sqrt",
        "square",
        "sin",
        "cos",
        "tan",
        "tanh",
        "sinh",
        "cosh",
        "asin",
        "arcsin",
        "acos",
        "arccos",
        "atan",
        "arctan",
        "asinh",
        "arcsinh",
        "acosh",
        "arccosh",
        "atanh",
        "arctanh",
        "floor",
        "ceil",
        "round",
        "sign",
        "reciprocal",
        "sigmoid",
        "relu",
        "relu6",
        "selu",
        "elu",
        "celu",
        "gelu",
        "silu",
        "mish",
        "softplus",
        "hardsigmoid",
        "hardswish",
        "logsigmoid",
        "erf",
        "erfc",
        "digamma",
        "lgamma",
        "gamma",
        "gammaln",
    }:
        value = POS if canonical in {"log", "sqrt", "reciprocal", "acosh", "arccosh"} else X
        if canonical == "hardshrink":
            value = np.array([[1.0, -2.0, 3.0], [4.0, 0.75, -6.0]], dtype=np.float32)
        return (tensor(value, lib),), {}

    if category == "linalg":
        return (tensor(MAT, lib),), {}
    if category in {"generic", "reduction"}:
        return (tensor(X, lib),), {}

    raise UnsupportedCall(f"no generic input plan for {canonical}/{category}")


def encode(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.resolve_conj().detach().cpu().numpy()
        if isinstance(value, torch.dtype):
            return str(value).replace("torch.", "")
    except Exception:
        pass
    if hasattr(value, "asnumpy"):
        value = value.asnumpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    else:
        array_attr = getattr(value, "array", None)
        if array_attr is not None:
            value = array_attr.get() if hasattr(array_attr, "get") else array_attr
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if isinstance(value, str):
        return value
    if isinstance(value, np.dtype):
        return str(value)
    if isinstance(value, type):
        return value.__name__
    arr = np.asarray(value)
    if arr.dtype == object:
        return str(value)
    if arr.dtype.kind == "c":
        stacked = np.stack([arr.real, arr.imag], axis=-1)
        return {"shape": list(arr.shape), "dtype": str(arr.dtype), "value": stacked.tolist()}
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "value": arr.tolist()}


def array_value(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.resolve_conj().detach().cpu().numpy()
    except Exception:
        pass
    if hasattr(value, "asnumpy"):
        return value.asnumpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    array_attr = getattr(value, "array", None)
    if array_attr is not None:
        return array_attr.get() if hasattr(array_attr, "get") else np.asarray(array_attr)
    return np.asarray(value)


def real_if_close(value: Any) -> np.ndarray:
    arr = array_value(value)
    if arr.dtype.kind == "c" and np.allclose(arr.imag, 0, atol=1e-4, rtol=1e-4):
        return arr.real
    return arr


def normalize_dtype_name(value: Any) -> str:
    text = str(value).replace("torch.", "").replace("mindspore.", "").strip()
    text = text.replace("<class '", "").replace("'>", "")
    return text.lower()


def sort_1d_values(value: Any) -> np.ndarray:
    arr = np.asarray(real_if_close(value))
    if arr.ndim != 1:
        return arr
    if arr.dtype.kind == "c":
        order = np.lexsort((arr.imag, arr.real))
    else:
        order = np.argsort(arr, kind="stable")
    return arr[order]


def trim_trailing_zeros(value: Any) -> np.ndarray:
    arr = np.asarray(real_if_close(value)).reshape(-1)
    end = arr.size
    while end > 1 and np.allclose(arr[end - 1], 0, atol=1e-6, rtol=1e-6):
        end -= 1
    return arr[:end]


def global_mean(value: Any) -> np.ndarray:
    arr = np.asarray(array_value(value), dtype=np.float64)
    return np.asarray(np.mean(arr), dtype=np.float64)


def config_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "class_name" in value:
            return str(value["class_name"])
        if "config" in value and isinstance(value["config"], dict) and "name" in value["config"]:
            return str(value["config"]["name"])
        return json.dumps(value, sort_keys=True)
    if callable(value):
        return getattr(value, "__name__", value.__class__.__name__)
    return value.__class__.__name__


def normalize_result(api: Api, canonical: str, value: Any) -> Any:
    qname = api.qualified_name
    if canonical in {"get", "serialize", "deserialize"}:
        return config_name(value)
    if canonical == "shape":
        return np.asarray(value, dtype=np.int64)
    if canonical in {"dtype", "get_default_dtype", "promote_types", "result_type"}:
        return normalize_dtype_name(value)
    if canonical == "get_default_device":
        return str(value).replace("device(type=", "").strip("')")
    if canonical == "array_repr":
        return str(value).replace("ArrayImpl", "array")
    if canonical in {"conv", "conv1d", "conv2d", "conv3d", "conv_transpose", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d", "max_pool1d", "max_pool2d", "max_pool3d", "avg_pool1d", "avg_pool2d", "avg_pool3d"}:
        return np.squeeze(array_value(value))
    if canonical in {"eig", "eigh"}:
        if isinstance(value, (list, tuple)):
            value = value[0]
        return sort_1d_values(value)
    if canonical in {"sqrtm", "logm"}:
        return real_if_close(value)
    if canonical == "roots":
        return sort_1d_values(value)
    if canonical == "svd" and isinstance(value, (list, tuple)):
        if api.library in {"tensorflow", "mindspore"}:
            return value[0]
        return value[1]
    if canonical == "svd_lowrank" and isinstance(value, (list, tuple)):
        return value[1]
    if canonical == "pca_lowrank" and isinstance(value, (list, tuple)):
        return value[1]
    if canonical in {"sort", "mode", "kthvalue", "median", "nanmedian", "cummax", "cummin", "topk", "adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d"} and isinstance(value, (list, tuple)):
        return value[0]
    if canonical in {"histogram", "histogramdd"} and isinstance(value, (list, tuple)):
        return value[0]
    if canonical == "scan" and isinstance(value, (list, tuple)):
        return value[1]
    if canonical in {"fft", "ifft", "fft2", "ifft2", "fftn", "ifftn", "rfft"} and isinstance(value, (list, tuple)):
        return array_value(value[0]) + 1j * array_value(value[1])
    if canonical == "stft":
        if isinstance(value, (list, tuple)):
            return value[2]
        result = array_value(value)
        if api.library == "torch":
            result = result / 2.0
        return result
    if canonical == "istft":
        if isinstance(value, (list, tuple)):
            return value[1]
        return value
    if canonical == "csd" and isinstance(value, (list, tuple)):
        return [value[0], real_if_close(value[1])]
    if canonical in {"indices", "diag_indices", "tril_indices", "triu_indices"} and isinstance(value, (list, tuple)):
        return np.stack([array_value(item) for item in value], axis=0)
    if canonical == "nonzero" and isinstance(value, (list, tuple)):
        return np.stack([array_value(item) for item in value], axis=1)
    if canonical == "clip_by_global_norm" and isinstance(value, (list, tuple)):
        if len(value) == 2 and isinstance(value[0], (list, tuple)):
            return value[0]
        if len(value) == 2:
            try:
                if np.asarray(array_value(value[1])).shape == ():
                    return value[0]
            except Exception:
                pass
        return list(value)
    if canonical in {"l1_loss", "mse_loss", "binary_cross_entropy", "kl_divergence"}:
        return global_mean(value)
    if canonical == "lu_factor" and isinstance(value, (list, tuple)):
        lu, pivots = array_value(value[0]), array_value(value[1]).astype(np.int64)
        if api.library == "torch":
            pivots = pivots - 1
        return [lu, pivots]
    if canonical == "lu" and isinstance(value, (list, tuple)):
        if len(value) == 3:
            p, l, u = (array_value(item) for item in value)
            return p @ l @ u
        if len(value) == 2:
            packed, pivots = array_value(value[0]), array_value(value[1]).astype(np.int64)
            n = packed.shape[-1]
            l = np.tril(packed, -1) + np.eye(n, dtype=packed.dtype)
            u = np.triu(packed)
            if pivots.size and pivots.max() >= n:
                pivots = pivots - 1
            if sorted(pivots.tolist()) == list(range(n)) and len(set(pivots.tolist())) == n:
                perm = pivots.tolist()
            else:
                perm = list(range(n))
                for idx, pivot in enumerate(pivots.tolist()):
                    perm[idx], perm[pivot] = perm[pivot], perm[idx]
            p = np.eye(n, dtype=packed.dtype)[perm]
            return p @ l @ u
    if canonical == "cho_factor" and isinstance(value, (list, tuple)):
        factor = array_value(value[0])
        lower = bool(np.asarray(array_value(value[1])).item())
        factor = np.tril(factor) if lower else np.triu(factor)
        return factor @ factor.T if lower else factor.T @ factor
    if canonical == "lstsq":
        if hasattr(value, "solution"):
            return value.solution
        if isinstance(value, (list, tuple)):
            return value[0]
    if canonical == "slogdet":
        if qname.rpartition(".")[2] == "logdet":
            raise SkipCall("logdet returns only the log determinant, not slogdet's sign/logabs pair")
        if isinstance(value, (list, tuple)):
            sign, logabs = value[0], value[1]
            return np.stack([array_value(sign), array_value(logabs)])
        return value
    if canonical == "unravel_index" and isinstance(value, (list, tuple)):
        return np.stack([array_value(item) for item in value])
    if canonical == "unique":
        if isinstance(value, (list, tuple)):
            value = value[0]
        return sort_1d_values(value)
    if canonical == "partition":
        return np.sort(np.asarray(real_if_close(value)).reshape(-1))
    if canonical == "polydiv" and isinstance(value, (list, tuple)):
        return [trim_trailing_zeros(value[0]), trim_trailing_zeros(value[1])]
    return value


def equal_encoded(left: Any, right: Any) -> bool:
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(equal_encoded(a, b) for a, b in zip(left, right))
        )
    if not isinstance(left, dict) or not isinstance(right, dict):
        if isinstance(left, str) and isinstance(right, str):
            return normalize_dtype_name(left) == normalize_dtype_name(right)
        return left == right
    a = np.asarray(left["value"])
    b = np.asarray(right["value"])
    if a.shape != b.shape:
        if a.size == b.size == 1:
            a = a.reshape(())
            b = b.reshape(())
        else:
            return False
    if a.dtype.kind in "OUS" or b.dtype.kind in "OUS":
        return np.array_equal(a, b)
    if a.dtype.kind in "bui" and b.dtype.kind in "bui":
        return np.array_equal(a, b)
    return bool(np.allclose(a, b, atol=1e-4, rtol=1e-3, equal_nan=True))


def run_external_api(api: Api, canonical: str, category: str) -> Any:
    python = EXTERNAL_LIBRARY_PYTHONS.get(api.library)
    if not python or not os.path.exists(python):
        raise UnsupportedCall(f"external Python for {api.library} is unavailable")

    payload = {
        "api": {
            "library": api.library,
            "namespace": api.namespace,
            "name": api.name,
            "qualified_name": api.qualified_name,
            "arity": api.arity,
            "parameters": api.parameters,
            "roles": api.roles,
            "doc": api.doc,
        },
        "state": export_runtime_state(),
        "canonical": canonical,
        "category": category,
    }
    proc = external_worker(api.library, python)
    try:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        response = read_external_response(proc)
    except TimeoutError:
        stop_external_worker(api.library)
        raise
    except Exception:
        stop_external_worker(api.library)
        proc = external_worker(api.library, python)
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        response = read_external_response(proc)

    status = response.get("status")
    if status == "ok":
        return response.get("value")
    if status == "skip":
        raise SkipCall(response.get("message", "external runner skipped"))
    if status == "unsupported":
        raise UnsupportedCall(response.get("message", "external runner unsupported"))
    raise RuntimeError(response.get("message", f"external {api.library} runner failed"))


def external_worker(library: str, python: str) -> subprocess.Popen[str]:
    proc = EXTERNAL_WORKERS.get(library)
    if proc is not None and proc.poll() is None:
        return proc

    code = r'''
from __future__ import annotations
import json, os, sys
tools_dir = os.environ["XAMT_TOOLS_DIR"]
sys.path.insert(0, tools_dir)
os.environ["XAMT_EXTERNAL_RUNNER"] = "1"
from compare_api_matchers import Api
import diff_static_candidate_groups as runner

for line in sys.stdin:
    try:
        payload = json.loads(line)
        runner.apply_runtime_state(payload.get("state"))
        api = Api(**payload["api"])
        value = runner.run_api(api, payload["canonical"], payload["category"])
    except runner.SkipCall as exc:
        response = {"status": "skip", "message": str(exc)}
    except runner.UnsupportedCall as exc:
        response = {"status": "unsupported", "message": str(exc)}
    except Exception as exc:
        response = {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
    else:
        response = {"status": "ok", "value": value}
    print(json.dumps(response), flush=True)
'''
    env = os.environ.copy()
    env[EXTERNAL_RUNNER_FLAG] = "1"
    env["XAMT_TOOLS_DIR"] = os.path.dirname(os.path.abspath(__file__))
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [python, "-B", "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )
    EXTERNAL_WORKERS[library] = proc
    return proc


def read_external_response(proc: subprocess.Popen[str]) -> dict[str, Any]:
    if proc.stdout is None:
        raise RuntimeError("external runner has no stdout")
    deadline = time.monotonic() + EXTERNAL_RESPONSE_TIMEOUT_SECONDS
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"external runner response timed out after {EXTERNAL_RESPONSE_TIMEOUT_SECONDS:g}s")
        ready, _, _ = select.select([proc.stdout], [], [], remaining)
        if not ready:
            raise TimeoutError(f"external runner response timed out after {EXTERNAL_RESPONSE_TIMEOUT_SECONDS:g}s")
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("external runner exited before returning output")
        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(response, dict):
            raise RuntimeError("external runner returned non-object json")
        return response


def stop_external_worker(library: str) -> None:
    proc = EXTERNAL_WORKERS.pop(library, None)
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        proc.kill()


def stop_external_workers() -> None:
    for library in list(EXTERNAL_WORKERS):
        stop_external_worker(library)


atexit.register(stop_external_workers)


def run_api(api: Api, canonical: str, category: str) -> Any:
    if EXTERNAL_LIBRARY_PYTHONS.get(api.library) and not os.environ.get(EXTERNAL_RUNNER_FLAG):
        return run_external_api(api, canonical, category)
    qname = api.qualified_name
    if qname == "torch.binary_cross_entropy_with_logits":
        raise SkipCall("torch.binary_cross_entropy_with_logits is a low-level binding; use torch.nn.functional instead")
    if qname == "torch.matrix_rank":
        import torch

        return encode(torch.linalg.matrix_rank(tensor(SPD, "torch")))
    if qname == "torch.lstsq":
        import torch

        return encode(torch.linalg.lstsq(tensor(MAT, "torch"), tensor(RHS, "torch")).solution)
    if qname == "torch.ctc_loss":
        import torch

        log_probs = tensor(np.log(np.array([[[0.6, 0.3, 0.1]], [[0.6, 0.3, 0.1]]], dtype=np.float32)), "torch")
        labels = torch.tensor([1], dtype=torch.long)
        input_lengths = torch.tensor([2], dtype=torch.long)
        label_lengths = torch.tensor([1], dtype=torch.long)
        return encode(torch.ctc_loss(log_probs, labels, input_lengths, label_lengths, 0, 0, True))
    if canonical == "cond" and category != "linalg":
        if api.library == "torch":
            import torch

            return encode(torch.cond(torch.tensor(True), lambda x: x + 1, lambda x: x - 1, (tensor(VEC_A, "torch"),)))
        if api.library == "tensorflow":
            import tensorflow as tf

            return encode(tf.cond(tf.constant(True), lambda: tensor(VEC_A, "tensorflow") + 1, lambda: tensor(VEC_A, "tensorflow") - 1))
        if api.library == "keras":
            import keras

            return encode(keras.ops.cond(True, lambda: tensor(VEC_A, "keras") + 1, lambda: tensor(VEC_A, "keras") - 1))
        if api.library == "jax":
            import jax

            return encode(jax.lax.cond(True, lambda x: x + 1, lambda x: x - 1, tensor(VEC_A, "jax")))
    if canonical == "while_loop":
        if api.library == "torch":
            import torch

            out = torch.while_loop(lambda i, x: i < 3, lambda i, x: (i + 1, x + 1), (torch.tensor(0), tensor(VEC_A, "torch")))
            return encode(out[1])
        if api.library == "tensorflow":
            import tensorflow as tf

            out = tf.while_loop(lambda i, x: i < 3, lambda i, x: (i + 1, x + 1), (tf.constant(0), tensor(VEC_A, "tensorflow")))
            return encode(out[1])
        if api.library == "keras":
            import keras

            out = keras.ops.while_loop(lambda i, x: i < 3, lambda i, x: (i + 1, x + 1), (0, tensor(VEC_A, "keras")))
            return encode(out[1])
        if api.library == "jax":
            import jax
            import jax.numpy as jnp

            out = jax.lax.while_loop(lambda state: state[0] < 3, lambda state: (state[0] + 1, state[1] + 1), (jnp.array(0), tensor(VEC_A, "jax")))
            return encode(out[1])
    if qname == "tensorflow.linalg.lu_solve":
        import tensorflow as tf

        lu, perm = tf.linalg.lu(tensor(MAT, "tensorflow"))
        return encode(tf.linalg.lu_solve(lu, perm, tensor(RHS, "tensorflow")))
    if qname == "numpy.random.shuffle":
        arr = np.array([1.0], dtype=np.float32)
        resolve(api)(arr)
        return encode(arr)
    if qname == "numpy.fill_diagonal":
        arr = np.zeros((3, 3), dtype=np.float32)
        np.fill_diagonal(arr, 7.0)
        return encode(arr)
    if qname == "numpy.put_along_axis":
        arr = np.zeros((2, 3), dtype=np.float32)
        indices = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int64)
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        np.put_along_axis(arr, indices, values, axis=1)
        return encode(arr)
    if canonical == "custom_gradient":
        if api.library == "tensorflow":
            import tensorflow as tf

            @tf.custom_gradient
            def square(x):
                def grad(upstream):
                    return upstream * 2.0 * x
                return x * x, grad

            return encode(square(tensor(VEC_A, "tensorflow")))
        if api.library == "keras":
            import keras

            @keras.ops.custom_gradient
            def square(x):
                def grad(upstream):
                    return upstream * 2.0 * x
                return x * x, grad

            return encode(square(tensor(VEC_A, "keras")))
    if canonical == "vectorize":
        if api.library == "keras":
            raise SkipCall("keras.ops.vectorize is backend-dependent and unsupported on the current TensorFlow backend")
        fn = resolve(api)(lambda z: z + 1)
        return encode(fn(tensor(VEC_A, api.library)))
    if canonical == "input":
        obj = resolve(api)(shape=(3,), dtype="float32")
        dims = [-1 if dim is None else int(dim) for dim in obj.shape]
        return encode(np.asarray(dims, dtype=np.int64))
    fn = resolve(api)
    args, kwargs = args_for(api, canonical, category)
    result = fn(*args, **kwargs)
    result = normalize_result(api, canonical, result)
    if canonical == "broadcast_shapes":
        result = np.asarray(result, dtype=np.int64)
    return encode(result)


def run_group(key: tuple[Any, ...], apis: list[Api]) -> tuple[str, dict[str, Any], dict[str, str]]:
    canonical = str(key[0])
    category = str(key[1]) if len(key) > 1 else "generic"
    by_lib: dict[str, list[Api]] = {}
    for api in sorted(apis, key=candidate_priority):
        by_lib.setdefault(api.library, []).append(api)

    outputs: dict[str, Any] = {}
    errors: dict[str, str] = {}
    skipped: dict[str, str] = {}
    chosen: dict[str, str] = {}
    for lib, candidates in sorted(by_lib.items()):
        last_error = ""
        for api in candidates:
            try:
                outputs[lib] = run_api(api, canonical, category)
                chosen[lib] = api.qualified_name
                break
            except SkipCall as exc:
                skipped[lib] = f"SKIP: {exc}"
                last_error = ""
                break
            except UnsupportedCall as exc:
                last_error = f"UNSUPPORTED: {exc}"
                break
            except Exception as exc:
                last_error = f"{api.qualified_name}: {type(exc).__name__}: {exc}"
        if lib not in outputs and lib not in skipped:
            errors[lib] = last_error or "no callable candidate"

    data = {"chosen": chosen, "outputs": outputs, "skipped": skipped}
    if errors:
        return "ERROR", data, errors
    if len(outputs) < 2:
        return "SKIP", data, skipped or {"group": "fewer than two executable libraries"}
    values = list(outputs.values())
    if all(equal_encoded(values[0], value) for value in values[1:]):
        return "PASS", data, {}
    return "DIFF", data, {}


def pairwise_adapter_groups(apis: list[Api]) -> list[tuple[tuple[Any, ...], list[Api]]]:
    selected: list[tuple[tuple[Any, ...], list[Api]]] = []
    loose_groups = cross_library_groups(
        group_apis(apis, use_aliases=True, use_category=True, use_arity=False)
    )
    for key, group in loose_groups:
        candidates = sorted(group, key=lambda api: (api.library, candidate_priority(api)))
        parent = list(range(len(candidates)))
        active: set[int] = set()

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for left_index, left in enumerate(candidates):
            for right_index in range(left_index + 1, len(candidates)):
                right = candidates[right_index]
                if left.library == right.library:
                    continue
                state, data, _ = run_group(key, [left, right])
                if state != "PASS":
                    continue
                chosen = data.get("chosen", {})
                if len(chosen) != 2:
                    continue
                active.update({left_index, right_index})
                union(left_index, right_index)

        components: dict[int, list[Api]] = {}
        for index in sorted(active):
            components.setdefault(find(index), []).append(candidates[index])
        component_index = 0
        for component in components.values():
            if len({api.library for api in component}) < 2:
                continue
            component_index += 1
            selected.append((tuple(key) + ("component", component_index), component))
    stop_external_workers()
    return selected


def build_candidate_groups(apis: list[Api], strategy: str) -> list[tuple[tuple[Any, ...], list[Api]]]:
    if strategy == "alias-category-arity":
        return cross_library_groups(
            group_apis(apis, use_aliases=True, use_category=True, use_arity=True)
        )
    if strategy == "alias-category":
        return cross_library_groups(
            group_apis(apis, use_aliases=True, use_category=True, use_arity=False)
        )
    if strategy == "role-aware":
        return role_aware_groups(apis)
    if strategy == "pairwise-adapter-aware":
        return pairwise_adapter_groups(apis)
    if strategy == "role-adapter-aware":
        selected: list[tuple[tuple[Any, ...], list[Api]]] = []
        for key, group in role_aware_groups(apis):
            state, _, _ = run_group(key, group)
            if state != "ERROR":
                selected.append((key, group))
                continue
            selected.extend(
                cross_library_groups(
                    group_apis(group, use_aliases=True, use_category=True, use_arity=True)
                )
            )
        return selected
    if strategy != "adapter-aware":
        raise ValueError(f"unknown strategy: {strategy}")

    selected: list[tuple[tuple[Any, ...], list[Api]]] = []
    loose_groups = cross_library_groups(
        group_apis(apis, use_aliases=True, use_category=True, use_arity=False)
    )
    for key, group in loose_groups:
        state, _, _ = run_group(key, group)
        if state != "ERROR":
            selected.append((key, group))
            continue
        selected.extend(
            cross_library_groups(
                group_apis(group, use_aliases=True, use_category=True, use_arity=True)
            )
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", type=int, default=20)
    parser.add_argument(
        "--strategy",
        choices=("alias-category-arity", "alias-category", "role-aware", "role-adapter-aware", "adapter-aware", "pairwise-adapter-aware"),
        default="alias-category-arity",
    )
    args = parser.parse_args()

    apis = collect_apis()
    groups = build_candidate_groups(apis, args.strategy)
    confidence_counts = Counter(confidence_band(group_confidence(group)) for _, group in groups)
    status: dict[str, str] = {}
    apis_by_status: Counter[str] = Counter()
    details: list[tuple[str, tuple[Any, ...], dict[str, Any], dict[str, str]]] = []
    confidence_by_status: dict[str, Counter[str]] = {}
    for idx, (key, group) in enumerate(groups, start=1):
        name = f"{idx:04d}:{key}"
        state, data, errors = run_group(key, group)
        status[name] = state
        apis_by_status[state] += len(group)
        confidence_by_status.setdefault(state, Counter())[confidence_band(group_confidence(group))] += 1
        if state != "PASS":
            details.append((name, key, data, errors))

    counts = Counter(status.values())
    ordered_statuses = ["PASS", "DIFF", "ERROR", "SKIP"]
    print("strategy:", args.strategy)
    print("groups:", len(groups))
    print("confidence:", json.dumps(dict(sorted(confidence_counts.items())), sort_keys=True))
    print("summary:", json.dumps({k: counts.get(k, 0) for k in ordered_statuses}, sort_keys=True))
    print(
        "confidence_by_status:",
        json.dumps(
            {state: dict(sorted(confidence_by_status.get(state, Counter()).items())) for state in ordered_statuses},
            sort_keys=True,
        ),
    )
    print("apis_by_status:", json.dumps({k: apis_by_status.get(k, 0) for k in ordered_statuses}, sort_keys=True))
    print("executed_without_error:", counts.get("PASS", 0) + counts.get("DIFF", 0))
    print("executed_api_count:", apis_by_status.get("PASS", 0) + apis_by_status.get("DIFF", 0))
    print()

    print("DIFF_DETAILS")
    shown = 0
    for name, key, data, errors in details:
        if status[name] != "DIFF":
            continue
        print(f"{name}: {key}")
        for lib, api_name in data["chosen"].items():
            print(f"  {lib:10s} {api_name} -> {data['outputs'][lib]}")
        shown += 1
        if shown >= args.details:
            break
    if sum(1 for name, *_ in details if status[name] == "DIFF") > shown:
        print(f"  ... {sum(1 for name, *_ in details if status[name] == 'DIFF') - shown} more")
    print()

    print("SKIP_DETAILS")
    shown = 0
    for name, key, data, errors in details:
        if status[name] != "SKIP":
            continue
        print(f"{name}: {key}")
        for lib, reason in (data.get("skipped") or errors).items():
            print(f"  {lib:10s} {reason}")
        shown += 1
        if shown >= args.details:
            break
    if sum(1 for name, *_ in details if status[name] == "SKIP") > shown:
        print(f"  ... {sum(1 for name, *_ in details if status[name] == 'SKIP') - shown} more")
    print()

    print("ERROR_DETAILS")
    shown = 0
    for name, key, data, errors in details:
        if status[name] != "ERROR":
            continue
        print(f"{name}: {key}")
        for lib, error in errors.items():
            print(f"  {lib:10s} {error}")
        shown += 1
        if shown >= args.details:
            break
    if sum(1 for name, *_ in details if status[name] == "ERROR") > shown:
        print(f"  ... {sum(1 for name, *_ in details if status[name] == 'ERROR') - shown} more")


if __name__ == "__main__":
    main()
