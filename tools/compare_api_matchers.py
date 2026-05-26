"""Compare API matching strategies on installed libraries.

The experiment is deliberately small and reproducible:

* mine callable APIs from installed DL libraries;
* compare raw-name, alias-name, category-aware, and role/doc-aware candidate grouping;
* dynamically validate a set of high-value equivalence classes.

The dynamic validation stage is the important part: it turns a high-recall
candidate group into an executable equivalence claim.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Optional

try:
    from .api_match_common import CATEGORY_TERMS, NAMESPACES, normalize_name, parameter_roles, role_jaccard
except ImportError:  # pragma: no cover - allows running as a plain script.
    from api_match_common import CATEGORY_TERMS, NAMESPACES, normalize_name, parameter_roles, role_jaccard

import numpy as np


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


@dataclass(frozen=True)
class Api:
    library: str
    namespace: str
    name: str
    qualified_name: str
    arity: Optional[int]
    parameters: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    doc: str = ""



CATEGORY_OVERRIDES = {
    "batch_norm": "nn",
    "group_norm": "nn",
    "instance_norm": "nn",
    "layer_norm": "nn",
    "conv": "nn",
    "conv1d": "nn",
    "conv2d": "nn",
    "conv3d": "nn",
    "conv_transpose": "nn",
    "conv_transpose1d": "nn",
    "conv_transpose2d": "nn",
    "conv_transpose3d": "nn",
    "avg_pool": "nn",
    "avg_pool1d": "nn",
    "avg_pool2d": "nn",
    "avg_pool3d": "nn",
    "max_pool1d": "nn",
    "max_pool2d": "nn",
    "max_pool3d": "nn",
    "ctc": "loss",
}


def category_term_matches(text: str, term: str) -> bool:
    return re.search(rf"(^|[._]){re.escape(term)}($|[._])", text) is not None


def infer_category(api: Api, canonical_name: str) -> str:
    if canonical_name == "cond" and api.qualified_name == "paddle.tensor.cond":
        return "linalg"
    if canonical_name in CATEGORY_OVERRIDES:
        return CATEGORY_OVERRIDES[canonical_name]
    namespace = api.namespace.lower()
    if canonical_name in {"get", "serialize", "deserialize"}:
        if ".activations" in namespace:
            return "activation_config"
        if ".layers" in namespace:
            return "layer_config"
        if ".losses" in namespace:
            return "loss_config"
        if ".metrics" in namespace:
            return "metric_config"
        if ".initializers" in namespace:
            return "initializer_config"
        if ".regularizers" in namespace:
            return "regularizer_config"
        if ".optimizers" in namespace:
            return "optimizer_config"
    if ".losses" in namespace or ".metrics" in namespace:
        return "loss"
    text = f"{api.namespace}.{canonical_name}".lower()
    for category, terms in CATEGORY_TERMS.items():
        if any(category_term_matches(text, term) for term in terms):
            return category
    return "generic"


def signature_parameters(obj: object) -> tuple[str, ...]:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return ()
    return tuple(
        name
        for name, param in sig.parameters.items()
        if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    )


def arity(obj: object) -> Optional[int]:
    params = signature_parameters(obj)
    if params:
        return len(params)
    try:
        inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    return 0


DOC_STOPWORDS = {
    "about",
    "above",
    "according",
    "across",
    "after",
    "again",
    "against",
    "also",
    "always",
    "and",
    "another",
    "any",
    "are",
    "arg",
    "args",
    "argument",
    "arguments",
    "array_like",
    "available",
    "based",
    "been",
    "before",
    "being",
    "below",
    "between",
    "bool",
    "boolean",
    "both",
    "can",
    "cannot",
    "case",
    "class",
    "containing",
    "default",
    "defined",
    "description",
    "does",
    "dtype",
    "each",
    "either",
    "element",
    "elements",
    "example",
    "examples",
    "false",
    "first",
    "float",
    "following",
    "for",
    "from",
    "function",
    "given",
    "has",
    "have",
    "input",
    "inputs",
    "int",
    "into",
    "its",
    "kwargs",
    "like",
    "list",
    "may",
    "method",
    "must",
    "none",
    "not",
    "numpy",
    "object",
    "optional",
    "other",
    "output",
    "parameters",
    "param",
    "please",
    "return",
    "returned",
    "returns",
    "same",
    "see",
    "self",
    "shape",
    "should",
    "specified",
    "tensor",
    "the",
    "their",
    "this",
    "torch",
    "true",
    "type",
    "used",
    "using",
    "value",
    "values",
    "when",
    "where",
    "which",
    "with",
}


def safe_doc(obj: object, *, limit: int = 1200) -> str:
    try:
        doc = inspect.getdoc(obj) or ""
    except Exception:
        return ""
    doc = re.sub(r"\s+", " ", doc).strip()
    return doc[:limit]


def doc_terms(api: Api) -> tuple[str, ...]:
    text = api.doc.lower()
    tokens = re.findall(r"[a-z][a-z0-9_]{2,}", text)
    terms = []
    for token in tokens:
        term = normalize_name(token, use_aliases=True)
        if len(term) < 3 or term in DOC_STOPWORDS:
            continue
        terms.append(term)
    return tuple(sorted(set(terms)))


def doc_similarity(left: Api, right: Api) -> float:
    if not left.doc or not right.doc:
        return 0.0
    return role_jaccard(doc_terms(left), doc_terms(right))


def collect_namespace_apis(library: str, namespace: str) -> list[Api]:
    try:
        module = importlib.import_module(namespace)
    except Exception:
        return []

    apis: list[Api] = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(module, name)
        except Exception:
            continue
        if not (
            inspect.isfunction(obj)
            or inspect.isbuiltin(obj)
            or inspect.isroutine(obj)
        ):
            continue
        params = signature_parameters(obj)
        apis.append(
            Api(
                library=library,
                namespace=namespace,
                name=name,
                qualified_name=f"{namespace}.{name}",
                arity=arity(obj),
                parameters=params,
                roles=parameter_roles(params),
                doc=safe_doc(obj),
            )
        )
    return apis


def collect_external_apis(library: str, namespaces: list[str]) -> list[Api]:
    python = EXTERNAL_LIBRARY_PYTHONS.get(library)
    if not python or not os.path.exists(python):
        return []

    code = r'''
from __future__ import annotations
import importlib, inspect, json, os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
library = sys.argv[1]
namespaces = json.loads(sys.argv[2])

def signature_parameters(obj):
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return []
    return [
        name
        for name, param in sig.parameters.items()
        if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    ]

def arity(obj):
    params = signature_parameters(obj)
    if params:
        return len(params)
    try:
        inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    return 0

def safe_doc(obj, limit=1200):
    try:
        doc = inspect.getdoc(obj) or ""
    except Exception:
        return ""
    return " ".join(doc.split())[:limit]

result = []
for namespace in namespaces:
    try:
        module = importlib.import_module(namespace)
    except Exception:
        continue
    for name in dir(module):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(module, name)
        except Exception:
            continue
        if not (inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.isroutine(obj)):
            continue
        result.append({
            "library": library,
            "namespace": namespace,
            "name": name,
            "qualified_name": f"{namespace}.{name}",
            "arity": arity(obj),
            "parameters": signature_parameters(obj),
            "doc": safe_doc(obj),
        })
print(json.dumps(result))
'''
    try:
        proc = subprocess.run(
            [python, "-c", code, library, json.dumps(namespaces)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    try:
        rows = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    apis = []
    for row in rows:
        params = tuple(row.get("parameters", ()))
        apis.append(Api(**{**row, "parameters": params, "roles": parameter_roles(params)}))
    return apis


def collect_apis() -> list[Api]:
    apis: list[Api] = []
    for library, namespaces in NAMESPACES.items():
        before = len(apis)
        for namespace in namespaces:
            apis.extend(collect_namespace_apis(library, namespace))
        if len(apis) == before:
            apis.extend(collect_external_apis(library, namespaces))
    return apis


def group_apis(
    apis: list[Api],
    *,
    use_aliases: bool,
    use_category: bool,
    use_arity: bool = False,
    use_roles: bool = False,
) -> dict[tuple[Any, ...], list[Api]]:
    groups: dict[tuple[Any, ...], list[Api]] = defaultdict(list)
    for api in apis:
        canonical = normalize_name(api.name, use_aliases=use_aliases)
        key: list[Any] = [canonical]
        if use_category:
            key.append(infer_category(api, canonical))
        if use_arity:
            key.append(api.arity)
        if use_roles:
            key.append(api.roles or ("unknown",))
        groups[tuple(key)].append(api)
    return groups


def arity_similarity(left: Optional[int], right: Optional[int]) -> float:
    if left is None or right is None:
        return 0.5
    if left == 0 and right == 0:
        return 1.0
    return 1.0 - abs(left - right) / max(left, right, 1)


def pair_confidence(left: Api, right: Api) -> float:
    left_name = normalize_name(left.name, use_aliases=True)
    right_name = normalize_name(right.name, use_aliases=True)
    name_score = 1.0 if left_name == right_name else 0.0
    category_score = 1.0 if infer_category(left, left_name) == infer_category(right, right_name) else 0.0
    role_score = role_jaccard(left.roles, right.roles)
    if not left.roles or not right.roles:
        role_score = max(role_score, 0.45)
    arity_score = arity_similarity(left.arity, right.arity)
    base_score = 0.35 * name_score + 0.20 * category_score + 0.30 * role_score + 0.15 * arity_score
    # Documentation text is noisy across frameworks, so it is treated as positive
    # evidence only. It can raise confidence but should not reject a match.
    doc_bonus = min(0.08, max(0.0, doc_similarity(left, right) - 0.10) * 0.16)
    return min(1.0, base_score + doc_bonus)


def group_confidence(group: list[Api]) -> float:
    pairs = []
    for idx, left in enumerate(group):
        for right in group[idx + 1 :]:
            if left.library == right.library:
                continue
            pairs.append(pair_confidence(left, right))
    if not pairs:
        return 0.0
    return sum(pairs) / len(pairs)


def role_compatible(left: Api, right: Api) -> bool:
    score = pair_confidence(left, right)
    if score >= 0.78:
        return True
    if left.roles and right.roles and role_jaccard(left.roles, right.roles) >= 0.50:
        return True
    if not left.roles or not right.roles:
        return arity_similarity(left.arity, right.arity) >= 0.50
    return False


def role_aware_groups(apis: list[Api]) -> list[tuple[tuple[Any, ...], list[Api]]]:
    selected: list[tuple[tuple[Any, ...], list[Api]]] = []
    loose_groups = cross_library_groups(
        group_apis(apis, use_aliases=True, use_category=True, use_arity=False)
    )
    for key, group in loose_groups:
        remaining = set(range(len(group)))
        while remaining:
            stack = [remaining.pop()]
            component_indices = []
            while stack:
                idx = stack.pop()
                component_indices.append(idx)
                current = group[idx]
                linked = {other for other in remaining if role_compatible(current, group[other])}
                remaining -= linked
                stack.extend(linked)
            component = [group[idx] for idx in component_indices]
            if len({api.library for api in component}) >= 2:
                confidence = round(group_confidence(component), 3)
                selected.append((tuple(key) + ("conf", confidence), component))
    return selected


def confidence_band(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.70:
        return "medium"
    return "low"


def summarize_confidence(label: str, groups: list[tuple[tuple[Any, ...], list[Api]]]) -> None:
    bands: dict[str, int] = defaultdict(int)
    for _, group in groups:
        bands[confidence_band(group_confidence(group))] += 1
    print(f"{label:24s} confidence={dict(sorted(bands.items()))}")


def cross_library_groups(groups: dict[tuple[Any, ...], list[Api]]) -> list[tuple[tuple[Any, ...], list[Api]]]:
    return [
        (key, group)
        for key, group in groups.items()
        if len({api.library for api in group}) >= 2
    ]


def summarize(label: str, groups: list[tuple[tuple[Any, ...], list[Api]]]) -> None:
    width: dict[int, int] = defaultdict(int)
    for _, group in groups:
        width[len({api.library for api in group})] += 1
    print(
        f"{label:24s} groups={len(groups):4d} "
        f"apis={sum(len(group) for _, group in groups):4d} "
        f"width={dict(sorted(width.items()))}"
    )


def np_array(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def compare_outputs(outputs: dict[str, Any], *, atol: float = 1e-5, rtol: float = 1e-5) -> bool:
    arrays = {name: np_array(value) for name, value in outputs.items()}
    first = next(iter(arrays.values()))
    return all(
        array.shape == first.shape
        and np.allclose(first, array, atol=atol, rtol=rtol, equal_nan=True)
        for array in arrays.values()
    )


def dynamic_specs() -> dict[str, Callable[[], dict[str, Any]]]:
    import jax
    import jax.numpy as jnp
    import keras
    import tensorflow as tf
    import torch

    x = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]], dtype=np.float32)
    y = np.array([[2.0, 3.0, -1.0], [0.5, -4.0, 2.0]], dtype=np.float32)
    pos = np.array([[0.25, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=np.float32)
    mat_left = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mat_right = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    spd = np.array([[2.0, 0.3], [0.3, 1.0]], dtype=np.float32)
    solve_rhs = np.array([[1.0], [2.0]], dtype=np.float32)
    bool_mask = np.array([[True, False, True], [False, True, False]])

    return {
        "abs": lambda: {
            "torch": torch.abs(torch.tensor(x)),
            "tensorflow": tf.abs(tf.constant(x)),
            "keras": keras.ops.abs(x),
            "jax": jnp.abs(jnp.array(x)),
        },
        "negative": lambda: {
            "torch": torch.negative(torch.tensor(x)),
            "tensorflow": tf.negative(tf.constant(x)),
            "keras": keras.ops.negative(x),
            "jax": jnp.negative(jnp.array(x)),
        },
        "exp": lambda: {
            "torch": torch.exp(torch.tensor(x)),
            "tensorflow": tf.exp(tf.constant(x)),
            "keras": keras.ops.exp(x),
            "jax": jnp.exp(jnp.array(x)),
        },
        "log": lambda: {
            "torch": torch.log(torch.tensor(pos)),
            "tensorflow": tf.math.log(tf.constant(pos)),
            "keras": keras.ops.log(pos),
            "jax": jnp.log(jnp.array(pos)),
        },
        "sqrt": lambda: {
            "torch": torch.sqrt(torch.tensor(pos)),
            "tensorflow": tf.sqrt(tf.constant(pos)),
            "keras": keras.ops.sqrt(pos),
            "jax": jnp.sqrt(jnp.array(pos)),
        },
        "square": lambda: {
            "torch": torch.square(torch.tensor(x)),
            "tensorflow": tf.square(tf.constant(x)),
            "keras": keras.ops.square(x),
            "jax": jnp.square(jnp.array(x)),
        },
        "sin": lambda: {
            "torch": torch.sin(torch.tensor(x)),
            "tensorflow": tf.sin(tf.constant(x)),
            "keras": keras.ops.sin(x),
            "jax": jnp.sin(jnp.array(x)),
        },
        "cos": lambda: {
            "torch": torch.cos(torch.tensor(x)),
            "tensorflow": tf.cos(tf.constant(x)),
            "keras": keras.ops.cos(x),
            "jax": jnp.cos(jnp.array(x)),
        },
        "tanh": lambda: {
            "torch": torch.tanh(torch.tensor(x)),
            "tensorflow": tf.tanh(tf.constant(x)),
            "keras": keras.ops.tanh(x),
            "jax": jnp.tanh(jnp.array(x)),
        },
        "sigmoid": lambda: {
            "torch": torch.sigmoid(torch.tensor(x)),
            "tensorflow": tf.math.sigmoid(tf.constant(x)),
            "keras": keras.ops.sigmoid(x),
            "jax": jax.nn.sigmoid(jnp.array(x)),
        },
        "relu": lambda: {
            "torch": torch.nn.functional.relu(torch.tensor(x)),
            "tensorflow": tf.nn.relu(tf.constant(x)),
            "keras": keras.ops.relu(x),
            "jax": jax.nn.relu(jnp.array(x)),
        },
        "softplus": lambda: {
            "torch": torch.nn.functional.softplus(torch.tensor(x)),
            "tensorflow": tf.nn.softplus(tf.constant(x)),
            "keras": keras.ops.softplus(x),
            "jax": jax.nn.softplus(jnp.array(x)),
        },
        "gelu": lambda: {
            "torch": torch.nn.functional.gelu(torch.tensor(x), approximate="none"),
            "tensorflow": tf.nn.gelu(tf.constant(x), approximate=False),
            "keras": keras.ops.gelu(x, approximate=False),
            "jax": jax.nn.gelu(jnp.array(x), approximate=False),
        },
        "sum": lambda: {
            "torch": torch.sum(torch.tensor(x), dim=1, keepdim=True),
            "tensorflow": tf.reduce_sum(tf.constant(x), axis=1, keepdims=True),
            "keras": keras.ops.sum(x, axis=1, keepdims=True),
            "jax": jnp.sum(jnp.array(x), axis=1, keepdims=True),
        },
        "mean": lambda: {
            "torch": torch.mean(torch.tensor(x), dim=0),
            "tensorflow": tf.reduce_mean(tf.constant(x), axis=0),
            "keras": keras.ops.mean(x, axis=0),
            "jax": jnp.mean(jnp.array(x), axis=0),
        },
        "max": lambda: {
            "torch": torch.max(torch.tensor(x), dim=1).values,
            "tensorflow": tf.reduce_max(tf.constant(x), axis=1),
            "keras": keras.ops.max(x, axis=1),
            "jax": jnp.max(jnp.array(x), axis=1),
        },
        "min": lambda: {
            "torch": torch.min(torch.tensor(x), dim=1).values,
            "tensorflow": tf.reduce_min(tf.constant(x), axis=1),
            "keras": keras.ops.min(x, axis=1),
            "jax": jnp.min(jnp.array(x), axis=1),
        },
        "prod": lambda: {
            "torch": torch.prod(torch.tensor(x), dim=1),
            "tensorflow": tf.reduce_prod(tf.constant(x), axis=1),
            "keras": keras.ops.prod(x, axis=1),
            "jax": jnp.prod(jnp.array(x), axis=1),
        },
        "std_population": lambda: {
            "torch": torch.std(torch.tensor(x), dim=1, correction=0),
            "tensorflow": tf.math.reduce_std(tf.constant(x), axis=1),
            "keras": keras.ops.std(x, axis=1),
            "jax": jnp.std(jnp.array(x), axis=1),
        },
        "var_population": lambda: {
            "torch": torch.var(torch.tensor(x), dim=1, correction=0),
            "tensorflow": tf.math.reduce_variance(tf.constant(x), axis=1),
            "keras": keras.ops.var(x, axis=1),
            "jax": jnp.var(jnp.array(x), axis=1),
        },
        "argmax": lambda: {
            "torch": torch.argmax(torch.tensor(x), dim=1),
            "tensorflow": tf.argmax(tf.constant(x), axis=1),
            "keras": keras.ops.argmax(x, axis=1),
            "jax": jnp.argmax(jnp.array(x), axis=1),
        },
        "argmin": lambda: {
            "torch": torch.argmin(torch.tensor(x), dim=1),
            "tensorflow": tf.argmin(tf.constant(x), axis=1),
            "keras": keras.ops.argmin(x, axis=1),
            "jax": jnp.argmin(jnp.array(x), axis=1),
        },
        "cumsum": lambda: {
            "torch": torch.cumsum(torch.tensor(x), dim=1),
            "tensorflow": tf.cumsum(tf.constant(x), axis=1),
            "keras": keras.ops.cumsum(x, axis=1),
            "jax": jnp.cumsum(jnp.array(x), axis=1),
        },
        "cumprod": lambda: {
            "torch": torch.cumprod(torch.tensor(x), dim=1),
            "tensorflow": tf.math.cumprod(tf.constant(x), axis=1),
            "keras": keras.ops.cumprod(x, axis=1),
            "jax": jnp.cumprod(jnp.array(x), axis=1),
        },
        "add": lambda: {
            "torch": torch.add(torch.tensor(x), torch.tensor(y)),
            "tensorflow": tf.add(tf.constant(x), tf.constant(y)),
            "keras": keras.ops.add(x, y),
            "jax": jnp.add(jnp.array(x), jnp.array(y)),
        },
        "subtract": lambda: {
            "torch": torch.subtract(torch.tensor(x), torch.tensor(y)),
            "tensorflow": tf.subtract(tf.constant(x), tf.constant(y)),
            "keras": keras.ops.subtract(x, y),
            "jax": jnp.subtract(jnp.array(x), jnp.array(y)),
        },
        "multiply": lambda: {
            "torch": torch.multiply(torch.tensor(x), torch.tensor(y)),
            "tensorflow": tf.multiply(tf.constant(x), tf.constant(y)),
            "keras": keras.ops.multiply(x, y),
            "jax": jnp.multiply(jnp.array(x), jnp.array(y)),
        },
        "divide": lambda: {
            "torch": torch.divide(torch.tensor(x), torch.tensor(pos)),
            "tensorflow": tf.divide(tf.constant(x), tf.constant(pos)),
            "keras": keras.ops.divide(x, pos),
            "jax": jnp.divide(jnp.array(x), jnp.array(pos)),
        },
        "maximum": lambda: {
            "torch": torch.maximum(torch.tensor(x), torch.tensor(y)),
            "tensorflow": tf.maximum(tf.constant(x), tf.constant(y)),
            "keras": keras.ops.maximum(x, y),
            "jax": jnp.maximum(jnp.array(x), jnp.array(y)),
        },
        "minimum": lambda: {
            "torch": torch.minimum(torch.tensor(x), torch.tensor(y)),
            "tensorflow": tf.minimum(tf.constant(x), tf.constant(y)),
            "keras": keras.ops.minimum(x, y),
            "jax": jnp.minimum(jnp.array(x), jnp.array(y)),
        },
        "power": lambda: {
            "torch": torch.pow(torch.tensor(pos), 1.5),
            "tensorflow": tf.pow(tf.constant(pos), 1.5),
            "keras": keras.ops.power(pos, 1.5),
            "jax": jnp.power(jnp.array(pos), 1.5),
        },
        "equal": lambda: {
            "torch": torch.eq(torch.tensor(x), torch.tensor(y)),
            "tensorflow": tf.equal(tf.constant(x), tf.constant(y)),
            "keras": keras.ops.equal(x, y),
            "jax": jnp.equal(jnp.array(x), jnp.array(y)),
        },
        "greater_equal": lambda: {
            "torch": torch.ge(torch.tensor(x), torch.tensor(y)),
            "tensorflow": tf.greater_equal(tf.constant(x), tf.constant(y)),
            "keras": keras.ops.greater_equal(x, y),
            "jax": jnp.greater_equal(jnp.array(x), jnp.array(y)),
        },
        "matmul": lambda: {
            "torch": torch.matmul(torch.tensor(mat_left), torch.tensor(mat_right)),
            "tensorflow": tf.matmul(tf.constant(mat_left), tf.constant(mat_right)),
            "keras": keras.ops.matmul(mat_left, mat_right),
            "jax": jnp.matmul(jnp.array(mat_left), jnp.array(mat_right)),
        },
        "einsum": lambda: {
            "torch": torch.einsum("ij,jk->ik", torch.tensor(mat_left), torch.tensor(mat_right)),
            "tensorflow": tf.einsum("ij,jk->ik", tf.constant(mat_left), tf.constant(mat_right)),
            "keras": keras.ops.einsum("ij,jk->ik", mat_left, mat_right),
            "jax": jnp.einsum("ij,jk->ik", jnp.array(mat_left), jnp.array(mat_right)),
        },
        "reshape": lambda: {
            "torch": torch.reshape(torch.tensor(x), (3, 2)),
            "tensorflow": tf.reshape(tf.constant(x), (3, 2)),
            "keras": keras.ops.reshape(x, (3, 2)),
            "jax": jnp.reshape(jnp.array(x), (3, 2)),
        },
        "transpose": lambda: {
            "torch": torch.transpose(torch.tensor(x), 0, 1),
            "tensorflow": tf.transpose(tf.constant(x), perm=(1, 0)),
            "keras": keras.ops.transpose(x, (1, 0)),
            "jax": jnp.transpose(jnp.array(x), (1, 0)),
        },
        "squeeze": lambda: {
            "torch": torch.squeeze(torch.reshape(torch.tensor(x), (1, 2, 3, 1))),
            "tensorflow": tf.squeeze(tf.reshape(tf.constant(x), (1, 2, 3, 1))),
            "keras": keras.ops.squeeze(keras.ops.reshape(x, (1, 2, 3, 1))),
            "jax": jnp.squeeze(jnp.reshape(jnp.array(x), (1, 2, 3, 1))),
        },
        "expand_dims": lambda: {
            "torch": torch.unsqueeze(torch.tensor(x), 0),
            "tensorflow": tf.expand_dims(tf.constant(x), axis=0),
            "keras": keras.ops.expand_dims(x, axis=0),
            "jax": jnp.expand_dims(jnp.array(x), axis=0),
        },
        "concatenate": lambda: {
            "torch": torch.cat([torch.tensor(x), torch.tensor(x)], dim=0),
            "tensorflow": tf.concat([tf.constant(x), tf.constant(x)], axis=0),
            "keras": keras.ops.concatenate([x, x], axis=0),
            "jax": jnp.concatenate([jnp.array(x), jnp.array(x)], axis=0),
        },
        "stack": lambda: {
            "torch": torch.stack([torch.tensor(x), torch.tensor(x)], dim=0),
            "tensorflow": tf.stack([tf.constant(x), tf.constant(x)], axis=0),
            "keras": keras.ops.stack([x, x], axis=0),
            "jax": jnp.stack([jnp.array(x), jnp.array(x)], axis=0),
        },
        "clip": lambda: {
            "torch": torch.clamp(torch.tensor(x), -1.0, 2.0),
            "tensorflow": tf.clip_by_value(tf.constant(x), -1.0, 2.0),
            "keras": keras.ops.clip(x, -1.0, 2.0),
            "jax": jnp.clip(jnp.array(x), -1.0, 2.0),
        },
        "where": lambda: {
            "torch": torch.where(torch.tensor(bool_mask), torch.tensor(x), torch.tensor(y)),
            "tensorflow": tf.where(tf.constant(bool_mask), tf.constant(x), tf.constant(y)),
            "keras": keras.ops.where(bool_mask, x, y),
            "jax": jnp.where(jnp.array(bool_mask), jnp.array(x), jnp.array(y)),
        },
        "sort": lambda: {
            "torch": torch.sort(torch.tensor(x), dim=1).values,
            "tensorflow": tf.sort(tf.constant(x), axis=1),
            "keras": keras.ops.sort(x, axis=1),
            "jax": jnp.sort(jnp.array(x), axis=1),
        },
        "argsort": lambda: {
            "torch": torch.argsort(torch.tensor(x), dim=1),
            "tensorflow": tf.argsort(tf.constant(x), axis=1),
            "keras": keras.ops.argsort(x, axis=1),
            "jax": jnp.argsort(jnp.array(x), axis=1),
        },
        "softmax": lambda: {
            "torch": torch.nn.functional.softmax(torch.tensor(x), dim=1),
            "tensorflow": tf.nn.softmax(tf.constant(x), axis=1),
            "keras": keras.activations.softmax(x, axis=1),
            "jax": jax.nn.softmax(jnp.array(x), axis=1),
        },
        "log_softmax": lambda: {
            "torch": torch.nn.functional.log_softmax(torch.tensor(x), dim=1),
            "tensorflow": tf.nn.log_softmax(tf.constant(x), axis=1),
            "keras": keras.ops.log_softmax(x, axis=1),
            "jax": jax.nn.log_softmax(jnp.array(x), axis=1),
        },
        "logsumexp": lambda: {
            "torch": torch.logsumexp(torch.tensor(x), dim=1),
            "tensorflow": tf.reduce_logsumexp(tf.constant(x), axis=1),
            "keras": keras.ops.logsumexp(x, axis=1),
            "jax": jax.nn.logsumexp(jnp.array(x), axis=1),
        },
        "det": lambda: {
            "torch": torch.linalg.det(torch.tensor(mat_left)),
            "tensorflow": tf.linalg.det(tf.constant(mat_left)),
            "keras": keras.ops.det(mat_left),
            "jax": jnp.linalg.det(jnp.array(mat_left)),
        },
        "inv": lambda: {
            "torch": torch.linalg.inv(torch.tensor(mat_left)),
            "tensorflow": tf.linalg.inv(tf.constant(mat_left)),
            "keras": keras.ops.inv(mat_left),
            "jax": jnp.linalg.inv(jnp.array(mat_left)),
        },
        "solve": lambda: {
            "torch": torch.linalg.solve(torch.tensor(mat_left), torch.tensor(solve_rhs)),
            "tensorflow": tf.linalg.solve(tf.constant(mat_left), tf.constant(solve_rhs)),
            "keras": keras.ops.solve(mat_left, solve_rhs),
            "jax": jnp.linalg.solve(jnp.array(mat_left), jnp.array(solve_rhs)),
        },
        "cholesky": lambda: {
            "torch": torch.linalg.cholesky(torch.tensor(spd)),
            "tensorflow": tf.linalg.cholesky(tf.constant(spd)),
            "keras": keras.ops.cholesky(spd),
            "jax": jnp.linalg.cholesky(jnp.array(spd)),
        },
        "eigh": lambda: {
            "torch": torch.linalg.eigh(torch.tensor(spd)).eigenvalues,
            "tensorflow": tf.linalg.eigh(tf.constant(spd))[0],
            "keras": keras.ops.eigh(spd)[0],
            "jax": jnp.linalg.eigh(jnp.array(spd))[0],
        },
        "svd": lambda: {
            "torch": torch.linalg.svdvals(torch.tensor(mat_left)),
            "tensorflow": tf.linalg.svd(tf.constant(mat_left), compute_uv=False),
            "jax": jnp.linalg.svd(jnp.array(mat_left), compute_uv=False),
        },
        "norm": lambda: {
            "torch": torch.linalg.norm(torch.tensor(mat_left)),
            "tensorflow": tf.linalg.norm(tf.constant(mat_left)),
            "keras": keras.ops.norm(mat_left),
            "jax": jnp.linalg.norm(jnp.array(mat_left)),
        },
    }


def validate_dynamic_specs() -> None:
    print("\ndynamic validation")
    passed = 0
    specs = dynamic_specs()
    for name, run in specs.items():
        try:
            outputs = run()
            ok = compare_outputs(outputs)
        except Exception as exc:
            ok = False
            print(f"  {name:12s} FAIL {type(exc).__name__}: {exc}")
        else:
            print(f"  {name:12s} {'PASS' if ok else 'FAIL'} libs={list(outputs)}")
        passed += int(ok)
    print(f"dynamic pass rate: {passed}/{len(specs)}")


def main() -> None:
    apis = collect_apis()
    print("mined function APIs")
    for library in sorted({api.library for api in apis}):
        print(f"  {library:10s} {sum(api.library == library for api in apis)}")

    strategies = [
        ("raw-name", dict(use_aliases=False, use_category=False)),
        ("alias-name", dict(use_aliases=True, use_category=False)),
        ("alias+category", dict(use_aliases=True, use_category=True)),
        ("alias+category+arity", dict(use_aliases=True, use_category=True, use_arity=True)),
        ("alias+category+roles", dict(use_aliases=True, use_category=True, use_roles=True)),
    ]
    print("\nstatic candidate grouping")
    grouped: dict[str, list[tuple[tuple[Any, ...], list[Api]]]] = {}
    for label, kwargs in strategies:
        groups = cross_library_groups(group_apis(apis, **kwargs))
        grouped[label] = groups
        summarize(label, groups)
        if label.startswith("alias+category"):
            summarize_confidence(label, groups)

    calibrated_groups = role_aware_groups(apis)
    grouped["role-aware"] = calibrated_groups
    summarize("role-aware", calibrated_groups)
    summarize_confidence("role-aware", calibrated_groups)

    print("\nhigh-width alias+category examples")
    examples = sorted(
        grouped["alias+category"],
        key=lambda item: (-len({api.library for api in item[1]}), str(item[0])),
    )[:20]
    for key, group in examples:
        seen = set()
        shown = []
        for api in sorted(group, key=lambda item: (item.library, item.qualified_name)):
            if api.library in seen:
                continue
            seen.add(api.library)
            shown.append(api.qualified_name)
        print(f"  {key!s:32s} {len(seen)} libs -> " + " | ".join(shown))

    validate_dynamic_specs()


if __name__ == "__main__":
    main()
