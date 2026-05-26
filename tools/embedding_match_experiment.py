"""Small semantic-retrieval experiment for API matching.

This uses a locally cached MiniLM model through transformers. It does not call
any external LLM service. The purpose is to compare name/alias retrieval with a
language-model embedding retriever on representative API matching targets.
"""

from __future__ import annotations

import importlib
import inspect
import os
from dataclasses import dataclass

try:
    from .api_match_common import NAMESPACES, normalize_name
except ImportError:  # pragma: no cover - allows running as a plain script.
    from api_match_common import NAMESPACES, normalize_name

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


@dataclass(frozen=True)
class Api:
    library: str
    qualified_name: str
    name: str
    text: str



def safe_signature(obj: object) -> str:
    try:
        return str(inspect.signature(obj))
    except Exception:
        return ""


def collect() -> list[Api]:
    apis: list[Api] = []
    for library, namespaces in NAMESPACES.items():
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
                if not (
                    inspect.isfunction(obj)
                    or inspect.isbuiltin(obj)
                    or inspect.isroutine(obj)
                ):
                    continue
                doc = inspect.getdoc(obj) or ""
                qname = f"{namespace}.{name}"
                text = f"{qname} {safe_signature(obj)} {doc[:700]}"
                apis.append(Api(library=library, qualified_name=qname, name=name, text=text))
    return apis


def embed_texts(texts: list[str]) -> np.ndarray:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    model.eval()
    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), 64):
            batch = texts[start : start + 64]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=192,
                return_tensors="pt",
            )
            output = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy())
    return np.vstack(vectors)


TARGETS = {
    "torch.sum": {"tensorflow.math.reduce_sum", "keras.ops.sum", "jax.numpy.sum"},
    "torch.mean": {"tensorflow.math.reduce_mean", "keras.ops.mean", "jax.numpy.mean"},
    "torch.max": {"tensorflow.math.reduce_max", "keras.ops.max", "jax.numpy.max"},
    "torch.min": {"tensorflow.math.reduce_min", "keras.ops.min", "jax.numpy.min"},
    "torch.prod": {"tensorflow.math.reduce_prod", "keras.ops.prod", "jax.numpy.prod"},
    "torch.cumsum": {"tensorflow.cumsum", "keras.ops.cumsum", "jax.numpy.cumsum"},
    "torch.cumprod": {"tensorflow.math.cumprod", "keras.ops.cumprod", "jax.numpy.cumprod"},
    "torch.cat": {"tensorflow.concat", "keras.ops.concatenate", "jax.numpy.concatenate"},
    "torch.stack": {"tensorflow.stack", "keras.ops.stack", "jax.numpy.stack"},
    "torch.reshape": {"tensorflow.reshape", "keras.ops.reshape", "jax.numpy.reshape"},
    "torch.transpose": {"tensorflow.transpose", "keras.ops.transpose", "jax.numpy.transpose"},
    "torch.squeeze": {"tensorflow.squeeze", "keras.ops.squeeze", "jax.numpy.squeeze"},
    "torch.unsqueeze": {"tensorflow.expand_dims", "keras.ops.expand_dims", "jax.numpy.expand_dims"},
    "torch.add": {"tensorflow.add", "keras.ops.add", "jax.numpy.add"},
    "torch.subtract": {"tensorflow.subtract", "keras.ops.subtract", "jax.numpy.subtract"},
    "torch.multiply": {"tensorflow.multiply", "keras.ops.multiply", "jax.numpy.multiply"},
    "torch.divide": {"tensorflow.divide", "keras.ops.divide", "jax.numpy.divide"},
    "torch.maximum": {"tensorflow.math.maximum", "keras.ops.maximum", "jax.numpy.maximum"},
    "torch.minimum": {"tensorflow.math.minimum", "keras.ops.minimum", "jax.numpy.minimum"},
    "torch.pow": {"tensorflow.pow", "keras.ops.power", "jax.numpy.power"},
    "torch.argmax": {"tensorflow.argmax", "keras.ops.argmax", "jax.numpy.argmax"},
    "torch.argmin": {"tensorflow.argmin", "keras.ops.argmin", "jax.numpy.argmin"},
    "torch.argsort": {"tensorflow.argsort", "keras.ops.argsort", "jax.numpy.argsort"},
    "torch.sort": {"tensorflow.sort", "keras.ops.sort", "jax.numpy.sort"},
    "torch.clip": {"tensorflow.clip_by_value", "keras.ops.clip", "jax.numpy.clip"},
    "torch.where": {"tensorflow.where", "keras.ops.where", "jax.numpy.where"},
    "torch.sigmoid": {"tensorflow.math.sigmoid", "keras.ops.sigmoid", "jax.nn.sigmoid"},
    "torch.nn.functional.relu": {"tensorflow.nn.relu", "keras.ops.relu", "keras.activations.relu", "jax.nn.relu"},
    "torch.nn.functional.softplus": {"tensorflow.nn.softplus", "keras.ops.softplus", "keras.activations.softplus", "jax.nn.softplus"},
    "torch.nn.functional.gelu": {"tensorflow.nn.gelu", "keras.ops.gelu", "keras.activations.gelu", "jax.nn.gelu"},
    "torch.nn.functional.softmax": {"tensorflow.nn.softmax", "keras.activations.softmax", "keras.ops.softmax", "jax.nn.softmax"},
    "torch.nn.functional.log_softmax": {"tensorflow.nn.log_softmax", "keras.ops.log_softmax", "jax.nn.log_softmax"},
    "torch.logsumexp": {"tensorflow.math.reduce_logsumexp", "keras.ops.logsumexp", "jax.scipy.special.logsumexp"},
    "torch.linalg.det": {"tensorflow.linalg.det", "keras.ops.det", "jax.numpy.linalg.det"},
    "torch.linalg.inv": {"tensorflow.linalg.inv", "keras.ops.inv", "jax.numpy.linalg.inv"},
    "torch.linalg.eigh": {"tensorflow.linalg.eigh", "keras.ops.eigh", "jax.numpy.linalg.eigh"},
    "torch.linalg.solve": {"tensorflow.linalg.solve", "keras.ops.solve", "jax.numpy.linalg.solve"},
    "torch.linalg.cholesky": {"tensorflow.linalg.cholesky", "keras.ops.cholesky", "jax.numpy.linalg.cholesky"},
    "torch.linalg.svd": {"tensorflow.linalg.svd", "keras.ops.svd", "jax.numpy.linalg.svd"},
    "torch.linalg.norm": {"tensorflow.linalg.norm", "keras.ops.norm", "jax.numpy.linalg.norm"},
}


def alias_candidates(target: Api, apis: list[Api]) -> set[str]:
    key = normalize_name(target.name)
    return {
        api.qualified_name
        for api in apis
        if api.library != target.library and normalize_name(api.name) == key
    }


def embedding_candidates(target_idx: int, apis: list[Api], vectors: np.ndarray, top_k: int = 12) -> list[str]:
    target = apis[target_idx]
    sims = vectors @ vectors[target_idx]
    ranked = np.argsort(-sims)
    result = []
    seen_libs = set()
    for idx in ranked:
        api = apis[idx]
        if api.qualified_name == target.qualified_name or api.library == target.library:
            continue
        if api.library in seen_libs:
            continue
        seen_libs.add(api.library)
        result.append(api.qualified_name)
        if len(result) >= top_k:
            break
    return result


def main() -> None:
    apis = collect()
    index = {api.qualified_name: i for i, api in enumerate(apis)}
    print(f"apis={len(apis)}")
    vectors = embed_texts([api.text for api in apis])

    alias_hit_total = 0
    embed_hit_total = 0
    expected_total = 0
    print("\nretrieval comparison")
    for target_name, expected in TARGETS.items():
        if target_name not in index:
            print(f"{target_name}: missing target")
            continue
        target = apis[index[target_name]]
        alias = alias_candidates(target, apis)
        embed = set(embedding_candidates(index[target_name], apis, vectors))
        alias_hits = len(alias & expected)
        embed_hits = len(embed & expected)
        expected_total += len(expected)
        alias_hit_total += alias_hits
        embed_hit_total += embed_hits
        print(
            f"{target_name:28s} alias={alias_hits}/{len(expected)} "
            f"embedding={embed_hits}/{len(expected)}"
        )
        if embed_hits < len(expected):
            print("  embedding top:", ", ".join(list(embed)[:6]))

    print(
        f"\nsummary alias_hits={alias_hit_total}/{expected_total} "
        f"embedding_hits={embed_hit_total}/{expected_total}"
    )


if __name__ == "__main__":
    main()
