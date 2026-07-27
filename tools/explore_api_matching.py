
from __future__ import annotations

import importlib
import inspect
import os
from collections import defaultdict
from dataclasses import dataclass

try:
    from .api_match_common import NAMESPACES, normalize_name
except ImportError:
    from api_match_common import NAMESPACES, normalize_name


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


@dataclass(frozen=True)
class Api:
    library: str
    namespace: str
    name: str
    qualified_name: str
    kind: str
    arity: int | None



def arity(obj: object) -> int | None:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    count = 0
    for param in sig.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        count += 1
    return count


def collect_apis() -> list[Api]:
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
                if inspect.isclass(obj):
                    kind = "class"
                elif inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.isroutine(obj):
                    kind = "function"
                else:
                    continue
                apis.append(
                    Api(
                        library=library,
                        namespace=namespace,
                        name=name,
                        qualified_name=f"{namespace}.{name}",
                        kind=kind,
                        arity=arity(obj),
                    )
                )
    return apis


def groups_by_key(apis: list[Api], use_arity: bool) -> dict[tuple[object, ...], list[Api]]:
    groups: dict[tuple[object, ...], list[Api]] = defaultdict(list)
    for api in apis:
        key: tuple[object, ...]
        if use_arity:
            key = (normalize_name(api.name), api.arity)
        else:
            key = (normalize_name(api.name),)
        groups[key].append(api)
    return groups


def cross_library_groups(groups: dict[tuple[object, ...], list[Api]]) -> list[list[Api]]:
    result = []
    for apis in groups.values():
        libs = {api.library for api in apis}
        if len(libs) >= 2:
            result.append(apis)
    return result


def summarize(groups: list[list[Api]], title: str) -> None:
    group_count = len(groups)
    api_count = sum(len(group) for group in groups)
    by_width: dict[int, int] = defaultdict(int)
    for group in groups:
        by_width[len({api.library for api in group})] += 1
    print(f"\n{title}")
    print(f"  groups: {group_count}")
    print(f"  APIs in groups: {api_count}")
    print("  width distribution:", dict(sorted(by_width.items())))


def print_examples(groups: list[list[Api]], limit: int = 20) -> None:
    ranked = sorted(
        groups,
        key=lambda group: (-len({api.library for api in group}), normalize_name(group[0].name)),
    )
    print("\nexamples:")
    for group in ranked[:limit]:
        libs = sorted({api.library for api in group})
        key = normalize_name(group[0].name)
        shown = []
        seen = set()
        for api in sorted(group, key=lambda item: (item.library, item.qualified_name)):
            if api.library in seen:
                continue
            seen.add(api.library)
            shown.append(api.qualified_name)
        print(f"  {key:28s} {len(libs)} libs -> " + " | ".join(shown))


def main() -> None:
    apis = collect_apis()
    libs = sorted({api.library for api in apis})
    print("libraries:", ", ".join(libs))
    for lib in libs:
        count = sum(1 for api in apis if api.library == lib)
        print(f"  {lib:10s} {count}")

    exact = cross_library_groups(groups_by_key(apis, use_arity=False))
    exact_with_arity = cross_library_groups(groups_by_key(apis, use_arity=True))

    summarize(exact, "normalized-name candidate groups")
    summarize(exact_with_arity, "normalized-name + arity candidate groups")
    print_examples(exact_with_arity)


if __name__ == "__main__":
    main()
