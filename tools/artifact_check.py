from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "README.md",
    "ARTIFACT.md",
    "CITATION.cff",
    "LICENSE",
    "requirements-main.txt",
    "tools/api_match_common.py",
    "tools/compare_api_matchers.py",
    "tools/diff_static_candidate_groups.py",
    "tools/timed_group_fuzz.py",
    "tools/build_artifact.py",
]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    failures: list[str] = []

    for relpath in REQUIRED_FILES:
        require((ROOT / relpath).is_file(), f"missing required file: {relpath}", failures)

    for relpath in REQUIRED_FILES:
        if not relpath.endswith(".py"):
            continue
        target = ROOT / relpath
        if not target.is_file():
            continue
        try:
            ast.parse(target.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            failures.append(f"syntax error in {relpath}: {exc}")

    api_common = read("tools/api_match_common.py")
    target_libraries = re.findall(r'^    "([^"]+)": \[', api_common, flags=re.MULTILINE)
    require(bool(target_libraries), "no target libraries configured", failures)
    require("numpy" not in target_libraries, "numpy must not be a target library", failures)
    require("scipy" not in target_libraries, "scipy must not be a target library", failures)

    print("artifact_status:", "ok" if not failures else "failed")
    print("target_libraries:", ",".join(target_libraries))

    if failures:
        print("failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
