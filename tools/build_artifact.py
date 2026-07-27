
from __future__ import annotations

import argparse
import os
import subprocess
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "dist" / "xamtplusplus-artifact.tar.gz"


EXCLUDE_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "rank_0",
    "results",
    "dist",
    "build",
}

EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".tmp",
    ".log",
    ".jsonl",
}


def included(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDE_PARTS for part in rel.parts):
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    if path.name.endswith(".egg-info"):
        return False
    return True


def run_check() -> None:
    subprocess.run(
        ["python", "-B", str(ROOT / "tools" / "artifact_check.py")],
        cwd=ROOT,
        check=True,
    )


def add_tree(archive: tarfile.TarFile, prefix: str) -> int:
    count = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or not included(path):
            continue
        archive.add(path, arcname=str(Path(prefix) / path.relative_to(ROOT)))
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output .tar.gz path")
    parser.add_argument("--prefix", default="xamtplusplus-artifact", help="archive top-level directory")
    parser.add_argument("--skip-check", action="store_true", help="skip tools/artifact_check.py before archiving")
    args = parser.parse_args()

    output = Path(args.out)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_check:
        run_check()

    with tarfile.open(output, "w:gz") as archive:
        file_count = add_tree(archive, args.prefix)

    size = os.path.getsize(output)
    print("artifact_archive:", output)
    print("files:", file_count)
    print("bytes:", size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
