"""Sanity checks for the Xamt++ release artifact.

The check is intentionally lightweight: it only uses the standard library and
does not import framework packages. It verifies that the release metadata,
coverage summaries, and reproduction files are internally consistent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "README.md",
    "ARTIFACT.md",
    "RELEASE_MANIFEST.md",
    "PAIRWISE_ADAPTER_SUMMARY.md",
    "ALL_BUG_CANDIDATES.md",
    "REAL_BUG_AUDIT.md",
    "bug_repros/README.md",
    "bug_repros/metadata.json",
    "tools/api_match_common.py",
    "tools/compare_api_matchers.py",
    "tools/diff_static_candidate_groups.py",
    "tools/timed_group_fuzz.py",
]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def extract_int(pattern: str, text: str, label: str) -> int:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"missing {label}")
    return int(match.group(1))


def count_bug_scripts() -> int:
    return len(sorted((ROOT / "bug_repros").glob("bug_[0-9][0-9][0-9]_*.py")))


def main() -> int:
    failures: list[str] = []

    for relpath in REQUIRED_FILES:
        require((ROOT / relpath).is_file(), f"missing required file: {relpath}", failures)

    summary = read("PAIRWISE_ADAPTER_SUMMARY.md")
    api_common = read("tools/api_match_common.py")
    all_candidates = read("ALL_BUG_CANDIDATES.md")
    audit = read("REAL_BUG_AUDIT.md")
    repro_readme = read("bug_repros/README.md")

    metadata = json.loads(read("bug_repros/metadata.json"))
    records = metadata.get("records", [])

    target_libraries = re.findall(r'^    "([^"]+)": \[', api_common, flags=re.MULTILINE)
    pairwise_groups = extract_int(r"- Groups: (\d+)", summary, "pairwise groups")
    pairwise_unique_apis = extract_int(r"- Unique APIs: (\d+)", summary, "pairwise unique APIs")
    pairwise_libraries = extract_int(r"- Libraries: (\d+)", summary, "library count")
    pairwise_diff = extract_int(r"\| DIFF \| (\d+) \|", summary, "pairwise DIFF groups")
    total_candidates = extract_int(r"Total current bug candidates: \*\*(\d+)\*\*", all_candidates, "candidate count")
    likely_real = extract_int(r"Likely real differential bugs: (\d+) keys", audit, "likely real bugs")
    strict_real = extract_int(r"Strong/reportable .*: (\d+) keys", audit, "strict bugs")
    verified_live = extract_int(r"Verified live DIFF bugs: (\d+)", repro_readme, "verified live bugs")
    audited_scripts = extract_int(r"Candidate scripts audited: (\d+)", repro_readme, "audited scripts")

    script_count = count_bug_scripts()
    metadata_count = int(metadata.get("count", -1))

    require(len(target_libraries) == 8, f"expected 8 target DL libraries, got {len(target_libraries)}", failures)
    require("numpy" not in target_libraries, "numpy must not be a target library", failures)
    require("scipy" not in target_libraries, "scipy must not be a target library", failures)
    require(pairwise_groups > 0, f"expected positive pairwise groups, got {pairwise_groups}", failures)
    require(pairwise_unique_apis > 0, f"expected positive unique APIs, got {pairwise_unique_apis}", failures)
    require(pairwise_libraries >= len(target_libraries), f"recorded library count {pairwise_libraries} is smaller than target count {len(target_libraries)}", failures)
    require(pairwise_diff >= 0, f"expected non-negative DIFF groups, got {pairwise_diff}", failures)
    require(total_candidates >= likely_real >= strict_real, "candidate audit counts are not monotonic", failures)
    require(metadata_count == 188, f"expected metadata count 188, got {metadata_count}", failures)
    require(len(records) == metadata_count, f"metadata count {metadata_count} != record rows {len(records)}", failures)
    require(script_count == metadata_count, f"script count {script_count} != metadata count {metadata_count}", failures)
    require(audited_scripts == metadata_count, f"audited scripts {audited_scripts} != metadata count {metadata_count}", failures)
    require(0 <= verified_live <= metadata_count, f"verified live count {verified_live} is outside metadata range", failures)

    print("artifact_status:", "ok" if not failures else "failed")
    print("target_libraries:", len(target_libraries), ",".join(target_libraries))
    print("recorded_pairwise_libraries:", pairwise_libraries)
    print("pairwise_groups:", pairwise_groups)
    print("pairwise_unique_apis:", pairwise_unique_apis)
    print("pairwise_diff_groups:", pairwise_diff)
    print("bug_candidates:", total_candidates)
    print("likely_real_bugs:", likely_real)
    print("strict_reportable_bugs:", strict_real)
    print("bug_records:", metadata_count)
    print("bug_scripts:", script_count)
    print("verified_live_diff_bugs:", verified_live)

    if failures:
        print("failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
