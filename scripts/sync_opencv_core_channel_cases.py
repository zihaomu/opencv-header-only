#!/usr/bin/env python3
"""
Sync selected OpenCV core channel-related test cases into this repository.

This script extracts exact TEST(...) blocks from upstream OpenCV sources and
writes read-only snapshots under:

  test/upstream/opencv/core/<opencv-commit>/

The extracted files are not compiled yet; they are migration artifacts for
"upstream case parity" tracking.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from typing import Dict, List, Sequence, Tuple


@dataclasses.dataclass(frozen=True)
class CaseSpec:
    suite: str
    name: str
    status: str
    reason: str
    unblock_by: str


CASE_SPECS: Dict[str, List[CaseSpec]] = {
    "test_mat.cpp": [
        CaseSpec("Core_Merge", "shape_operations", "PASS_NOW", "ported and runnable in test/core/mat_upstream_channel_port_test.cpp", "none"),
        CaseSpec("Core_Split", "shape_operations", "PASS_NOW", "ported and runnable in test/core/mat_upstream_channel_port_test.cpp", "none"),
        CaseSpec("Core_Merge", "hang_12171", "PASS_NOW", "ported and runnable in test/core/mat_upstream_channel_port_test.cpp", "none"),
        CaseSpec("Core_Split", "hang_12171", "PASS_NOW", "ported and runnable in test/core/mat_upstream_channel_port_test.cpp", "none"),
        CaseSpec("Core_Split", "crash_12171", "PASS_NOW", "ported and runnable in test/core/mat_upstream_channel_port_test.cpp", "none"),
        CaseSpec("Core_Merge", "bug_13544", "PASS_NOW", "ported and runnable in test/core/mat_upstream_channel_port_test.cpp", "none"),
        CaseSpec("Core_Mat", "reinterpret_Mat_8UC3_8SC3", "PASS_NOW", "ported and runnable in test/core/mat_upstream_channel_port_test.cpp", "none"),
        CaseSpec("Core_Mat", "reinterpret_Mat_8UC4_32FC1", "PASS_NOW", "ported and runnable in test/core/mat_upstream_channel_port_test.cpp", "none"),
        CaseSpec(
            "Core_Mat",
            "reinterpret_OutputArray_8UC3_8SC3",
            "PENDING_CHANNEL",
            "by-design gap: OutputArray compatibility is out of scope in Mat-only v1",
            "non-goal (or add thin OutputArray adapter if strategy changes)",
        ),
        CaseSpec(
            "Core_Mat",
            "reinterpret_OutputArray_8UC4_32FC1",
            "PENDING_CHANNEL",
            "by-design gap: OutputArray compatibility is out of scope in Mat-only v1",
            "non-goal (or add thin OutputArray adapter if strategy changes)",
        ),
        CaseSpec("Core_MatExpr", "issue_16655", "PASS_NOW", "ported and runnable in test/core/mat_upstream_channel_port_test.cpp", "none"),
    ],
    "test_arithm.cpp": [
        CaseSpec("Subtract", "scalarc1_matc3", "PASS_NOW", "ported and runnable in test/core/mat_upstream_channel_port_test.cpp", "none"),
        CaseSpec("Subtract", "scalarc4_matc4", "PASS_NOW", "ported and runnable in test/core/mat_upstream_channel_port_test.cpp", "none"),
        CaseSpec("Compare", "empty", "PASS_NOW", "ported and runnable in test/core/mat_upstream_channel_port_test.cpp", "none"),
        CaseSpec("Compare", "regression_8999", "PASS_NOW", "ported and runnable in test/core/mat_upstream_channel_port_test.cpp", "none"),
        CaseSpec("Compare", "regression_16F_do_not_crash", "PASS_NOW", "ported and runnable in test/core/mat_upstream_channel_port_test.cpp", "none"),
        CaseSpec(
            "Core_LUT",
            "accuracy",
            "PASS_NOW",
            "ported and runnable in test/core/mat_upstream_channel_port_test.cpp (fixed-parameter subset)",
            "none",
        ),
        CaseSpec(
            "Core_LUT",
            "accuracy_multi",
            "PASS_NOW",
            "ported and runnable in test/core/mat_upstream_channel_port_test.cpp (fixed-parameter subset)",
            "none",
        ),
        CaseSpec(
            "Core_LUT",
            "accuracy_multi2",
            "PASS_NOW",
            "ported and runnable in test/core/mat_upstream_channel_port_test.cpp (fixed-parameter subset)",
            "none",
        ),
    ],
    "test_operations.cpp": [
        CaseSpec(
            "Core_Array",
            "expressions",
            "PASS_NOW",
            "ported and runnable in test/core/mat_upstream_channel_port_test.cpp",
            "none",
        ),
    ],
}

REQUIRED_MIN_CASES_PER_SOURCE: Dict[str, int] = {
    "test_mat.cpp": 1,
    "test_arithm.cpp": 1,
    "test_operations.cpp": 1,
}


TEST_RE = re.compile(
    r"^\s*TEST(?:_P)?\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync upstream OpenCV channel test cases.")
    parser.add_argument(
        "--opencv-root",
        default="/home/moo/work/github/opencv",
        help="Path to OpenCV repository root",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to opencv-header-only repository root",
    )
    return parser.parse_args()


def run_git_short_head(repo_root: pathlib.Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        head = out.strip()
        if head:
            return head
    except Exception:
        pass

    stem = repo_root.name.strip().replace(" ", "_")
    return stem if stem else "opencv-local"


def find_case_block(lines: Sequence[str], suite: str, name: str) -> Tuple[int, int]:
    for idx, line in enumerate(lines):
        m = TEST_RE.match(line)
        if not m:
            continue
        if m.group(1) != suite or m.group(2) != name:
            continue

        start = idx
        open_found = False
        brace_depth = 0
        end = -1

        for j in range(idx, len(lines)):
            text = lines[j]
            for ch in text:
                if ch == "{":
                    brace_depth += 1
                    open_found = True
                elif ch == "}":
                    brace_depth -= 1
                    if open_found and brace_depth == 0:
                        end = j
                        break
            if end != -1:
                break

        if end == -1:
            raise RuntimeError(f"Failed to close TEST block for {suite}.{name}")
        return start, end

    raise RuntimeError(f"TEST block not found for {suite}.{name}")


def extract_cases(file_path: pathlib.Path, cases: Sequence[CaseSpec]) -> List[dict]:
    lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
    extracted = []
    for case in cases:
        start, end = find_case_block(lines, case.suite, case.name)
        block = "".join(lines[start : end + 1])
        digest = hashlib.sha256(block.encode("utf-8")).hexdigest()
        extracted.append(
            {
                "suite": case.suite,
                "name": case.name,
                "status": case.status,
                "reason": case.reason,
                "unblock_by": case.unblock_by,
                "source_line_start": start + 1,
                "source_line_end": end + 1,
                "sha256": digest,
                "body": block,
            }
        )
    return extracted


def write_snapshot(
    out_file: pathlib.Path, source_rel: str, extracted: Sequence[dict], upstream_commit: str
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        f.write("// AUTO-GENERATED by scripts/sync_opencv_core_channel_cases.py\n")
        f.write(f"// Upstream OpenCV commit: {upstream_commit}\n")
        f.write(f"// Source: modules/core/test/{source_rel}\n")
        f.write("// This file intentionally stores exact upstream TEST blocks.\n")
        f.write("// Do not edit manually; re-run sync script.\n\n")
        for case in extracted:
            f.write(f"// BEGIN {case['suite']}.{case['name']}\n")
            f.write(case["body"])
            if not case["body"].endswith("\n"):
                f.write("\n")
            f.write(f"// END {case['suite']}.{case['name']}\n\n")


def main() -> int:
    args = parse_args()
    opencv_root = pathlib.Path(args.opencv_root).resolve()
    repo_root = pathlib.Path(args.repo_root).resolve()

    if not opencv_root.exists():
        print(f"OpenCV root does not exist: {opencv_root}", file=sys.stderr)
        return 2

    upstream_commit = run_git_short_head(opencv_root)
    out_dir = repo_root / "test" / "upstream" / "opencv" / "core" / upstream_commit
    manifest_path = repo_root / "test" / "upstream" / "opencv" / "core" / "channel_manifest.json"

    manifest = {
        "upstream_repo": str(opencv_root),
        "upstream_commit": upstream_commit,
        "generated_by": "scripts/sync_opencv_core_channel_cases.py",
        "cases": [],
    }

    source_case_counts: Dict[str, int] = {}

    for src_file, case_specs in CASE_SPECS.items():
        required_min = REQUIRED_MIN_CASES_PER_SOURCE.get(src_file, 0)
        if len(case_specs) < required_min:
            raise RuntimeError(
                f"source {src_file} has {len(case_specs)} configured cases, "
                f"but policy requires at least {required_min}"
            )

        if not case_specs:
            source_case_counts[src_file] = 0
            continue
        source_path = opencv_root / "modules" / "core" / "test" / src_file
        if not source_path.exists():
            raise RuntimeError(f"Missing upstream source file: {source_path}")
        extracted = extract_cases(source_path, case_specs)
        source_case_counts[src_file] = len(extracted)
        out_file = out_dir / src_file.replace(".cpp", ".channel_cases.cpp")
        write_snapshot(out_file, src_file, extracted, upstream_commit)
        for item in extracted:
            manifest["cases"].append(
                {
                    "source_file": f"modules/core/test/{src_file}",
                    "source_line_start": item["source_line_start"],
                    "source_line_end": item["source_line_end"],
                    "suite": item["suite"],
                    "name": item["name"],
                    "status": item["status"],
                    "reason": item["reason"],
                    "unblock_by": item["unblock_by"],
                    "sha256": item["sha256"],
                    "snapshot_file": str(out_file.relative_to(repo_root)),
                }
            )

    for src_file, required_min in REQUIRED_MIN_CASES_PER_SOURCE.items():
        found = source_case_counts.get(src_file, 0)
        if found < required_min:
            raise RuntimeError(
                f"source {src_file} extracted {found} cases, policy requires >= {required_min}"
            )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Synced {len(manifest['cases'])} channel cases from OpenCV {upstream_commit}")
    print(f"Manifest: {manifest_path}")
    print(f"Snapshots: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
