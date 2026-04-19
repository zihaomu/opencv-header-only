#!/usr/bin/env python3
"""
Sync selected OpenCV imgproc test cases into this repository.

Extract exact TEST(...) blocks from upstream OpenCV sources and write snapshots:
  test/upstream/opencv/imgproc/<opencv-commit>/
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
    source_file: str
    suite: str
    name: str
    status: str
    reason: str


CASE_SPECS: List[CaseSpec] = [
    CaseSpec(
        source_file="test_imgwarp.cpp",
        suite="Resize",
        name="nearest_regression_15075",
        status="PASS_NOW",
        reason="Covered by test/imgproc/imgproc_resize_contract_test.cpp",
    ),
    CaseSpec(
        source_file="test_color.cpp",
        suite="ImgProc_cvtColor_InvalidNumOfChannels",
        name="regression_25971",
        status="PASS_NOW",
        reason="Covered by test/imgproc/imgproc_cvtcolor_contract_test.cpp",
    ),
    CaseSpec(
        source_file="test_resize_bitexact.cpp",
        suite="Resize_Bitexact",
        name="Nearest8U",
        status="PASS_NOW",
        reason="Covered by test/imgproc/imgproc_resize_contract_test.cpp (INTER_NEAREST_EXACT).",
    ),
    CaseSpec(
        source_file="test_imgwarp.cpp",
        suite="Imgproc_WarpAffine",
        name="accuracy",
        status="PASS_NOW",
        reason="Covered by test/imgproc/imgproc_warp_affine_contract_test.cpp (fixed-parameter subset for geometry, inverse map, border, and argument validation).",
    ),
    CaseSpec(
        source_file="test_imgwarp.cpp",
        suite="Imgproc_Warp",
        name="regression_19566",
        status="PASS_NOW",
        reason="Covered by test/imgproc/imgproc_warp_affine_contract_test.cpp (constant-border multi-channel subset).",
    ),
    CaseSpec(
        source_file="test_thresh.cpp",
        suite="Imgproc_Threshold",
        name="threshold_dryrun",
        status="PASS_NOW",
        reason="Covered by test/imgproc/imgproc_threshold_contract_test.cpp (THRESH_DRYRUN contract).",
    ),
    CaseSpec(
        source_file="test_filter.cpp",
        suite="Imgproc_Blur",
        name="borderTypes",
        status="PASS_NOW",
        reason="Covered by test/imgproc/imgproc_filter_contract_test.cpp",
    ),
    CaseSpec(
        source_file="test_filter.cpp",
        suite="Imgproc_GaussianBlur",
        name="borderTypes",
        status="PASS_NOW",
        reason="Covered by test/imgproc/imgproc_filter_contract_test.cpp",
    ),
    CaseSpec(
        source_file="test_filter.cpp",
        suite="Imgproc_GaussianBlur",
        name="regression_11303",
        status="PASS_NOW",
        reason="Covered by test/imgproc/imgproc_filter_contract_test.cpp (CV_32F constant-image sigma path).",
    ),
    CaseSpec(
        source_file="test_filter.cpp",
        suite="Imgproc_Filter2D",
        name="dftFilter2d_regression_13179",
        status="PASS_NOW",
        reason="Covered by test/imgproc/imgproc_filter2d_contract_test.cpp (filter2D regression subset).",
    ),
    CaseSpec(
        source_file="test_filter.cpp",
        suite="Imgproc_sepFilter2D",
        name="identity",
        status="PASS_NOW",
        reason="Covered by test/imgproc/imgproc_sep_filter2d_contract_test.cpp (identity kernel semantics).",
    ),
    CaseSpec(
        source_file="test_contours.cpp",
        suite="Imgproc_FindContours",
        name="border",
        status="PASS_NOW",
        reason="Covered by test/imgproc/imgproc_copy_make_border_contract_test.cpp (upstream preamble semantics subset).",
    ),
    CaseSpec(
        source_file="test_smooth_bitexact.cpp",
        suite="GaussianBlur_Bitexact",
        name="regression_15015",
        status="PASS_NOW",
        reason="Covered by test/imgproc/imgproc_filter_contract_test.cpp (constant-image regression).",
    ),
    CaseSpec(
        source_file="test_filter.cpp",
        suite="Imgproc_Morphology",
        name="iterated",
        status="PASS_NOW",
        reason="Covered by test/imgproc/imgproc_morph_gradient_contract_test.cpp (iterative morphology equivalence).",
    ),
    CaseSpec(
        source_file="test_filter.cpp",
        suite="Imgproc",
        name="filter_empty_src_16857",
        status="PASS_NOW",
        reason="Covered by test/imgproc/imgproc_morph_gradient_contract_test.cpp (implemented-op empty input coverage).",
    ),
    CaseSpec(
        source_file="test_filter.cpp",
        suite="Imgproc",
        name="morphologyEx_small_input_22893",
        status="PASS_NOW",
        reason="Covered by test/imgproc/imgproc_morph_gradient_contract_test.cpp (MORPH_DILATE small input regression).",
    ),
    CaseSpec(
        source_file="test_filter.cpp",
        suite="Imgproc_MorphEx",
        name="hitmiss_regression_8957",
        status="PASS_NOW",
        reason="Covered by test/imgproc/imgproc_morph_gradient_contract_test.cpp (MORPH_HITMISS regression).",
    ),
    CaseSpec(
        source_file="test_filter.cpp",
        suite="Imgproc_MorphEx",
        name="hitmiss_zero_kernel",
        status="PASS_NOW",
        reason="Covered by test/imgproc/imgproc_morph_gradient_contract_test.cpp (MORPH_HITMISS zero-kernel behavior).",
    ),
    CaseSpec(
        source_file="test_filter.cpp",
        suite="Imgproc_Sobel",
        name="borderTypes",
        status="PASS_NOW",
        reason="Covered by test/imgproc/imgproc_morph_gradient_contract_test.cpp (ROI BORDER_ISOLATED semantics).",
    ),
    CaseSpec(
        source_file="test_filter.cpp",
        suite="Imgproc_Sobel",
        name="s16_regression_13506",
        status="PASS_NOW",
        reason="Covered by test/imgproc/imgproc_morph_gradient_contract_test.cpp (CV_16S ksize=5 regression).",
    ),
    CaseSpec(
        source_file="test_canny.cpp",
        suite="Canny_Modes",
        name="accuracy",
        status="PASS_NOW",
        reason="Covered by test/imgproc/imgproc_canny_contract_test.cpp (fixed-parameter upstream subset).",
    ),
]


TEST_RE = re.compile(
    r"^\s*TEST(?:_P)?\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync upstream OpenCV imgproc test case snapshots.")
    parser.add_argument(
        "--opencv-root",
        default="/home/moo/work/github/opencv",
        help="Path to OpenCV repository root.",
    )
    parser.add_argument("--repo-root", default=".", help="Path to opencv-header-only repository root.")
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

    # Fallback for source trees copied from release archives (no .git).
    # Keep commit-like directory stable and traceable.
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


def write_snapshot(
    out_file: pathlib.Path,
    source_rel: str,
    upstream_commit: str,
    extracted: List[dict],
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        f.write("// AUTO-GENERATED by scripts/sync_opencv_imgproc_cases.py\n")
        f.write(f"// Upstream OpenCV commit: {upstream_commit}\n")
        f.write(f"// Source: modules/imgproc/test/{source_rel}\n")
        f.write("// This file intentionally stores exact upstream TEST blocks.\n")
        f.write("// Do not edit manually; re-run sync script.\n\n")
        for case in extracted:
            f.write(f"// BEGIN {case['suite']}.{case['name']} ({case['status']})\n")
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
    out_dir = repo_root / "test" / "upstream" / "opencv" / "imgproc" / upstream_commit
    manifest_path = repo_root / "test" / "upstream" / "opencv" / "imgproc" / "case_manifest.json"

    grouped: Dict[str, List[CaseSpec]] = {}
    for spec in CASE_SPECS:
        grouped.setdefault(spec.source_file, []).append(spec)

    manifest = {
        "upstream_repo": str(opencv_root),
        "upstream_commit": upstream_commit,
        "generated_by": "scripts/sync_opencv_imgproc_cases.py",
        "cases": [],
    }

    for source_file, specs in grouped.items():
        source_path = opencv_root / "modules" / "imgproc" / "test" / source_file
        if not source_path.exists():
            raise RuntimeError(f"Missing upstream source file: {source_path}")

        lines = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
        extracted = []
        for spec in specs:
            start, end = find_case_block(lines, spec.suite, spec.name)
            body = "".join(lines[start : end + 1])
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            extracted.append(
                {
                    "suite": spec.suite,
                    "name": spec.name,
                    "status": spec.status,
                    "reason": spec.reason,
                    "source_line_start": start + 1,
                    "source_line_end": end + 1,
                    "sha256": digest,
                    "body": body,
                }
            )

        out_file = out_dir / source_file.replace(".cpp", ".cases.cpp")
        write_snapshot(out_file, source_file, upstream_commit, extracted)

        for case in extracted:
            manifest["cases"].append(
                {
                    "source_file": f"modules/imgproc/test/{source_file}",
                    "source_line_start": case["source_line_start"],
                    "source_line_end": case["source_line_end"],
                    "suite": case["suite"],
                    "name": case["name"],
                    "status": case["status"],
                    "reason": case["reason"],
                    "sha256": case["sha256"],
                    "snapshot_file": str(out_file.relative_to(repo_root)),
                }
            )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    print(f"synced {len(manifest['cases'])} imgproc case snapshot(s) from OpenCV {upstream_commit}")
    print(f"manifest: {manifest_path}")
    print(f"snapshots: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
