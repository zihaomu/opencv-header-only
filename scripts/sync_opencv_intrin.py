#!/usr/bin/env python3
"""Sync the vendored OpenCV Universal Intrinsics whitelist."""

from __future__ import annotations

import argparse
import filecmp
import shutil
import subprocess
from pathlib import Path


DEFAULT_OPENCV_ROOT = Path("/Users/zmu/work/my_project/ocvh/opencv")
REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO_ROOT / "include" / "cvh" / "3rdparty" / "opencv_intrin"

WHITELIST = {
    "modules/core/include/opencv2/core/hal/intrin.hpp": "opencv2/core/hal/intrin.hpp",
    "modules/core/include/opencv2/core/hal/intrin_cpp.hpp": "opencv2/core/hal/intrin_cpp.hpp",
    "modules/core/include/opencv2/core/hal/intrin_forward.hpp": "opencv2/core/hal/intrin_forward.hpp",
    "modules/core/include/opencv2/core/hal/intrin_math.hpp": "opencv2/core/hal/intrin_math.hpp",
    "modules/core/include/opencv2/core/hal/intrin_neon.hpp": "opencv2/core/hal/intrin_neon.hpp",
    "modules/core/include/opencv2/core/hal/simd_utils.impl.hpp": "opencv2/core/hal/simd_utils.impl.hpp",
    "LICENSE": "LICENSE.opencv",
}


def git_value(opencv_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(opencv_root), *args],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--opencv-root",
        type=Path,
        default=DEFAULT_OPENCV_ROOT,
        help="Path to the local OpenCV source tree",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify that vendored files match the OpenCV source tree",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    opencv_root = args.opencv_root.resolve()
    if not (opencv_root / "modules" / "core" / "include" / "opencv2" / "core" / "hal" / "intrin.hpp").exists():
        raise SystemExit(f"OpenCV source tree not found: {opencv_root}")

    changed = []
    missing = []
    for source_rel, dest_rel in WHITELIST.items():
        source = opencv_root / source_rel
        dest = VENDOR_ROOT / dest_rel
        if not source.exists():
            raise SystemExit(f"Missing upstream file: {source}")
        if args.check:
            if not dest.exists():
                missing.append(str(dest.relative_to(REPO_ROOT)))
            elif not filecmp.cmp(source, dest, shallow=False):
                changed.append(str(dest.relative_to(REPO_ROOT)))
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            changed.append(str(dest.relative_to(REPO_ROOT)))

    if args.check:
        if missing or changed:
            for path in missing:
                print(f"[opencv-intrin] missing: {path}")
            for path in changed:
                print(f"[opencv-intrin] out-of-sync: {path}")
            return 1
        head = git_value(opencv_root, "rev-parse", "HEAD")
        describe = git_value(opencv_root, "describe", "--tags", "--always", "--dirty")
        print(f"[opencv-intrin] OK: {describe} {head}")
        return 0

    head = git_value(opencv_root, "rev-parse", "HEAD")
    describe = git_value(opencv_root, "describe", "--tags", "--always", "--dirty")
    for path in changed:
        print(f"[opencv-intrin] synced: {path}")
    print(f"[opencv-intrin] upstream: {describe} {head}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
