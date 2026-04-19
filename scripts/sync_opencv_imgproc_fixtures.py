#!/usr/bin/env python3
"""
Sync selected OpenCV extra fixtures for imgproc tests.

Default source:
  /home/moo/work/github/opencv_extra-4.x/testdata

Destination:
  test/imgproc/data/opencv_extra
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
from typing import Iterable, List, Set


REQUIRED_FIXTURES = [
    "cv/shared/pic1.png",
    "cv/shared/fruits.png",
    "cv/imgproc/stuff.jpg",
    "cv/grabcut/image1652.ppm",
    "cv/grabcut/mask1652.ppm",
]

OPTIONAL_LARGE_FIXTURES = [
    "cv/shared/lena.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync selected OpenCV extra fixtures for imgproc tests.")
    parser.add_argument(
        "--opencv-extra-root",
        default="/home/moo/work/github/opencv_extra-4.x",
        help="Path to opencv_extra root (or directly to testdata directory).",
    )
    parser.add_argument("--repo-root", default=".", help="Path to opencv-header-only repository root.")
    parser.add_argument(
        "--with-large",
        action="store_true",
        help="Also sync large optional fixtures (e.g. cv/shared/lena.png).",
    )
    return parser.parse_args()


def resolve_testdata_root(opencv_extra_root: pathlib.Path) -> pathlib.Path:
    if (opencv_extra_root / "testdata").is_dir():
        return opencv_extra_root / "testdata"
    if opencv_extra_root.name == "testdata" and opencv_extra_root.is_dir():
        return opencv_extra_root
    raise FileNotFoundError(
        f"Cannot find testdata under {opencv_extra_root}. "
        "Pass --opencv-extra-root as opencv_extra root or testdata directory."
    )


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def iter_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file())


def main() -> int:
    args = parse_args()
    repo_root = pathlib.Path(args.repo_root).resolve()
    opencv_extra_root = pathlib.Path(args.opencv_extra_root).resolve()
    testdata_root = resolve_testdata_root(opencv_extra_root)

    dst_root = repo_root / "test" / "imgproc" / "data" / "opencv_extra"
    manifest_path = repo_root / "test" / "imgproc" / "data" / "manifest.json"
    dst_root.mkdir(parents=True, exist_ok=True)

    selected: List[str] = list(REQUIRED_FIXTURES)
    if args.with_large:
        selected.extend(OPTIONAL_LARGE_FIXTURES)

    copied_records = []
    kept_relpaths: Set[pathlib.Path] = set()

    for rel in selected:
        src = testdata_root / rel
        if not src.exists():
            print(f"missing source fixture: {src}", file=sys.stderr)
            return 2

        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        rel_path = dst.relative_to(dst_root)
        kept_relpaths.add(rel_path)
        copied_records.append(
            {
                "relative_path": rel,
                "optional_large": rel in OPTIONAL_LARGE_FIXTURES,
                "source_file": str(src),
                "snapshot_file": str(dst.relative_to(repo_root)),
                "size": dst.stat().st_size,
                "sha256": sha256_file(dst),
            }
        )

    for stale in iter_files(dst_root):
        rel_path = stale.relative_to(dst_root)
        if rel_path not in kept_relpaths:
            stale.unlink()

    manifest = {
        "generated_by": "scripts/sync_opencv_imgproc_fixtures.py",
        "opencv_extra_root": str(opencv_extra_root),
        "opencv_extra_testdata_root": str(testdata_root),
        "fixtures": copied_records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    print(f"synced {len(copied_records)} fixture(s) into {dst_root}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
