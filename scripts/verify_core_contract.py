#!/usr/bin/env python3
"""Validate contract_v0 compatibility gate against channel manifest.

Outputs a gtest filter for OpenCVUpstreamChannelPort_TEST cases when requested.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Dict, Iterable, List, Set, Tuple

CaseKey = Tuple[str, str]

ALLOWED_CASE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify core contract gate against channel manifest")
    parser.add_argument("--repo-root", default=".", help="Repository root")
    parser.add_argument(
        "--manifest",
        default="test/upstream/opencv/core/channel_manifest.json",
        help="Path to channel manifest (relative to repo root unless absolute)",
    )
    parser.add_argument(
        "--contract",
        default="test/upstream/opencv/core/contract_v0.json",
        help="Path to contract json (relative to repo root unless absolute)",
    )
    parser.add_argument(
        "--emit-gtest-filter",
        action="store_true",
        help="Emit gtest filter on stdout",
    )
    return parser.parse_args()


def _resolve(root: pathlib.Path, path: str) -> pathlib.Path:
    p = pathlib.Path(path)
    if p.is_absolute():
        return p
    return root / p


def _validate_case_id(case: Dict[str, str], label: str) -> Tuple[str, str]:
    suite = str(case.get("suite", "")).strip()
    name = str(case.get("name", "")).strip()
    if not suite or not name:
        raise ValueError(f"{label}: suite/name must be non-empty")
    if not ALLOWED_CASE_RE.match(suite) or not ALLOWED_CASE_RE.match(name):
        raise ValueError(f"{label}: invalid suite/name format: {suite}.{name}")
    return suite, name


def load_manifest_cases(manifest: Dict[str, object]) -> Dict[CaseKey, Dict[str, str]]:
    cases: Dict[CaseKey, Dict[str, str]] = {}
    for raw in manifest.get("cases", []):
        if not isinstance(raw, dict):
            raise ValueError("manifest: case entry must be object")
        suite, name = _validate_case_id(raw, "manifest case")
        key = (suite, name)
        if key in cases:
            raise ValueError(f"manifest: duplicate case {suite}.{name}")
        cases[key] = {
            "status": str(raw.get("status", "")).strip(),
            "source_file": str(raw.get("source_file", "")).strip(),
        }
    return cases


def load_contract_cases(contract: Dict[str, object]) -> Tuple[Set[CaseKey], Set[CaseKey], int]:
    min_must_pass = int(contract.get("minimum_must_pass", 1))
    if min_must_pass < 1:
        raise ValueError("contract: minimum_must_pass must be >= 1")

    must_pass_raw = contract.get("must_pass", [])
    if not isinstance(must_pass_raw, list):
        raise ValueError("contract: must_pass must be list")

    must_pass: Set[CaseKey] = set()
    for idx, raw in enumerate(must_pass_raw):
        if not isinstance(raw, dict):
            raise ValueError(f"contract: must_pass[{idx}] must be object")
        key = _validate_case_id(raw, f"contract must_pass[{idx}]")
        if key in must_pass:
            raise ValueError(f"contract: duplicate must_pass entry {key[0]}.{key[1]}")
        must_pass.add(key)

    required_raw = contract.get("required_cases", [])
    if not isinstance(required_raw, list):
        raise ValueError("contract: required_cases must be list")

    required: Set[CaseKey] = set()
    for idx, raw in enumerate(required_raw):
        if not isinstance(raw, dict):
            raise ValueError(f"contract: required_cases[{idx}] must be object")
        key = _validate_case_id(raw, f"contract required_cases[{idx}]")
        required.add(key)

    if len(must_pass) < min_must_pass:
        raise ValueError(
            f"contract: must_pass size {len(must_pass)} is less than minimum_must_pass {min_must_pass}"
        )

    missing_required = sorted(required - must_pass)
    if missing_required:
        joined = ", ".join(f"{s}.{n}" for s, n in missing_required)
        raise ValueError(f"contract: required_cases missing from must_pass: {joined}")

    return must_pass, required, min_must_pass


def to_gtest_case_name(case: CaseKey) -> str:
    suite, name = case
    return f"OpenCVUpstreamChannelPort_TEST.{suite}_{name}"


def build_gtest_filter(cases: Iterable[CaseKey]) -> str:
    ordered = sorted(cases)
    return ":".join(to_gtest_case_name(case) for case in ordered)


def main() -> int:
    args = parse_args()
    root = pathlib.Path(args.repo_root).resolve()

    manifest_path = _resolve(root, args.manifest)
    contract_path = _resolve(root, args.contract)

    if not manifest_path.exists():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    if not contract_path.exists():
        print(f"contract not found: {contract_path}", file=sys.stderr)
        return 2

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = json.loads(contract_path.read_text(encoding="utf-8"))

        manifest_cases = load_manifest_cases(manifest)
        must_pass, _, _ = load_contract_cases(contract)

        missing_in_manifest = sorted(case for case in must_pass if case not in manifest_cases)
        if missing_in_manifest:
            joined = ", ".join(f"{s}.{n}" for s, n in missing_in_manifest)
            raise ValueError(f"contract: cases missing in manifest: {joined}")

        wrong_status = sorted(
            case
            for case in must_pass
            if manifest_cases[case].get("status") != "PASS_NOW"
        )
        if wrong_status:
            details = ", ".join(
                f"{s}.{n}={manifest_cases[(s, n)].get('status', 'unknown')}" for s, n in wrong_status
            )
            raise ValueError(f"contract: must_pass cases must be PASS_NOW: {details}")

        if args.emit_gtest_filter:
            print(build_gtest_filter(must_pass))
        else:
            print(
                f"verified contract with {len(must_pass)} must-pass cases "
                f"against manifest {manifest_path.name}"
            )
        return 0
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"contract verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
