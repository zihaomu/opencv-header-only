#!/usr/bin/env python3
"""Read benchmark gate policy and emit shell variables for CI scripts."""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import sys
from typing import Dict, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read benchmark gate policy")
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    parser.add_argument(
        "--policy",
        default="benchmark/gate_policy.json",
        help="Policy JSON path (relative to repo root unless absolute)",
    )
    parser.add_argument("--profile", default="quick", help="Imgproc profile name")
    parser.add_argument(
        "--emit",
        choices=("shell", "json"),
        default="shell",
        help="Output format",
    )
    return parser.parse_args()


def resolve_path(root: pathlib.Path, path: str) -> pathlib.Path:
    p = pathlib.Path(path)
    if p.is_absolute():
        return p
    return root / p


def _require_dict(parent: Dict[str, object], key: str, label: str) -> Dict[str, object]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{label}.{key} must be an object")
    return value


def _require_number(parent: Dict[str, object], key: str, label: str) -> float:
    value = parent.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{label}.{key} must be numeric")
    return float(value)


def _require_int(parent: Dict[str, object], key: str, label: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{label}.{key} must be integer")
    return value


def _require_string(parent: Dict[str, object], key: str, label: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{label}.{key} must be string")
    return value


def load_policy(path: pathlib.Path, profile: str) -> Tuple[Dict[str, object], Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"policy file not found: {path}")

    policy = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(policy, dict):
        raise ValueError("policy root must be object")

    core_contract = _require_dict(policy, "core_contract", "policy")
    contract_path = _require_string(core_contract, "path", "policy.core_contract").strip()
    if not contract_path:
        raise ValueError("policy.core_contract.path must not be empty")

    imgproc = _require_dict(policy, "imgproc", "policy")
    enforce_fingerprint = imgproc.get("enforce_fingerprint", True)
    if not isinstance(enforce_fingerprint, bool):
        raise ValueError("policy.imgproc.enforce_fingerprint must be boolean")

    profiles = _require_dict(imgproc, "profiles", "policy.imgproc")
    profile_cfg = profiles.get(profile)
    if not isinstance(profile_cfg, dict):
        available = ", ".join(sorted(profiles.keys()))
        raise ValueError(f"unknown profile '{profile}', available: {available}")

    warmup = _require_int(profile_cfg, "default_warmup", f"policy.imgproc.profiles.{profile}")
    iters = _require_int(profile_cfg, "default_iters", f"policy.imgproc.profiles.{profile}")
    repeats = _require_int(profile_cfg, "default_repeats", f"policy.imgproc.profiles.{profile}")
    threads = _require_int(profile_cfg, "default_threads", f"policy.imgproc.profiles.{profile}")
    minimum_samples = _require_int(profile_cfg, "minimum_samples", f"policy.imgproc.profiles.{profile}")
    max_slowdown = _require_number(profile_cfg, "default_max_slowdown", f"policy.imgproc.profiles.{profile}")
    max_slowdown_by_op_depth = _require_string(
        profile_cfg, "default_max_slowdown_by_op_depth", f"policy.imgproc.profiles.{profile}"
    )

    if warmup < 0 or iters < 1 or repeats < 1 or threads < 1:
        raise ValueError("warmup must be >=0, iters/repeats/threads must be >=1")
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be >=1")
    if max_slowdown < 0.0:
        raise ValueError("default_max_slowdown must be >= 0")

    normalized = {
        "version": int(policy.get("version", 0)),
        "core_contract_path": contract_path,
        "imgproc_enforce_fingerprint": enforce_fingerprint,
        "imgproc_profile": profile,
        "imgproc_default_warmup": warmup,
        "imgproc_default_iters": iters,
        "imgproc_default_repeats": repeats,
        "imgproc_default_threads": threads,
        "imgproc_default_max_slowdown": max_slowdown,
        "imgproc_default_max_slowdown_by_op_depth": max_slowdown_by_op_depth.strip(),
        "imgproc_minimum_samples": minimum_samples,
    }
    return policy, normalized


def emit_shell(normalized: Dict[str, object]) -> str:
    variables = {
        "CVH_POLICY_CORE_CONTRACT_PATH": str(normalized["core_contract_path"]),
        "CVH_POLICY_IMGPROC_ENFORCE_FINGERPRINT": "true"
        if normalized["imgproc_enforce_fingerprint"]
        else "false",
        "CVH_POLICY_IMGPROC_PROFILE": str(normalized["imgproc_profile"]),
        "CVH_POLICY_IMGPROC_WARMUP": str(normalized["imgproc_default_warmup"]),
        "CVH_POLICY_IMGPROC_ITERS": str(normalized["imgproc_default_iters"]),
        "CVH_POLICY_IMGPROC_REPEATS": str(normalized["imgproc_default_repeats"]),
        "CVH_POLICY_IMGPROC_THREADS": str(normalized["imgproc_default_threads"]),
        "CVH_POLICY_IMGPROC_MAX_SLOWDOWN": str(normalized["imgproc_default_max_slowdown"]),
        "CVH_POLICY_IMGPROC_MAX_SLOWDOWN_BY_OP_DEPTH": str(
            normalized["imgproc_default_max_slowdown_by_op_depth"]
        ),
        "CVH_POLICY_IMGPROC_MINIMUM_SAMPLES": str(normalized["imgproc_minimum_samples"]),
    }
    return "\n".join(f"{k}={shlex.quote(v)}" for k, v in variables.items())


def main() -> int:
    args = parse_args()
    root = pathlib.Path(args.repo_root).resolve()
    policy_path = resolve_path(root, args.policy)

    try:
        _policy, normalized = load_policy(policy_path, args.profile)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"failed to read gate policy: {exc}", file=sys.stderr)
        return 1

    if args.emit == "json":
        print(json.dumps(normalized, sort_keys=True))
    else:
        print(emit_shell(normalized))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
