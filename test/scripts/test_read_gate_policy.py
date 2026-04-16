from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "read_gate_policy.py"
POLICY_FIXTURE = REPO_ROOT / "test" / "scripts" / "fixtures" / "gate_policy" / "policy_valid.json"


class ReadGatePolicyScriptTests(unittest.TestCase):
    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        cmd = [sys.executable, str(SCRIPT), *args]
        return subprocess.run(cmd, check=False, capture_output=True, text=True)

    def test_emit_json_for_quick_profile(self) -> None:
        result = self.run_script(
            "--repo-root",
            str(REPO_ROOT),
            "--policy",
            str(POLICY_FIXTURE),
            "--profile",
            "quick",
            "--emit",
            "json",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["imgproc_profile"], "quick")
        self.assertEqual(payload["imgproc_default_warmup"], 2)
        self.assertEqual(payload["imgproc_default_max_slowdown"], 0.08)
        self.assertEqual(payload["core_contract_path"], "test/upstream/opencv/core/contract_v0.json")

    def test_emit_shell_contains_expected_exports(self) -> None:
        result = self.run_script(
            "--repo-root",
            str(REPO_ROOT),
            "--policy",
            str(POLICY_FIXTURE),
            "--profile",
            "quick",
            "--emit",
            "shell",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("CVH_POLICY_CORE_CONTRACT_PATH=test/upstream/opencv/core/contract_v0.json", result.stdout)
        self.assertIn("CVH_POLICY_IMGPROC_PROFILE=quick", result.stdout)
        self.assertIn("CVH_POLICY_IMGPROC_WARMUP=2", result.stdout)

    def test_fails_for_unknown_profile(self) -> None:
        result = self.run_script(
            "--repo-root",
            str(REPO_ROOT),
            "--policy",
            str(POLICY_FIXTURE),
            "--profile",
            "nightly",
            "--emit",
            "json",
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("unknown profile", result.stderr)


if __name__ == "__main__":
    unittest.main()
