from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "verify_core_contract.py"
FIXTURES = REPO_ROOT / "test" / "scripts" / "fixtures" / "core_contract"


class VerifyCoreContractScriptTests(unittest.TestCase):
    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        cmd = [sys.executable, str(SCRIPT), *args]
        return subprocess.run(cmd, check=False, capture_output=True, text=True)

    def test_emit_gtest_filter_for_valid_contract(self) -> None:
        result = self.run_script(
            "--repo-root",
            str(REPO_ROOT),
            "--manifest",
            str(FIXTURES / "manifest_valid.json"),
            "--contract",
            str(FIXTURES / "contract_valid.json"),
            "--emit-gtest-filter",
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            (
                "OpenCVUpstreamChannelPort_TEST.Compare_regression_8999:"
                "OpenCVUpstreamChannelPort_TEST.Core_Merge_shape_operations"
            ),
        )

    def test_fails_when_must_pass_case_is_not_pass_now(self) -> None:
        result = self.run_script(
            "--repo-root",
            str(REPO_ROOT),
            "--manifest",
            str(FIXTURES / "manifest_valid.json"),
            "--contract",
            str(FIXTURES / "contract_bad_status.json"),
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("must_pass cases must be PASS_NOW", result.stderr)


if __name__ == "__main__":
    unittest.main()
