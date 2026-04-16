from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_imgproc_benchmark_regression.py"
FIXTURES = REPO_ROOT / "test" / "scripts" / "fixtures" / "imgproc_bench"


class CheckImgprocBenchmarkRegressionTests(unittest.TestCase):
    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        cmd = [sys.executable, str(SCRIPT), *args]
        return subprocess.run(cmd, check=False, capture_output=True, text=True)

    def test_passes_when_within_default_threshold(self) -> None:
        result = self.run_script(
            "--baseline",
            str(FIXTURES / "baseline.csv"),
            "--current",
            str(FIXTURES / "current_pass.csv"),
            "--max-slowdown",
            "0.08",
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("[imgproc-benchmark-regression] PASS", result.stdout)

    def test_fails_when_slowdown_exceeds_threshold(self) -> None:
        result = self.run_script(
            "--baseline",
            str(FIXTURES / "baseline.csv"),
            "--current",
            str(FIXTURES / "current_fail.csv"),
            "--max-slowdown",
            "0.08",
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("slowdown violations", result.stdout)

    def test_override_rule_can_relax_specific_case(self) -> None:
        result = self.run_script(
            "--baseline",
            str(FIXTURES / "baseline.csv"),
            "--current",
            str(FIXTURES / "current_fail.csv"),
            "--max-slowdown",
            "0.08",
            "--max-slowdown-by-op-depth",
            "THRESH_BINARY_F32:CV_32F=0.30",
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("[imgproc-benchmark-regression] PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
