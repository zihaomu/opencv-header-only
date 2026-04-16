# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Added
- Trust Pipeline PR1 foundation:
  - `core_basic` runs by default on `push/pull_request`; `imgproc_quick_gate` runs by default on `pull_request`.
  - `contract_v0` hard compatibility gate for upstream core must-pass subset.
  - `benchmark/gate_policy.json` as single gate policy source.
  - Script-level unittest fixtures for gate/policy/regression scripts.
- Packaging and release baseline (PR2):
  - CMake install/export package (`opencv_header_onlyConfig.cmake`, version file, exported targets).
  - `VERSION.txt` as single project version source.
  - Minimal release playbook in `doc/release.md`.

## [0.1.0] - 2026-04-16

### Added
- Dual mode architecture (`CVH_LITE` + `CVH_FULL`) with core/imgproc/imgcodecs/highgui slices.
- Core compatibility and contract test suites, including upstream channel-port coverage.
- Imgproc benchmark and benchmark regression gates (quick/full profiles).
- Cross-platform `parallel_for` runtime path with serial/std::thread/OpenMP behavior checks.
