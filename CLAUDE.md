# CLAUDE.md

## Project Execution Notes

- Default CI modules:
  - `lite_all` on every `push`/`pull_request` (`scripts/ci_lite_all.sh`)
  - `full_all` on every `push`/`pull_request` (`scripts/ci_full_all.sh`)
  - `opencv_compare` optional (PR label `ci/run-opencv-compare`, or `workflow_dispatch`)

## Local Validation

```bash
./scripts/ci_lite_all.sh
./scripts/ci_full_all.sh
./scripts/ci_compare_log_only.sh
```
