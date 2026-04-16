# CLAUDE.md

## Project Execution Notes

- Default CI trust gates:
  - `smoke` on every `push`/`pull_request`
  - `core_basic` on every `push`/`pull_request`
  - `imgproc_quick_gate` on every `pull_request` (manual on `workflow_dispatch`)
  - `imgproc_full_gate` is manual (`workflow_dispatch`)
- Core hard compatibility gate is defined by:
  - `test/upstream/opencv/core/contract_v0.json`
  - validated by `scripts/verify_core_contract.py`
- Gate policy single source of truth:
  - `benchmark/gate_policy.json`
  - parsed by `scripts/read_gate_policy.py`

## Local Validation

```bash
./scripts/ci_core_basic.sh
./scripts/ci_imgproc_quick_gate.sh
```
