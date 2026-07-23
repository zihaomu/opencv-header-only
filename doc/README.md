# Documentation Index

This directory keeps only current project documents.

- [design.md](design.md): product direction, public targets, module boundaries,
  and SIMD strategy.
- [mat-contract-v1.md](mat-contract-v1.md): current `cvh::Mat` behavior
  contract.
- [benchmark-refactor-implementation-plan.md](benchmark-refactor-implementation-plan.md):
  implementation plan for the two-mode benchmark framework.
- [core-cpp-cleanup-plan.md](core-cpp-cleanup-plan.md): cleanup and
  header-only migration plan for core arithmetic, transpose, GEMM, and
  overlapping legacy `.cpp` implementations.
- [opencv-ui-kernel-migration-checklist.md](opencv-ui-kernel-migration-checklist.md):
  checklist for porting OpenCV Universal Intrinsics kernel fragments.
- [opencv-universal-intrinsics-adapter-plan.md](opencv-universal-intrinsics-adapter-plan.md):
  current OpenCV UI SIMD status summary. The filename is kept for existing
  references, but the old adapter/facade execution log has been removed.

Historical rollout notes, old native-backend planning, old xsimd TODOs, and
stale compatibility-test plans have been removed from `doc/`. Current behavior
should be checked through README, this directory, and the header-only tests.
