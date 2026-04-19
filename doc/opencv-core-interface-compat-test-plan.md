# OpenCV Core Interface Compatibility Test Plan (Header-Only)

- Scope: `cvh` core compatibility validation against OpenCV upstream tests
- Upstream baseline: OpenCV commit `d719c6d`
- Upstream repo path: `/home/moo/work/github/opencv`
- Last update: 2026-03-25

## 1. Goal

Build a repeatable test system to validate:

1. API completeness: claimed OpenCV-like interfaces exist in `cvh`.
2. Behavior completeness: migrated upstream test cases pass (or are explicitly tracked as pending).
3. No silent regressions: every unsupported case is visible and traceable via status.

## 2. Test Layers

### L0. Signature Compile Checks

- Compile-only checks for key APIs (`Mat`, type/channel helpers, arithmetic entry points).
- Fails fast on missing declarations or signature drift.

### L1. Upstream Raw Case Snapshots

- Store exact upstream `TEST(...)` blocks in repository.
- No manual edits to test body.
- Case metadata tracked in `channel_manifest.json`.

### L2. Ported Runtime Tests

- For APIs already supported, migrate case wrappers into runnable `gtest` targets.
- Preserve original assertions and case names whenever possible.

### L3. Pending/XFAIL Registry

- Cases for future APIs are tracked with explicit status:
  - `PASS_NOW`
  - `PENDING_CHANNEL`
- No silent skip. Pending reason and unblock condition are mandatory.

## 3. Channel-Focused Migration (Future-Critical)

`Mat` channel semantics are a mandatory future capability.  
Channel-related upstream cases are now tracked under:

- Manifest: `test/upstream/opencv/core/channel_manifest.json`
- Snapshots: `test/upstream/opencv/core/d719c6d/*.channel_cases.cpp`

Current policy:

- Keep these cases as raw upstream snapshots immediately.
- Promote each case from `PENDING_CHANNEL` -> `PASS_NOW` only when corresponding `cvh` API lands.

## 4. Initial Channel Case Set (from OpenCV `d719c6d`)

From `modules/core/test/test_mat.cpp`:

- `Core_Merge.shape_operations`
- `Core_Split.shape_operations`
- `Core_Merge.hang_12171`
- `Core_Split.hang_12171`
- `Core_Split.crash_12171`
- `Core_Merge.bug_13544`
- `Core_Mat.reinterpret_Mat_8UC3_8SC3`
- `Core_Mat.reinterpret_Mat_8UC4_32FC1`
- `Core_Mat.reinterpret_OutputArray_8UC3_8SC3`
- `Core_Mat.reinterpret_OutputArray_8UC4_32FC1`
- `Core_MatExpr.issue_16655`

From `modules/core/test/test_arithm.cpp`:

- `Subtract.scalarc1_matc3`
- `Subtract.scalarc4_matc4`
- `Compare.empty`
- `Compare.regression_8999`
- `Compare.regression_16F_do_not_crash`

From `modules/core/test/test_operations.cpp`:

- `Core_Array.expressions`

## 5. Current Gaps Mapped to Pending Cases

- Missing `merge/split` APIs in `cvh` core.
- Missing scalar-overload style subtraction path equivalent to `cv::subtract(Scalar, Mat, dst)`.
- `compare` is currently not implemented.
- `Mat::reinterpret` / `OutputArray::reinterpret` compatibility is absent.
- `MatExpr` multi-channel compare/type propagation is incomplete.

## 6. Tooling

### Sync upstream snapshots

```bash
python3 scripts/sync_opencv_core_channel_cases.py --repo-root .
```

## 7. Promotion Rule: Pending -> Runnable

A pending channel case can be promoted to runnable only when:

1. Required API exists in public headers.
2. Runtime behavior is implemented for channel semantics.
3. Case executes in CI and passes.
4. Manifest status is updated from `PENDING_CHANNEL` to `PASS_NOW`.

## 8. CI Recommendation

Add these checks into core CI:

1. Snapshot sync/verification step:
   - run `sync_opencv_core_channel_cases.py` and ensure `channel_manifest.json` + snapshot files are up to date in PR diff
2. Runtime layer step:
   - execute currently runnable compatibility tests
3. Policy step:
   - fail if a claimed supported API still maps only to `PENDING_CHANNEL` cases.
