# OpenCV Upstream Case Snapshots

This directory stores upstream OpenCV `modules/core/test` case snapshots used for
`cvh` compatibility tracking.

Rules:

- Snapshot sources are copied from local OpenCV checkout at
  `/home/moo/work/github/opencv`.
- Case bodies are kept exact ("raw upstream") and are not manually edited.
- Current migration scope in this folder focuses on **Mat channel-related** cases,
  currently extracted from `test_mat.cpp`, `test_arithm.cpp`, and
  `test_operations.cpp` (`Core_Array.expressions` channel path), including
  `Core_LUT.{accuracy,accuracy_multi,accuracy_multi2}` snapshots.
- Status is tracked in `channel_manifest.json`:
  - `PASS_NOW`: expected to run and pass in current `cvh`.
  - `PENDING_CHANNEL`: accepted future requirement, currently blocked by missing APIs.

Regeneration:

```bash
python3 scripts/sync_opencv_core_channel_cases.py --repo-root .
```

Outputs:

- `channel_manifest.json`: source location, case id, status, blocker, hash.
- `<opencv-commit>/*.channel_cases.cpp`: exact upstream TEST blocks.
