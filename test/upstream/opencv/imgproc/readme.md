# `test/upstream/opencv/imgproc` 说明

该目录用于保存从 upstream OpenCV `modules/imgproc/test` 截取的原始测试片段快照，
用于追踪 `cvh` 的对齐进度。

## 生成方式

```bash
python3 scripts/sync_opencv_imgproc_cases.py
```

## 当前策略

- `PASS_NOW`：当前 `cvh` 已覆盖的行为（通过本仓库 GTest 合同测试验证）。
- `PENDING_*`：尚未纳入当前 `cvh` 支持范围的上游行为（作为后续里程碑跟踪）。
