# Imgproc Rollout History (Merged)

更新时间：2026-04-19

本文件合并了以下重复文档：

- `doc/filter-acceleration-plan.md`
- `doc/filter-acceleration-execution-plan-2026-04-15.md`
- `doc/m1-morph-gradient-execution-plan-2026-04-18.md`
- `doc/m2-operator-rollout-plan-2026-04-18.md`

## 1. 历史目标（已完成）

`imgproc` 从 fallback 为主，逐步补齐为“可调用 + 可测试 + 可对比”的可交付能力：

1. 先建立 benchmark/回归门禁与执行闭环。
2. 先 correctness，再逐步推进性能优化。
3. 关键算子按阶段落地并纳入 compare 报告。

## 2. 阶段性结果

### M1（Morphology + Gradient）

已完成：

- `Sobel`
- `erode`
- `dilate`
- `morphologyEx`

交付闭环：API、合同测试、upstream 子集移植、compare 结果可见。

### M2（Operator Rollout）

已完成：

- `copyMakeBorder`
- `LUT`
- `warpAffine`
- `filter2D`
- `sepFilter2D`

交付闭环：API、合同测试、upstream 子集移植、compare 结果可见。

## 3. 当前对外查看入口（单一事实来源）

性能对比不再以 `doc/` 下阶段快照为主，统一看：

- `opencv_compare/opencv_compare_quick.md`
- `opencv_compare/opencv_compare_stable.md`
- `opencv_compare/opencv_compare_baseline_stable.md`
- `opencv_compare/README.md`

## 4. 后续建议

如继续推进 `imgproc`，建议只维护一份执行主文档，避免再按阶段复制新计划文档：

1. 在本文件追加里程碑（时间 + 算子 + 验收结果）。
2. 详细实验数据仅放 `opencv_compare/results/*.csv` 与自动生成的 Markdown 报告。
3. 算子性能优化与正确性回归保持同一 PR 闭环。
