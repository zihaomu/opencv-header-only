# `benchmark` 目录规划

## 目录职责

提供性能基准与回归检查，防止优化退化和跨版本性能漂移。

## 规划目标

- 建立可重复的 benchmark 入口（固定数据、固定线程、固定输出格式）。
- 优先覆盖 `core` 热点算子，再覆盖 `imgproc`。
- 可选对比 OpenCV 或仓库历史版本。

## 阶段计划

### P1：基准框架

- 统一 CLI 参数：输入规模、线程数、warmup、repeat。
- 输出统一为 `csv/json`，便于 CI 归档。

### P2：Core 基线

- 覆盖 `add/mul/convertTo/transpose/gemm` 等高频路径。
- 分离标量路径与 SIMD 路径数据。

### P3：Imgproc 基线

- 覆盖 `cvtColor/resize/GaussianBlur`。
- 建立不同分辨率下的吞吐与延迟对比。

## 完成定义（DoD）

- 关键算子有可追踪性能曲线。
- PR 可基于 benchmark 报告判断是否有明显性能回退。
