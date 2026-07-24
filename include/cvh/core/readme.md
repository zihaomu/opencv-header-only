# `include/cvh/core` 目录规划

## 目录职责

`core` 是整个项目的底座，负责 `Mat`、基础数据类型、错误处理、基础算子和工具函数。  
该目录最终必须独立支撑 header-only 使用，不依赖 `src/` 才能工作。

## 当前状态（2026-07-24）

- 已有 `Mat`、`define`、`system`、`basic_op` 等基础头文件。
- `src/core` 的 Mat/system/算术/transpose/GEMM/MatExpr 实现已迁入
  ODR-safe headers，公共 core 不再依赖编译单元。
- `Mat` 已支持 `CV_64F` 存储、ROI 和基础标量 dispatch；几何类型已提供
  `Point2i/Point2f/Point2d` 与 `Size2i/Size2f/Size2d`。
- 逐元素层已提供 `absdiff`、bitwise 系列、`inRange`、`min/max`，并覆盖
  Mat/Scalar、浮点 raw-bit、mask 和非连续 ROI 基线。
- 数学层已提供缩放转换、FP16 bits 转换、F32/F64
  `sqrt/pow/exp/log`、`checkRange` 与 F32 `patchNaNs`。
- 归约层已提供 `norm/sum/mean/meanStdDev`、non-zero 谓词、extrema、
  axis reduce/arg-reduce 与 `normalize`，覆盖文档约定的 mask、ROI、
  C1/C3/C4 和代表深度。
- 布局层已提供 mask copy、channel routing、2D/N-D flip、rotate、repeat、
  concat、broadcast、swap 与公开 border interpolation；共享存储写入会先
  保留 source snapshot。
- API 语义与 OpenCV 仍未完全对齐；已完成 type/channel 宏、连续多通道以及首批 2D submat+非连续步长语义，未完成项主要在通用 ND ROI 与高阶算子行为。

## 阶段计划

### P0：基线清理（必须先做）

- 统一命名空间为 `cvh`，清理历史命名残留。
- 保证公共头可独立编译，不引入测试路径依赖。
- 明确哪些头是稳定 API，哪些是 `detail`/过渡实现。

### P1：Mat 合同冻结

- 明确并固定 `Mat` 的 type/channel/shape/stride/ROI 行为。
- 补齐 `clone/copyTo/convertTo/reshape` 行为一致性。
- 对齐 OpenCV 风格的错误与断言接口。
- 合同基线文档：`doc/mat-contract-v1.md`。
- 第一阶段执行计划：`doc/opencv-core-imgproc-phase1-implementation-plan.md`。

### P2：Core 能力闭环

- 补齐高频逐元素操作、类型转换、规约等基础能力。
- 保持 core accepted API 只由 header 提供，禁止重新引入 `src/core`
  链接依赖。
- 建立每个核心 API 的测试和示例映射关系。

### P3：性能层引入

- 在不破坏 API 的前提下引入 direct OpenCV Universal Intrinsics 写法和必要的平台专项 header-only 优化。
- 历史 xsimd 路径已从当前代码面移除，不再作为本目录的优化方向。
- 不再把二次 SIMD facade 作为未来主路线；OpenCV UI 只作为 `cvh` 内部 SIMD dialect，不构成用户公开 API。
- 性能路径与显式标量 fallback 保持统一语义并双向回归测试。

## 非目标与边界

- 本目录不承接 `imgproc/imgcodecs` 的具体业务逻辑。
- `softmax/rmsnorm/rope/silu` 这类偏推理算子不作为主线核心 API 扩展依据。

## 完成定义（DoD）

- `include/cvh/core/*` 可独立作为公开接口使用。
- `test/core` 覆盖核心行为与边界条件。
- 新增核心接口必须同时提供最小示例和单元测试。
