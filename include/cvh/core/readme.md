# `include/cvh/core` 目录规划

## 目录职责

`core` 是整个项目的底座，负责 `Mat`、基础数据类型、错误处理、基础算子和工具函数。  
该目录最终必须独立支撑 header-only 使用，不依赖 `src/` 才能工作。

## 当前状态（2026-03-10）

- 已有 `Mat`、`define`、`system`、`basic_op` 等基础头文件。
- 仍存在历史命名与实现迁移债务（例如旧命名残留、部分能力在 `src/core`）。
- API 语义与 OpenCV 还未完全对齐（特别是通道、步长、错误模型）。

## 阶段计划

### P0：基线清理（必须先做）

- 统一命名空间为 `cvh`，清理历史命名残留。
- 保证公共头可独立编译，不引入测试路径依赖。
- 明确哪些头是稳定 API，哪些是 `detail`/过渡实现。

### P1：Mat 合同冻结

- 明确并固定 `Mat` 的 type/channel/shape/stride/ROI 行为。
- 补齐 `clone/copyTo/convertTo/reshape` 行为一致性。
- 对齐 OpenCV 风格的错误与断言接口。

### P2：Core 能力闭环

- 补齐高频逐元素操作、类型转换、规约等基础能力。
- 将 `src/core` 里保留的通用实现逐步迁入 header-only 结构。
- 建立每个核心 API 的测试和示例映射关系。

### P3：性能层引入

- 在不破坏 API 的前提下引入 `OpenMP`/`xsimd` 可选优化。
- 性能路径与标量路径保持统一语义并双向回归测试。

## 非目标与边界

- 本目录不承接 `imgproc/imgcodecs` 的具体业务逻辑。
- `softmax/rmsnorm/rope/silu` 这类偏推理算子不作为主线核心 API 扩展依据。

## 完成定义（DoD）

- `include/cvh/core/*` 可独立作为公开接口使用。
- `test/core` 覆盖核心行为与边界条件。
- 新增核心接口必须同时提供最小示例和单元测试。
