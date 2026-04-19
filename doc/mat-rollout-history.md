# Mat Rollout History (Merged)

更新时间：2026-04-19

本文件合并了以下重复规划文档：

- `doc/mat-basic-op-acceleration-plan.md`
- `doc/mat-channel-plan-a.md`
- `doc/scalar-mat-integration-plan.md`

## 1. 历史主线

`Mat` 相关工作过去分为三条并行线：

1. 语义冻结线：通道语义、shape/dims 合同、layout 边界。
2. 能力补齐线：`Scalar` 接入、`Mat-Scalar` 运算、compare 路径。
3. 性能优化线：dispatch 可观测、xsimd 覆盖扩展、平台专项优化。

三条线当前已可统一理解为：

- 先保证语义一致和可测试，再做性能优化。
- 优先覆盖高频 `type/channel/layout` 组合。
- 每次优化都必须有 benchmark 与 correctness 证据。

## 2. 已沉淀共识

### 2.1 Mat 语义共识

- `type = depth + channels`（OpenCV 风格）。
- `channels` 不是 `dims` 的某一轴。
- `shape/dims` 只表达几何维度。
- 任何 `CHW/NCHW` 语义由独立适配层承载，不隐式混入 `Mat`。

### 2.2 Scalar 协作共识

- `Scalar` 与 `Mat` 的协作应显式、可预测。
- 优先补齐 `Mat=Scalar`、`setTo(Scalar)`、`Mat-Scalar` 常用算子。
- 对超出 v1 能力边界的组合，采用“明确报错”而非隐式行为。

### 2.3 性能推进共识

- 先修 dispatch 与可观测性，再补 kernel 覆盖。
- 优先 `Mat-Mat`，再复用到 `Mat-Scalar`。
- `xsimd` 为主线，平台专项优化后置且必须有回归证据。

## 3. 当前文档归一（单一事实来源）

`Mat` 相关后续以以下文档为准：

- 语义/行为合同：`doc/mat-contract-v1.md`
- 项目总设计：`doc/design.md`
- 性能与对比结果：`opencv_compare/README.md` 及自动生成报告

本文件仅保留历史整合语境，不再作为执行脚本清单。

## 4. 后续维护建议

1. `Mat` 的行为变更先改合同文档，再改代码与测试。
2. 性能优化只在一个执行文档持续追加，避免再拆分多份平行计划。
3. `Scalar`/channel/layout 的边界规则始终与 `mat-contract-v1` 保持同步。
