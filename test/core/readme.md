# `test/core` 目录规划

## 目录职责

验证 `core` 模块行为，包括 `Mat` 语义、基础算子、错误路径和数值稳定性。

## 核心覆盖目标

- `Mat` 生命周期：创建、拷贝、释放、浅拷贝/深拷贝。
- `Mat` 结构语义：shape、type、channel、stride、reshape、transpose。
- 基础算子：`add/sub/mul/div/compare/convertTo`。
- 异常语义：非法输入、维度不匹配、类型不支持。

## 阶段计划

### P0：现有用例治理

- 清理对旧 backend 头文件的依赖。
- 将测试按 API 分组，提升定位效率。

### P1：Mat 合同测试

- 增加通道语义和边界行为测试。
- 建立与 OpenCV 行为对照的关键 case（选取高频路径）。

### P2：算子回归

- 每个公开算子至少包含：
  - 一条正常路径 case
  - 一条边界或异常路径 case

## 完成定义（DoD）

- `core` 公开 API 均有可追溯测试。
- 任何 `Mat` 语义变更都会触发对应回归测试。
