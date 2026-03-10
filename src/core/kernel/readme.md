# `src/core/kernel` 目录规划（性能内核过渡区）

## 目录职责

存放当前 `xsimd`/CPU 内核实现，支撑 `core` 里的性能路径。  
该目录当前仍属于过渡形态，后续需与 header-only 架构统一。

## 阶段计划

### P0：边界收敛

- 明确哪些内核仍被主线 API 使用。
- 删除无调用路径或仅服务旧 backend 的内核。

### P1：接口标准化

- 统一内核函数签名与命名规则。
- 将 kernel 与业务算子解耦，避免一对一硬编码。

### P2：迁移与并行化

- 可 header-only 的内核迁入 `include/cvh/core/detail`。
- 不适合头文件化的实现保留为可选编译单元，并给出开关。

### P3：性能治理

- 每个关键内核需要 benchmark 基线和正确性对照。
- 优化前后必须有数据，避免不可回归优化。

## 风险控制

- 防止 `kernel` 直接依赖测试数据格式或旧工程路径。
- 防止不同 ISA 分支行为不一致。

## 完成定义（DoD）

- 主线算子的标量/向量路径结果一致。
- `benchmark` 可稳定输出 kernel 性能对比结果。

## 已完成迁移样例

- `openmp_utils.h` 已迁移到 `include/cvh/core/detail/openmp_utils.h`
- `xsimd_kernel_utils.h` 已迁移到 `include/cvh/core/detail/xsimd_kernel_utils.h`
