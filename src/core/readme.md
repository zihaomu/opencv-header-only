# `src/core` 目录规划（过渡区）

## 目录职责

`src/core` 是历史实现的迁移缓冲区，不是长期功能承载区。  
目标是把可保留实现逐步迁移到 `include/cvh/core`，最终清空或仅保留极少数可选后端源文件。

## 规则

- 不在本目录新增长期 API。
- 允许做迁移期修复，但必须同步记录迁移目标头文件。
- 每个 `.cpp` 都应有“迁移去向 + 删除条件”。

## 阶段计划

### P0：盘点与标注

- 逐文件标注：可迁移、待重写、可删除。
- 清理明显旧依赖（例如历史 backend include）。

### P1：分批迁移

- 先迁移稳定且通用的实现到 `include/cvh/core/*.inl.h` / `detail/*.h`。
- 对难迁移逻辑先抽象接口，再迁移具体实现。

### P2：退场收口

- 保留项只限“确实不适合 header-only 的可选实现”。
- 达到门禁后，默认构建关闭 `CVH_BUILD_LEGACY_CORE`。

## 风险控制

- 防止“新功能继续堆在 src”的路径依赖。
- 防止迁移过程中 header 和 source 双份逻辑长期分叉。

## 完成定义（DoD）

- 公开 API 不依赖 `src/core` 即可使用。
- `src/core` 剩余文件都有明确存在理由和关闭计划。

## 迁移追踪

- 文件级迁移状态见：`doc/src-core-migration-tracker.md`
- 已完成低风险迁移样例：
  - `src/core/kernel/openmp_utils.h` -> `include/cvh/core/detail/openmp_utils.h`
  - `src/core/kernel/xsimd_kernel_utils.h` -> `include/cvh/core/detail/xsimd_kernel_utils.h`
