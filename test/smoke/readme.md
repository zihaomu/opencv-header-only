# `test/smoke` 目录规划

## 目录职责

提供最快速的可用性验证，确保入口头和最小运行链路不被破坏。

## 阶段计划

### P0：基础 smoke

- `cvh_header_compile_smoke`：验证公开头可编译。
- `cvh_include_only_smoke`：验证仅 `-Iinclude` 场景可编译运行。
- `cvh_headers_fast_smoke`：验证 `cvh::headers_fast` 只启用 OpenCV Universal Intrinsics，不启用 xsimd 或 `.cpp` 模式。
- `cvh_legacy_xsimd_facade_smoke`：只在 `CVH_BUILD_LEGACY_XSIMD_TESTS=ON` 时构建，用于隔离验证历史 xsimd adapter。
- legacy `.cpp` smoke target 只验证历史实验链路，不属于公开 header-only 产品面。

### P1：主线优先

- 增加纯 header-only 路径 smoke。

### P2：发布门禁

- 将 smoke 作为 CI 必跑项。
- 任意公共头破坏必须在 smoke 阶段被拦截。

## 完成定义（DoD）

- 新环境下可通过 smoke 快速确认项目可用性。
