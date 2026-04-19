# `test/smoke` 目录规划

## 目录职责

提供最快速的可用性验证，确保入口头和最小运行链路不被破坏。

## 阶段计划

### P0：基础 smoke

- `cvh_header_compile_smoke`：验证公开头可编译。
- `cvh_include_only_smoke`：验证仅 `-Iinclude` 场景可编译运行。
- `cvh_full_backend_smoke`：验证 Full backend 目标链路可用。

### P1：主线优先

- 增加纯 header-only 路径 smoke。

### P2：发布门禁

- 将 smoke 作为 CI 必跑项。
- 任意公共头破坏必须在 smoke 阶段被拦截。

## 完成定义（DoD）

- 新环境下可通过 smoke 快速确认项目可用性。
