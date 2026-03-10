# Phase 0 执行计划（基线收口）

- 时间范围：2026-03-10 ~ 2026-03-21（10 个工作日）
- 对应里程碑：`doc/header-only-opencv-plan.md` 的 Phase 0
- 目标：把项目从“可运行原型”收敛到“可持续迭代基线”

## 1. Phase 0 目标与退出条件

### 目标

- 统一构建与测试主干流程。
- 保证公共头可独立编译。
- 清理 `minfer` 残留路径。
- 为 `src/core` 迁移建立台账和退场标准。

### 退出条件（全部满足）

- `cvh/cvh.h` 在仅 `-Iinclude` 情况下可编译。
- `configure -> build -> test` 默认链路可稳定运行（至少 `smoke`；如 core 已接线则包含 core）。
- `include/src/test` 不再存在有效 `minfer` 引用路径。
- `src/core` 每个文件均有“迁移去向 + 删除条件”记录。

## 2. 工作分解（WBS）

### P0-01（D1-D2）：工程门禁固化

- 任务：
  - 固化标准命令链：`configure -> build -> test`。
  - CI 中加入最小必跑项（先 `smoke`，core 接线后并入）。
- 交付物：
  - 统一执行命令文档（可放入 `README` 或 `doc/`）。
  - CI job 配置或本地脚本入口。
- 验收：
  - 新环境按文档可一键跑通最小链路。

### P0-02（D3-D4）：公开头独立编译

- 任务：
  - 清理公共头对测试目录和历史路径的依赖。
  - 新增 include-only smoke（仅 `-Iinclude` 编译）。
- 交付物：
  - 最小 include-only 测试入口（建议放 `test/smoke`）。
- 验收：
  - include-only smoke 持续通过。

### P0-03（D5-D6）：`minfer` 残留清理

- 任务：
  - 扫描并替换 `minfer` 命名空间、宏、错误处理入口。
  - 清理历史文案和注释中的误导信息。
- 交付物：
  - 清理清单（文件 + 处理方式）。
- 验收：
  - `rg -n "\\bminfer\\b" include src test` 仅保留允许的历史注释（或归零）。

### P0-04（D7-D8）：测试去旧依赖

- 任务：
  - 清理 `test/core` 对 `backend/*` 等旧路径依赖。
  - 将测试对象对齐到当前公开 API。
- 交付物：
  - 可运行的 `core` 基础测试子集（若未完全接线，先形成可执行清单）。
- 验收：
  - 测试不依赖旧工程目录即可编译运行。

### P0-05（D9-D10）：`src/core` 迁移台账

- 任务：
  - 为 `src/core` 每个文件建立迁移状态：`迁移中/待重写/待删除`。
  - 选择 1-2 个低风险文件先完成迁移闭环。
- 交付物：
  - 迁移台账文档（建议 `doc/src-core-migration-tracker.md`）。
  - 至少 1 次“迁移后验证”记录。
- 验收：
  - 台账完整，且有实际迁移样例闭环。

## 3. 执行状态

| 任务 | 状态 | 更新时间 | 备注 |
|---|---|---|---|
| P0-01 | 已完成 | 2026-03-10 | 已新增统一脚本 `scripts/ci_smoke.sh`，接入 `.github/workflows/ci.yml`，并验证 `configure -> build -> test` |
| P0-02 | 已完成 | 2026-03-10 | 已新增 `cvh_include_only_smoke` 与 `scripts/check_public_headers.sh`，并在 CI 脚本中执行 |
| P0-03 | 已完成 | 2026-03-10 | 已清理 `system.h/system.cpp/mat_convert.cpp` 与相关测试脚本中的历史命名残留，并通过 smoke 验证 |
| P0-04 | 已完成 | 2026-03-10 | 已清理 `test/core` 对旧 backend 的直接依赖，并接线 `cvh_core_basic_tests` 可运行子集 |
| P0-05 | 已完成 | 2026-03-10 | 已新增 `doc/src-core-migration-tracker.md`，并完成 2 个低风险头文件迁移闭环 |

## 4. 每日执行节奏（建议）

- 每天开始：
  - 确认当日任务编号（P0-01 ~ P0-05）。
  - 先跑一次最小基线（build + smoke）。
- 每天结束：
  - 更新当日状态：完成项、阻塞项、次日计划。
  - 如发生范围变更，更新对应目录 `readme.md` 与总规划。

## 5. 风险与应对

- 风险：测试改造时被旧 backend 头文件反向锁定。  
  应对：先切分“可独立运行的最小 core 子集”，再逐步替换其余测试。

- 风险：header-only 迁移导致短期重复实现（`src` 与 `include` 双份）。  
  应对：所有迁移 PR 必须附“删除旧实现时间点”。

- 风险：清理 `minfer` 时误伤行为一致性。  
  应对：每次批量替换后跑 smoke + core 子集回归。

## 6. 追踪模板（可直接复制）

```md
## YYYY-MM-DD
- 今日任务：P0-0X
- 完成：
- 阻塞：
- 风险：
- 明日计划：
```

## 7. 建议命令基线

```bash
cmake -S . -B build
cmake --build build -j
cmake --build build --target test
```

如环境提供 `ctest`，可等价执行：`ctest --test-dir build --output-on-failure`。
