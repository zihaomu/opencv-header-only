# Phase 1 执行计划（Core MVP）

- 时间范围：2026-03-10 ~ 2026-03-24（10 个工作日）
- 对应里程碑：`doc/header-only-opencv-plan.md` 的 Phase 1
- 目标：冻结 `Mat` 合同并推动实现与测试对齐，形成可稳定迭代的 `core` MVP 基线

## 1. Phase 1 目标与退出条件

### 目标

- 冻结 `Mat` 的 v1 行为合同，并作为后续实现/测试唯一语义基线。
- 将 `Mat` 关键 API（`create/clone/copyTo/convertTo/reshape`）对齐合同语义。
- 形成可重复执行的 `core` 合同回归测试入口。

### 退出条件（全部满足）

- `doc/mat-contract-v1.md` 冻结并纳入主规划索引。
- `Mat` 关键 API 的实现行为与合同一致（至少覆盖单通道连续内存语义）。
- `test/core` 覆盖合同清单中的成功路径与失败路径。
- `./scripts/ci_smoke.sh` 与 `./scripts/ci_core_basic.sh` 稳定通过。

## 2. 工作分解（WBS）

### P1-01（D1）：Phase 1 任务拆解与门禁固化

- 任务：
  - 建立 Phase 1 执行条目、顺序、门禁与验收命令。
  - 明确后续任务编号（P1-02 ~ P1-05）的输入输出关系。
- 交付物：
  - 本文档 `doc/phase1-execution-plan.md`。
- 验收：
  - 能基于本文档直接推进 P1-02 ~ P1-05。

### P1-02（D1-D2）：Mat 合同冻结

- 任务：
  - 固化 `Mat` v1 语义边界（type/channel/shape/ownership/continuous/error）。
  - 明确 v1 非目标，防止范围漂移。
- 交付物：
  - `doc/mat-contract-v1.md`。
- 验收：
  - 合同覆盖关键 API 行为与失败语义，并在规划文档中可追踪。

### P1-03（D3-D5）：实现差异清理（合同对齐）

- 任务：
  - 对齐 `include/cvh/core/mat.h`、`include/cvh/core/mat.inl.h`、`src/core/mat.cpp` 与合同差异。
  - 清理与合同冲突的隐式行为（优先错误路径与边界路径）。
- 交付物：
  - `Mat` 关键 API 对齐补丁（以合同为准）。
- 验收：
  - 合同条目对应实现均可被测试验证，不存在 silent wrong result。

### P1-04（D6-D7）：Header-only 迁移首批闭环

- 任务：
  - 将合同范围内的 `Mat` 稳定实现优先迁至 `include/cvh/core/*.inl.h` / `detail/*.h`。
  - 缩减 `src/core/mat.cpp` 对默认能力的承载范围。
- 交付物：
  - 首批迁移闭环记录（迁移前后能力和限制说明）。
- 验收：
  - 默认使用路径对 `src/core` 依赖进一步下降，行为与测试不回退。

### P1-05（D8-D10）：合同测试闭环

- 任务：
  - 按合同第 7 节补齐测试：`clone/copyTo/reshape/convertTo` 成功与失败路径。
  - 增加 `channels != 1` 的明确失败行为测试。
- 交付物：
  - `test/core` 新增或更新的合同测试用例。
- 验收：
  - 合同测试可稳定复现，CI 可运行且结果一致。

## 3. 执行状态

| 任务 | 状态 | 更新时间 | 备注 |
|---|---|---|---|
| P1-01 | 已完成 | 2026-03-10 | 已建立 Phase 1 执行文档，固化任务与门禁 |
| P1-02 | 已完成 | 2026-03-10 | 已新增 `doc/mat-contract-v1.md`，并完成索引与 core 文档联动 |
| P1-03 | 待开始 | 2026-03-10 | 下一步对齐 `Mat` 实现与合同差异 |
| P1-04 | 待开始 | 2026-03-10 | 依赖 P1-03 的差异清理结果 |
| P1-05 | 待开始 | 2026-03-10 | 依赖 P1-03 的行为稳定后补齐 |

## 4. 风险与应对

- 风险：合同冻结后仍出现“实现先变更、文档后补”的倒挂。  
  应对：P1 期间 `Mat` 行为变更必须先改合同，再改代码和测试。

- 风险：`src` 与 `include` 双份实现短期分叉。  
  应对：P1-04 每次迁移都写明删除条件并同步台账。

- 风险：测试覆盖偏正常路径，边界/错误路径缺失。  
  应对：P1-05 以合同第 7 节为强制清单逐项验收。

## 5. 建议验收命令基线

```bash
./scripts/ci_smoke.sh
CVH_WARNING_BUDGET=0 ./scripts/ci_core_basic.sh
```
