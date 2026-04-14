# 测试未通过台账（接口未对齐）

- 更新时间：2026-04-14
- 作用：记录当前 `test/` 下未通过、未运行或显式挂起（skip/pending）的测试及原因。

## 1. 构建状态（2026-03-25）

已解锁此前阻塞 `cvh_test_core` 的编译问题，并完成首批 channel case 提升：

- `Error::Code` 增加了 `StsOutOfMem` / `StsBadType` 兼容别名，`cvh_test_core` 可完成编译与执行。
- 当前 `cvh_test_core` 可稳定执行；channel pending 用例均为显式 `GTEST_SKIP`。

## 2. Upstream Channel 迁移用例（已纳入台账但当前未通过/未完成）

来源：`test/upstream/opencv/core/channel_manifest.json`  
当前状态：`PASS_NOW = 15`，`PENDING_CHANNEL = 2`

### 2.1 OutputArray 相关（设计性不对齐，非目标）（2）

| 用例 | 当前状态 | 原因 | 处理策略 |
|---|---|---|---|
| `Core_Mat.reinterpret_OutputArray_8UC3_8SC3` | `PENDING_CHANNEL` | 采用 Mat-only 接口路线，不引入 OutputArray 兼容层 | 维持 pending（非目标） |
| `Core_Mat.reinterpret_OutputArray_8UC4_32FC1` | `PENDING_CHANNEL` | 同上 | 同上 |

### 2.2 已提升为 PASS_NOW（15）

| 用例 | 当前状态 | 说明 |
|---|---|---|
| `Core_Merge.shape_operations` | `PASS_NOW` | 已在 `mat_upstream_channel_port_test` 落地并可运行 |
| `Core_Split.shape_operations` | `PASS_NOW` | 已在 `mat_upstream_channel_port_test` 落地并可运行 |
| `Core_Merge.hang_12171` | `PASS_NOW` | 已在 `mat_upstream_channel_port_test` 落地并可运行 |
| `Core_Split.hang_12171` | `PASS_NOW` | 已在 `mat_upstream_channel_port_test` 落地并可运行 |
| `Core_Split.crash_12171` | `PASS_NOW` | 已在 `mat_upstream_channel_port_test` 落地并可运行 |
| `Core_Merge.bug_13544` | `PASS_NOW` | 已在 `mat_upstream_channel_port_test` 落地并可运行 |
| `Core_Mat.reinterpret_Mat_8UC3_8SC3` | `PASS_NOW` | 已在 `mat_upstream_channel_port_test` 落地并可运行 |
| `Core_Mat.reinterpret_Mat_8UC4_32FC1` | `PASS_NOW` | 已在 `mat_upstream_channel_port_test` 落地并可运行 |
| `Core_MatExpr.issue_16655` | `PASS_NOW` | 已在 `mat_upstream_channel_port_test` 落地并可运行 |
| `Subtract.scalarc1_matc3` | `PASS_NOW` | 已在 `mat_upstream_channel_port_test` 落地并可运行 |
| `Subtract.scalarc4_matc4` | `PASS_NOW` | 已在 `mat_upstream_channel_port_test` 落地并可运行 |
| `Compare.empty` | `PASS_NOW` | 已在 `mat_upstream_channel_port_test` 落地并可运行 |
| `Compare.regression_8999` | `PASS_NOW` | 已在 `mat_upstream_channel_port_test` 落地并可运行 |
| `Compare.regression_16F_do_not_crash` | `PASS_NOW` | 已在 `mat_upstream_channel_port_test` 落地并可运行 |
| `Core_Array.expressions` | `PASS_NOW` | 已在 `mat_upstream_channel_port_test` 落地并可运行 |

## 3. 已接线但受阻说明

- `test/core/mat_upstream_channel_port_test.cpp` 已新增 17 个上游对齐入口测试（其中 15 项已提升为 `PASS_NOW`）。
- `cvh_test_core` 已可运行；当前剩余 pending 用例均通过 `GTEST_SKIP` 显式暴露。
- 其中 OutputArray 2 项是“设计性不对齐”（Mat-only 路线下非目标），并非实现故障。
- 后续每解锁一个 API 能力，应把对应 case 从 `PENDING_CHANNEL` 提升为 `PASS_NOW` 并更新 manifest。

## 4. 维护规则

1. 每次修复一个 pending 能力后，先把对应测试改为可运行，再更新 `channel_manifest.json` 状态为 `PASS_NOW`。
2. 本文档与 `channel_manifest.json` 需同步更新，避免“文档说已修，台账仍 pending”。
