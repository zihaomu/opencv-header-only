# Dual-Mode 落地差距跟踪（CVH_LITE / CVH_FULL）

- 更新时间：2026-03-25
- 目标来源：`doc/design.md`

## 本次已落地（PR-0 + PR-1 Lite 最小链路）

1. 模式基础设施

- 新增 `include/cvh/detail/config.h`
- 默认模式：未定义时进入 `CVH_LITE`
- Full 模式：构建 Full backend 目标时注入 `CVH_FULL`
- 互斥约束：`CVH_LITE` 与 `CVH_FULL` 同时开启会编译报错

2. 构建图切分

- 新增 `CVH_BUILD_FULL_BACKEND` 开关
- `cvh_test_core` 改为仅在 Full backend 存在时构建
- 新增模式 smoke：
  - `cvh_mode_lite_smoke`
  - `cvh_mode_full_smoke`

3. 当前阻塞修复

- `Error::Code` 增加 `StsOutOfMem` / `StsBadType` 兼容别名
- 解除 `cvh_test_core` 编译阻塞

4. 测试台账一致性修正

- `Subtract.scalarc1_matc3` / `Subtract.scalarc4_matc4` 从“临时可运行替代实现”改为显式 pending（`GTEST_SKIP`）
- `test/failing-tests.md` 已同步更新构建状态

5. Lite API 最小链路（fallback API 已接线）

- 新增 `imgproc` fallback API：
  - `resize`
  - `cvtColor`（`BGR2GRAY` / `GRAY2BGR`）
  - `threshold`（`THRESH_BINARY` / `THRESH_BINARY_INV`）
- 新增 `imgcodecs` fallback API：
  - `imread`
  - `imwrite`
- 新增 Lite 端到端 smoke：
  - `cvh_lite_pipeline_smoke`（`imread -> resize -> cvtColor -> threshold -> imwrite`）
  - 现阶段该 smoke 仍链接 `cvh::legacy_core`（根因：`Mat`/`system` 尚未完全 header-only）

## 当前差距（相对 design.md）

1. Lite 真实无链接运行仍未达成

- `imgproc/imgcodecs` fallback API 已落地，但运行仍依赖 `cvh::legacy_core`（`Mat` 当前仍主要在 `src/core/*.cpp`）。
- 仍缺少 `GaussianBlur/boxFilter` 的 Lite fallback 落地。

2. dispatch/registry 机制尚未开始

- 目前仍是传统直接实现方式，尚未引入算子级 fallback + backend 注册模型。

3. imgproc/imgcodecs 仍是占位

- `include/cvh/imgproc`、`include/cvh/imgcodecs` 只有 readme，无稳定 API。

4. upstream channel 案例仍全部处于 manifest pending

- `PASS_NOW=0`，`PENDING_CHANNEL=17`。

## 下一步分批建议

### PR-1（最小可运行链路）

- 在 Lite 中打通一个最小 imgproc/imgcodecs 闭环（建议：`imread` + `resize` + `cvtColor(BGR2GRAY)` + `threshold` + `imwrite`）
- 增加 Lite 路径 E2E 测试（二进制可执行，不依赖 Full backend）

### PR-2（dispatch 样板）

- 先选一个算子（建议 `resize`）做完整样板：
  - API 层
  - fallback 实现（include）
  - dispatch 函数指针/注册入口
  - Full backend 注册实现（src）
- 增加 Lite/Full 等价语义测试（同输入同输出容差）

### PR-3（channel 关键能力收敛）

- 优先解锁 `merge/split`、`compare`、`Mat::reinterpret`
- 每解锁一项，升级对应 upstream case：`PENDING_CHANNEL -> PASS_NOW`

## 大坑预判与规避

1. 静态注册顺序风险

- 坑：backend 注册时序不确定导致偶发 fallback。
- 规避：提供显式 `register_all_backends()` 入口，测试中强制调用。

2. Lite/Full 语义分叉风险

- 坑：参数检查、边界行为在两条路径不一致。
- 规避：为每个 dispatch 算子建立 shared contract tests（同一测试集跑 Lite/Full）。

3. ROI/stride/channel 组合缺陷

- 坑：非连续内存 + 多通道组合最容易出现隐式内存越界。
- 规避：所有 channel 相关 API 必须包含 ROI/non-continuous 组合测试。

4. CI 覆盖错觉

- 坑：只跑 smoke 但误以为 Lite/Full 都健康。
- 规避：CI 拆分 `lite-smoke`、`full-core`、`compat-upstream` 三条必跑流水线。
