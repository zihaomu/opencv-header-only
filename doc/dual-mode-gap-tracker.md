# Dual-Mode 落地差距跟踪（CVH_LITE / CVH_FULL）

- 更新时间：2026-03-27
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
  - 该 smoke 已切为纯头文件目标（仅链接 `cvh::headers`）

6. Lite 基础设施闭环（2026-03-27）

- `Mat` 在 `CVH_LITE` 下提供头文件实现（不依赖 `src/core/mat*.cpp`）
- `system` 在 `CVH_LITE` 下提供头文件实现（`error/format/Exception` 等）
- `cvh_include_only_smoke` 新增 `Mat create/setTo/clone/pixelPtr` 运行校验

7. PR-2 dispatch 样板已启动（2026-03-27）

- `resize/cvtColor/threshold` 改为 `fallback + dispatch + backend 注册` 三层结构
- Full backend 新增 `register_all_backends()`，由 API 首次调用懒加载注册
- 新增 `cvh_resize_dispatch_lite_smoke / cvh_resize_dispatch_full_smoke` 验证 Lite/Full 语义与注册状态

## 当前差距（相对 design.md）

1. Lite 能力覆盖仍未完成

- `imgproc/imgcodecs` 最小链路已可纯头运行，但仍缺少 `GaussianBlur/boxFilter` 的 Lite fallback。

2. dispatch/registry 机制已启动但覆盖面不足

- 当前 `resize/cvtColor/threshold` 已完成样板，其他算子仍未切到统一 dispatch 管线。

3. imgproc/imgcodecs 覆盖仍有限

- 已有最小稳定 API（`resize/cvtColor/threshold/imread/imwrite`），但缺少更广算子族与行为对齐测试。

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
