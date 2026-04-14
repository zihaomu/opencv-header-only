# CVH 当前差距审计（2026-04-14）

更新时间：2026-04-14（含首轮收敛结果）  
审计范围：`opencv-header-only` 当前 `main` 分支（本地）

## 1. 本轮已完成（已验证）

本轮已落地并通过回归：

1. Lite/Full 构建隔离修复  
   - `scripts/ci_smoke.sh`、`scripts/ci_core_basic.sh` 显式加入  
     `-DCVH_BUILD_FULL_BACKEND=OFF`，避免 Lite 流水线误构建 Full。

2. 默认测试门禁补齐  
   - `CMakeLists.txt` 已将 `cvh_test_imgproc` 纳入 `add_test(...)`。

3. Full 编译阻塞修复  
   - `src/core/kernel/binary_kernel_xsimd.cpp` small-int 路径中  
     `saturate_cast<T>(long long)` 改为 `int64` 入参，消除重载歧义。

4. Full 测试稳定性修复  
   - `src/highgui/highgui.cpp`：Framebuffer 测试模式优先走 fb 分支，避免被 X11 抢占。  
   - `src/core/mat.cpp`：深度支持收口到 `[CV_8U..CV_16F]`，与合同测试一致。

5. Channel 对齐用例增量推进  
   - `merge/split` 已补齐 core API，并完成 6 个 upstream case（含 ROI/stride 路径）提升为 `PASS_NOW`。  
   - `subtract(double, Mat)`/`subtract(Mat, double)` 已补齐，`Subtract.scalarc1_matc3` 提升为 `PASS_NOW`。  
   - compare 语义已补齐：`compare(empty,empty)->empty` 与 `compare(Mat,Mat)` 的 `CV_16F` 异常路径对齐，2 个 compare case 提升为 `PASS_NOW`。  
   - `Mat::reinterpret`（Mat-only）已补齐，2 个 reinterpret-Mat case 提升为 `PASS_NOW`。  
   - `PASS_NOW/PENDING_CHANNEL` 已从 `4/13` 推进至 `15/2`（剩余 2 项均为 OutputArray 非目标）。

回归结果：

- `./scripts/ci_smoke.sh`：通过
- `./scripts/ci_core_basic.sh`：通过
- `cmake --build build-full-check --target test`：`13/13` 通过

---

## 2. 当前剩余差距（主线）

当前已经没有 P0 级构建阻塞，核心剩余差距集中在覆盖度与文档一致性。

## P1（功能覆盖）

### GAP-P1-1：Core channel 兼容仍有 2 项 pending（共 17 项）

- 证据：
  - `test/upstream/opencv/core/channel_manifest.json`
  - `scripts/verify_opencv_core_channel_cases.py` 输出：`PASS_NOW=15, PENDING_CHANNEL=2`
- 缺口分组：
  - `OutputArray`（2，设计性不对齐）
- 影响：
  - Mat-only 可实现项已收敛完成，剩余缺口仅为 OutputArray 非目标项
- DoD：
  - 维持 `PASS_NOW=15, PENDING_CHANNEL=2(非目标)` 的稳定状态
  - 若策略切换到 OpenCV 接口级对齐，再引入薄 OutputArray 适配层

### GAP-P1-2：dispatch 样板覆盖面仍局部

- 已有：`resize/cvtColor/threshold` 三层（fallback + dispatch + backend 注册）
- 未完成：更多算子统一进 dispatch 管线
- DoD：
  - 下一批建议：`blur/GaussianBlur/boxFilter`
  - Lite/Full 同输入合同测试齐备

## P2（工程一致性）

### GAP-P2-1：文档引用与仓库实际不一致

- 现象：
  - 多处文档引用缺失文件（如 `doc/phase0-execution-plan.md`,
    `doc/phase1-execution-plan.md`, `doc/src-core-migration-tracker.md`）
- 影响：
  - 计划跟踪链路断裂，协作成本上升
- DoD：
  - 清理失效引用或补齐缺失文档
  - 维护单一“文档索引真源”

### GAP-P2-2：对外完成度口径仍需统一

- 现象：
  - 代码已具备 dual-mode + header-first 能力，但 README 与计划文档未同步“当前已完成项/已知限制/推荐构建矩阵”
- DoD：
  - README 增加“当前完成度声明 + 已知限制 + 推荐构建矩阵”

---

## 3. 建议下一步（按优先级）

1. 维持 Mat-only 对齐口径（`PASS_NOW=15`），并将剩余 2 项按“非目标”长期跟踪
2. 再扩展 dispatch 第二批算子（`blur/GaussianBlur/boxFilter`）
3. 最后做文档索引与 README 口径收口

---

## 4. 验收指标（当前建议）

- Lite job：
  - `CVH_BUILD_FULL_BACKEND=OFF`
  - `ctest` 覆盖 smoke + imgproc + imgcodecs + highgui(lite)
- Full job：
  - 默认 Full 编译通过
  - Full `ctest` 全通过
- Compat job：
  - `verify_opencv_core_channel_cases.py` 必跑
  - 阶段门禁：`CVH_MIN_PASS_NOW>=1`（后续逐步提升）

---

## 5. 本轮修改涉及文件

- `CMakeLists.txt`
- `scripts/ci_smoke.sh`
- `scripts/ci_core_basic.sh`
- `include/cvh/core/basic_op.h`
- `include/cvh/core/mat.h`
- `include/cvh/core/mat_lite_impl.h`
- `src/core/basic_op_scalar.cpp`
- `src/core/kernel/binary_kernel_xsimd.cpp`
- `src/core/mat.cpp`
- `src/highgui/highgui.cpp`
- `README.md`
- `test/core/mat_upstream_channel_port_test.cpp`
- `test/upstream/opencv/core/channel_manifest.json`
- `test/failing-tests.md`
- `doc/current-gap-audit-2026-04-14.md`
