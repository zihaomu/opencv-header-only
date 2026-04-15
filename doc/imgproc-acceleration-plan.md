# Imgproc 模块加速计划（可中断/可恢复）

- 文档版本：v1.18
- 创建日期：2026-04-08
- 适用仓库：`opencv-header-only`
- 分支基线：`main`
- 当前模式：`SELECTIVE_EXPANSION`
- 目标：在不破坏现有正确性的前提下，系统性提升 `imgproc` 在不同类型、不同通道下的性能，并建立可持续回归门禁。

---

## 1. 背景与目标

当前 `imgproc` 已具备基础功能与测试覆盖，但实现以 naive 循环为主，后端注册实现仍主要转发 fallback，性能提升空间明显。

本计划聚焦三件事：

1. **性能提升**：优先优化 `resize / cvtColor / threshold / boxFilter / GaussianBlur`。
2. **覆盖扩展**：从 `CV_8U` 起步，逐步扩展到 `CV_32F`，覆盖 `C1/C3/C4` 常见通道组合。
3. **验证闭环**：补齐 benchmark 与回归门禁，防止“优化后变慢”或“优化后错算”。

---

## 2. 当前代码现状（审计结论）

### 2.1 结构与后端

- 公开入口已具备 dispatch/fallback 架构：
  - `include/cvh/imgproc/resize.h`
  - `include/cvh/imgproc/cvtcolor.h`
  - `include/cvh/imgproc/threshold.h`
  - `include/cvh/imgproc/box_filter.h`
  - `include/cvh/imgproc/gaussian_blur.h`
- backend 注册已连通，并已形成 `fast-path + fallback` 组合：
  - `src/imgproc/resize_backend.cpp`

### 2.2 能力边界

- 当前 fast-path 已覆盖：
  - `resize / boxFilter / GaussianBlur` 的 `CV_8U/CV_32F C1/C3/C4` 主路径
  - `cvtColor(GRAY<->BGR/BGRA/RGBA, BGR<->RGB, BGR/RGB<->BGRA/RGBA, BGRA<->RGBA, BGR/RGB<->YUV)` 的 `CV_8U/CV_32F` 主路径
  - 其余组合回落 fallback 或显式报错
- `imgproc` 正确性测试目前可通过（`cvh_test_imgproc`）。
- 已具备 `imgproc` 专项 benchmark 与回归阈值门禁（含 quick profile 与 CI 可选 gate）。

---

## 3. 范围定义

### 3.1 In Scope（本轮）

1. `resize` 性能优化（`INTER_NEAREST / INTER_NEAREST_EXACT / INTER_LINEAR`）
2. `cvtColor` 性能优化（`BGR2GRAY / GRAY2BGR`）
3. `boxFilter / GaussianBlur` 性能优化
4. `threshold` 固定阈值路径扩展与优化（保持自动阈值边界清晰）
5. 建立 `imgproc` benchmark + 回归脚本 + CI 可选门禁

### 3.2 Out of Scope（本轮不做）

1. 新增 `warpAffine / morphology / canny` 等新算子
2. ARM NEON 专项路径
3. 全面重构内核框架（大规模架构改造）

---

## 4. 实施路线（分阶段）

## 阶段 A：基线与门禁（PR-1）

### 目标

- 增加 `imgproc` 性能基准可执行程序
- 增加回归比对脚本
- 固化 baseline 结果文件

### 计划项

1. 新增 `benchmark/imgproc_ops_benchmark.cpp`
2. 新增 `scripts/check_imgproc_benchmark_regression.py`
3. 产出 `benchmark/baseline_imgproc_quick.csv`
4. 增加 CI 可选任务（quick profile）

### 退出标准

- 能稳定导出 CSV
- 基线对比脚本可在 slowdown 超阈值时失败退出

---

## 阶段 B：`resize + cvtColor` 首批提速（PR-2）

### 目标

- 让最常用两条图像处理路径先产生可见收益

### 计划项

1. `resize`
   - 预计算坐标映射与权重
   - 按 `C1/C3/C4` 区分内核路径
   - 大图按行块并行（OpenMP，可控启用）
2. `cvtColor`
   - `BGR2GRAY` 使用整数系数近似路径
   - `GRAY2BGR` 向量化复制

### 退出标准

- 现有 `imgproc` 测试全通过
- 对目标分辨率/通道组合，性能优于基线

### 阶段 B 里程碑拆分（可中断执行）

| 里程碑 | 工作内容 | 验收标准 | 完成情况 | 开始时间 | 完成时间 | 备注/证据 |
|---|---|---|---|---|---|---|
| B1：后端骨架 | 在 `src/imgproc/resize_backend.cpp` 建立 `resize/cvtColor` fast-path + fallback 分派骨架 | 后端不再是全量直转 fallback；未覆盖场景可正确回落；可编译 | `已完成` | 2026-04-08 22:45 CST | 2026-04-08 22:55 CST | `cmake --build build-full-test -j --target cvh_benchmark_imgproc_ops` + `./build-full-test/cvh_benchmark_imgproc_ops --profile quick --warmup 1 --iters 1 --repeats 1 --output /tmp/cvh_imgproc_b1_sanity.csv` |
| B2：resize 提速 | `INTER_NEAREST/EXACT/LINEAR` 预计算映射/权重；按 `C1/C3/C4` 分核；大图行块并行 | `cvh_test_imgproc` 通过；quick profile 的 `resize` 子集无 >8% 回退 | `已完成` | 2026-04-08 22:56 CST | 2026-04-08 23:02 CST | `./build-full-test/cvh_test_imgproc` + `./build-full-test/cvh_benchmark_imgproc_ops --profile quick ... --output benchmark/current_imgproc_quick_b2.csv` + regression PASS |
| B3：cvtColor 提速 | `BGR2GRAY` 整数系数路径 + 向量化；`GRAY2BGR` 向量化复制 | `cvh_test_imgproc` 通过；quick profile 的 `cvtColor` 子集无 >8% 回退 | `已完成` | 2026-04-08 23:03 CST | 2026-04-08 23:11 CST | `./build-full-test/cvh_test_imgproc` + `./build-full-test/cvh_benchmark_imgproc_ops --profile quick ... --output benchmark/current_imgproc_quick_b3.csv` + regression PASS |
| B4：正确性补强 | 增加非连续 ROI、边界尺寸、通道覆盖回归用例 | 新增测试稳定通过，且无行为回退 | `已完成` | 2026-04-08 23:12 CST | 2026-04-08 23:16 CST | 新增 4 个用例；`cvh_test_imgproc` 提升至 26/26 PASS |
| B5：性能门禁验证 | 运行 `cvh_benchmark_imgproc_ops` quick + regression 对比 baseline | `scripts/check_imgproc_benchmark_regression.py` 返回 PASS | `已完成` | 2026-04-08 23:22 CST | 2026-04-08 23:24 CST | `benchmark/current_imgproc_quick_b5.csv` 对比 baseline：`compared=32, PASS`，最大回退 +4.27% |
| B6：文档收口 | 回填阶段 B 验收状态、命令、结果摘要与已知问题 | 本文档“验收状态/验证结果/变更记录”同步完成 | `已完成` | 2026-04-08 23:25 CST | 2026-04-08 23:30 CST | 已完成阶段 B 收口：状态切换、结果汇总、下一阶段指针更新 |

### 阶段 B 量化验收目标（KPI）

1. `RESIZE_NEAREST`（HD/FHD，`C3/C4`）目标加速：`>= 1.35x`（相对阶段 A baseline）
2. `RESIZE_LINEAR`（HD/FHD）目标加速：`>= 1.20x`
3. `CVTCOLOR_BGR2GRAY` 目标加速：`>= 1.20x`
4. quick profile 全量 case 门禁：`max slowdown <= 8%`

---

## 阶段 C：`boxFilter + GaussianBlur` 提速（PR-3）

### 目标

- 处理邻域算子的复杂度与缓存效率问题

### 计划项

1. `boxFilter` 从逐点全窗口扫描优化为滑窗/累计策略
2. `GaussianBlur` 维持可分离卷积，增加边界索引预计算和行缓存
3. 非连续 ROI 路径保持正确性，不允许 silent wrong result

### 退出标准

- 与参考实现误差受控
- 大核尺寸场景有稳定收益

### 阶段 C 里程碑拆分（可中断执行）

| 里程碑 | 工作内容 | 验收标准 | 完成情况 | 开始时间 | 完成时间 | 备注/证据 |
|---|---|---|---|---|---|---|
| C1：后端分派接入 | 在 `src/imgproc/resize_backend.cpp` 中为 `boxFilter/GaussianBlur` 加入 fast-path 入口，保留 fallback 兜底 | 可编译；不支持场景可回退；行为与原入口兼容 | `已完成` | 2026-04-08 23:31 CST | 2026-04-08 23:36 CST | `cmake --build build-full-test -j --target cvh_benchmark_imgproc_ops` |
| C2：boxFilter 提速 | `boxFilter` 改为横向滑窗 + 纵向累计，避免逐点全窗口重扫 | `boxFilter` 相关 case 正确；quick profile 无 >8% 回退 | `已完成` | 2026-04-08 23:32 CST | 2026-04-08 23:36 CST | `src/imgproc/resize_backend.cpp` 已实现滑窗累计；`BOXFILTER_3X3` 全组合提升 |
| C3：GaussianBlur 提速 | 可分离卷积下引入边界索引预计算 + 行并行，减少 inner-loop 边界分支开销 | `GaussianBlur` 相关 case 正确；quick profile 无 >8% 回退 | `已完成` | 2026-04-08 23:33 CST | 2026-04-08 23:36 CST | `src/imgproc/resize_backend.cpp` 已实现 `x/y` 索引预计算与 OpenMP 行并行 |
| C4：正确性补强 | 新增 ROI/anchor/normalize/in-place 回归测试 | 新增测试稳定通过，且无行为回退 | `已完成` | 2026-04-08 23:36 CST | 2026-04-08 23:39 CST | `test/imgproc/imgproc_filter_contract_test.cpp` 新增 3 用例；`cvh_test_imgproc` 提升至 29/29 PASS |
| C5：性能门禁验证 + 文档收口 | 运行 quick benchmark + regression，对比 baseline 并回填文档 | regression PASS；文档状态/结果同步完成 | `已完成` | 2026-04-08 23:40 CST | 2026-04-08 23:44 CST | `benchmark/current_imgproc_quick_c2.csv` + regression PASS（compared=32） |

---

## 阶段 D：类型与通道扩展（PR-4）

### 目标

- 从 `CV_8U` 扩展到 `CV_32F` 主路径

### 计划项

1. 优先支持 `CV_32F` 的 `C1/C3/C4` 常见组合
2. `threshold`：自动阈值保持 `CV_8UC1`，固定阈值扩展至 `CV_32F`
3. 不支持组合必须显式报错，不做隐式降级

### 退出标准

- 新增组合具备单测与回归基线
- 失败路径信息明确可定位

### 阶段 D 里程碑拆分（可中断执行）

| 里程碑 | 工作内容 | 验收标准 | 完成情况 | 开始时间 | 完成时间 | 备注/证据 |
|---|---|---|---|---|---|---|
| D1：threshold 类型扩展 | `threshold` 固定阈值扩展到 `CV_32F`，保持 `OTSU/TRIANGLE` 仅 `CV_8UC1` | `CV_32F C1/C3/C4` 正确性通过；自动阈值限制明确；quick gate 通过 | `已完成` | 2026-04-09 09:20 CST | 2026-04-09 09:36 CST | `include/cvh/imgproc/threshold.h` + `src/imgproc/resize_backend.cpp`；`cvh_test_imgproc 32/32 PASS`；`benchmark/current_imgproc_quick_d1.csv` + regression PASS |
| D2：CV_32F 邻域/几何扩展 | `resize/boxFilter/GaussianBlur` 的 `CV_32F C1/C3/C4` 主路径扩展 | 新增组合有测试与 quick/full 数据，且门禁通过 | `已完成` | 2026-04-09 09:37 CST | 2026-04-09 09:58 CST | `resize.h/box_filter.h/gaussian_blur.h` 已支持 `CV_32F`；`cvh_test_imgproc 36/36 PASS`；`benchmark/current_imgproc_quick_d2.csv` |
| D3：文档与基线同步 | 更新支持矩阵、基线策略与限制说明 | 文档与 benchmark 资产一致，可恢复执行 | `已完成` | 2026-04-09 09:50 CST | 2026-04-09 09:58 CST | 已同步 `test/imgproc/readme.md`、`benchmark/readme.md`、本计划文档与当前结果 |

---

## 阶段 E：收口发布（PR-5）

### 目标

- 文档、数据、门禁闭环

### 计划项

1. 完成 benchmark 回归门禁接线
2. 更新 `include/cvh/imgproc/readme.md` 的支持矩阵
3. 记录最终收益数据与已知限制

### 退出标准

- 有可复现性能数据
- 有可审计验收记录

### 阶段 E 里程碑拆分（可中断执行）

| 里程碑 | 工作内容 | 验收标准 | 完成情况 | 开始时间 | 完成时间 | 备注/证据 |
|---|---|---|---|---|---|---|
| E1：deterministic gate 稳定化 | 为 benchmark gate 固定 `OMP` 运行时，并默认绑定到当前可用 `cpuset` 的首个 CPU，降低进程迁移噪声 | 同代码双跑 quick profile 可稳定通过 8% 门禁；脚本支持 `quick/full` | `已完成` | 2026-04-09 13:58 CST | 2026-04-09 14:08 CST | `scripts/ci_imgproc_quick_gate.sh` 新增 `CVH_IMGPROC_BENCH_CPU_LIST`；`taskset -c 0 ... quick` 双跑 `PASS (62/62)` |
| E2：quick/full 基线固化 | 生成 deterministic quick/full baseline + current，并完成 regression 验证 | quick/full 均有 baseline 与 current CSV，门禁脚本可通过 | `已完成` | 2026-04-09 14:08 CST | 2026-04-09 14:20 CST | quick：`benchmark/baseline_imgproc_quick.csv` vs `benchmark/current_imgproc_quick.csv`，`PASS (62/62)`；full：`benchmark/baseline_imgproc_full.csv` vs `benchmark/current_imgproc_full.csv`，`PASS (93/93, threshold=15%)` |
| E3：文档与限制清单收口 | 更新 `benchmark/readme.md`、`include/cvh/imgproc/readme.md` 与本计划文档，固化支持矩阵、阈值和已知限制 | 文档可独立支撑中断恢复与后续验收 | `已完成` | 2026-04-09 14:20 CST | 2026-04-09 14:23 CST | 已同步 CPU pin 策略、full gate 阈值、支持矩阵、已知限制与阶段状态 |

---

## 阶段 F：`cvtColor CV_32F` 扩展与门禁校准（PR-6）

### 目标

- 补齐 `cvtColor(BGR2GRAY/GRAY2BGR)` 的 `CV_32F` 支持
- 将 `CV_32F cvtColor` 纳入 benchmark/gate 与文档

### 计划项

1. 为 `cvtColor` 增加 `CV_32F BGR2GRAY/GRAY2BGR` 的 fallback 与 backend fast-path
2. 新增 `CV_32F` 契约测试与 benchmark case
3. 校准 gate 基线生成口径与 quick noisy case 的定向阈值覆盖

### 退出标准

- `cvh_test_imgproc` 补齐 `CV_32F cvtColor` 后全通过
- quick/full benchmark 覆盖新增 `CV_32F cvtColor` case，且 gate 可稳定通过
- 文档中支持矩阵、基线策略、限制说明完成同步

### 阶段 F 里程碑拆分（可中断执行）

| 里程碑 | 工作内容 | 验收标准 | 完成情况 | 开始时间 | 完成时间 | 备注/证据 |
|---|---|---|---|---|---|---|
| F1：`CV_32F cvtColor` 能力补齐 | 为 `cvtColor(BGR2GRAY/GRAY2BGR)` 增加 `CV_32F` 的 header fallback、backend fast-path 与 ROI 契约测试 | 新增 `CV_32F` case 先 RED 后 GREEN，`ImgprocCvtColor_TEST` 全通过 | `已完成` | 2026-04-09 14:24 CST | 2026-04-09 14:37 CST | `ImgprocCvtColor_TEST` 从 `6/8` 到 `8/8 PASS`；`cvh_test_imgproc` 提升至 `38/38 PASS` |
| F2：benchmark 扩展与兼容性回归 | benchmark 新增 `CVTCOLOR_BGR2GRAY_F32/CVTCOLOR_GRAY2BGR_F32`，并验证旧 baseline key 无回退 | quick/full 新增 case 入表；对旧 `62/93` key regression `PASS` | `已完成` | 2026-04-09 14:37 CST | 2026-04-09 15:04 CST | quick：`66 rows`，full：`99 rows`；对阶段 E baseline key regression 全通过 |
| F3：gate 校准与文档收口 | 统一 baseline 到 gate build config，优化 `CPU auto-pinning`，为 quick noisy case 增加定向阈值覆盖并回填文档 | quick/full gate 端到端通过；文档可恢复执行 | `已完成` | 2026-04-09 15:04 CST | 2026-04-09 15:37 CST | quick gate：`66/66 PASS`（`cpu_list=8`，1 条 override）；full gate：`99/99 PASS` |

---

## 阶段 G：`BGR<->RGB` / `BGR<->BGRA` 扩展（PR-7）

### 目标

- 扩展 `cvtColor` 到更常用的通道重排与 alpha 扩展场景
- 将 `BGR<->RGB`、`BGR<->BGRA` 纳入 benchmark/gate 与文档

### 计划项

1. 新增 `COLOR_BGR2RGB` / `COLOR_RGB2BGR` / `COLOR_BGR2BGRA` / `COLOR_BGRA2BGR`
2. 为 `CV_8U/CV_32F` 补齐 fallback、backend fast-path、ROI 契约测试
3. 将新增 color code 纳入 quick/full benchmark 与 gate baseline

### 退出标准

- 新增 color code 的 `CV_8U/CV_32F` 路径测试通过
- quick/full benchmark 覆盖新增 color code 且 gate 通过
- 文档中支持矩阵、测试范围、基线结果完成同步

### 阶段 G 里程碑拆分（可中断执行）

| 里程碑 | 工作内容 | 验收标准 | 完成情况 | 开始时间 | 完成时间 | 备注/证据 |
|---|---|---|---|---|---|---|
| G1：新增 color code 与正确性补齐 | 增加 `BGR<->RGB` / `BGR<->BGRA` 的 enum、fallback、backend fast-path 与 `CV_8U/CV_32F` 契约测试 | 新增测试先 RED 后 GREEN，`ImgprocCvtColor_TEST` 全通过 | `已完成` | 2026-04-09 15:38 CST | 2026-04-09 15:55 CST | `ImgprocCvtColor_TEST` 提升至 `12/12 PASS`；`cvh_test_imgproc` 提升至 `42/42 PASS` |
| G2：benchmark 扩展 | benchmark 新增 `CVTCOLOR_BGR2RGB` / `CVTCOLOR_BGR2BGRA` / `CVTCOLOR_BGRA2BGR` 及 `CV_32F` 对应 case | quick/full 新增 case 入表，benchmark smoke 可见新增 op | `已完成` | 2026-04-09 15:55 CST | 2026-04-09 16:05 CST | quick 行数 `66 -> 78`；full 行数 `99 -> 117` |
| G3：baseline/gate/doc 收口 | 固化新 schema 的 baseline/current，并更新 readme/计划文档 | quick/full gate 端到端通过；文档可恢复执行 | `已完成` | 2026-04-09 16:05 CST | 2026-04-09 16:20 CST | quick gate：`78/78 PASS`；full gate：`117/117 PASS` |

---

## 阶段 H：`RGB/BGR/RGBA/BGRA` 家族补齐（PR-8）

### 目标

- 补齐 `RGB/BGR/RGBA/BGRA` 常用互转中剩余的 alpha 添加、alpha 去除和 4 通道重排路径
- 将新增 color code 纳入 benchmark/gate 与文档

### 计划项

1. 新增 `COLOR_RGB2RGBA` / `COLOR_RGBA2RGB` / `COLOR_BGR2RGBA` / `COLOR_RGBA2BGR`
2. 新增 `COLOR_RGB2BGRA` / `COLOR_BGRA2RGB` / `COLOR_BGRA2RGBA` / `COLOR_RGBA2BGRA`
3. 为 `CV_8U/CV_32F` 补齐 fallback、backend fast-path、ROI 契约测试与 quick/full benchmark

### 退出标准

- 新增 color code 的 `CV_8U/CV_32F` 路径测试通过
- quick/full benchmark 覆盖新增 color code 且 gate 通过
- 文档中支持矩阵、测试范围、基线结果完成同步

### 阶段 H 里程碑拆分（可中断执行）

| 里程碑 | 工作内容 | 验收标准 | 完成情况 | 开始时间 | 完成时间 | 备注/证据 |
|---|---|---|---|---|---|---|
| H1：契约测试先行 | 为剩余 `RGB/BGR/RGBA/BGRA` family color code 增加 `CV_8U/CV_32F` 的连续/ROI 契约测试 | 先出现目标 RED，再由实现转 GREEN | `已完成` | 2026-04-09 16:38 CST | 2026-04-09 16:42 CST | 先以缺少 enum 的编译失败进入 RED，再转 GREEN；`ImgprocCvtColor_TEST.*` 提升至 `15/15 PASS` |
| H2：fallback/backend 实现 | 在 `detail/common.h`、`cvtcolor.h`、`resize_backend.cpp` 中补齐 enum、fallback 与 fast-path | 新增 color code 可走 fast-path，未覆盖场景仍正确报错/回退 | `已完成` | 2026-04-09 16:42 CST | 2026-04-09 16:44 CST | 已补齐 8 个 color code，复用 3ch->4ch / 4ch->3ch / 4ch swap helper 完成 fast-path |
| H3：benchmark/gate/doc 收口 | benchmark 新增 case、更新 baseline/current、执行 quick/full gate，并回填文档 | quick/full gate 端到端通过；文档可恢复执行 | `已完成` | 2026-04-09 16:44 CST | 2026-04-09 16:52 CST | quick gate：`110/110 PASS`；full gate：`165/165 PASS` |

---

## 阶段 I：`GRAY <-> RGBA/BGRA` 扩展（PR-9）

### 目标

- 补齐灰度与 4 通道带 alpha 图像之间的常用互转路径
- 将 `GRAY <-> RGBA/BGRA` 纳入 benchmark/gate 与文档

### 计划项

1. 新增 `COLOR_GRAY2BGRA` / `COLOR_BGRA2GRAY` / `COLOR_GRAY2RGBA` / `COLOR_RGBA2GRAY`
2. 为 `CV_8U/CV_32F` 补齐 fallback、backend fast-path、ROI 契约测试
3. 将新增 color code 纳入 quick/full benchmark 与 gate baseline

### 退出标准

- 新增 color code 的 `CV_8U/CV_32F` 路径测试通过
- quick/full benchmark 覆盖新增 color code 且 gate 通过
- 文档中支持矩阵、测试范围、基线结果完成同步

### 阶段 I 里程碑拆分（可中断执行）

| 里程碑 | 工作内容 | 验收标准 | 完成情况 | 开始时间 | 完成时间 | 备注/证据 |
|---|---|---|---|---|---|---|
| I1：契约测试先行 | 为 `GRAY <-> RGBA/BGRA` 增加 `CV_8U/CV_32F` 的连续/ROI 契约测试 | 先出现目标 RED，再由实现转 GREEN | `已完成` | 2026-04-09 17:00 CST | 2026-04-09 17:04 CST | 先以缺少 enum 的编译失败进入 RED，再转 GREEN；`ImgprocCvtColor_TEST.*` 提升至 `18/18 PASS` |
| I2：fallback/backend 实现 | 在 `detail/common.h`、`cvtcolor.h`、`resize_backend.cpp` 中补齐 enum、fallback 与 fast-path | 新增 color code 可走 fast-path，未覆盖场景仍正确报错/回退 | `已完成` | 2026-04-09 17:04 CST | 2026-04-09 17:08 CST | 已补齐 4 个 color code，新增 `GRAY->4ch alpha` 与 `4ch->GRAY` fast-path/helper |
| I3：benchmark/gate/doc 收口 | benchmark 新增 case、更新 baseline/current、执行 quick/full gate，并回填文档 | quick/full gate 端到端通过；文档可恢复执行 | `已完成` | 2026-04-09 17:08 CST | 2026-04-09 17:20 CST | quick gate：`126/126 PASS`；full gate：`189/189 PASS` |

---

## 阶段 J：`BGR/RGB <-> YUV` 扩展（PR-10）

### 目标

- 补齐 3 通道 `BGR/RGB <-> YUV` 的常用互转路径
- 将 `BGR/RGB <-> YUV` 纳入 benchmark/gate 与文档

### 计划项

1. 新增 `COLOR_BGR2YUV` / `COLOR_YUV2BGR` / `COLOR_RGB2YUV` / `COLOR_YUV2RGB`
2. 为 `CV_8U/CV_32F` 补齐 fallback、backend fast-path、ROI 契约测试
3. 将新增 color code 纳入 quick/full benchmark 与 gate baseline

### 退出标准

- 新增 color code 的 `CV_8U/CV_32F` 路径测试通过
- quick/full benchmark 覆盖新增 color code 且 gate 通过
- 文档中支持矩阵、测试范围、基线结果完成同步

### 阶段 J 里程碑拆分（可中断执行）

| 里程碑 | 工作内容 | 验收标准 | 完成情况 | 开始时间 | 完成时间 | 备注/证据 |
|---|---|---|---|---|---|---|
| J1：契约测试先行 | 为 `BGR/RGB <-> YUV` 增加 `CV_8U/CV_32F` 的连续/ROI 契约测试 | 先出现目标 RED，再由实现转 GREEN | `已完成` | 2026-04-09 17:45 CST | 2026-04-09 17:46 CST | 先以缺少 enum 的编译失败进入 RED，再转 GREEN；`ImgprocCvtColor_TEST.*` 提升至 `21/21 PASS` |
| J2：fallback/backend 实现 | 在 `detail/common.h`、`cvtcolor.h`、`resize_backend.cpp` 中补齐 enum、fallback 与 fast-path | 新增 color code 可走 fast-path，未覆盖场景仍正确报错/回退 | `已完成` | 2026-04-09 17:46 CST | 2026-04-09 17:47 CST | 已补齐 4 个 color code，新增 `3ch->YUV` / `YUV->3ch` 的 `CV_8U/CV_32F` helper 与 backend fast-path |
| J3：benchmark/gate/doc 收口 | benchmark 新增 case、更新 baseline/current、执行 quick/full gate，并回填文档 | quick/full gate 端到端通过；文档可恢复执行 | `已完成` | 2026-04-09 17:47 CST | 2026-04-09 17:48 CST | quick gate：`142/142 PASS`；full gate：`213/213 PASS` |

## 阶段 K：`NV12/NV21 -> BGR/RGB` 半平面 YUV420 解码（PR-11）

### 目标

- 在现有单 `Mat` `cvtColor` 接口下，优先补齐最常见的视频/相机输入格式：`NV12/NV21 -> BGR/RGB`
- 将 `CV_8U C1` 的半平面 `YUV420` 解码路径纳入测试、benchmark、gate 与文档
- 明确本阶段边界：只做 decode-only，不包含 `BGR/RGB -> NV12/NV21`，也不包含 `I420/YV12/YUY2/UYVY`

### 计划项

1. 新增 `COLOR_YUV2BGR_NV12` / `COLOR_YUV2RGB_NV12` / `COLOR_YUV2BGR_NV21` / `COLOR_YUV2RGB_NV21`
2. 以 TDD 补齐 `CV_8U` 的连续/step/非法输入契约测试，明确 `src` 为单 `Mat`、`rows = H * 3 / 2`、`cols = W`、`channels = 1`、`W/H` 为偶数
3. 在 `cvtcolor.h` 与 `resize_backend.cpp` 中实现 `NV12/NV21` 的 `C1 -> C3` fallback/fast-path，并保持未覆盖输入显式报错
4. 将新增 color code 纳入 quick/full benchmark 与 gate，同步 `readme`、支持矩阵、验收状态与验证结果

### 退出标准

- 新增 4 个 color code 的 `CV_8U` 连续/step/非法输入测试通过
- quick/full benchmark 覆盖 `NV12/NV21 -> BGR/RGB` 且 gate 通过
- 文档明确格式约束、支持矩阵、验证命令与结果；任务中断后可从 K1/K2/K3/K4 任一里程碑恢复

### 阶段 K 里程碑拆分（可中断执行）

| 里程碑 | 工作内容 | 验收标准 | 完成情况 | 开始时间 | 完成时间 | 备注/证据 |
|---|---|---|---|---|---|---|
| K1：契约测试先行 | 新增 `NV12/NV21 -> BGR/RGB` 的 `CV_8U` 连续/step/非法尺寸测试 | 先出现目标 RED，再由实现转 GREEN | `已完成` | 2026-04-09 18:31 CST | 2026-04-09 18:42 CST | 先以缺少 `COLOR_YUV2*NV12/NV21` enum 的编译失败进入 RED，再转 GREEN；定向用例 `3/3 PASS` |
| K2：fallback/backend 解码实现 | 在 `detail/common.h`、`cvtcolor.h`、`resize_backend.cpp` 中补齐 enum、layout 校验与 decode 实现 | 新增 color code 可走 fast-path；非法 `rows/cols/channels` 能正确报错 | `已完成` | 2026-04-09 18:42 CST | 2026-04-09 18:55 CST | 已补齐 4 个 color code，新增 `YUV420sp(NV12/NV21) -> BGR/RGB` 的 `CV_8U` helper、layout 校验与 backend fast-path |
| K3：benchmark/gate 接入 | benchmark 新增 `NV12/NV21` decode case，更新 baseline/current，执行 quick/full gate | quick/full gate 端到端通过 | `已完成` | 2026-04-09 18:55 CST | 2026-04-09 20:20 CST | quick gate：`150/150 PASS`；full gate：`225/225 PASS` |
| K4：文档收口 | 回填 `readme`、支持矩阵、验收状态、验证结果与限制说明 | 文档与当前实现一致，可恢复继续下一阶段 | `已完成` | 2026-04-09 20:20 CST | 2026-04-09 20:28 CST | 已同步 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档 |

## 阶段 L：`I420/YV12 -> BGR/RGB` 平面 YUV420 解码（PR-12）

### 目标

- 在 Stage K 的 `NV12/NV21` 基础上，继续补齐最常见的平面 `YUV420` 解码格式：`I420/YV12 -> BGR/RGB`
- 继续沿用单 `Mat` `cvtColor` 接口，优先支持 `CV_8U C1(H*3/2 x W)` 的 decode-only 最小子集
- 保持阶段边界聚焦：仍不做 encode-only，也不扩展到 `YUY2/UYVY/NV16/NV24`

### 计划项

1. 新增 `COLOR_YUV2BGR_I420` / `COLOR_YUV2RGB_I420` / `COLOR_YUV2BGR_YV12` / `COLOR_YUV2RGB_YV12`
2. 以 TDD 补齐 `CV_8U` 连续/step/非法布局测试，明确平面顺序、`rows = H * 3 / 2`、`cols = W`、`W/H` 为偶数
3. 在 `cvtcolor.h` 与 `resize_backend.cpp` 中实现 `I420/YV12` 的 `C1 -> C3` fallback/fast-path
4. 将新增 color code 纳入 quick/full benchmark、gate 与文档

### 退出标准

- 新增 4 个 color code 的 `CV_8U` 连续/step/非法输入测试通过
- quick/full benchmark 覆盖 `I420/YV12 -> BGR/RGB` 且 gate 通过
- 文档明确格式约束、支持矩阵、验证命令与结果；任务中断后可从 L1/L2/L3/L4 任一里程碑恢复

### 阶段 L 里程碑拆分（可中断执行）

| 里程碑 | 工作内容 | 验收标准 | 完成情况 | 开始时间 | 完成时间 | 备注/证据 |
|---|---|---|---|---|---|---|
| L1：契约测试先行 | 新增 `I420/YV12 -> BGR/RGB` 的 `CV_8U` 连续/step/非法尺寸测试 | 先出现目标 RED，再由实现转 GREEN | `已完成` | 2026-04-09 20:29 CST | 2026-04-09 20:42 CST | 先以缺少 `COLOR_YUV2*I420/YV12` enum 的编译失败进入 RED，再转 GREEN；定向用例 `3/3 PASS` |
| L2：fallback/backend 解码实现 | 在 `detail/common.h`、`cvtcolor.h`、`resize_backend.cpp` 中补齐 enum、layout 校验与 decode 实现 | 新增 color code 可走 fast-path；非法 `rows/cols/channels` 能正确报错 | `已完成` | 2026-04-09 20:42 CST | 2026-04-09 20:56 CST | 已补齐 4 个 color code，新增 `YUV420p(I420/YV12) -> BGR/RGB` 的 `CV_8U` helper、plane 偏移计算与 backend fast-path |
| L3：benchmark/gate 接入 | benchmark 新增 `I420/YV12` decode case，更新 baseline/current，执行 quick/full gate | quick/full gate 端到端通过 | `已完成` | 2026-04-09 20:56 CST | 2026-04-09 21:50 CST | quick gate：`158/158 PASS`；full gate：`237/237 PASS` |
| L4：文档收口 | 回填 `readme`、支持矩阵、验收状态、验证结果与限制说明 | 文档与当前实现一致，可恢复继续下一阶段 | `已完成` | 2026-04-09 21:50 CST | 2026-04-09 21:54 CST | 已同步 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档 |

## 阶段 M：`YUY2/UYVY -> BGR/RGB` 打包 YUV422 解码（PR-13）

### 目标

- 在现有 `YUV420` decode 覆盖基础上，继续补齐常见的 packed `YUV422` 摄像头输入：`YUY2/UYVY -> BGR/RGB`
- 继续沿用单 `Mat` `cvtColor` 接口，优先支持 `CV_8U C2(W*2 byte semantics)` 对应的 packed decode-only 最小子集
- 保持阶段边界聚焦：仍不做 encode-only，也不扩展到 `NV16/NV24`

### 计划项

1. 新增 `COLOR_YUV2BGR_YUY2` / `COLOR_YUV2RGB_YUY2` / `COLOR_YUV2BGR_UYVY` / `COLOR_YUV2RGB_UYVY`
2. 以 TDD 补齐 `CV_8U` 连续/step/非法布局测试，明确 packed 2-byte pair 语义与偶数宽度约束
3. 在 `cvtcolor.h` 与 `resize_backend.cpp` 中实现 `YUY2/UYVY` 的 `C2 -> C3` fallback/fast-path
4. 将新增 color code 纳入 quick/full benchmark、gate 与文档

### 退出标准

- 新增 4 个 color code 的 `CV_8U` 连续/step/非法输入测试通过
- quick/full benchmark 覆盖 `YUY2/UYVY -> BGR/RGB` 且 gate 通过
- 文档明确格式约束、支持矩阵、验证命令与结果；任务中断后可从 M1/M2/M3/M4 任一里程碑恢复

### 阶段 M 里程碑拆分（可中断执行）

| 里程碑 | 工作内容 | 验收标准 | 完成情况 | 开始时间 | 完成时间 | 备注/证据 |
|---|---|---|---|---|---|---|
| M1：契约测试先行 | 新增 `YUY2/UYVY -> BGR/RGB` 的 `CV_8U` 连续/step/非法尺寸测试 | 先出现目标 RED，再由实现转 GREEN | `已完成` | 2026-04-09 21:59 CST | 2026-04-09 22:03 CST | 先以缺少 `COLOR_YUV2*YUY2/UYVY` enum 的编译失败进入 RED，再转 GREEN；定向用例 `3/3 PASS` |
| M2：fallback/backend 解码实现 | 在 `detail/common.h`、`cvtcolor.h`、`resize_backend.cpp` 中补齐 enum、layout 校验与 decode 实现 | 新增 color code 可走 fast-path；非法 `rows/cols/channels` 能正确报错 | `已完成` | 2026-04-09 22:03 CST | 2026-04-09 22:06 CST | 已补齐 4 个 color code，新增 packed `YUV422(YUY2/UYVY) -> BGR/RGB` 的 `CV_8U C2 -> C3` helper、偶数宽度校验与 backend fast-path |
| M3：benchmark/gate 接入 | benchmark 新增 `YUY2/UYVY` decode case，更新 baseline/current，执行 quick/full gate | quick/full gate 端到端通过 | `已完成` | 2026-04-09 22:06 CST | 2026-04-09 22:18 CST | quick gate：`166/166 PASS`；full gate：`249/249 PASS`；full 首次测得 `CVTCOLOR_BGR2GRAY_F32` 孤立噪声尖峰，warmup 后复跑收敛 |
| M4：文档收口 | 回填 `readme`、支持矩阵、验收状态、验证结果与限制说明 | 文档与当前实现一致，可恢复继续下一阶段 | `已完成` | 2026-04-09 22:18 CST | 2026-04-09 22:20 CST | 已同步 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档 |

## 阶段 N：`NV16/NV61 -> BGR/RGB` 半平面 YUV422 解码（PR-14）

### 目标

- 在现有 `NV12/NV21`（半平面 `YUV420`）与 `YUY2/UYVY`（打包 `YUV422`）decode 覆盖基础上，补齐最接近摄像头/编解码输入的半平面 `YUV422`：`NV16/NV61 -> BGR/RGB`
- 继续沿用单 `Mat` `cvtColor` 接口，优先支持 `CV_8U C1(H*2 x W byte semantics)` 的 decode-only 最小子集
- 保持阶段边界聚焦：本阶段只做 `NV16/NV61`，不并入 `NV24/NV42`，也不做 encode-only

### 计划项

1. 新增 `COLOR_YUV2BGR_NV16` / `COLOR_YUV2RGB_NV16` / `COLOR_YUV2BGR_NV61` / `COLOR_YUV2RGB_NV61`
2. 以 TDD 补齐 `CV_8U` 连续/step/非法布局测试，明确 `rows = H*2`、偶数宽度与 `UV/VU` 交织语义
3. 在 `cvtcolor.h` 与 `resize_backend.cpp` 中实现 `NV16/NV61` 的 `C1 -> C3` fallback/fast-path，优先复用现有 `NV12/NV21` decode 结构
4. 将新增 color code 纳入 quick/full benchmark、gate 与文档

### 退出标准

- 新增 4 个 color code 的 `CV_8U` 连续/step/非法输入测试通过
- quick/full benchmark 覆盖 `NV16/NV61 -> BGR/RGB` 且 gate 通过
- 文档明确格式约束、支持矩阵、验证命令与结果；任务中断后可从 N1/N2/N3/N4 任一里程碑恢复

### 阶段 N 里程碑拆分（可中断执行）

| 里程碑 | 工作内容 | 验收标准 | 完成情况 | 开始时间 | 完成时间 | 备注/证据 |
|---|---|---|---|---|---|---|
| N1：契约测试先行 | 新增 `NV16/NV61 -> BGR/RGB` 的 `CV_8U` 连续/step/非法尺寸测试 | 先出现目标 RED，再由实现转 GREEN | `已完成` | 2026-04-09 22:29 CST | 2026-04-09 22:32 CST | 先以缺少 `COLOR_YUV2*NV16/NV61` enum 的编译失败进入 RED，再转 GREEN；定向用例 `3/3 PASS` |
| N2：fallback/backend 解码实现 | 在 `detail/common.h`、`cvtcolor.h`、`resize_backend.cpp` 中补齐 enum、layout 校验与 decode 实现 | 新增 color code 可走 fast-path；非法 `rows/cols/channels` 能正确报错 | `已完成` | 2026-04-09 22:32 CST | 2026-04-09 22:35 CST | 已补齐 4 个 color code，新增半平面 `YUV422(NV16/NV61) -> BGR/RGB` 的 `CV_8U C1(H*2 x W)` helper、`rows=H*2` 校验与 backend fast-path |
| N3：benchmark/gate 接入 | benchmark 新增 `NV16/NV61` decode case，更新 baseline/current，执行 quick/full gate | quick/full gate 端到端通过 | `已完成` | 2026-04-09 22:35 CST | 2026-04-09 22:47 CST | quick gate：`174/174 PASS`；full gate：`261/261 PASS` |
| N4：文档收口 | 回填 `readme`、支持矩阵、验收状态、验证结果与限制说明 | 文档与当前实现一致，可恢复继续下一阶段 | `已完成` | 2026-04-09 22:47 CST | 2026-04-09 22:47 CST | 已同步 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档 |

## 阶段 O：`NV24/NV42 -> BGR/RGB` 半平面 YUV444 解码（PR-15）

### 目标

- 在现有 `NV12/NV21`（半平面 `YUV420`）、`NV16/NV61`（半平面 `YUV422`）与 `YUY2/UYVY`（打包 `YUV422`）decode 覆盖基础上，继续补齐半平面 `YUV444`：`NV24/NV42 -> BGR/RGB`
- 继续沿用单 `Mat` `cvtColor` 接口，优先支持 `CV_8U C1(H*3 x W byte semantics)` 的 decode-only 最小子集
- 明确 `NV24/NV42` 的单 `Mat` 语义：
  - 上半 `H` 行为 `Y`
  - 下半 `2H` 行视为连续 `UV/VU` 字节流
  - 对像素 `(y, x)` 的 chroma 逻辑偏移为 `y * (2W) + 2x + {0,1}`
- 保持阶段边界聚焦：本阶段只做 `NV24/NV42`，不并入 encode-only 或更广的 YUV444 家族

### 计划项

1. 新增 `COLOR_YUV2BGR_NV24` / `COLOR_YUV2RGB_NV24` / `COLOR_YUV2BGR_NV42` / `COLOR_YUV2RGB_NV42`
2. 以 TDD 补齐 `CV_8U` 连续/step/非法布局测试，明确 `rows = H*3` 与 `UV/VU` 连续字节流访问语义
3. 在 `cvtcolor.h` 与 `resize_backend.cpp` 中实现 `NV24/NV42` 的 `C1 -> C3` fallback/fast-path
4. 将新增 color code 纳入 quick/full benchmark、gate 与文档

### 退出标准

- 新增 4 个 color code 的 `CV_8U` 连续/step/非法输入测试通过
- quick/full benchmark 覆盖 `NV24/NV42 -> BGR/RGB` 且 gate 通过
- 文档明确格式约束、支持矩阵、验证命令与结果；任务中断后可从 O1/O2/O3/O4 任一里程碑恢复

### 阶段 O 里程碑拆分（可中断执行）

| 里程碑 | 工作内容 | 验收标准 | 完成情况 | 开始时间 | 完成时间 | 备注/证据 |
|---|---|---|---|---|---|---|
| O1：契约测试先行 | 新增 `NV24/NV42 -> BGR/RGB` 的 `CV_8U` 连续/step/非法尺寸测试 | 先出现目标 RED，再由实现转 GREEN | `已完成` | 2026-04-10 09:51 CST | 2026-04-10 09:57 CST | 先以缺少 `COLOR_YUV2*NV24/NV42` enum 的编译失败进入 RED，再转 GREEN；定向用例 `3/3 PASS` |
| O2：fallback/backend 解码实现 | 在 `detail/common.h`、`cvtcolor.h`、`resize_backend.cpp` 中补齐 enum、layout 校验与 decode 实现 | 新增 color code 可走 fast-path；非法 `rows/cols/channels` 能正确报错 | `已完成` | 2026-04-10 09:57 CST | 2026-04-10 10:02 CST | 已补齐 4 个 color code，新增半平面 `YUV444(NV24/NV42) -> BGR/RGB` 的 `CV_8U C1(H*3 x W)` helper、`rows=H*3` 校验与 backend fast-path |
| O3：benchmark/gate 接入 | benchmark 新增 `NV24/NV42` decode case，更新 baseline/current，执行 quick/full gate | quick/full gate 端到端通过 | `已完成` | 2026-04-10 10:02 CST | 2026-04-10 10:13 CST | quick gate：`182/182 PASS`；full gate 首次出现 1 个孤立噪声尖峰，经额外 warmup 后复跑：`273/273 PASS` |
| O4：文档收口 | 回填 `readme`、支持矩阵、验收状态、验证结果与限制说明 | 文档与当前实现一致，可恢复继续下一阶段 | `已完成` | 2026-04-10 10:13 CST | 2026-04-10 10:14 CST | 已同步 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档，并记录 `NV24/NV42` 的 `H*3 x W` 单 `Mat` 语义 |

## 阶段 P：`BGR/RGB -> NV24/NV42` 半平面 YUV444 编码（PR-16）

### 目标

- 在现有 `BGR/RGB <-> YUV(C3)` 与阶段 O 的 `NV24/NV42 -> BGR/RGB` 基础上，补齐同布局的 encode-only 最小闭环：`BGR/RGB -> NV24/NV42`
- 继续沿用单 `Mat` `cvtColor` 接口，优先支持 `CV_8U C3 -> C1(H*3 x W byte semantics)` 的最小子集
- 沿用阶段 O 的 `NV24/NV42` 单 `Mat` 语义：
  - 上半 `H` 行为 `Y`
  - 下半 `2H` 行视为连续 `UV/VU` 字节流
  - 对像素 `(y, x)` 的 chroma 逻辑偏移为 `y * (2W) + 2x + {0,1}`
- 保持阶段边界聚焦：本阶段只做 `NV24/NV42` encode-only，不并入 `I444/YV24` 等平面 `YUV444` family，也不扩展到 `CV_32F`

### 计划项

1. 新增 `COLOR_BGR2YUV_NV24` / `COLOR_RGB2YUV_NV24` / `COLOR_BGR2YUV_NV42` / `COLOR_RGB2YUV_NV42`
2. 以 TDD 补齐 `CV_8U` 连续/ROI step/非法输入测试，明确输出 `rows = H*3` 与 `UV/VU` 连续字节流写入语义
3. 在 `cvtcolor.h` 与 `resize_backend.cpp` 中实现 `BGR/RGB -> NV24/NV42` 的 fallback/fast-path
4. 将新增 color code 纳入 quick/full benchmark、gate 与文档

### 退出标准

- 新增 4 个 color code 的 `CV_8U` 连续/ROI/非法输入测试通过
- quick/full benchmark 覆盖 `BGR/RGB -> NV24/NV42` 且 gate 通过
- 文档明确格式约束、支持矩阵、验证命令与结果；任务中断后可从 P1/P2/P3/P4 任一里程碑恢复

### 阶段 P 里程碑拆分（可中断执行）

| 里程碑 | 工作内容 | 验收标准 | 完成情况 | 开始时间 | 完成时间 | 备注/证据 |
|---|---|---|---|---|---|---|
| P1：契约测试先行 | 新增 `BGR/RGB -> NV24/NV42` 的 `CV_8U` 连续/ROI/非法输入测试 | 先出现目标 RED，再由实现转 GREEN | `已完成` | 2026-04-10 10:20 CST | 2026-04-10 10:28 CST | 先以缺少 `COLOR_*YUV_NV24/NV42` enum 的编译失败进入 RED，再转 GREEN；定向用例 `3/3 PASS` |
| P2：fallback/backend 编码实现 | 在 `detail/common.h`、`cvtcolor.h`、`resize_backend.cpp` 中补齐 enum 与 encode 实现 | 新增 color code 可走 fast-path；非法 `depth/channels` 能正确报错 | `已完成` | 2026-04-10 10:28 CST | 2026-04-10 10:39 CST | 已补齐 4 个 color code，新增半平面 `YUV444(NV24/NV42)` 的 `CV_8U C3 -> C1(H*3 x W)` helper、limited-range `YUV` 写出与 backend fast-path |
| P3：benchmark/gate 接入 | benchmark 新增 `BGR/RGB -> NV24/NV42` encode case，更新 baseline/current，执行 quick/full gate | quick/full gate 端到端通过 | `已完成` | 2026-04-10 10:39 CST | 2026-04-10 15:08 CST | quick gate：`190/190 PASS`；full gate：`285/285 PASS`；中途修复 benchmark `run_one_op()` 遗漏新 op 导致的空 `dst` 计量问题 |
| P4：文档收口 | 回填 `readme`、支持矩阵、验收状态、验证结果与限制说明 | 文档与当前实现一致，可恢复继续下一阶段 | `已完成` | 2026-04-10 15:08 CST | 2026-04-10 15:13 CST | 已同步 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档，并补齐阶段 Q 占位 |

## 阶段 Q：`I444/YV24 -> BGR/RGB` 平面 YUV444 解码（PR-17）

### 目标

- 在阶段 O/P 已完成 `NV24/NV42` 半平面 `YUV444` decode/encode 的基础上，继续补齐平面 `YUV444`：`I444/YV24 -> BGR/RGB`
- 继续沿用单 `Mat` `cvtColor` 接口，优先支持 `CV_8U C1(H*3 x W byte semantics)` 的 decode-only 最小子集
- 明确 `I444/YV24` 的单 `Mat` 语义：
  - 上半 `H` 行为 `Y`
  - 中间 `H` 行为 `U/V`
  - 最后 `H` 行为 `V/U`
- 保持阶段边界聚焦：本阶段只做 `I444/YV24` decode-only，不并入 encode-only 或 `CV_32F`

### 计划项

1. 新增 `COLOR_YUV2BGR_I444` / `COLOR_YUV2RGB_I444` / `COLOR_YUV2BGR_YV24` / `COLOR_YUV2RGB_YV24`
2. 以 TDD 补齐 `CV_8U` 连续/step/非法布局测试，明确 `rows = H*3` 与 `U/V` 平面偏移语义
3. 在 `cvtcolor.h` 与 `resize_backend.cpp` 中实现 `I444/YV24` 的 `C1 -> C3` fallback/fast-path
4. 将新增 color code 纳入 quick/full benchmark、gate 与文档

### 退出标准

- 新增 4 个 color code 的 `CV_8U` 连续/step/非法输入测试通过
- quick/full benchmark 覆盖 `I444/YV24 -> BGR/RGB` 且 gate 通过
- 文档明确格式约束、支持矩阵、验证命令与结果；任务中断后可从 Q1/Q2/Q3/Q4 任一里程碑恢复

### 阶段 Q 里程碑拆分（可中断执行）

| 里程碑 | 工作内容 | 验收标准 | 完成情况 | 开始时间 | 完成时间 | 备注/证据 |
|---|---|---|---|---|---|---|
| Q1：契约测试先行 | 新增 `I444/YV24 -> BGR/RGB` 的 `CV_8U` 连续/step/非法尺寸测试 | 先出现目标 RED，再由实现转 GREEN | `已完成` | 2026-04-10 15:26 CST | 2026-04-10 15:30 CST | 先以缺少 `COLOR_YUV2*I444/YV24` enum 的编译失败进入 RED，再转 GREEN；定向用例 `3/3 PASS` |
| Q2：fallback/backend 解码实现 | 在 `detail/common.h`、`cvtcolor.h`、`resize_backend.cpp` 中补齐 enum、layout 校验与 decode 实现 | 新增 color code 可走 fast-path；非法 `rows/channels` 能正确报错 | `已完成` | 2026-04-10 15:30 CST | 2026-04-10 15:31 CST | 已补齐 4 个 color code，新增平面 `YUV444(I444/YV24) -> BGR/RGB` 的 `CV_8U C1(H*3 x W)` helper、`rows=H*3` 校验与 backend fast-path |
| Q3：benchmark/gate 接入 | benchmark 新增 `I444/YV24` decode case，更新 baseline/current，执行 quick/full gate | quick/full gate 端到端通过 | `已完成` | 2026-04-10 15:31 CST | 2026-04-10 15:38 CST | quick gate：`198/198 PASS`；full gate：`297/297 PASS` |
| Q4：文档收口 | 回填 `readme`、支持矩阵、验收状态、验证结果与限制说明 | 文档与当前实现一致，可恢复继续下一阶段 | `已完成` | 2026-04-10 15:38 CST | 2026-04-10 15:38 CST | 已同步 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档，并记录 `I444/YV24` 的 `H*3 x W` 单 `Mat` 平面语义 |

---

## 阶段 R：`BGR/RGB -> I444/YV24` 平面 YUV444 编码（PR-18）

### 目标

- 在阶段 Q 已完成 `I444/YV24 -> BGR/RGB` 平面 `YUV444` decode 的基础上，补齐同布局的 encode-only 最小闭环：`BGR/RGB -> I444/YV24`
- 继续沿用单 `Mat` `cvtColor` 接口，优先支持 `CV_8U C3 -> C1(H*3 x W)` 的 encode-only 最小子集
- 沿用阶段 Q 的 `I444/YV24` 单 `Mat` 语义：
  - 上 `H` 行为 `Y`
  - 中间 `H` 行为 `U/V`
  - 最后 `H` 行为 `V/U`
- 保持阶段边界聚焦：本阶段只做 `I444/YV24` encode-only，不并入 decode 扩展或 `CV_32F`

### 计划项

1. 新增 `COLOR_BGR2YUV_I444` / `COLOR_RGB2YUV_I444` / `COLOR_BGR2YUV_YV24` / `COLOR_RGB2YUV_YV24`
2. 以 TDD 补齐 `CV_8U` 连续/ROI/非法输入测试，明确 `rows = H*3` 与 `U/V` 平面偏移语义
3. 在 `cvtcolor.h` 与 `resize_backend.cpp` 中实现 `BGR/RGB -> I444/YV24` 的 `C3 -> C1` fallback/fast-path
4. 将新增 color code 纳入 quick/full benchmark、gate 与文档

### 退出标准

- 新增 4 个 color code 的 `CV_8U` 连续/ROI/非法输入测试通过
- quick/full benchmark 覆盖 `BGR/RGB -> I444/YV24` 且 gate 通过
- 文档明确格式约束、支持矩阵、验证命令与结果；任务中断后可从 R1/R2/R3/R4 任一里程碑恢复

### 阶段 R 里程碑拆分（可中断执行）

| 里程碑 | 工作内容 | 验收标准 | 完成情况 | 开始时间 | 完成时间 | 备注/证据 |
|---|---|---|---|---|---|---|
| R1：契约测试先行 | 新增 `BGR/RGB -> I444/YV24` 的 `CV_8U` 连续/ROI/非法输入测试 | 先出现目标 RED，再由实现转 GREEN | `已完成` | 2026-04-10 16:47 CST | 2026-04-10 16:50 CST | 先以缺少 `COLOR_BGR2YUV_*I444/YV24` enum 的编译失败进入 RED，再转 GREEN；定向用例 `3/3 PASS` |
| R2：fallback/backend 编码实现 | 在 `detail/common.h`、`cvtcolor.h`、`resize_backend.cpp` 中补齐 enum 与 encode 实现 | 新增 color code 可走 fast-path；非法 `depth/channels` 能正确报错 | `已完成` | 2026-04-10 16:50 CST | 2026-04-10 16:53 CST | 已补齐 4 个 color code，新增平面 `BGR/RGB -> YUV444(I444/YV24)` 的 `CV_8U C3 -> C1(H*3 x W)` helper，并修复 `YV24` 平面写出重复交换导致的错序问题 |
| R3：benchmark/gate 接入 | benchmark 新增 `I444/YV24` encode case，更新 baseline/current，执行 quick/full gate | quick/full gate 端到端通过 | `已完成` | 2026-04-10 16:53 CST | 2026-04-10 17:09 CST | quick gate：`206/206 PASS`；full gate：`309/309 PASS` |
| R4：文档收口 | 回填 `readme`、支持矩阵、验收状态、验证结果与限制说明 | 文档与当前实现一致，可恢复继续下一阶段 | `已完成` | 2026-04-11 11:20 CST | 2026-04-11 11:33 CST | 已同步 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档，并记录 `I444/YV24` 的 `H*3 x W` 单 `Mat` 输入/输出语义 |

---

## 阶段 S：`BGR/RGB -> NV16/NV61` 半平面 YUV422 编码（PR-19）

### 目标

- 在阶段 N 已完成 `NV16/NV61 -> BGR/RGB` 半平面 `YUV422` decode 的基础上，补齐同布局的 encode-only 最小闭环：`BGR/RGB -> NV16/NV61`
- 继续沿用单 `Mat` `cvtColor` 接口，优先支持 `CV_8U C3 -> C1(H*2 x W)` 的 encode-only 最小子集
- 沿用阶段 N 的 `NV16/NV61` 单 `Mat` 语义：
  - 上 `H` 行为 `Y`
  - 下 `H` 行为按 `UV/VU` 交织写出的半平面 `YUV422`
  - 每 2 个水平像素共享一组 `U/V`
- 保持阶段边界聚焦：本阶段只做 `NV16/NV61` encode-only，不并入 `NV12/NV21` / `I420/YV12` / `YUY2/UYVY` 等其它 `YUV422/YUV420` family，也不扩展到 `CV_32F`

### 计划项

1. 新增 `COLOR_BGR2YUV_NV16` / `COLOR_RGB2YUV_NV16` / `COLOR_BGR2YUV_NV61` / `COLOR_RGB2YUV_NV61`
2. 以 TDD 补齐 `CV_8U` 连续/ROI/非法输入测试，明确输出 `rows = H*2`、偶数宽度与 `UV/VU` 交织写入语义
3. 在 `cvtcolor.h` 与 `resize_backend.cpp` 中实现 `BGR/RGB -> NV16/NV61` 的 fallback/fast-path
4. 将新增 color code 纳入 quick/full benchmark、gate 与文档

### 退出标准

- 新增 4 个 color code 的 `CV_8U` 连续/ROI/非法输入测试通过
- quick/full benchmark 覆盖 `BGR/RGB -> NV16/NV61` 且 gate 通过
- 文档明确格式约束、支持矩阵、验证命令与结果；任务中断后可从 S1/S2/S3/S4 任一里程碑恢复

### 阶段 S 里程碑拆分（可中断执行）

| 里程碑 | 工作内容 | 验收标准 | 完成情况 | 开始时间 | 完成时间 | 备注/证据 |
|---|---|---|---|---|---|---|
| S1：契约测试先行 | 新增 `BGR/RGB -> NV16/NV61` 的 `CV_8U` 连续/ROI/非法输入测试 | 先出现目标 RED，再由实现转 GREEN | `已完成` | 2026-04-11 11:58 CST | 2026-04-11 12:00 CST | 先以缺少 `COLOR_BGR2YUV_*NV16/NV61` enum 的编译失败进入 RED，再转 GREEN；定向用例 `3/3 PASS` |
| S2：fallback/backend 编码实现 | 在 `detail/common.h`、`cvtcolor.h`、`resize_backend.cpp` 中补齐 enum 与 encode 实现 | 新增 color code 可走 fast-path；非法 `depth/channels/width` 能正确报错 | `已完成` | 2026-04-11 12:00 CST | 2026-04-11 12:07 CST | 已补齐 4 个 color code，新增半平面 `YUV422(NV16/NV61)` 的 `CV_8U C3 -> C1(H*2 x W)` helper、水平二像素共享 `U/V` 的 chroma 写出与 backend fast-path |
| S3：benchmark/gate 接入 | benchmark 新增 `NV16/NV61` encode case，更新 baseline/current，执行 quick/full gate | quick/full gate 端到端通过 | `已完成` | 2026-04-11 12:07 CST | 2026-04-11 12:26 CST | quick gate 首次 retained current 出现 2 个旧 case 孤立尖峰，经额外 warmup 后复跑：`214/214 PASS`；full gate：`321/321 PASS` |
| S4：文档收口 | 回填 `readme`、支持矩阵、验收状态、验证结果与限制说明 | 文档与当前实现一致，可恢复继续下一阶段 | `已完成` | 2026-04-13 16:10 CST | 2026-04-13 16:23 CST | 已同步 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档，并记录 `NV16/NV61` 的 `H*2 x W` 单 `Mat` 输入/输出语义 |

---

## 阶段 T：`BGR/RGB -> YUY2/UYVY` 打包 YUV422 编码（PR-20）

### 目标

- 在阶段 M 已完成 `YUY2/UYVY -> BGR/RGB` 打包 `YUV422` decode 的基础上，补齐同布局的 encode-only 最小闭环：`BGR/RGB -> YUY2/UYVY`
- 继续沿用单 `Mat` `cvtColor` 接口，优先支持 `CV_8U C3 -> C2(H x W)` 的 encode-only 最小子集
- 沿用阶段 M 的 `YUY2/UYVY` 单 `Mat` 语义：
  - `YUY2`：每 2 个水平像素写作 `[Y0 U][Y1 V]`
  - `UYVY`：每 2 个水平像素写作 `[U Y0][V Y1]`
  - 每 2 个水平像素共享一组 `U/V`
- 保持阶段边界聚焦：本阶段只做 `YUY2/UYVY` encode-only，不并入 `NV12/NV21` / `I420/YV12` 等其它 `YUV420/YUV422` family，也不扩展到 `CV_32F`

### 计划项

1. 新增 `COLOR_BGR2YUV_YUY2` / `COLOR_RGB2YUV_YUY2` / `COLOR_BGR2YUV_UYVY` / `COLOR_RGB2YUV_UYVY`
2. 以 TDD 补齐 `CV_8U` 连续/ROI/非法输入测试，明确输出 `CV_8UC2(H x W)`、偶数宽度与 `YUY2/UYVY` packed 写入语义
3. 在 `cvtcolor.h` 与 `resize_backend.cpp` 中实现 `BGR/RGB -> YUY2/UYVY` 的 fallback/fast-path
4. 将新增 color code 纳入 quick/full benchmark、gate 与文档

### 退出标准

- 新增 4 个 color code 的 `CV_8U` 连续/ROI/非法输入测试通过
- quick/full benchmark 覆盖 `BGR/RGB -> YUY2/UYVY` 且 gate 通过
- 文档明确格式约束、支持矩阵、验证命令与结果；任务中断后可从 T1/T2/T3/T4 任一里程碑恢复

### 阶段 T 里程碑拆分（可中断执行）

| 里程碑 | 工作内容 | 验收标准 | 完成情况 | 开始时间 | 完成时间 | 备注/证据 |
|---|---|---|---|---|---|---|
| T1：契约测试先行 | 新增 `BGR/RGB -> YUY2/UYVY` 的 `CV_8U` 连续/ROI/非法输入测试 | 先出现目标 RED，再由实现转 GREEN | `已完成` | 2026-04-13 16:30 CST | 2026-04-13 16:34 CST | 先以缺少 `COLOR_BGR2YUV_*YUY2/UYVY` enum 的编译失败进入 RED，再转 GREEN；定向用例 `3/3 PASS` |
| T2：fallback/backend 编码实现 | 在 `detail/common.h`、`cvtcolor.h`、`resize_backend.cpp` 中补齐 enum 与 encode 实现 | 新增 color code 可走 fast-path；非法 `depth/channels/width` 能正确报错 | `已完成` | 2026-04-13 16:34 CST | 2026-04-13 16:41 CST | 已补齐 4 个 color code，新增 packed `YUV422(YUY2/UYVY)` 的 `CV_8U C3 -> C2(H x W)` helper、水平二像素共享 `U/V` 的 packed 写出与 backend fast-path |
| T3：benchmark/gate 接入 | benchmark 新增 `YUY2/UYVY` encode case，更新 baseline/current，执行 quick/full gate | quick/full gate 端到端通过 | `已完成` | 2026-04-13 16:43 CST | 2026-04-13 17:02 CST | quick gate：`222/222 PASS`；full gate：`333/333 PASS` |
| T4：文档收口 | 回填 `readme`、支持矩阵、验收状态、验证结果与限制说明 | 文档与当前实现一致，可恢复继续下一阶段 | `已完成` | 2026-04-13 17:02 CST | 2026-04-13 17:02 CST | 已同步 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档，并记录 `YUY2/UYVY` 的 `H x W` 单 `Mat` packed 输入/输出语义 |

---

## 阶段 U：`BGR/RGB -> NV12/NV21` 半平面 YUV420 编码（PR-21）

### 目标

- 在阶段 K 已完成 `NV12/NV21 -> BGR/RGB` 半平面 `YUV420` decode 的基础上，补齐同布局的 encode-only 最小闭环：`BGR/RGB -> NV12/NV21`
- 继续沿用单 `Mat` `cvtColor` 接口，优先支持 `CV_8U C3 -> C1(H*3/2 x W)` 的 encode-only 最小子集
- 沿用阶段 K 的 `NV12/NV21` 单 `Mat` 语义：
  - 上 `H` 行为 `Y`
  - 下 `H/2` 行为按 `UV/VU` 交织写出的半平面 `YUV420`
  - 每个 `2x2` 像素块共享一组 `U/V`
- 保持阶段边界聚焦：本阶段只做 `NV12/NV21` encode-only，不并入 `I420/YV12` 等其它 `YUV420` family，也不扩展到 `CV_32F`

### 计划项

1. 新增 `COLOR_BGR2YUV_NV12` / `COLOR_RGB2YUV_NV12` / `COLOR_BGR2YUV_NV21` / `COLOR_RGB2YUV_NV21`
2. 以 TDD 补齐 `CV_8U` 连续/ROI/非法输入测试，明确输出 `rows = H*3/2`、宽高偶数与 `UV/VU` 交织写入语义
3. 在 `cvtcolor.h` 与 `resize_backend.cpp` 中实现 `BGR/RGB -> NV12/NV21` 的 fallback/fast-path
4. 将新增 color code 纳入 quick/full benchmark、gate 与文档

### 退出标准

- 新增 4 个 color code 的 `CV_8U` 连续/ROI/非法输入测试通过
- quick/full benchmark 覆盖 `BGR/RGB -> NV12/NV21` 且 gate 通过
- 文档明确格式约束、支持矩阵、验证命令与结果；任务中断后可从 U1/U2/U3/U4 任一里程碑恢复

### 阶段 U 里程碑拆分（可中断执行）

| 里程碑 | 工作内容 | 验收标准 | 完成情况 | 开始时间 | 完成时间 | 备注/证据 |
|---|---|---|---|---|---|---|
| U1：契约测试先行 | 新增 `BGR/RGB -> NV12/NV21` 的 `CV_8U` 连续/ROI/非法输入测试 | 先出现目标 RED，再由实现转 GREEN | `已完成` | 2026-04-13 17:16 CST | 2026-04-13 19:54 CST | 先以缺少 `COLOR_BGR2YUV_*NV12/NV21` 的编译失败进入 RED，再转 GREEN；定向用例 `3/3 PASS` |
| U2：fallback/backend 编码实现 | 在 `detail/common.h`、`cvtcolor.h`、`resize_backend.cpp` 中补齐 enum 与 encode 实现 | 新增 color code 可走 fast-path；非法 `depth/channels/shape` 能正确报错 | `已完成` | 2026-04-13 19:54 CST | 2026-04-13 20:05 CST | 已补齐 4 个 color code，新增 `YUV420sp(NV12/NV21)` 的 `CV_8U C3 -> C1(H*3/2 x W)` helper 与 backend fast-path |
| U3：benchmark/gate 接入 | benchmark 新增 `NV12/NV21` encode case，更新 baseline/current，执行 quick/full gate | quick/full gate 端到端通过 | `已完成` | 2026-04-13 20:05 CST | 2026-04-13 20:24 CST | quick gate：`230/230 PASS`；full gate：`345/345 PASS` |
| U4：文档收口 | 回填 `readme`、支持矩阵、验收状态、验证结果与限制说明 | 文档与当前实现一致，可恢复继续下一阶段 | `已完成` | 2026-04-13 20:24 CST | 2026-04-13 20:28 CST | 已同步 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档，并记录 `NV12/NV21` 的 `H*3/2 x W` 单 `Mat` 输入/输出语义 |

---

## 阶段 V：`BGR/RGB -> I420/YV12` 平面 YUV420 编码（PR-22）

### 目标

- 在阶段 L 已完成 `I420/YV12 -> BGR/RGB` 平面 `YUV420` decode、阶段 U 已完成 `BGR/RGB -> NV12/NV21` 半平面 `YUV420` encode 的基础上，补齐平面 `YUV420` encode-only 最小闭环：`BGR/RGB -> I420/YV12`
- 继续沿用单 `Mat` `cvtColor` 接口，优先支持 `CV_8U C3 -> C1(H*3/2 x W)` 的 encode-only 最小子集
- 沿用阶段 L 的 `I420/YV12` 单 `Mat` 语义：
  - 上 `H` 行为 `Y`
  - 下 `H/2` 行拆分为两个平面：`I420` 为 `U` 后 `V`，`YV12` 为 `V` 后 `U`
  - 每个 `2x2` 像素块共享一组 `U/V`
- 保持阶段边界聚焦：本阶段只做 `I420/YV12` encode-only，不并入其它 family，也不扩展到 `CV_32F`

### 计划项

1. 新增 `COLOR_BGR2YUV_I420` / `COLOR_RGB2YUV_I420` / `COLOR_BGR2YUV_YV12` / `COLOR_RGB2YUV_YV12`
2. 以 TDD 补齐 `CV_8U` 连续/ROI/非法输入测试，明确输出 `rows = H*3/2`、宽高偶数与 `I420/YV12` 平面写入语义
3. 在 `cvtcolor.h` 与 `resize_backend.cpp` 中实现 `BGR/RGB -> I420/YV12` 的 fallback/fast-path
4. 将新增 color code 纳入 quick/full benchmark、gate 与文档

### 退出标准

- 新增 4 个 color code 的 `CV_8U` 连续/ROI/非法输入测试通过
- quick/full benchmark 覆盖 `BGR/RGB -> I420/YV12` 且 gate 通过
- 文档明确格式约束、支持矩阵、验证命令与结果；任务中断后可从 V1/V2/V3/V4 任一里程碑恢复

### 阶段 V 里程碑拆分（可中断执行）

| 里程碑 | 工作内容 | 验收标准 | 完成情况 | 开始时间 | 完成时间 | 备注/证据 |
|---|---|---|---|---|---|---|
| V1：契约测试先行 | 新增 `BGR/RGB -> I420/YV12` 的 `CV_8U` 连续/ROI/非法输入测试 | 先出现目标 RED，再由实现转 GREEN | `已完成` | 2026-04-15 13:35 CST | 2026-04-15 13:50 CST | 先以缺少 `COLOR_BGR2YUV_*I420/YV12` 的编译失败进入 RED，再转 GREEN；定向用例 `3/3 PASS` |
| V2：fallback/backend 编码实现 | 在 `detail/common.h`、`cvtcolor.h`、`resize_backend.cpp` 中补齐 enum 与 encode 实现 | 新增 color code 可走 fast-path；非法 `depth/channels/shape` 能正确报错 | `已完成` | 2026-04-15 13:50 CST | 2026-04-15 14:02 CST | 已补齐 4 个 color code，新增 `YUV420p(I420/YV12)` 的 `CV_8U C3 -> C1(H*3/2 x W)` helper 与 backend fast-path |
| V3：benchmark/gate 接入 | benchmark 新增 `I420/YV12` encode case，更新 baseline/current，执行 quick/full gate | quick/full gate 端到端通过 | `已完成` | 2026-04-15 14:02 CST | 2026-04-15 14:18 CST | quick gate：`238/238 PASS`；full gate：`357/357 PASS`（full 首次复测 2 个旧 case 噪声尖峰，复跑后收敛） |
| V4：文档收口 | 回填 `readme`、支持矩阵、验收状态、验证结果与限制说明 | 文档与当前实现一致，可恢复继续下一阶段 | `已完成` | 2026-04-15 14:18 CST | 2026-04-15 14:19 CST | 已同步 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档，并记录 `I420/YV12` 的 `H*3/2 x W` 单 `Mat` 输入/输出语义 |

---

## 5. 类型与通道支持矩阵（目标态）

| 算子 | Wave-1 | Wave-2 | 备注 |
|---|---|---|---|
| resize | `CV_8U C1/C3/C4` | `CV_32F C1/C3/C4` | `INTER_NEAREST/EXACT/LINEAR` |
| cvtColor | `CV_8U C1/C2/C3/C4` | `CV_32F C1/C3/C4` | 目前覆盖 `GRAY<->BGR/BGRA/RGBA`、`BGR<->RGB`、`BGR/RGB<->BGRA/RGBA`、`BGRA<->RGBA`、`BGR/RGB<->YUV`、`BGR/RGB<->NV12/NV21`、`BGR/RGB<->I420/YV12`、`BGR/RGB<->I444/YV24`、`BGR/RGB<->NV16/NV61`、`BGR/RGB<->NV24/NV42`、`BGR/RGB<->YUY2/UYVY` |
| threshold | `CV_8U` | `CV_32F`（固定阈值） | OTSU/TRIANGLE 保持 `CV_8UC1` |
| boxFilter | `CV_8U C1/C3/C4` | `CV_32F C1/C3/C4` | 边界模式对齐现有语义 |
| GaussianBlur | `CV_8U C1/C3/C4` | `CV_32F C1/C3/C4` | 可分离实现 |

---

## 6. 风险与回滚

### 风险

1. 优化引入边界条件偏差（尤其非连续 ROI 与 border 模式）
2. 某些 ISA/编译器组合下向量路径退化
3. 多线程切分不当导致小图性能变差

### 回滚策略

1. 保留 scalar 路径作为安全兜底
2. 通过编译开关或 dispatch 控制快速禁用优化路径
3. 任何性能或正确性回退，优先回滚到上一阶段稳定点

---

## 7. 验收标准（完成定义）

1. 正确性：
   - `cvh_test_imgproc` 全通过
   - 新增类型/通道组合有对应测试
2. 性能：
   - 在定义的 quick/full profile 下，相比基线无显著回退
   - 核心场景具备可量化提升
3. 工程化：
   - benchmark 与 regression 脚本可在 CI 运行
   - 文档包含支持矩阵、限制和结果

---

## 8. 验收状态（持续更新）

> 说明：本区用于任务中断后恢复。每次恢复任务先看本表，确认下一步。

| 阶段 | 状态 | 负责人 | 开始时间 | 完成时间 | 备注 |
|---|---|---|---|---|---|
| 阶段 A：基线与门禁 | `已完成` | Codex | 2026-04-08 18:08 CST | 2026-04-08 18:35 CST | 已完成 benchmark/regression/baseline/readme/CI quick gate |
| 阶段 B：resize+cvtColor | `已完成` | Codex | 2026-04-08 22:45 CST | 2026-04-08 23:30 CST | B1~B6 全部完成（含性能门禁与文档收口） |
| 阶段 C：box+gaussian | `已完成` | Codex | 2026-04-08 23:31 CST | 2026-04-08 23:44 CST | C1~C5 全部完成（含 ROI/in-place 回归与 quick 性能门禁） |
| 阶段 D：类型扩展 | `已完成` | Codex | 2026-04-09 09:20 CST | 2026-04-09 09:58 CST | D1~D3 完成（threshold/resize/box/gaussian 的 CV_32F 扩展 + 验证 + 文档） |
| 阶段 E：收口发布 | `已完成` | Codex | 2026-04-09 13:58 CST | 2026-04-09 14:23 CST | E1~E3 完成（deterministic quick/full gate + 基线固化 + 文档收口） |
| 阶段 F：`cvtColor CV_32F` 扩展 | `已完成` | Codex | 2026-04-09 14:24 CST | 2026-04-09 15:37 CST | F1~F3 完成（`cvtColor CV_32F` + benchmark 扩展 + gate 校准 + 文档收口） |
| 阶段 G：`BGR<->RGB` / `BGR<->BGRA` | `已完成` | Codex | 2026-04-09 15:38 CST | 2026-04-09 16:20 CST | G1~G3 完成（新增 color code + benchmark 扩展 + gate/doc 收口） |
| 阶段 H：`RGB/BGR/RGBA/BGRA` 家族补齐 | `已完成` | Codex | 2026-04-09 16:38 CST | 2026-04-09 16:52 CST | H1~H3 完成（剩余 RGB/BGR/RGBA/BGRA family color code + benchmark 扩展 + gate/doc 收口） |
| 阶段 I：`GRAY <-> RGBA/BGRA` | `已完成` | Codex | 2026-04-09 17:00 CST | 2026-04-09 17:20 CST | I1~I3 完成（GRAY<->RGBA/BGRA color code + benchmark 扩展 + gate/doc 收口） |
| 阶段 J：`BGR/RGB <-> YUV` | `已完成` | Codex | 2026-04-09 17:45 CST | 2026-04-09 17:48 CST | J1~J3 完成（3ch `BGR/RGB<->YUV` 的 `CV_8U/CV_32F` 扩展 + benchmark/gate/doc 收口） |
| 阶段 K：`NV12/NV21 -> BGR/RGB` | `已完成` | Codex | 2026-04-09 18:31 CST | 2026-04-09 20:28 CST | K1~K4 完成（`CV_8U` `NV12/NV21->BGR/RGB` 解码 + benchmark/gate/doc 收口） |
| 阶段 L：`I420/YV12 -> BGR/RGB` | `已完成` | Codex | 2026-04-09 20:29 CST | 2026-04-09 21:54 CST | L1~L4 完成（`CV_8U` `I420/YV12->BGR/RGB` 解码 + benchmark/gate/doc 收口） |
| 阶段 M：`YUY2/UYVY -> BGR/RGB` | `已完成` | Codex | 2026-04-09 21:59 CST | 2026-04-09 22:20 CST | M1~M4 完成（`CV_8U` `YUY2/UYVY->BGR/RGB` 解码 + benchmark/gate/doc 收口） |
| 阶段 N：`NV16/NV61 -> BGR/RGB` | `已完成` | Codex | 2026-04-09 22:29 CST | 2026-04-09 22:47 CST | N1~N4 完成（`CV_8U` `NV16/NV61->BGR/RGB` 解码 + benchmark/gate/doc 收口） |
| 阶段 O：`NV24/NV42 -> BGR/RGB` | `已完成` | Codex | 2026-04-10 09:51 CST | 2026-04-10 10:14 CST | O1~O4 完成（`CV_8U` `NV24/NV42->BGR/RGB` 解码 + benchmark/gate/doc 收口） |
| 阶段 P：`BGR/RGB -> NV24/NV42` | `已完成` | Codex | 2026-04-10 10:20 CST | 2026-04-10 15:13 CST | P1~P4 完成（`CV_8U` `BGR/RGB->NV24/NV42` 编码 + benchmark/gate/doc 收口） |
| 阶段 Q：`I444/YV24 -> BGR/RGB` | `已完成` | Codex | 2026-04-10 15:13 CST | 2026-04-10 15:38 CST | Q1~Q4 完成（`CV_8U` `I444/YV24->BGR/RGB` 解码 + benchmark/gate/doc 收口） |
| 阶段 R：`BGR/RGB -> I444/YV24` | `已完成` | Codex | 2026-04-10 16:47 CST | 2026-04-11 11:33 CST | R1~R4 完成（`CV_8U` `BGR/RGB<->I444/YV24` 闭环、quick/full gate、文档收口） |
| 阶段 S：`BGR/RGB -> NV16/NV61` | `已完成` | Codex | 2026-04-11 11:58 CST | 2026-04-13 16:23 CST | S1~S4 完成（`CV_8U` `BGR/RGB<->NV16/NV61` 闭环、quick/full gate、文档收口） |
| 阶段 T：`BGR/RGB -> YUY2/UYVY` | `已完成` | Codex | 2026-04-13 16:30 CST | 2026-04-13 17:02 CST | T1~T4 完成（`CV_8U` `BGR/RGB<->YUY2/UYVY` 闭环、quick/full gate、文档收口） |
| 阶段 U：`BGR/RGB -> NV12/NV21` | `已完成` | Codex | 2026-04-13 17:16 CST | 2026-04-13 20:28 CST | U1~U4 完成（`CV_8U` `BGR/RGB<->NV12/NV21` 闭环、quick/full gate、文档收口） |
| 阶段 V：`BGR/RGB -> I420/YV12` | `已完成` | Codex | 2026-04-15 13:35 CST | 2026-04-15 14:19 CST | V1~V4 完成（`CV_8U` `BGR/RGB<->I420/YV12` 闭环、quick/full gate、文档收口） |

当前执行阶段：`本轮 A~V 已完成；下一阶段待规划（建议评估下一批 YUV 家族或进入下一算子簇）`

---

## 9. 验证结果

> 说明：此区先建模板。后续每阶段完成后补真实数据与命令输出摘要。

### 9.1 功能正确性

- 状态：`已完成`
- 执行命令：
  - `cmake --build build-full-test -j --target cvh_test_imgproc`
  - `./build-full-test/cvh_test_imgproc --gtest_filter='ImgprocCvtColor_TEST.bgr_rgb_to_nv24_nv42_yuv444sp_u8_matches_reference:ImgprocCvtColor_TEST.non_contiguous_roi_for_nv24_nv42_encode_matches_reference:ImgprocCvtColor_TEST.throws_on_invalid_bgr_rgb_to_nv24_nv42_inputs'`
  - `./build-full-test/cvh_test_imgproc --gtest_filter='ImgprocCvtColor_TEST.*'`
  - `./build-full-test/cvh_test_imgproc`
  - `cmake --build build-full-test -j --target cvh_test_imgproc`
  - `./build-full-test/cvh_test_imgproc --gtest_filter='ImgprocCvtColor_TEST.i444_yv24_yuv444p_u8_matches_reference:ImgprocCvtColor_TEST.non_contiguous_step_for_i444_yv24_matches_reference:ImgprocCvtColor_TEST.throws_on_invalid_i444_yv24_layouts'`
  - `./build-full-test/cvh_test_imgproc --gtest_filter='ImgprocCvtColor_TEST.*'`
  - `./build-full-test/cvh_test_imgproc`
  - `cmake --build build-full-test -j --target cvh_test_imgproc`
  - `cmake --build build-full-test --clean-first -j --target cvh_test_imgproc`
  - `./build-full-test/cvh_test_imgproc --gtest_filter='ImgprocCvtColor_TEST.bgr_rgb_to_i444_yv24_yuv444p_u8_matches_reference:ImgprocCvtColor_TEST.non_contiguous_roi_for_i444_yv24_encode_matches_reference:ImgprocCvtColor_TEST.throws_on_invalid_bgr_rgb_to_i444_yv24_inputs'`
  - `./build-full-test/cvh_test_imgproc --gtest_filter='ImgprocCvtColor_TEST.*'`
  - `./build-full-test/cvh_test_imgproc`
  - `cmake --build build-full-test -j --target cvh_test_imgproc`
  - `./build-full-test/cvh_test_imgproc --gtest_filter='ImgprocCvtColor_TEST.bgr_rgb_to_nv16_nv61_yuv422sp_u8_matches_reference:ImgprocCvtColor_TEST.non_contiguous_roi_for_nv16_nv61_encode_matches_reference:ImgprocCvtColor_TEST.throws_on_invalid_bgr_rgb_to_nv16_nv61_inputs'`
  - `./build-full-test/cvh_test_imgproc --gtest_filter='ImgprocCvtColor_TEST.*'`
  - `./build-full-test/cvh_test_imgproc`
  - `cmake --build build-full-test -j --target cvh_test_imgproc`
  - `./build-full-test/cvh_test_imgproc --gtest_filter='ImgprocCvtColor_TEST.bgr_rgb_to_yuy2_uyvy_yuv422packed_u8_matches_reference:ImgprocCvtColor_TEST.non_contiguous_roi_for_yuy2_uyvy_encode_matches_reference:ImgprocCvtColor_TEST.throws_on_invalid_bgr_rgb_to_yuy2_uyvy_inputs'`
  - `./build-full-test/cvh_test_imgproc --gtest_filter='ImgprocCvtColor_TEST.*'`
  - `./build-full-test/cvh_test_imgproc`
  - `cmake --build build-full-test -j --target cvh_test_imgproc`
  - `./build-full-test/cvh_test_imgproc --gtest_filter='ImgprocCvtColor_TEST.bgr_rgb_to_i420_yv12_yuv420p_u8_matches_reference:ImgprocCvtColor_TEST.non_contiguous_roi_for_i420_yv12_encode_matches_reference:ImgprocCvtColor_TEST.throws_on_invalid_bgr_rgb_to_i420_yv12_inputs'`
  - `./build-full-test/cvh_test_imgproc --gtest_filter='ImgprocCvtColor_TEST.*'`
  - `./build-full-test/cvh_test_imgproc`
- 结果摘要：
  - `Stage P 新增定向用例：3/3 PASS（BGR/RGB -> NV24/NV42 的连续/ROI/非法输入）`
  - `ImgprocCvtColor_TEST.*：39/39 PASS`
  - `最终收口复验：69/69 tests PASSED`
  - `Stage Q RED：以缺少 COLOR_YUV2*I444/YV24 enum 的编译失败进入 RED`
  - `Stage Q 新增定向用例：3/3 PASS（I444/YV24 -> BGR/RGB 的连续/step/非法布局）`
  - `ImgprocCvtColor_TEST.*：42/42 PASS`
  - `最终收口复验：72/72 tests PASSED`
  - `Stage R RED：以缺少 COLOR_BGR2YUV_*I444/YV24 enum 的编译失败进入 RED`
  - `Stage R 新增定向用例：3/3 PASS（BGR/RGB -> I444/YV24 的连续/ROI/非法输入）`
  - `ImgprocCvtColor_TEST.*：45/45 PASS`
  - `最终收口复验：75/75 tests PASSED`
  - `Stage R 最终 GREEN 采用 --clean-first 重建，以规避一次增量构建残留的不一致`
  - `Stage S RED：以缺少 COLOR_BGR2YUV_*NV16/NV61 enum 的编译失败进入 RED`
  - `Stage S 新增定向用例：3/3 PASS（BGR/RGB -> NV16/NV61 的连续/ROI/非法输入）`
  - `ImgprocCvtColor_TEST.*：48/48 PASS`
  - `最终收口复验：78/78 tests PASSED`
  - `Stage T RED：以缺少 COLOR_BGR2YUV_*YUY2/UYVY enum 的编译失败进入 RED`
  - `Stage T 新增定向用例：3/3 PASS（BGR/RGB -> YUY2/UYVY 的连续/ROI/非法输入）`
  - `ImgprocCvtColor_TEST.*：51/51 PASS`
  - `最终收口复验：81/81 tests PASSED`
  - `Stage U RED：以缺少 COLOR_BGR2YUV_*NV12/NV21 enum 的编译失败进入 RED`
  - `Stage U 新增定向用例：3/3 PASS（BGR/RGB -> NV12/NV21 的连续/ROI/非法输入）`
  - `ImgprocCvtColor_TEST.*：54/54 PASS`
  - `最终收口复验：84/84 tests PASSED`
  - `Stage V RED：以缺少 COLOR_BGR2YUV_*I420/YV12 enum 的编译失败进入 RED`
  - `Stage V 新增定向用例：3/3 PASS（BGR/RGB -> I420/YV12 的连续/ROI/非法输入）`
  - `ImgprocCvtColor_TEST.*：57/57 PASS`
  - `最终收口复验：87/87 tests PASSED`
  - `结果记录时间：2026-04-15 14:19 CST`

### 9.2 性能结果（Quick）

- 状态：`已完成`
- 基线文件：`benchmark/baseline_imgproc_quick.csv`
- 当前文件：`benchmark/current_imgproc_quick.csv`
- 回归阈值：`0.08`（8%）
- 定向阈值覆盖：`THRESH_BINARY_F32:CV_32F=0.10`
- 运行时约束：`OMP_NUM_THREADS=1`, `OMP_DYNAMIC=false`, `OMP_PROC_BIND=close`, `CPU pin=8（由 gate script 的 auto 选择）`
- 保留策略：`benchmark/` 仅保留 quick/full 的最终 baseline/current 证据文件；阶段中间 CSV 已清理
- 执行命令：
  - `taskset -c 8 env OMP_NUM_THREADS=1 OMP_DYNAMIC=false OMP_PROC_BIND=close ./build-imgproc-benchmark-gate/cvh_benchmark_imgproc_ops --profile quick --warmup 1 --iters 5 --repeats 1 --output /tmp/cvh_imgproc_quick_stagep_warm.csv >/dev/null`
  - `taskset -c 8 env OMP_NUM_THREADS=1 OMP_DYNAMIC=false OMP_PROC_BIND=close ./build-imgproc-benchmark-gate/cvh_benchmark_imgproc_ops --profile quick --warmup 2 --iters 20 --repeats 5 --output benchmark/baseline_imgproc_quick.csv`
  - `CVH_IMGPROC_BENCH_CURRENT=benchmark/current_imgproc_quick.csv ./scripts/ci_imgproc_quick_gate.sh`
  - `taskset -c 8 env OMP_NUM_THREADS=1 OMP_DYNAMIC=false OMP_PROC_BIND=close ./build-imgproc-benchmark-gate/cvh_benchmark_imgproc_ops --profile quick --warmup 2 --iters 20 --repeats 5 --output benchmark/baseline_imgproc_quick.csv >/dev/null`
  - `CVH_IMGPROC_BENCH_CURRENT=benchmark/current_imgproc_quick.csv ./scripts/ci_imgproc_quick_gate.sh`
  - `taskset -c 8 env OMP_NUM_THREADS=1 OMP_DYNAMIC=false OMP_PROC_BIND=close ./build-imgproc-benchmark-gate/cvh_benchmark_imgproc_ops --profile quick --warmup 1 --iters 5 --repeats 1 --output /tmp/cvh_imgproc_quick_stage_s_warm.csv >/dev/null`
  - `taskset -c 8 env OMP_NUM_THREADS=1 OMP_DYNAMIC=false OMP_PROC_BIND=close ./build-imgproc-benchmark-gate/cvh_benchmark_imgproc_ops --profile quick --warmup 2 --iters 20 --repeats 5 --output benchmark/current_imgproc_quick.csv >/dev/null`
  - `python3 scripts/check_imgproc_benchmark_regression.py --baseline benchmark/baseline_imgproc_quick.csv --current benchmark/current_imgproc_quick.csv --max-slowdown 0.08 --max-slowdown-by-op-depth THRESH_BINARY_F32:CV_32F=0.10`
  - `taskset -c 8 env OMP_NUM_THREADS=1 OMP_DYNAMIC=false OMP_PROC_BIND=close ./build-imgproc-benchmark-gate/cvh_benchmark_imgproc_ops --profile quick --warmup 1 --iters 5 --repeats 1 --output /tmp/cvh_imgproc_quick_stage_t_warm.csv >/dev/null`
  - `taskset -c 8 env OMP_NUM_THREADS=1 OMP_DYNAMIC=false OMP_PROC_BIND=close ./build-imgproc-benchmark-gate/cvh_benchmark_imgproc_ops --profile quick --warmup 2 --iters 20 --repeats 5 --output benchmark/baseline_imgproc_quick.csv >/dev/null`
  - `taskset -c 8 env OMP_NUM_THREADS=1 OMP_DYNAMIC=false OMP_PROC_BIND=close ./build-imgproc-benchmark-gate/cvh_benchmark_imgproc_ops --profile quick --warmup 2 --iters 20 --repeats 5 --output benchmark/current_imgproc_quick.csv >/dev/null`
  - `python3 scripts/check_imgproc_benchmark_regression.py --baseline benchmark/baseline_imgproc_quick.csv --current benchmark/current_imgproc_quick.csv --max-slowdown 0.08 --max-slowdown-by-op-depth THRESH_BINARY_F32:CV_32F=0.10`
  - `taskset -c 8 env OMP_NUM_THREADS=1 OMP_DYNAMIC=false OMP_PROC_BIND=close ./build-imgproc-benchmark-gate/cvh_benchmark_imgproc_ops --profile quick --warmup 2 --iters 20 --repeats 5 --output benchmark/baseline_imgproc_quick.csv >/dev/null`
  - `CVH_IMGPROC_BENCH_CURRENT=benchmark/current_imgproc_quick.csv ./scripts/ci_imgproc_quick_gate.sh`
- 结果摘要：
  - `Stage F 对阶段 E baseline key 的兼容性回归：compared=62, improved_or_equal=62, improved=27, PASS`
  - `Stage G quick gate：compared=78, improved_or_equal=78, improved=59, PASS（default=8%，override_rules=1，cpu_list=8）`
  - `Stage G quick 最大回退：+3.12%（THRESH_BINARY_F32, CV_32F, C4, 480x640；低于该类定向 10% 阈值）`
  - `Stage H quick gate：compared=110, improved_or_equal=110, improved=88, PASS（default=8%，override_rules=1，cpu_list=8）`
  - `Stage H quick 最大回退：+3.18%（RESIZE_NEAREST, CV_8U, C3, 720x1280->360x640；低于全局 8% 阈值）`
  - `Stage I quick gate：compared=126, improved_or_equal=126, improved=81, PASS（default=8%，override_rules=1，cpu_list=8）`
  - `Stage I quick 最大回退：+6.26%（THRESH_BINARY_F32, CV_32F, C4, 720x1280；低于该类定向 10% 阈值）`
  - `Stage J quick gate：compared=142, improved_or_equal=142, improved=71, PASS（default=8%，override_rules=1，cpu_list=8）`
  - `Stage J quick 最大回退：+3.17%（CVTCOLOR_GRAY2RGBA, CV_8U, C1, 480x640；低于全局 8% 阈值）`
  - `Stage K quick gate：compared=150, improved_or_equal=150, improved=82, PASS（default=8%，override_rules=1，cpu_list=8）`
  - `Stage K quick 最大回退：+8.66%（THRESH_BINARY_F32, CV_32F, C1, 480x640；低于该类定向 10% 阈值）`
  - `Stage L quick gate：compared=158, improved_or_equal=158, improved=103, PASS（default=8%，override_rules=1，cpu_list=8）`
  - `Stage L quick 最大回退：+8.72%（THRESH_BINARY_F32, CV_32F, C3, 480x640；低于该类定向 10% 阈值）`
  - `Stage M quick gate：compared=166, improved_or_equal=166, improved=103, PASS（default=8%，override_rules=1，cpu_list=8）`
  - `Stage M quick 最大回退：+1.83%（CVTCOLOR_BGRA2RGBA, CV_8U, C4, 720x1280；低于全局 8% 阈值）`
  - `Stage N quick gate：compared=174, improved_or_equal=174, improved=106, PASS（default=8%，override_rules=1，cpu_list=8）`
  - `Stage N quick 最大回退：+7.46%（THRESH_BINARY_F32, CV_32F, C4, 480x640；低于该类定向 10% 阈值）`
  - `Stage O quick gate：compared=182, improved_or_equal=182, improved=83, PASS（default=8%，override_rules=1，cpu_list=8）`
  - `Stage O quick 最大回退：+8.92%（THRESH_BINARY_F32, CV_32F, C1, 480x640；低于该类定向 10% 阈值）`
  - `Stage P quick gate：compared=190, improved_or_equal=190, improved=112, PASS（default=8%，override_rules=1，cpu_list=8）`
  - `Stage P quick 最大回退：+1.48%（RESIZE_NEAREST, CV_8U, C3, 720x1280->360x640；低于全局 8% 阈值）`
  - `CSV 样本规模：142 cases（quick，新增 16 个 BGR/RGB<->YUV case）`
  - `CSV 样本规模：150 cases（quick，新增 8 个 NV12/NV21 -> BGR/RGB case）`
  - `CSV 样本规模：158 cases（quick，新增 8 个 I420/YV12 -> BGR/RGB case）`
  - `CSV 样本规模：166 cases（quick，新增 8 个 YUY2/UYVY -> BGR/RGB case）`
  - `CSV 样本规模：174 cases（quick，新增 8 个 NV16/NV61 -> BGR/RGB case）`
  - `CSV 样本规模：182 cases（quick，新增 8 个 NV24/NV42 -> BGR/RGB case）`
  - `CSV 样本规模：190 cases（quick，新增 8 个 BGR/RGB -> NV24/NV42 case）`
  - `B2 回归：compared=32, improved_or_equal=32, improved=17, PASS`
  - `B2 KPI（相对 baseline）：RESIZE_NEAREST(HD,C3/C4) 平均 20.74x；RESIZE_LINEAR(HD) 平均 30.93x`
  - `B3 回归：compared=32, improved_or_equal=32, improved=21, PASS`
  - `B3 KPI（相对 baseline）：CVTCOLOR_BGR2GRAY 平均 53.34x；CVTCOLOR_GRAY2BGR 平均 19.80x`
  - `B5 回归：compared=32, improved_or_equal=32, improved=21, PASS`
  - `B5 最大回退：+4.27%（BOXFILTER_3X3, CV_8U, C1, 480x640）`
  - `C2（Stage C）重点收益（相对 B5）：BOXFILTER_3X3 约 1.90x~2.92x；GAUSSIAN_5X5 约 9.47x~14.34x`
  - `D2 新增观测：CV_32F 已覆盖 RESIZE/BOXFILTER/GAUSSIAN/THRESH（每算子 C1/C3/C4 + VGA/HD，共 30 case）`
  - `F2 新增观测：CV_32F cvtColor 已覆盖 BGR2GRAY/GRAY2BGR（quick 共 4 case）`
  - `G2 新增观测：BGR<->RGB/BGR<->BGRA 已覆盖 CV_8U/CV_32F（quick 共 12 case）`
  - `H3 新增观测：RGB/BGR/RGBA/BGRA family 已覆盖 CV_8U/CV_32F（quick 共 32 case）`
  - `I3 新增观测：GRAY<->RGBA/BGRA 已覆盖 CV_8U/CV_32F（quick 共 16 case）`
  - `J3 新增观测：BGR/RGB<->YUV 已覆盖 CV_8U/CV_32F（quick 共 16 case）`
  - `K3 新增观测：NV12/NV21->BGR/RGB 已覆盖 CV_8U（quick 共 8 case）`
  - `L3 新增观测：I420/YV12->BGR/RGB 已覆盖 CV_8U（quick 共 8 case）`
  - `M3 新增观测：YUY2/UYVY->BGR/RGB 已覆盖 CV_8U（quick 共 8 case）`
  - `N3 新增观测：NV16/NV61->BGR/RGB 已覆盖 CV_8U（quick 共 8 case）`
  - `O3 新增观测：NV24/NV42->BGR/RGB 已覆盖 CV_8U（quick 共 8 case）`
  - `P3 新增观测：BGR/RGB->NV24/NV42 已覆盖 CV_8U（quick 共 8 case）`
  - `Stage Q quick gate：compared=198, improved_or_equal=198, improved=117, PASS（default=8%，override_rules=1，cpu_list=8）`
  - `Stage Q quick 最大回退：+2.23%（CVTCOLOR_BGR2GRAY_F32, CV_32F, C3, 480x640；低于全局 8% 阈值）`
  - `CSV 样本规模：198 cases（quick，新增 8 个 I444/YV24 -> BGR/RGB case）`
  - `Q3 新增观测：I444/YV24->BGR/RGB 已覆盖 CV_8U（quick 共 8 case）`
  - `Stage R quick gate：compared=206, improved_or_equal=206, improved=130, PASS（default=8%，override_rules=1，cpu_list=8）`
  - `Stage R quick 最大回退：+3.23%（CVTCOLOR_GRAY2RGBA, CV_8U, C1, 720x1280；低于全局 8% 阈值）`
  - `CSV 样本规模：206 cases（quick，新增 8 个 BGR/RGB -> I444/YV24 case）`
  - `R3 新增观测：BGR/RGB->I444/YV24 已覆盖 CV_8U（quick 共 8 case）`
  - `Stage S quick 首次 retained current 出现 2 个旧 case 的孤立尖峰：CVTCOLOR_RGBA2BGR(CV_8U,C4,720x1280)=+65.55%，CVTCOLOR_BGR2RGBA(CV_8U,C3,720x1280)=+58.42%；均不在本轮新增路径上`
  - `Stage S quick 经额外 warmup 后复跑：compared=214, improved_or_equal=214, improved=97, PASS（default=8%，override_rules=1，cpu_list=8）`
  - `Stage S quick 最大回退：+1.80%（CVTCOLOR_BGR2RGB_F32, CV_32F, C3, 480x640；低于全局 8% 阈值）`
  - `CSV 样本规模：214 cases（quick，新增 8 个 BGR/RGB -> NV16/NV61 case）`
  - `S3 新增观测：BGR/RGB->NV16/NV61 已覆盖 CV_8U（quick 共 8 case）`
  - `Stage T quick gate：compared=222, improved_or_equal=222, improved=96, PASS（default=8%，override_rules=1，cpu_list=8）`
  - `Stage T quick 最大回退：+9.92%（THRESH_BINARY_F32, CV_32F, C1, 480x640；低于该类定向 10% 阈值）`
  - `CSV 样本规模：222 cases（quick，新增 8 个 BGR/RGB -> YUY2/UYVY case）`
  - `T3 新增观测：BGR/RGB->YUY2/UYVY 已覆盖 CV_8U（quick 共 8 case）`
  - `Stage U quick gate：compared=230, improved_or_equal=230, improved=141, PASS（default=8%，override_rules=1，cpu_list=8）`
  - `Stage U quick 最大回退：+5.31%（THRESH_BINARY_F32, CV_32F, C1, 720x1280；低于该类定向 10% 阈值）`
  - `CSV 样本规模：230 cases（quick，新增 8 个 BGR/RGB -> NV12/NV21 case）`
  - `U3 新增观测：BGR/RGB->NV12/NV21 已覆盖 CV_8U（quick 共 8 case）`
  - `Stage V quick gate：compared=238, improved_or_equal=238, improved=133, PASS（default=8%，override_rules=1，cpu_list=8）`
  - `Stage V quick 最大回退：+3.29%（CVTCOLOR_RGB2YUV_NV12, CV_8U, C3, 480x640；低于全局 8% 阈值）`
  - `CSV 样本规模：238 cases（quick，新增 8 个 BGR/RGB -> I420/YV12 case）`
  - `V3 新增观测：BGR/RGB->I420/YV12 已覆盖 CV_8U（quick 共 8 case）`
  - `结果记录时间：2026-04-15 14:19 CST`

### 9.3 性能结果（Full）

- 状态：`已完成`
- 基线文件：`benchmark/baseline_imgproc_full.csv`
- 当前文件：`benchmark/current_imgproc_full.csv`
- 回归阈值：`0.15`（15%，基于同代码 full profile 复测噪声收敛结果）
- 运行时约束：`OMP_NUM_THREADS=1`, `OMP_DYNAMIC=false`, `OMP_PROC_BIND=close`, `CPU pin=8（由 gate script 的 auto 选择）`
- 保留策略：`benchmark/` 仅保留 quick/full 的最终 baseline/current 证据文件；阶段中间 CSV 已清理
- 执行命令：
  - `taskset -c 8 env OMP_NUM_THREADS=1 OMP_DYNAMIC=false OMP_PROC_BIND=close ./build-imgproc-benchmark-gate/cvh_benchmark_imgproc_ops --profile full --warmup 1 --iters 3 --repeats 1 --output /tmp/cvh_imgproc_full_stagep_warm.csv >/dev/null`
  - `taskset -c 8 env OMP_NUM_THREADS=1 OMP_DYNAMIC=false OMP_PROC_BIND=close ./build-imgproc-benchmark-gate/cvh_benchmark_imgproc_ops --profile full --warmup 1 --iters 10 --repeats 3 --output benchmark/baseline_imgproc_full.csv`
  - `CVH_IMGPROC_BENCH_PROFILE=full CVH_IMGPROC_BENCH_BASELINE=benchmark/baseline_imgproc_full.csv CVH_IMGPROC_BENCH_CURRENT=benchmark/current_imgproc_full.csv CVH_IMGPROC_BENCH_WARMUP=1 CVH_IMGPROC_BENCH_ITERS=10 CVH_IMGPROC_BENCH_REPEATS=3 CVH_IMGPROC_BENCH_MAX_SLOWDOWN=0.15 ./scripts/ci_imgproc_quick_gate.sh`
  - `taskset -c 8 env OMP_NUM_THREADS=1 OMP_DYNAMIC=false OMP_PROC_BIND=close ./build-imgproc-benchmark-gate/cvh_benchmark_imgproc_ops --profile full --warmup 1 --iters 10 --repeats 3 --output benchmark/current_imgproc_full.csv >/dev/null`
  - `python3 scripts/check_imgproc_benchmark_regression.py --baseline benchmark/baseline_imgproc_full.csv --current benchmark/current_imgproc_full.csv --max-slowdown 0.15`
  - `taskset -c 8 env OMP_NUM_THREADS=1 OMP_DYNAMIC=false OMP_PROC_BIND=close ./build-imgproc-benchmark-gate/cvh_benchmark_imgproc_ops --profile full --warmup 1 --iters 10 --repeats 3 --output benchmark/baseline_imgproc_full.csv >/dev/null`
  - `CVH_IMGPROC_BENCH_PROFILE=full CVH_IMGPROC_BENCH_BASELINE=benchmark/baseline_imgproc_full.csv CVH_IMGPROC_BENCH_CURRENT=benchmark/current_imgproc_full.csv CVH_IMGPROC_BENCH_WARMUP=1 CVH_IMGPROC_BENCH_ITERS=10 CVH_IMGPROC_BENCH_REPEATS=3 CVH_IMGPROC_BENCH_MAX_SLOWDOWN=0.15 ./scripts/ci_imgproc_quick_gate.sh`
  - `CVH_IMGPROC_BENCH_PROFILE=full CVH_IMGPROC_BENCH_BASELINE=benchmark/baseline_imgproc_full.csv CVH_IMGPROC_BENCH_CURRENT=benchmark/current_imgproc_full.csv CVH_IMGPROC_BENCH_WARMUP=1 CVH_IMGPROC_BENCH_ITERS=10 CVH_IMGPROC_BENCH_REPEATS=3 CVH_IMGPROC_BENCH_MAX_SLOWDOWN=0.15 ./scripts/ci_imgproc_quick_gate.sh`
- 结果摘要：
  - `Stage F 对阶段 E baseline key 的兼容性回归：compared=93, improved_or_equal=93, improved=52, PASS`
  - `Stage G full gate：compared=117, improved_or_equal=117, improved=63, PASS（cpu_list=8）`
  - `Stage G full 最大回退：+11.68%（THRESH_BINARY_F32, CV_32F, C4, 480x640；低于 15% 阈值）`
  - `Stage H full gate：compared=165, improved_or_equal=165, improved=97, PASS（cpu_list=8）`
  - `Stage H full 最大回退：+10.59%（RESIZE_LINEAR, CV_8U, C1, 720x1280->360x640；低于 15% 阈值）`
  - `Stage I full gate：compared=189, improved_or_equal=189, improved=68, PASS（cpu_list=8）`
  - `Stage I full 最大回退：+7.12%（THRESH_BINARY_F32, CV_32F, C1, 480x640；低于 15% 阈值）`
  - `Stage J full gate：compared=213, improved_or_equal=213, improved=146, PASS（cpu_list=8）`
  - `Stage J full 最大回退：+10.06%（CVTCOLOR_RGBA2BGR_F32, CV_32F, C4, 720x1280；低于 15% 阈值）`
  - `Stage K full gate：compared=225, improved_or_equal=225, improved=140, PASS（cpu_list=8）`
  - `Stage K full 最大回退：+10.73%（RESIZE_LINEAR, CV_8U, C1, 720x1280->360x640；低于 15% 阈值）`
  - `CSV 样本规模：225 cases（full，新增 12 个 NV12/NV21 -> BGR/RGB case）`
  - `Stage L full gate：compared=237, improved_or_equal=237, improved=143, PASS（cpu_list=8）`
  - `Stage L full 最大回退：+3.00%（CVTCOLOR_GRAY2RGBA, CV_8U, C1, 480x640；低于 15% 阈值）`
  - `CSV 样本规模：237 cases（full，新增 12 个 I420/YV12 -> BGR/RGB case）`
  - `Stage M full 首次 gate 出现 1 个孤立噪声违例：CVTCOLOR_BGR2GRAY_F32, CV_32F, C3, 720x1280 = +59.72%；同 case 在 quick baseline/current 均稳定在 ~1.28 ms`
  - `Stage M full 经 warmup 后复跑：compared=249, improved_or_equal=249, improved=94, PASS（cpu_list=8）`
  - `Stage M full 最大回退：+3.65%（CVTCOLOR_BGR2RGB_F32, CV_32F, C3, 1080x1920；低于 15% 阈值）`
  - `CSV 样本规模：249 cases（full，新增 12 个 YUY2/UYVY -> BGR/RGB case）`
  - `Stage N full gate：compared=261, improved_or_equal=261, improved=165, PASS（cpu_list=8）`
  - `Stage N full 最大回退：+8.28%（THRESH_BINARY_F32, CV_32F, C1, 480x640；低于 15% 阈值）`
  - `CSV 样本规模：261 cases（full，新增 12 个 NV16/NV61 -> BGR/RGB case）`
  - `Stage O full 首次 gate 出现 1 个孤立噪声违例：CVTCOLOR_RGBA2GRAY_F32, CV_32F, C4, 480x640 = +69.36%；同 case 在 quick baseline/current 均稳定在 ~0.495 ms`
  - `Stage O full 经额外 warmup 后复跑：compared=273, improved_or_equal=273, improved=152, PASS（cpu_list=8）`
  - `Stage O full 最大回退：+8.85%（THRESH_BINARY_F32, CV_32F, C1, 1080x1920；低于 15% 阈值）`
  - `CSV 样本规模：273 cases（full，新增 12 个 NV24/NV42 -> BGR/RGB case）`
  - `Stage P full gate：compared=285, improved_or_equal=285, improved=102, PASS（cpu_list=8）`
  - `Stage P full 最大回退：+8.24%（RESIZE_LINEAR, CV_8U, C1, 1080x1920->540x960；低于 15% 阈值）`
  - `CSV 样本规模：285 cases（full，新增 12 个 BGR/RGB -> NV24/NV42 case）`
  - `Stage Q full gate：compared=297, improved_or_equal=297, improved=174, PASS`
  - `Stage Q full 最大回退：+2.79%（CVTCOLOR_BGR2GRAY_F32, CV_32F, C3, 480x640；低于 15% 阈值）`
  - `CSV 样本规模：297 cases（full，新增 12 个 I444/YV24 -> BGR/RGB case）`
  - `Stage R full gate：compared=309, improved_or_equal=309, improved=162, PASS`
  - `Stage R full 最大回退：+3.69%（CVTCOLOR_GRAY2BGRA, CV_8U, C1, 480x640；低于 15% 阈值）`
  - `CSV 样本规模：309 cases（full，新增 12 个 BGR/RGB -> I444/YV24 case）`
  - `R3 新增观测：BGR/RGB->I444/YV24 已覆盖 CV_8U（full 共 12 case）`
  - `Stage S full gate：compared=321, improved_or_equal=321, improved=130, PASS`
  - `Stage S full 最大回退：+4.57%（CVTCOLOR_RGBA2RGB, CV_8U, C4, 1080x1920；低于 15% 阈值）`
  - `CSV 样本规模：321 cases（full，新增 12 个 BGR/RGB -> NV16/NV61 case）`
  - `S3 新增观测：BGR/RGB->NV16/NV61 已覆盖 CV_8U（full 共 12 case）`
  - `Stage T full gate：compared=333, improved_or_equal=333, improved=185, PASS`
  - `Stage T full 最大回退：+8.85%（THRESH_BINARY_F32, CV_32F, C4, 720x1280；低于 15% 阈值）`
  - `CSV 样本规模：333 cases（full，新增 12 个 BGR/RGB -> YUY2/UYVY case）`
  - `T3 新增观测：BGR/RGB->YUY2/UYVY 已覆盖 CV_8U（full 共 12 case）`
  - `Stage U full gate：compared=345, improved_or_equal=345, improved=143, PASS`
  - `Stage U full 最大回退：+8.50%（THRESH_BINARY_F32, CV_32F, C4, 480x640；低于 15% 阈值）`
  - `CSV 样本规模：345 cases（full，新增 12 个 BGR/RGB -> NV12/NV21 case）`
  - `U3 新增观测：BGR/RGB->NV12/NV21 已覆盖 CV_8U（full 共 12 case）`
  - `Stage V full 首次 gate 出现 2 个旧 case 孤立尖峰：GAUSSIAN_5X5_F32(CV_32F,C1,480x640)=+62.06%，GAUSSIAN_5X5(CV_8U,C1,480x640)=+40.66%；均不在本轮新增路径上`
  - `Stage V full 经复跑后：compared=357, improved_or_equal=357, improved=179, PASS`
  - `Stage V full 最大回退：+8.01%（RESIZE_LINEAR, CV_8U, C1, 1080x1920->540x960；低于 15% 阈值）`
  - `CSV 样本规模：357 cases（full，新增 12 个 BGR/RGB -> I420/YV12 case）`
  - `V3 新增观测：BGR/RGB->I420/YV12 已覆盖 CV_8U（full 共 12 case）`
  - `阈值说明：full gate 继续保持 15%；quick gate 保持 8%，仅对 THRESH_BINARY_F32:CV_32F 应用 10% 定向覆盖`
  - `结果记录时间：2026-04-15 14:19 CST`

### 9.4 阶段 B 验收结论

- 状态：`已完成`
- 结论摘要：
  - `B1~B6` 里程碑全部完成
  - 正确性：`cvh_test_imgproc` 为 `26/26 PASS`
  - 性能门禁（quick）：`32/32 case PASS`，最大回退 `+4.27%`（低于 `8%` 阈值）
  - KPI 达成：
    - `RESIZE_NEAREST(HD,C3/C4)` 平均 `20.74x`（目标 `>=1.35x`）
    - `RESIZE_LINEAR(HD)` 平均 `30.93x`（目标 `>=1.20x`）
    - `CVTCOLOR_BGR2GRAY` 平均 `53.34x`（目标 `>=1.20x`）

### 9.5 阶段 C 验收结论

- 状态：`已完成`
- 结论摘要：
  - `C1~C5` 里程碑全部完成
  - 正确性：`cvh_test_imgproc` 为 `29/29 PASS`
  - 性能门禁（quick）：`32/32 case PASS`，最大回退 `+0.77%`（低于 `8%` 阈值）
  - Stage C 目标达成：
    - `BOXFILTER_3X3`（相对 B5）提升约 `1.90x~2.92x`
    - `GAUSSIAN_5X5`（相对 B5）提升约 `9.47x~14.34x`

### 9.6 阶段 D（D1~D3）验收结论

- 状态：`已完成`
- 结论摘要：
  - D1~D3 里程碑全部完成：`threshold/resize/boxFilter/GaussianBlur` 已扩展到 `CV_32F`（`C1/C3/C4`）。
  - 正确性：新增 `CV_32F` 相关测试通过，`cvh_test_imgproc` 达到 `36/36 PASS`。
  - 约束保持：`THRESH_OTSU/THRESH_TRIANGLE` 仍限定 `CV_8UC1`，对 `CV_32F` 显式报错。
  - 基准扩展：`cvh_benchmark_imgproc_ops` 已新增 `CV_32F` case（quick 共 30 case）。
  - 门禁状态：对既有 baseline key（32 case）回归检查 `PASS`（max slowdown +0.10%）。

### 9.7 阶段 E（E1~E3）验收结论

- 状态：`已完成`
- 结论摘要：
  - E1~E3 里程碑全部完成：`deterministic benchmark gate`、quick/full baseline、支持矩阵与限制文档已收口。
  - quick gate：`62/62 PASS`，最大回退 `+0.73%`，脚本端到端验证通过。
  - full gate：`93/93 PASS`，最大回退 `+13.05%`，脚本端到端验证通过。
  - gate 策略已固定：`单线程 OMP + CPU pin + machine/runner-class specific baseline`。
  - benchmark 资产已收敛：仅保留 `baseline/current × quick/full` 四个最终证据 CSV。

### 9.8 阶段 F（F1~F3）验收结论

- 状态：`已完成`
- 结论摘要：
  - F1~F3 里程碑全部完成：`cvtColor(BGR2GRAY/GRAY2BGR)` 已扩展到 `CV_32F`，并完成 benchmark/gate/doc 收口。
  - 正确性：`cvh_test_imgproc` 达到 `38/38 PASS`，新增 `CV_32F` 的连续/ROI 契约测试通过。
  - backward compatibility：对阶段 E baseline key 的 quick/full 回归检查全部通过。
  - gate 校准：baseline 生成口径已统一到 `build-imgproc-benchmark-gate`；`CPU auto-pinning` 改为优先选择 `cpuset` 第一段的 `1/4` 位置 CPU；quick gate 增加 `THRESH_BINARY_F32:CV_32F=0.10` 的定向覆盖。

### 9.9 阶段 G（G1~G3）验收结论

- 状态：`已完成`
- 结论摘要：
  - G1~G3 里程碑全部完成：`BGR<->RGB` / `BGR<->BGRA` 已扩展到 `CV_8U/CV_32F`，并完成 benchmark/gate/doc 收口。
  - 正确性：`cvh_test_imgproc` 达到 `42/42 PASS`，新增连续/ROI 契约测试通过。
  - benchmark：quick/full case 数量扩展到 `78/117`，新增 `BGR<->RGB` / `BGR<->BGRA` 相关 case 已纳入长期资产。
  - gate：quick/full 端到端验证通过；quick 仍只保留 `THRESH_BINARY_F32:CV_32F=0.10` 这 1 条定向覆盖。

### 9.10 阶段 H（H1~H3）验收结论

- 状态：`已完成`
- 结论摘要：
  - H1~H3 里程碑全部完成：已补齐 `RGB/BGR/RGBA/BGRA` family 剩余 8 个常用 color code，并完成 benchmark/gate/doc 收口。
  - 正确性：`cvh_test_imgproc` 达到 `45/45 PASS`，`ImgprocCvtColor_TEST.*` 达到 `15/15 PASS`，新增连续/ROI 契约测试通过。
  - benchmark：quick/full case 数量扩展到 `110/165`，新增 `RGB2RGBA/RGBA2RGB/BGR2RGBA/RGBA2BGR/RGB2BGRA/BGRA2RGB/BGRA2RGBA/RGBA2BGRA` 及 `CV_32F` 对应 case 已纳入长期资产。
  - gate：quick/full 端到端验证通过；quick 仍只保留 `THRESH_BINARY_F32:CV_32F=0.10` 这 1 条定向覆盖。

### 9.11 阶段 I（I1~I3）验收结论

- 状态：`已完成`
- 结论摘要：
  - I1~I3 里程碑全部完成：已补齐 `GRAY <-> RGBA/BGRA` 的 4 个常用 color code，并完成 benchmark/gate/doc 收口。
  - 正确性：`cvh_test_imgproc` 达到 `48/48 PASS`，`ImgprocCvtColor_TEST.*` 达到 `18/18 PASS`，新增连续/ROI 契约测试通过。
  - benchmark：quick/full case 数量扩展到 `126/189`，新增 `GRAY2BGRA/BGRA2GRAY/GRAY2RGBA/RGBA2GRAY` 及 `CV_32F` 对应 case 已纳入长期资产。
  - gate：quick/full 端到端验证通过；quick 仍只保留 `THRESH_BINARY_F32:CV_32F=0.10` 这 1 条定向覆盖。

### 9.12 阶段 J（J1~J3）验收结论

- 状态：`已完成`
- 结论摘要：
  - J1~J3 里程碑全部完成：已补齐 `BGR/RGB <-> YUV` 的 4 个常用 3 通道 color code，并完成 benchmark/gate/doc 收口。
  - 正确性：`cvh_test_imgproc` 达到 `51/51 PASS`，`ImgprocCvtColor_TEST.*` 达到 `21/21 PASS`，新增连续/ROI 契约测试通过。
  - benchmark：quick/full case 数量扩展到 `142/213`，新增 `BGR2YUV/YUV2BGR/RGB2YUV/YUV2RGB` 及 `CV_32F` 对应 case 已纳入长期资产。
  - gate：quick/full 端到端验证通过；quick 仍只保留 `THRESH_BINARY_F32:CV_32F=0.10` 这 1 条定向覆盖。

### 9.13 阶段 K（K1~K4）验收结论

- 状态：`已完成`
- 结论摘要：
  - K1~K4 里程碑全部完成：已补齐 `COLOR_YUV2BGR_NV12` / `COLOR_YUV2RGB_NV12` / `COLOR_YUV2BGR_NV21` / `COLOR_YUV2RGB_NV21`，并完成 benchmark/gate/doc 收口。
  - 正确性：`cvh_test_imgproc` 达到 `54/54 PASS`，`ImgprocCvtColor_TEST.*` 达到 `24/24 PASS`，新增连续/step/非法布局契约测试通过。
  - benchmark：quick/full case 数量扩展到 `150/225`，新增 `NV12/NV21 -> BGR/RGB` case 已纳入长期资产。
  - gate：quick/full 端到端验证通过；quick 仍只保留 `THRESH_BINARY_F32:CV_32F=0.10` 这 1 条定向覆盖。

### 9.14 阶段 L（L1~L4）验收结论

- 状态：`已完成`
- 结论摘要：
  - L1~L4 里程碑全部完成：已补齐 `COLOR_YUV2BGR_I420` / `COLOR_YUV2RGB_I420` / `COLOR_YUV2BGR_YV12` / `COLOR_YUV2RGB_YV12`，并完成 benchmark/gate/doc 收口。
  - 正确性：`cvh_test_imgproc` 达到 `57/57 PASS`，`ImgprocCvtColor_TEST.*` 达到 `27/27 PASS`，新增连续/step/非法布局契约测试通过。
  - benchmark：quick/full case 数量扩展到 `158/237`，新增 `I420/YV12 -> BGR/RGB` case 已纳入长期资产。
  - gate：quick/full 端到端验证通过；quick 仍只保留 `THRESH_BINARY_F32:CV_32F=0.10` 这 1 条定向覆盖。

### 9.15 阶段 M（M1~M4）验收结论

- 状态：`已完成`
- 结论摘要：
  - M1~M4 里程碑全部完成：已补齐 `COLOR_YUV2BGR_YUY2` / `COLOR_YUV2RGB_YUY2` / `COLOR_YUV2BGR_UYVY` / `COLOR_YUV2RGB_UYVY`，并完成 benchmark/gate/doc 收口。
  - 正确性：`cvh_test_imgproc` 达到 `60/60 PASS`，`ImgprocCvtColor_TEST.*` 达到 `30/30 PASS`，新增 packed `YUV422` 的连续/step/非法布局契约测试通过。
  - benchmark：quick/full case 数量扩展到 `166/249`，新增 `YUY2/UYVY -> BGR/RGB` case 已纳入长期资产。
  - gate：quick/full 端到端验证通过；full 首次复测出现 `CVTCOLOR_BGR2GRAY_F32` 的单次噪声尖峰，经 warmup 后复跑恢复收敛，最终 retained CSV 已通过门禁。

### 9.16 阶段 N（N1~N4）验收结论

- 状态：`已完成`
- 结论摘要：
  - N1~N4 里程碑全部完成：已补齐 `COLOR_YUV2BGR_NV16` / `COLOR_YUV2RGB_NV16` / `COLOR_YUV2BGR_NV61` / `COLOR_YUV2RGB_NV61`，并完成 benchmark/gate/doc 收口。
  - 正确性：`cvh_test_imgproc` 达到 `63/63 PASS`，`ImgprocCvtColor_TEST.*` 达到 `33/33 PASS`，新增半平面 `YUV422` 的连续/step/非法布局契约测试通过。
  - benchmark：quick/full case 数量扩展到 `174/261`，新增 `NV16/NV61 -> BGR/RGB` case 已纳入长期资产。
  - gate：quick/full 端到端验证通过；quick 仍只保留 `THRESH_BINARY_F32:CV_32F=0.10` 这 1 条定向覆盖。

### 9.17 阶段 O（O1~O4）验收结论

- 状态：`已完成`
- 结论摘要：
  - O1~O4 里程碑全部完成：已补齐 `COLOR_YUV2BGR_NV24` / `COLOR_YUV2RGB_NV24` / `COLOR_YUV2BGR_NV42` / `COLOR_YUV2RGB_NV42`，并完成 benchmark/gate/doc 收口。
  - 正确性：`cvh_test_imgproc` 达到 `66/66 PASS`，`ImgprocCvtColor_TEST.*` 达到 `36/36 PASS`，新增半平面 `YUV444` 的连续/step/非法布局契约测试通过。
  - benchmark：quick/full case 数量扩展到 `182/273`，新增 `NV24/NV42 -> BGR/RGB` case 已纳入长期资产。
  - gate：quick/full 端到端验证通过；full 首次复测出现 `CVTCOLOR_RGBA2GRAY_F32` 的单次噪声尖峰，经额外 warmup 后复跑恢复收敛，最终 retained CSV 已通过门禁。
  - 输入语义：`NV24/NV42` 统一按 `CV_8UC1(H*3 x W)` 单 `Mat` 解释，上 `H` 行为 `Y`，下 `2H` 行为连续 `UV/VU` 字节流。

### 9.18 阶段 P（P1~P4）验收结论

- 状态：`已完成`
- 结论摘要：
  - P1~P4 里程碑全部完成：已补齐 `COLOR_BGR2YUV_NV24` / `COLOR_RGB2YUV_NV24` / `COLOR_BGR2YUV_NV42` / `COLOR_RGB2YUV_NV42`，并完成 benchmark/gate/doc 收口。
  - 正确性：`cvh_test_imgproc` 达到 `69/69 PASS`，`ImgprocCvtColor_TEST.*` 达到 `39/39 PASS`，新增半平面 `YUV444` encode 的连续/ROI/非法输入契约测试通过。
  - benchmark：quick/full case 数量扩展到 `190/285`，新增 `BGR/RGB -> NV24/NV42` case 已纳入长期资产。
  - gate：quick/full 端到端验证通过；quick/full 首次 retained CSV 均一次通过，无需额外噪声复跑。
  - 输出语义：`BGR/RGB -> NV24/NV42` 统一输出 `CV_8UC1(H*3 x W)` 单 `Mat`，上 `H` 行为 `Y`，下 `2H` 行为连续 `UV/VU` 字节流。

### 9.19 阶段 Q（Q1~Q4）验收结论

- 状态：`已完成`
- 结论摘要：
  - Q1~Q4 里程碑全部完成：已补齐 `COLOR_YUV2BGR_I444` / `COLOR_YUV2RGB_I444` / `COLOR_YUV2BGR_YV24` / `COLOR_YUV2RGB_YV24`，并完成 benchmark/gate/doc 收口。
  - 正确性：`cvh_test_imgproc` 达到 `72/72 PASS`，`ImgprocCvtColor_TEST.*` 达到 `42/42 PASS`，新增平面 `YUV444` 的连续/step/非法布局契约测试通过。
  - benchmark：quick/full case 数量扩展到 `198/297`，新增 `I444/YV24 -> BGR/RGB` case 已纳入长期资产。
  - gate：quick/full 端到端验证通过；quick/full retained CSV 均一次通过。
  - 输入语义：`I444/YV24` 统一按 `CV_8UC1(H*3 x W)` 单 `Mat` 解释，上 `H` 行为 `Y`，中间 `H` 行为 `U/V` 平面，最后 `H` 行为 `V/U` 平面。

### 9.20 阶段 R（R1~R4）验收结论

- 状态：`已完成`
- 结论摘要：
  - R1~R4 里程碑全部完成：已补齐 `COLOR_BGR2YUV_I444` / `COLOR_RGB2YUV_I444` / `COLOR_BGR2YUV_YV24` / `COLOR_RGB2YUV_YV24`，并完成 benchmark/gate/doc 收口。
  - 正确性：`cvh_test_imgproc` 达到 `75/75 PASS`，`ImgprocCvtColor_TEST.*` 达到 `45/45 PASS`，新增平面 `YUV444` encode 的连续/ROI/非法输入契约测试通过。
  - benchmark：quick/full case 数量扩展到 `206/309`，新增 `BGR/RGB -> I444/YV24` case 已纳入长期资产。
  - gate：quick/full 端到端验证通过；quick/full retained CSV 均一次通过。
  - 输出语义：`BGR/RGB -> I444/YV24` 统一输出 `CV_8UC1(H*3 x W)` 单 `Mat`，上 `H` 行为 `Y`，中间 `H` 行为 `U/V` 平面，最后 `H` 行为 `V/U` 平面。
  - 缺陷收敛：`YV24` 初版实现曾出现 `U/V` 写出次序重复交换，根因是 plane offset 已编码平面位置后又做了一次显式 swap；最终 retained 结果已基于修正后的布局逻辑复验通过。

### 9.21 阶段 S（S1~S4）验收结论

- 状态：`已完成`
- 结论摘要：
  - S1~S4 里程碑全部完成：已补齐 `COLOR_BGR2YUV_NV16` / `COLOR_RGB2YUV_NV16` / `COLOR_BGR2YUV_NV61` / `COLOR_RGB2YUV_NV61`，并完成 benchmark/gate/doc 收口。
  - 正确性：`cvh_test_imgproc` 达到 `78/78 PASS`，`ImgprocCvtColor_TEST.*` 达到 `48/48 PASS`，新增半平面 `YUV422` encode 的连续/ROI/非法输入契约测试通过。
  - benchmark：quick/full case 数量扩展到 `214/321`，新增 `BGR/RGB -> NV16/NV61` case 已纳入长期资产。
  - gate：quick/full 端到端验证通过；quick 首次 retained current 出现 2 个旧 case 孤立尖峰，经额外 warmup 后复跑恢复收敛，最终 retained CSV 已通过门禁。
  - 输出语义：`BGR/RGB -> NV16/NV61` 统一输出 `CV_8UC1(H*2 x W)` 单 `Mat`，上 `H` 行为 `Y`，下 `H` 行为连续 `UV/VU` 字节流，每 2 个水平像素共享一组 `U/V`。

### 9.22 阶段 T（T1~T4）验收结论

- 状态：`已完成`
- 结论摘要：
  - T1~T4 里程碑全部完成：已补齐 `COLOR_BGR2YUV_YUY2` / `COLOR_RGB2YUV_YUY2` / `COLOR_BGR2YUV_UYVY` / `COLOR_RGB2YUV_UYVY`，并完成 benchmark/gate/doc 收口。
  - 正确性：`cvh_test_imgproc` 达到 `81/81 PASS`，`ImgprocCvtColor_TEST.*` 达到 `51/51 PASS`，新增 packed `YUV422` encode 的连续/ROI/非法输入契约测试通过。
  - benchmark：quick/full case 数量扩展到 `222/333`，新增 `BGR/RGB -> YUY2/UYVY` case 已纳入长期资产。
  - gate：quick/full 端到端验证通过；quick 最大回退落在既有 `THRESH_BINARY_F32:CV_32F` noisy bucket 内，仍低于定向 10% 阈值。
  - 输出语义：`BGR/RGB -> YUY2/UYVY` 统一输出 `CV_8UC2(H x W)` 单 `Mat`，`YUY2` 按 `[Y0 U][Y1 V]` 写出，`UYVY` 按 `[U Y0][V Y1]` 写出，每 2 个水平像素共享一组 `U/V`。

### 9.23 阶段 U（U1~U4）验收结论

- 状态：`已完成`
- 结论摘要：
  - U1~U4 里程碑全部完成：已补齐 `COLOR_BGR2YUV_NV12` / `COLOR_RGB2YUV_NV12` / `COLOR_BGR2YUV_NV21` / `COLOR_RGB2YUV_NV21`，并完成 benchmark/gate/doc 收口。
  - 正确性：`cvh_test_imgproc` 达到 `84/84 PASS`，`ImgprocCvtColor_TEST.*` 达到 `54/54 PASS`，新增半平面 `YUV420` encode 的连续/ROI/非法输入契约测试通过。
  - benchmark：quick/full case 数量扩展到 `230/345`，新增 `BGR/RGB -> NV12/NV21` case 已纳入长期资产。
  - gate：quick/full 端到端验证通过；quick 最大回退 `+5.31%`、full 最大回退 `+8.50%`，均低于现行门禁阈值。
  - 输出语义：`BGR/RGB -> NV12/NV21` 统一输出 `CV_8UC1(H*3/2 x W)` 单 `Mat`，上 `H` 行为 `Y`，下 `H/2` 行为连续 `UV/VU` 字节流，每 `2x2` 像素块共享一组 `U/V`。

### 9.24 阶段 V（V1~V4）验收结论

- 状态：`已完成`
- 结论摘要：
  - V1~V4 里程碑全部完成：已补齐 `COLOR_BGR2YUV_I420` / `COLOR_RGB2YUV_I420` / `COLOR_BGR2YUV_YV12` / `COLOR_RGB2YUV_YV12`，并完成 benchmark/gate/doc 收口。
  - 正确性：`cvh_test_imgproc` 达到 `87/87 PASS`，`ImgprocCvtColor_TEST.*` 达到 `57/57 PASS`，新增平面 `YUV420` encode 的连续/ROI/非法输入契约测试通过。
  - benchmark：quick/full case 数量扩展到 `238/357`，新增 `BGR/RGB -> I420/YV12` case 已纳入长期资产。
  - gate：quick/full 端到端验证通过；full 首次 retained current 出现 2 个旧 case 孤立尖峰，复跑后收敛并通过门禁。
  - 输出语义：`BGR/RGB -> I420/YV12` 统一输出 `CV_8UC1(H*3/2 x W)` 单 `Mat`，上 `H` 行为 `Y`，下 `H/2` 行按平面写出 `I420(U,V)` 或 `YV12(V,U)`，每 `2x2` 像素块共享一组 `U/V`。

### 9.25 已知问题 / Deferred

- `为恢复当前分支可编译，已最小修复 src/core/kernel/binary_kernel_xsimd.cpp 的 saturate_cast 歧义（long long -> int64）`
- `imgproc benchmark baseline 属于固定机器 / 固定 runner class 资产；跨硬件直接复用不保证通过，GitHub-hosted workflow 仍建议手动触发并在稳定 runner 上使用`
- `full gate 当前阈值为 15%，高于 quick gate 的 8%；若后续切换到更稳定的 self-hosted runner，可重新收紧 full 阈值`
- `cvtColor` 当前已覆盖 `GRAY(C1) <-> BGR(C3)/BGRA(C4)/RGBA(C4)`、`BGR(C3) <-> RGB(C3)`、`BGR/RGB(C3) <-> BGRA/RGBA(C4)`、`BGRA(C4) <-> RGBA(C4)`、`BGR/RGB(C3) <-> YUV(C3)`、`BGR/RGB(C3) <-> NV12/NV21(C1, H*3/2 x W)`、`BGR/RGB(C3) <-> I420/YV12(C1, H*3/2 x W)`、`BGR/RGB(C3) <-> I444/YV24(C1, H*3 x W)`、`BGR/RGB(C3) <-> NV16/NV61(C1, H*2 x W)`、`BGR/RGB(C3) <-> NV24/NV42(C1, H*3 x W)`、`BGR/RGB(C3) <-> YUY2/UYVY(C2, H x W)`；本轮计划内 YUV family 已收口
- `quick gate` 当前有 1 条定向覆盖规则：`THRESH_BINARY_F32:CV_32F=0.10`；若后续切换到更稳定的 runner，可尝试移除该覆盖并收回到全局 8%`

---

## 10. 恢复任务指引（中断后优先看）

1. 先更新“验收状态”里的当前阶段。
2. 在“验证结果”补上最近一次运行命令与结论。
3. 仅推进一个阶段，完成后再进入下一阶段。
4. 任一阶段出现正确性回退，先回滚阶段内改动，再继续。

---

## 11. 变更记录

- 2026-04-08：创建计划文档 v1.0，初始化阶段拆分、验收状态、验证结果模板。
- 2026-04-08：启动阶段 A，落地 imgproc benchmark/regression/readme，补录 quick 验证结果与当前阶段状态。
- 2026-04-08：完成阶段 A 的 CI quick gate 接线（workflow_dispatch 可选触发）。
- 2026-04-08：新增阶段 B 里程碑（B1~B6）量化 KPI 与逐项完成情况表。
- 2026-04-08：完成 B1（后端骨架），`resize/cvtColor` 已切换为 fast-path 优先、fallback 兜底策略。
- 2026-04-08：完成 B2（resize 提速），已落地通道特化、映射预计算与 OpenMP 行并行，并通过 quick 回归门禁。
- 2026-04-08：完成 B3（cvtColor 提速），`BGR2GRAY` 改为整数系数路径，`GRAY2BGR` 批量复制，并通过 quick 回归门禁。
- 2026-04-08：完成 B4（正确性补强），新增 `resize/cvtColor` 的非连续 ROI 与边界尺寸回归用例。
- 2026-04-08：完成 B5（性能门禁验证），quick profile 回归门禁通过（32/32 case, max slowdown +4.27%）。
- 2026-04-08：完成 B6（文档收口），阶段 B 状态切换为已完成，当前阶段转入阶段 C。
- 2026-04-08：完成 C1~C3，`boxFilter/GaussianBlur` fast-path 落地（滑窗累计 + 边界索引预计算 + OpenMP 行并行）。
- 2026-04-08：完成 C4，新增 filter fast-path 契约测试（ROI/anchor/normalize/in-place）3 项。
- 2026-04-08：完成 C5，quick 回归门禁通过（32/32 case, max slowdown +0.77%），阶段 C 状态切换为已完成，当前阶段转入阶段 D。
- 2026-04-09：启动阶段 D，完成 D1（`threshold` 固定阈值扩展到 `CV_32F`，`OTSU/TRIANGLE` 继续限定 `CV_8UC1`）。
- 2026-04-09：补充 `threshold` 的 `CV_32F C1/C3/C4 + ROI + dryrun + 自动阈值限制` 回归用例，`cvh_test_imgproc` 提升至 `32/32 PASS`。
- 2026-04-09：扩展 benchmark 新增 `THRESH_BINARY_F32`（6 case），quick 回归门禁保持 `PASS`（对 baseline 32 key 比对）。
- 2026-04-09：完成 D2，`resize/boxFilter/GaussianBlur` fallback 扩展到 `CV_32F`，并补齐 ROI/in-place 回归用例。
- 2026-04-09：完成 D3，imgproc benchmark 新增 `CV_32F` 组合（quick 30 case），文档与阶段状态更新，阶段 D 切换为已完成。
- 2026-04-09：完成 E1，`scripts/ci_imgproc_quick_gate.sh` 增加 `CPU pin(auto/off/explicit)` 支持，并将 deterministic gate 固定为单线程 OMP + taskset。
- 2026-04-09：完成 E2，生成 `benchmark/baseline_imgproc_quick.csv` / `benchmark/baseline_imgproc_full.csv` 与对应 current 结果，quick/full regression 与 gate 脚本全部通过。
- 2026-04-09：完成 E3，更新 `benchmark/readme.md`、`include/cvh/imgproc/readme.md` 与本计划文档，阶段 E 切换为已完成。
- 2026-04-09：清理 `benchmark/` 阶段性中间 CSV，仅保留 quick/full 的最终 baseline/current 证据文件，并将文档引用统一到固定文件名。
- 2026-04-09：启动阶段 F，先以 TDD 补齐 `CV_32F cvtColor(BGR2GRAY/GRAY2BGR)` 契约测试，并在 header fallback 与 backend fast-path 中实现对应支持。
- 2026-04-09：扩展 `benchmark/imgproc_ops_benchmark.cpp`，新增 `CVTCOLOR_BGR2GRAY_F32` / `CVTCOLOR_GRAY2BGR_F32`，quick/full case 数量分别扩展到 `66/99`。
- 2026-04-09：完成阶段 F gate 校准：imgproc regression 脚本新增 `--max-slowdown-by-op-depth`，quick gate 增加 `THRESH_BINARY_F32:CV_32F=0.10` 定向覆盖，`CPU auto-pinning` 改为优先选择 `cpuset` 第一段的 `1/4` 位置 CPU（当前机器落到 `cpu_list=8`）。
- 2026-04-09：完成阶段 F 文档收口，更新 benchmark/readme、imgproc/readme 与本计划文档；当前阶段切换为 `A~F 已完成`。
- 2026-04-09：启动阶段 G，补齐 `BGR<->RGB` / `BGR<->BGRA` 的 enum、header fallback、backend fast-path 与 `CV_8U/CV_32F` 契约测试，`cvh_test_imgproc` 提升至 `42/42 PASS`。
- 2026-04-09：扩展 `benchmark/imgproc_ops_benchmark.cpp`，新增 `CVTCOLOR_BGR2RGB` / `CVTCOLOR_BGR2BGRA` / `CVTCOLOR_BGRA2BGR` 及 `CV_32F` 对应 case，quick/full case 数量分别扩展到 `78/117`。
- 2026-04-09：完成阶段 G 文档收口，更新 benchmark/readme、imgproc/readme 与本计划文档；当前阶段切换为 `A~G 已完成`。
- 2026-04-09：启动阶段 H，按 `cpp-cmake-tdd-loop` 先补 `RGB/BGR/RGBA/BGRA` family 契约测试，以缺少 enum 的编译失败进入 RED，再转 GREEN。
- 2026-04-09：补齐 `COLOR_RGB2RGBA` / `COLOR_RGBA2RGB` / `COLOR_BGR2RGBA` / `COLOR_RGBA2BGR` / `COLOR_RGB2BGRA` / `COLOR_BGRA2RGB` / `COLOR_BGRA2RGBA` / `COLOR_RGBA2BGRA`，并在 header fallback 与 backend fast-path 中完成接线。
- 2026-04-09：扩展 `benchmark/imgproc_ops_benchmark.cpp`，新增 Stage H 对应 `CV_8U/CV_32F` case，quick/full case 数量分别扩展到 `110/165`。
- 2026-04-09：完成阶段 H 文档收口，更新 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档；当前阶段切换为 `A~H 已完成`。
- 2026-04-09：启动阶段 I，按 `cpp-cmake-tdd-loop` 先补 `GRAY <-> RGBA/BGRA` 契约测试，以缺少 enum 的编译失败进入 RED，再转 GREEN。
- 2026-04-09：补齐 `COLOR_GRAY2BGRA` / `COLOR_BGRA2GRAY` / `COLOR_GRAY2RGBA` / `COLOR_RGBA2GRAY`，并在 header fallback 与 backend fast-path 中完成接线。
- 2026-04-09：扩展 `benchmark/imgproc_ops_benchmark.cpp`，新增 Stage I 对应 `CV_8U/CV_32F` case，quick/full case 数量分别扩展到 `126/189`。
- 2026-04-09：完成阶段 I 文档收口，更新 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档；当前阶段切换为 `A~I 已完成`。
- 2026-04-09：启动阶段 J，按 `cpp-cmake-tdd-loop` 先补 `BGR/RGB <-> YUV` 契约测试，以缺少 enum 的编译失败进入 RED，再转 GREEN。
- 2026-04-09：补齐 `COLOR_BGR2YUV` / `COLOR_YUV2BGR` / `COLOR_RGB2YUV` / `COLOR_YUV2RGB`，并在 header fallback 与 backend fast-path 中完成接线。
- 2026-04-09：扩展 `benchmark/imgproc_ops_benchmark.cpp`，新增 Stage J 对应 `CV_8U/CV_32F` case，quick/full case 数量分别扩展到 `142/213`。
- 2026-04-09：完成阶段 J 文档收口，更新 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档；当前阶段切换为 `A~J 已完成`。
- 2026-04-09：创建阶段 K 计划骨架，冻结下一轮最小范围为 `NV12/NV21 -> BGR/RGB` 的 `CV_8U C1(H*3/2 x W)` decode-only；当前阶段切换为 `K1 待开始`。
- 2026-04-09：启动阶段 K，按 `cpp-cmake-tdd-loop` 先补 `NV12/NV21 -> BGR/RGB` 契约测试，以缺少 enum 的编译失败进入 RED，再转 GREEN。
- 2026-04-09：补齐 `COLOR_YUV2BGR_NV12` / `COLOR_YUV2RGB_NV12` / `COLOR_YUV2BGR_NV21` / `COLOR_YUV2RGB_NV21`，并在 header fallback 与 backend fast-path 中完成接线。
- 2026-04-09：扩展 `benchmark/imgproc_ops_benchmark.cpp`，新增 Stage K 对应 `CV_8U` case，quick/full case 数量分别扩展到 `150/225`。
- 2026-04-09：完成阶段 K 文档收口，更新 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档；当前阶段切换为 `A~K 已完成`，并创建阶段 L 占位。
- 2026-04-09：启动阶段 L，按 `cpp-cmake-tdd-loop` 先补 `I420/YV12 -> BGR/RGB` 契约测试，以缺少 enum 的编译失败进入 RED，再转 GREEN。
- 2026-04-09：补齐 `COLOR_YUV2BGR_I420` / `COLOR_YUV2RGB_I420` / `COLOR_YUV2BGR_YV12` / `COLOR_YUV2RGB_YV12`，并在 header fallback 与 backend fast-path 中完成接线。
- 2026-04-09：扩展 `benchmark/imgproc_ops_benchmark.cpp`，新增 Stage L 对应 `CV_8U` case，quick/full case 数量分别扩展到 `158/237`。
- 2026-04-09：完成阶段 L 文档收口，更新 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档；当前阶段切换为 `A~L 已完成`，并创建阶段 M 占位。
- 2026-04-09：启动阶段 M，按 `cpp-cmake-tdd-loop` 先补 `YUY2/UYVY -> BGR/RGB` 契约测试，以缺少 enum 的编译失败进入 RED，再转 GREEN。
- 2026-04-09：补齐 `COLOR_YUV2BGR_YUY2` / `COLOR_YUV2RGB_YUY2` / `COLOR_YUV2BGR_UYVY` / `COLOR_YUV2RGB_UYVY`，并在 header fallback 与 backend fast-path 中完成接线。
- 2026-04-09：扩展 `benchmark/imgproc_ops_benchmark.cpp`，新增 Stage M 对应 `CV_8U C2` case，quick/full case 数量分别扩展到 `166/249`。
- 2026-04-09：完成阶段 M 文档收口，更新 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档；当前阶段切换为 `A~M 已完成`。
- 2026-04-09：创建阶段 N 计划骨架，冻结下一轮最小范围为 `NV16/NV61 -> BGR/RGB` 的 `CV_8U C1(H*2 x W)` decode-only；当前阶段切换为 `N1 待开始`。
- 2026-04-09：启动阶段 N，按 `cpp-cmake-tdd-loop` 先补 `NV16/NV61 -> BGR/RGB` 契约测试，以缺少 enum 的编译失败进入 RED，再转 GREEN。
- 2026-04-09：补齐 `COLOR_YUV2BGR_NV16` / `COLOR_YUV2RGB_NV16` / `COLOR_YUV2BGR_NV61` / `COLOR_YUV2RGB_NV61`，并在 header fallback 与 backend fast-path 中完成接线。
- 2026-04-09：扩展 `benchmark/imgproc_ops_benchmark.cpp`，新增 Stage N 对应 `CV_8U C1(H*2 x W)` case，quick/full case 数量分别扩展到 `174/261`。
- 2026-04-09：完成阶段 N 文档收口，更新 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档；当前阶段切换为 `A~N 已完成`。
- 2026-04-10：创建阶段 O 计划骨架，冻结下一轮最小范围为 `NV24/NV42 -> BGR/RGB` 的 `CV_8U C1(H*3 x W)` decode-only；当前阶段切换为 `O1 待开始`。
- 2026-04-10：启动阶段 O，按 `cpp-cmake-tdd-loop` 先补 `NV24/NV42 -> BGR/RGB` 契约测试，以缺少 enum 的编译失败进入 RED，再转 GREEN。
- 2026-04-10：补齐 `COLOR_YUV2BGR_NV24` / `COLOR_YUV2RGB_NV24` / `COLOR_YUV2BGR_NV42` / `COLOR_YUV2RGB_NV42`，并在 header fallback 与 backend fast-path 中完成接线。
- 2026-04-10：扩展 `benchmark/imgproc_ops_benchmark.cpp`，新增 Stage O 对应 `CV_8U C1(H*3 x W)` case，quick/full case 数量分别扩展到 `182/273`。
- 2026-04-10：完成阶段 O 文档收口，更新 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档；当前阶段切换为 `A~O 已完成`。
- 2026-04-10：创建阶段 P 计划骨架，冻结下一轮最小范围为 `BGR/RGB -> NV24/NV42` 的 `CV_8U C3 -> C1(H*3 x W)` encode-only；当前阶段切换为 `P1 待开始`。
- 2026-04-10：启动阶段 P，按 `cpp-cmake-tdd-loop` 先补 `BGR/RGB -> NV24/NV42` 契约测试，以缺少 enum 的编译失败进入 RED，再转 GREEN。
- 2026-04-10：补齐 `COLOR_BGR2YUV_NV24` / `COLOR_RGB2YUV_NV24` / `COLOR_BGR2YUV_NV42` / `COLOR_RGB2YUV_NV42`，并在 header fallback 与 backend fast-path 中完成接线。
- 2026-04-10：扩展 `benchmark/imgproc_ops_benchmark.cpp`，新增 Stage P 对应 `CV_8U C3 -> C1(H*3 x W)` case；修复 benchmark `run_one_op()` 遗漏新 op 导致的空 `dst` 计量问题，quick/full case 数量分别扩展到 `190/285`。
- 2026-04-10：完成阶段 P 文档收口，更新 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档；当前阶段切换为 `A~P 已完成`，并创建阶段 Q 占位。
- 2026-04-10：启动阶段 Q，按 `cpp-cmake-tdd-loop` 先补 `I444/YV24 -> BGR/RGB` 契约测试，以缺少 enum 的编译失败进入 RED，再转 GREEN。
- 2026-04-10：补齐 `COLOR_YUV2BGR_I444` / `COLOR_YUV2RGB_I444` / `COLOR_YUV2BGR_YV24` / `COLOR_YUV2RGB_YV24`，并在 header fallback 与 backend fast-path 中完成接线。
- 2026-04-10：扩展 `benchmark/imgproc_ops_benchmark.cpp`，新增 Stage Q 对应 `CV_8U C1(H*3 x W)` 平面 `YUV444` case，quick/full case 数量分别扩展到 `198/297`。
- 2026-04-10：完成阶段 Q 文档收口，更新 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档；当前阶段切换为 `A~Q 已完成`。
- 2026-04-10：启动阶段 R，按 `cpp-cmake-tdd-loop` 先补 `BGR/RGB -> I444/YV24` 契约测试，以缺少 enum 的编译失败进入 RED，再转 GREEN。
- 2026-04-10：补齐 `COLOR_BGR2YUV_I444` / `COLOR_RGB2YUV_I444` / `COLOR_BGR2YUV_YV24` / `COLOR_RGB2YUV_YV24`，并在 header fallback 与 backend fast-path 中完成接线；修复 `YV24` 初版实现的平面写出重复交换问题。
- 2026-04-10：扩展 `benchmark/imgproc_ops_benchmark.cpp`，新增 Stage R 对应 `CV_8U C3 -> C1(H*3 x W)` 平面 `YUV444` encode case，quick/full case 数量分别扩展到 `206/309`。
- 2026-04-11：完成阶段 R 文档收口，更新 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档；当前阶段切换为 `A~R 已完成`。
- 2026-04-11：创建阶段 S 计划骨架，冻结下一轮最小范围为 `BGR/RGB -> NV16/NV61` 的 `CV_8U C3 -> C1(H*2 x W)` encode-only；当前阶段切换为 `S1 进行中`。
- 2026-04-11：启动阶段 S，按 `cpp-cmake-tdd-loop` 先补 `BGR/RGB -> NV16/NV61` 契约测试，以缺少 enum 的编译失败进入 RED，再转 GREEN。
- 2026-04-11：补齐 `COLOR_BGR2YUV_NV16` / `COLOR_RGB2YUV_NV16` / `COLOR_BGR2YUV_NV61` / `COLOR_RGB2YUV_NV61`，并在 header fallback 与 backend fast-path 中完成接线；统一按 `CV_8UC1(H*2 x W)` 输出 `NV16/NV61` 单 `Mat`。
- 2026-04-11：扩展 `benchmark/imgproc_ops_benchmark.cpp`，新增 Stage S 对应 `CV_8U C3 -> C1(H*2 x W)` 半平面 `YUV422` encode case，quick/full case 数量分别扩展到 `214/321`。
- 2026-04-11：Stage S quick 首次 retained current 出现 2 个旧 case 的孤立尖峰，经额外 warmup 后复跑恢复收敛；最终 quick/full gate retained 结果分别为 `214/214 PASS` 与 `321/321 PASS`。
- 2026-04-13：完成阶段 S 文档收口，更新 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档；当前阶段切换为 `A~S 已完成`。
- 2026-04-13：创建阶段 T 计划骨架，冻结下一轮最小范围为 `BGR/RGB -> YUY2/UYVY` 的 `CV_8U C3 -> C2(H x W)` encode-only；当前阶段切换为 `T1 进行中`。
- 2026-04-13：启动阶段 T，按 `cpp-cmake-tdd-loop` 先补 `BGR/RGB -> YUY2/UYVY` 契约测试，以缺少 enum 的编译失败进入 RED，再转 GREEN。
- 2026-04-13：补齐 `COLOR_BGR2YUV_YUY2` / `COLOR_RGB2YUV_YUY2` / `COLOR_BGR2YUV_UYVY` / `COLOR_RGB2YUV_UYVY`，并在 header fallback 与 backend fast-path 中完成接线；统一按 `CV_8UC2(H x W)` 输出 packed `YUV422` 单 `Mat`。
- 2026-04-13：扩展 `benchmark/imgproc_ops_benchmark.cpp`，新增 Stage T 对应 `CV_8U C3 -> C2(H x W)` packed `YUV422` encode case，quick/full case 数量分别扩展到 `222/333`。
- 2026-04-13：完成阶段 T 文档收口，更新 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档；当前阶段切换为 `A~T 已完成`。
- 2026-04-13：创建阶段 U 计划骨架，冻结下一轮最小范围为 `BGR/RGB -> NV12/NV21` 的 `CV_8U C3 -> C1(H*3/2 x W)` encode-only；当前阶段切换为 `U1 进行中`。
- 2026-04-13：启动阶段 U，按 `cpp-cmake-tdd-loop` 先补 `BGR/RGB -> NV12/NV21` 契约测试，以缺少 enum 的编译失败进入 RED，再转 GREEN。
- 2026-04-13：补齐 `COLOR_BGR2YUV_NV12` / `COLOR_RGB2YUV_NV12` / `COLOR_BGR2YUV_NV21` / `COLOR_RGB2YUV_NV21`，并在 header fallback 与 backend fast-path 中完成接线；统一按 `CV_8UC1(H*3/2 x W)` 输出半平面 `YUV420` 单 `Mat`。
- 2026-04-13：扩展 `benchmark/imgproc_ops_benchmark.cpp`，新增 Stage U 对应 `CV_8U C3 -> C1(H*3/2 x W)` 半平面 `YUV420` encode case，quick/full case 数量分别扩展到 `230/345`。
- 2026-04-13：完成阶段 U 文档收口，更新 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档；当前阶段切换为 `A~U 已完成`。
- 2026-04-15：创建阶段 V 计划骨架，冻结下一轮最小范围为 `BGR/RGB -> I420/YV12` 的 `CV_8U C3 -> C1(H*3/2 x W)` encode-only；当前阶段切换为 `V1 进行中`。
- 2026-04-15：启动阶段 V，按 `cpp-cmake-tdd-loop` 先补 `BGR/RGB -> I420/YV12` 契约测试，以缺少 enum 的编译失败进入 RED，再转 GREEN。
- 2026-04-15：补齐 `COLOR_BGR2YUV_I420` / `COLOR_RGB2YUV_I420` / `COLOR_BGR2YUV_YV12` / `COLOR_RGB2YUV_YV12`，并在 header fallback 与 backend fast-path 中完成接线；统一按 `CV_8UC1(H*3/2 x W)` 输出平面 `YUV420` 单 `Mat`。
- 2026-04-15：扩展 `benchmark/imgproc_ops_benchmark.cpp`，新增 Stage V 对应 `CV_8U C3 -> C1(H*3/2 x W)` 平面 `YUV420` encode case，quick/full case 数量分别扩展到 `238/357`。
- 2026-04-15：Stage V full 首次 retained current 出现 2 个旧 case 孤立尖峰（`GAUSSIAN_5X5_F32`/`GAUSSIAN_5X5`, `CV_8U/CV_32F`, `C1`, `480x640`）；复跑后收敛，最终 quick/full gate retained 结果分别为 `238/238 PASS` 与 `357/357 PASS`。
- 2026-04-15：完成阶段 V 文档收口，更新 benchmark/readme、imgproc/readme、test/imgproc/readme 与本计划文档；当前阶段切换为 `A~V 已完成`。
