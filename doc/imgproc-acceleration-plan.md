# Imgproc 模块加速计划（可中断/可恢复）

- 文档版本：v1.6
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

- 当前 fast-path 已覆盖 `resize / cvtColor / boxFilter / GaussianBlur` 的 `CV_8U C1/C3/C4` 主路径；其余组合回落 fallback。
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

## 5. 类型与通道支持矩阵（目标态）

| 算子 | Wave-1 | Wave-2 | 备注 |
|---|---|---|---|
| resize | `CV_8U C1/C3/C4` | `CV_32F C1/C3/C4` | `INTER_NEAREST/EXACT/LINEAR` |
| cvtColor | `CV_8U C1/C3` | （按需求扩展） | 先做 `BGR2GRAY/GRAY2BGR` |
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

当前执行阶段：`本轮 A~E 已完成，当前处于可验收/可继续下一轮优化状态`

---

## 9. 验证结果

> 说明：此区先建模板。后续每阶段完成后补真实数据与命令输出摘要。

### 9.1 功能正确性

- 状态：`已完成`
- 执行命令：
  - `cmake --build build-full-test -j --target cvh_test_imgproc && ./build-full-test/cvh_test_imgproc`
- 结果摘要：
  - `36/36 tests PASSED（含 D1/D2 新增 7 个 CV_32F 用例）`
  - `最终收口复验：36/36 tests PASSED`
  - `执行时间：2026-04-09 14:23 CST`

### 9.2 性能结果（Quick）

- 状态：`已完成`
- 基线文件：`benchmark/baseline_imgproc_quick.csv`
- 当前文件：`benchmark/current_imgproc_quick.csv`
- 回归阈值：`0.08`（8%）
- 运行时约束：`OMP_NUM_THREADS=1`, `OMP_DYNAMIC=false`, `OMP_PROC_BIND=close`, `CPU pin=0（由 taskset/auto 选择）`
- 保留策略：`benchmark/` 仅保留 quick/full 的最终 baseline/current 证据文件；阶段中间 CSV 已清理
- 执行命令：
  - `taskset -c 0 env OMP_NUM_THREADS=1 OMP_DYNAMIC=false OMP_PROC_BIND=close ./build-full-test/cvh_benchmark_imgproc_ops --profile quick --warmup 2 --iters 20 --repeats 5 --output benchmark/baseline_imgproc_quick.csv`
  - `taskset -c 0 env OMP_NUM_THREADS=1 OMP_DYNAMIC=false OMP_PROC_BIND=close ./build-full-test/cvh_benchmark_imgproc_ops --profile quick --warmup 2 --iters 20 --repeats 5 --output benchmark/current_imgproc_quick.csv`
  - `python3 scripts/check_imgproc_benchmark_regression.py --baseline benchmark/baseline_imgproc_quick.csv --current benchmark/current_imgproc_quick.csv --max-slowdown 0.08`
  - `./scripts/ci_imgproc_quick_gate.sh`
- 结果摘要：
  - `Stage E quick regression：compared=62, improved_or_equal=62, improved=33, PASS`
  - `Stage E quick 最大回退：+0.73%（THRESH_BINARY_F32, CV_32F, C3, 480x640；低于 8% 阈值）`
  - `CSV 样本规模：62 cases（quick，含 30 个 CV_32F case）`
  - `Stage E quick gate 脚本验证：compared=62, improved_or_equal=62, improved=39, PASS（cpu_list=0）`
  - `B2 回归：compared=32, improved_or_equal=32, improved=17, PASS`
  - `B2 KPI（相对 baseline）：RESIZE_NEAREST(HD,C3/C4) 平均 20.74x；RESIZE_LINEAR(HD) 平均 30.93x`
  - `B3 回归：compared=32, improved_or_equal=32, improved=21, PASS`
  - `B3 KPI（相对 baseline）：CVTCOLOR_BGR2GRAY 平均 53.34x；CVTCOLOR_GRAY2BGR 平均 19.80x`
  - `B5 回归：compared=32, improved_or_equal=32, improved=21, PASS`
  - `B5 最大回退：+4.27%（BOXFILTER_3X3, CV_8U, C1, 480x640）`
  - `C2（Stage C）重点收益（相对 B5）：BOXFILTER_3X3 约 1.90x~2.92x；GAUSSIAN_5X5 约 9.47x~14.34x`
  - `D2 新增观测：CV_32F 已覆盖 RESIZE/BOXFILTER/GAUSSIAN/THRESH（每算子 C1/C3/C4 + VGA/HD，共 30 case）`
  - `执行时间：2026-04-09 14:11 CST`

### 9.3 性能结果（Full）

- 状态：`已完成`
- 基线文件：`benchmark/baseline_imgproc_full.csv`
- 当前文件：`benchmark/current_imgproc_full.csv`
- 回归阈值：`0.15`（15%，基于同代码 full profile 复测噪声收敛结果）
- 运行时约束：`OMP_NUM_THREADS=1`, `OMP_DYNAMIC=false`, `OMP_PROC_BIND=close`, `CPU pin=0（由 taskset/auto 选择）`
- 保留策略：`benchmark/` 仅保留 quick/full 的最终 baseline/current 证据文件；阶段中间 CSV 已清理
- 执行命令：
  - `taskset -c 0 env OMP_NUM_THREADS=1 OMP_DYNAMIC=false OMP_PROC_BIND=close ./build-full-test/cvh_benchmark_imgproc_ops --profile full --warmup 1 --iters 10 --repeats 3 --output benchmark/baseline_imgproc_full.csv`
  - `taskset -c 0 env OMP_NUM_THREADS=1 OMP_DYNAMIC=false OMP_PROC_BIND=close ./build-full-test/cvh_benchmark_imgproc_ops --profile full --warmup 1 --iters 10 --repeats 3 --output benchmark/current_imgproc_full.csv`
  - `python3 scripts/check_imgproc_benchmark_regression.py --baseline benchmark/baseline_imgproc_full.csv --current benchmark/current_imgproc_full.csv --max-slowdown 0.15`
  - `CVH_IMGPROC_BENCH_PROFILE=full CVH_IMGPROC_BENCH_BASELINE=benchmark/baseline_imgproc_full.csv CVH_IMGPROC_BENCH_CURRENT=benchmark/current_imgproc_full.csv CVH_IMGPROC_BENCH_WARMUP=1 CVH_IMGPROC_BENCH_ITERS=10 CVH_IMGPROC_BENCH_REPEATS=3 CVH_IMGPROC_BENCH_MAX_SLOWDOWN=0.15 ./scripts/ci_imgproc_quick_gate.sh`
- 结果摘要：
  - `Stage E full regression：compared=93, improved_or_equal=93, improved=36, PASS`
  - `Stage E full 最大回退：+13.05%（RESIZE_LINEAR, CV_8U, C1, 720x1280->360x640；低于 15% 阈值）`
  - `CSV 样本规模：93 cases（full）`
  - `Stage E full gate 脚本验证：compared=93, improved_or_equal=93, improved=37, PASS（cpu_list=0）`
  - `阈值说明：同代码 full profile 在当前机器上的复测抖动集中在单个 case 的 11%~13%，因此将 full gate 固定为 15%，quick gate 保持 8%`
  - `执行时间：2026-04-09 14:20 CST`

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

### 9.8 已知问题 / Deferred

- `为恢复当前分支可编译，已最小修复 src/core/kernel/binary_kernel_xsimd.cpp 的 saturate_cast 歧义（long long -> int64）`
- `imgproc benchmark baseline 属于固定机器 / 固定 runner class 资产；跨硬件直接复用不保证通过，GitHub-hosted workflow 仍建议手动触发并在稳定 runner 上使用`
- `full gate 当前阈值为 15%，高于 quick gate 的 8%；若后续切换到更稳定的 self-hosted runner，可重新收紧 full 阈值`
- `cvtColor` 仍未扩展到 `CV_32F` 或更广的颜色空间组合，这一项留到下一轮类型扩展`

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
