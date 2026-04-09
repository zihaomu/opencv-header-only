# `benchmark` 目录规划

## 目录职责

提供性能基准与回归检查，防止优化退化和跨版本性能漂移。

## 规划目标

- 建立可重复的 benchmark 入口（固定数据、固定线程、固定输出格式）。
- 优先覆盖 `core` 热点算子，再覆盖 `imgproc`。
- 可选对比 OpenCV 或仓库历史版本。

## 阶段计划

### P1：基准框架

- 统一 CLI 参数：输入规模、线程数、warmup、repeat。
- 输出统一为 `csv/json`，便于 CI 归档。

### P2：Core 基线

- 覆盖 `add/mul/convertTo/transpose/gemm` 等高频路径。
- 分离标量路径与 SIMD 路径数据。

### P3：Imgproc 基线

- 覆盖 `cvtColor/resize/GaussianBlur`。
- 建立不同分辨率下的吞吐与延迟对比。

## 完成定义（DoD）

- 关键算子有可追踪性能曲线。
- PR 可基于 benchmark 报告判断是否有明显性能回退。

## 当前可用工具

- 可执行程序：`cvh_benchmark_core_ops`
  - 源码：`benchmark/core_ops_benchmark.cpp`
  - 覆盖：`core` 二元基础算子与 `compare(Mat,Mat)`（含多 `depth` / `channel` / `shape`）
  - 输出：标准 CSV（stdout）或 `--output <file>`

- 可执行程序：`cvh_benchmark_imgproc_ops`
  - 源码：`benchmark/imgproc_ops_benchmark.cpp`
  - 覆盖：`imgproc` 基础热点算子（`resize nearest/linear (CV_8U/CV_32F)`, `cvtColor`, `threshold(CV_8U/CV_32F fixed)`, `boxFilter(CV_8U/CV_32F)`, `GaussianBlur(CV_8U/CV_32F)`）
  - 数据维度：`CV_8U` + `C1/C3/C4`（按算子合法组合）+ `quick/full` 分辨率组合
  - 输出：标准 CSV（stdout）或 `--output <file>`

- 回归检查脚本：`scripts/check_core_benchmark_regression.py`
  - 对比 baseline/current CSV
  - 支持全局阈值 `--max-slowdown`
  - 支持分桶阈值 `--max-slowdown-by-op-depth`（按 `op+depth` 或单维覆盖）
  - 超过阈值时返回非 0（可用于 CI gate）

- 回归检查脚本：`scripts/check_imgproc_benchmark_regression.py`
  - 对比 baseline/current CSV
  - key：`profile+op+depth+channels+shape`
  - 支持全局阈值 `--max-slowdown`（默认 `0.08`，即 8%）
  - 超过阈值时返回非 0（可用于 CI gate）

## 使用示例

0. （可选）启用 OpenMP 构建（对 `compare` 等已并行化 kernel 提升明显）：

```bash
cmake -S . -B build-full-test -DCVH_USE_OPENMP=ON
cmake --build build-full-test -j --target cvh_benchmark_core_ops
```

1. 运行 quick 基准并导出结果：

```bash
./build-full-test/cvh_benchmark_core_ops --profile quick --warmup 2 --iters 20 --repeats 5 --output benchmark/current_quick.csv
```

2. 生成一次基线（例如当前主分支）：

```bash
cp benchmark/current_quick.csv benchmark/baseline_quick.csv
```

3. 对比回归（默认允许最多 8% 变慢）：

```bash
python3 scripts/check_core_benchmark_regression.py \
  --baseline benchmark/baseline_quick.csv \
  --current benchmark/current_quick.csv \
  --max-slowdown 0.08
```

4. 对热点分桶设置阈值（例如对 `CV_16F` 和 compare 单独放宽）：

```bash
python3 scripts/check_core_benchmark_regression.py \
  --baseline benchmark/baseline_quick.csv \
  --current benchmark/current_quick.csv \
  --max-slowdown 0.08 \
  --max-slowdown-by-op-depth CV_16F=0.20 \
  --max-slowdown-by-op-depth CMP_GT:CV_16F=0.25
```

## 建议流程

- 日常提交：使用 `quick` profile 做回归门禁。
- 周期性评估：使用 `full` profile 做深度扫描（更慢，但覆盖更多组合）。

## Imgproc 使用示例

1. 编译 imgproc benchmark：

```bash
cmake -S . -B build-full-test -DCVH_USE_OPENMP=ON
cmake --build build-full-test -j --target cvh_benchmark_imgproc_ops
```

2. 生成 quick 基线：

```bash
taskset -c 0 env OMP_NUM_THREADS=1 OMP_DYNAMIC=false OMP_PROC_BIND=close \
./build-full-test/cvh_benchmark_imgproc_ops --profile quick --warmup 2 --iters 20 --repeats 5 --output benchmark/baseline_imgproc_quick.csv
```

3. 生成当前版本结果：

```bash
taskset -c 0 env OMP_NUM_THREADS=1 OMP_DYNAMIC=false OMP_PROC_BIND=close \
./build-full-test/cvh_benchmark_imgproc_ops --profile quick --warmup 2 --iters 20 --repeats 5 --output benchmark/current_imgproc_quick.csv
```

4. 执行回归检查（默认允许最多 8% 变慢）：

```bash
python3 scripts/check_imgproc_benchmark_regression.py \
  --baseline benchmark/baseline_imgproc_quick.csv \
  --current benchmark/current_imgproc_quick.csv \
  --max-slowdown 0.08
```

## CI quick gate（可选）

- 脚本入口：`scripts/ci_imgproc_quick_gate.sh`
- workflow：`.github/workflows/ci.yml` 中 `imgproc_quick_gate`（`workflow_dispatch` 手动触发）
- 默认基线：`benchmark/baseline_imgproc_quick.csv`
- 默认使用固定 OpenMP 运行时参数：`OMP_NUM_THREADS=1`, `OMP_DYNAMIC=false`, `OMP_PROC_BIND=close`
- 如果系统提供 `taskset`，gate 脚本会默认将 benchmark 进程绑定到当前 `cpuset` 允许范围内的第一个 CPU，降低进程迁移带来的测量漂移
- 这样做是为了降低共享 CI/开发机上的测量噪声；本地多线程 exploratory benchmark 可手动覆盖
- baseline 属于“机器/runner 类别相关资产”，应在固定机器或固定 runner class 上生成与使用；跨硬件直接复用 baseline 不保证通过
- 可通过环境变量覆盖：
  - `CVH_IMGPROC_BENCH_BASELINE`
  - `CVH_IMGPROC_BENCH_CURRENT`
  - `CVH_IMGPROC_BENCH_PROFILE`
  - `CVH_IMGPROC_BENCH_WARMUP`
  - `CVH_IMGPROC_BENCH_ITERS`
  - `CVH_IMGPROC_BENCH_REPEATS`
  - `CVH_IMGPROC_BENCH_MAX_SLOWDOWN`
  - `CVH_IMGPROC_BENCH_THREADS`
  - `CVH_IMGPROC_BENCH_OMP_DYNAMIC`
  - `CVH_IMGPROC_BENCH_OMP_PROC_BIND`
  - `CVH_IMGPROC_BENCH_CPU_LIST`（`auto`/`off`/显式 CPU 列表）

## Imgproc Full Gate（可选）

- workflow：`.github/workflows/ci.yml` 中 `imgproc_full_gate`（`workflow_dispatch` 手动触发）
- 默认参数：`profile=full`, `warmup=1`, `iters=10`, `repeats=3`, `max_slowdown=0.15`
- 默认基线：`benchmark/baseline_imgproc_full.csv`
- 同样建议在固定机器或固定 runner class 上生成和使用 full baseline

## Imgproc 证据文件保留规则

- 版本库内仅保留 4 个最终证据 CSV：
  - `benchmark/baseline_imgproc_quick.csv`
  - `benchmark/current_imgproc_quick.csv`
  - `benchmark/baseline_imgproc_full.csv`
  - `benchmark/current_imgproc_full.csv`
- 阶段性中间产物（如 `*_b2.csv`、`*_d1.csv`、`*_e3.csv`、`*_ci_*.csv`）用于开发期验证，收口后应清理。
