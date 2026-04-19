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
  - 覆盖：`imgproc` 基础热点算子（`resize nearest/linear (CV_8U/CV_32F)`, `cvtColor BGR2GRAY/GRAY2BGR/GRAY2BGRA/BGRA2GRAY/GRAY2RGBA/RGBA2GRAY/BGR2RGB/BGR2BGRA/BGRA2BGR/RGB2RGBA/RGBA2RGB/BGR2RGBA/RGBA2BGR/RGB2BGRA/BGRA2RGB/BGRA2RGBA/RGBA2BGRA/BGR2YUV/YUV2BGR/RGB2YUV/YUV2RGB (CV_8U/CV_32F)`, `cvtColor BGR2YUV_NV12/RGB2YUV_NV12/BGR2YUV_NV21/RGB2YUV_NV21/BGR2YUV_I420/RGB2YUV_I420/BGR2YUV_YV12/RGB2YUV_YV12/BGR2YUV_NV16/RGB2YUV_NV16/BGR2YUV_NV61/RGB2YUV_NV61/BGR2YUV_YUY2/RGB2YUV_YUY2/BGR2YUV_UYVY/RGB2YUV_UYVY/BGR2YUV_NV24/RGB2YUV_NV24/BGR2YUV_NV42/RGB2YUV_NV42/BGR2YUV_I444/RGB2YUV_I444/BGR2YUV_YV24/RGB2YUV_YV24/YUV2BGR_NV12/YUV2RGB_NV12/YUV2BGR_NV21/YUV2RGB_NV21/YUV2BGR_I420/YUV2RGB_I420/YUV2BGR_YV12/YUV2RGB_YV12/YUV2BGR_I444/YUV2RGB_I444/YUV2BGR_YV24/YUV2RGB_YV24/YUV2BGR_NV16/YUV2RGB_NV16/YUV2BGR_NV61/YUV2RGB_NV61/YUV2BGR_NV24/YUV2RGB_NV24/YUV2BGR_NV42/YUV2RGB_NV42/YUV2BGR_YUY2/YUV2RGB_YUY2/YUV2BGR_UYVY/YUV2RGB_UYVY (CV_8U)`, `threshold(CV_8U/CV_32F fixed)`, `boxFilter(CV_8U/CV_32F)`, `GaussianBlur(CV_8U/CV_32F)`）
  - 数据维度：`CV_8U/CV_32F` + 合法 `channel` 组合 + `quick/full` 分辨率组合；`NV12/NV21` 与 `I420/YV12` decode/encode 均按 `C1(H*3/2 x W)` 布局生成，`I444/YV24` 与 `NV24/NV42` decode/encode 使用 `C1(H*3 x W)` 输入/输出布局，`NV16/NV61` decode/encode 使用 `C1(H*2 x W)` 输入/输出布局，`YUY2/UYVY` decode/encode 使用 `C2(H x W)` 输入/输出布局
  - 其中 `I444/YV24` 的 `C1(H*3 x W)` 语义为：上 `H` 行 `Y`，中间 `H` 行为 `U/V` 平面，最后 `H` 行为 `V/U` 平面；decode 以该布局作为输入，encode 以该布局作为输出
  - `NV24/NV42` 的 `C1(H*3 x W)` 语义为：上 `H` 行 `Y`，下 `2H` 行为连续 `UV` / `VU` 字节流；decode 以该布局作为输入，encode 以该布局作为输出
  - `NV12/NV21` 的 `C1(H*3/2 x W)` 语义为：上 `H` 行 `Y`，下 `H/2` 行为连续 `UV` / `VU` 字节流；decode 与 encode 都以该布局作为输入/输出，且每 `2x2` 像素共享一组 `U/V`
  - `I420/YV12` 的 `C1(H*3/2 x W)` 语义为：上 `H` 行 `Y`，下 `H/2` 行按平面排列；`I420` 为 `U` 后 `V`，`YV12` 为 `V` 后 `U`；decode 与 encode 都以该布局作为输入/输出，且每 `2x2` 像素共享一组 `U/V`
  - `NV16/NV61` 的 `C1(H*2 x W)` 语义为：上 `H` 行 `Y`，下 `H` 行为连续 `UV` / `VU` 字节流；decode 以该布局作为输入，encode 以该布局作为输出，且每 2 个水平像素共享一组 `U/V`
  - `YUY2/UYVY` 的 `C2(H x W)` 语义为：`YUY2` 按 `[Y0 U][Y1 V]` 写出，`UYVY` 按 `[U Y0][V Y1]` 写出；decode 以该布局作为输入，encode 以该布局作为输出，且每 2 个水平像素共享一组 `U/V`
  - 输出：标准 CSV（stdout）或 `--output <file>`

- 回归检查脚本：`scripts/check_core_benchmark_regression.py`
  - 对比 baseline/current CSV
  - 支持全局阈值 `--max-slowdown`
  - 支持分桶阈值 `--max-slowdown-by-op-depth`（按 `op+depth` 或单维覆盖）
  - 超过阈值时返回非 0（可用于 CI gate）

- 可执行程序：`cvh_benchmark_imgproc_filter`
  - 源码：`benchmark/imgproc_filter_benchmark.cpp`
  - 覆盖：`boxFilter/GaussianBlur`（`CV_8U`，`C1/C3/C4`，continuous/ROI）
  - 输出字段：`op,kernel,depth,channels,shape,layout,border,dispatch_path,ms_per_iter`
  - `dispatch_path` 典型值：`fallback / box3x3 / box_generic / gauss3x3 / gauss_separable`
  - 支持 `--dispatch auto|optimized-only|fallback-only`（用于 A/B 对照）

- 回归检查脚本：`scripts/check_imgproc_filter_benchmark_regression.py`
  - 对比 baseline/current CSV
  - 支持全局阈值 `--max-slowdown`
  - 支持分桶阈值 `--max-slowdown-by-op-kernel`（按 `op+kernel` 或单维覆盖）
  - 超过阈值时返回非 0（可用于 CI gate）

- 速度对照脚本：`scripts/report_imgproc_filter_speedup.py`
  - 对比 baseline/candidate CSV（按相同 case）
  - 输出 `geomean/median/min/max speedup`
  - 可用 `--fail-below-speedup` 设最低加速门槛
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

5. 运行 imgproc filter quick 基准并导出结果：

```bash
./build-full-test/cvh_benchmark_imgproc_filter --profile quick --output benchmark/current_filter_quick.csv
```

6. 生成一次 filter 基线（例如当前主分支）：

```bash
cp benchmark/current_filter_quick.csv benchmark/baseline_filter_quick.csv
```

7. 对比 filter 回归（默认允许最多 8% 变慢）：

```bash
python3 scripts/check_imgproc_filter_benchmark_regression.py \
  --baseline benchmark/baseline_filter_quick.csv \
  --current benchmark/current_filter_quick.csv \
  --max-slowdown 0.08
```

8. 量化 filter 加速（A/B 对照：forced fallback vs optimized）：

```bash
./build-full-test/cvh_benchmark_imgproc_filter --profile quick --dispatch fallback-only --output benchmark/filter_quick_fallback.csv
./build-full-test/cvh_benchmark_imgproc_filter --profile quick --dispatch optimized-only --output benchmark/filter_quick_optimized.csv

python3 scripts/report_imgproc_filter_speedup.py \
  --baseline benchmark/filter_quick_fallback.csv \
  --candidate benchmark/filter_quick_optimized.csv \
  --fail-below-speedup 1.0
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
taskset -c <stable-cpu> env OMP_NUM_THREADS=1 OMP_DYNAMIC=false OMP_PROC_BIND=close \
./build-full-test/cvh_benchmark_imgproc_ops --profile quick --warmup 2 --iters 20 --repeats 5 --output benchmark/baseline_imgproc_quick.csv
```

3. 生成当前版本结果：

```bash
taskset -c <stable-cpu> env OMP_NUM_THREADS=1 OMP_DYNAMIC=false OMP_PROC_BIND=close \
./build-full-test/cvh_benchmark_imgproc_ops --profile quick --warmup 2 --iters 20 --repeats 5 --output benchmark/current_imgproc_quick.csv
```

4. 使用 compare 工作流进行 CVH / OpenCV 对比：

```bash
./scripts/ci_compare_log_only.sh
```

## Imgproc 证据文件保留规则

- 版本库内仅保留 4 个最终证据 CSV：
  - `benchmark/baseline_imgproc_quick.csv`
  - `benchmark/current_imgproc_quick.csv`
  - `benchmark/baseline_imgproc_full.csv`
  - `benchmark/current_imgproc_full.csv`
- 阶段性中间产物（如 `*_b2.csv`、`*_d1.csv`、`*_e3.csv`、`*_ci_*.csv`）用于开发期验证，收口后应清理。
