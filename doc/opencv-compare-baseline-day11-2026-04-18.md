# OpenCV Compare Day11 基线规范（quick/stable/full）

更新时间：2026-04-18

## 1. 目标

锁定 `cvh_benchmark_compare` 的三档运行口径，确保：

- 开发期快速反馈（`quick`）
- 回归期低噪声对比（`stable`）
- 发布前扩展覆盖（`full`）

并为每次 compare 输出生成可追踪运行指纹（`*.meta.json`）。

## 2. 三档参数

- `quick`：`warmup=1, iters=5, repeats=1`
- `stable`：`warmup=2, iters=20, repeats=5`
- `full`：`warmup=1, iters=10, repeats=3`

说明：

- 上述参数为 `opencv_compare/run_compare.sh` 的 profile 默认值。
- 可被 CLI（`--warmup/--iters/--repeats`）或环境变量覆盖。

## 3. 产物约定

- 当前结果：
  - CSV：`opencv_compare/results/current_compare_<profile>.csv`
  - Markdown：`doc/opencv_compare_<profile>.md`
  - Metadata：`opencv_compare/results/current_compare_<profile>.csv.meta.json`
- 基线结果（`--baseline`）：
  - CSV：`opencv_compare/results/baseline_compare_<profile>.csv`
  - Markdown：`doc/opencv_compare_baseline_<profile>.md`
  - Metadata：`opencv_compare/results/baseline_compare_<profile>.csv.meta.json`

## 4. 指纹字段（meta）

`meta.json` 至少包含：

- 样本参数：`profile/warmup/iters/repeats/sample_count`
- 运行时：`threads/omp_dynamic/omp_proc_bind`
- 构建：`build_type`
- 环境：`system/arch/cpu_model`
- 版本：`repo_git_commit/opencv_git_commit`

并提供 `fingerprint` 子对象用于后续 gate 一致性校验。

## 5. 推荐命令

1. 生成稳定基线（建议）：

```bash
./opencv_compare/run_compare.sh --profile stable --baseline
```

2. 生成当前结果（同档位对比）：

```bash
./opencv_compare/run_compare.sh --profile stable
```

3. 本地快速迭代：

```bash
./opencv_compare/run_compare.sh --profile quick
```
