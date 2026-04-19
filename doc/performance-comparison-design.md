# Header-Only OpenCV 性能对比设计

## 测试来源（优先复用 OpenCV 官方用例）

- 测试数据来源：`/Users/moo/workssd/my_project/opencv_test/opencv_extra-4.x/testdata`
- 测试代码来源：`/Users/moo/workssd/my_project/opencv_test/opencv-4.13.0/modules/*/test`
- 复用策略：
  - 接口语义对齐优先移植官方 `test` case（contract 子集先行）。
  - 性能/回归继续使用本仓 benchmark + smoke，与官方 case 并行验证。

## 1. 背景与目标

项目目标不只是接口对齐，还需要给用户直观看到性能表现。  
本设计用于回答三个问题：

1. `cvh` 与 OpenCV 的行为是否一致（正确性基线）？
2. `cvh` 在关键算子上相对 OpenCV 的速度如何（性能对比）？
3. PR 是否引入明显性能回退（持续门禁）？

## 2. 设计原则

- 同机对比：同一 runner、同一线程配置下对比 `cvh` 与 OpenCV，避免跨机噪声。
- 先正确后性能：所有性能样本默认要求通过结果一致性校验。
- 可复现：固定输入、固定 warmup/repeat、固定线程与亲和性策略。
- 可解释：输出结构化结果，能定位到 `op + dtype + channels + shape + threads`。

## 3. 评测范围（首批）

### 3.1 Core 算子

- `add`
- `sub`
- `mul`
- `gemm`

### 3.2 Imgproc 算子

- `erode`
- `dilate`
- `sobel`
- `gaussian`（`GaussianBlur`）
- `canny`

### 3.3 维度矩阵

- dtype：`CV_8U`, `CV_32F`
- channels：`C1`, `C3`, `C4`
- shape：`256x256`, `640x480`, `1280x720`, `1920x1080`
- threads：`1`, `2`, `4`, `8`

### 3.4 边界 case（第二批纳入）

- ROI/non-contiguous
- 不同 kernel（3/5/7）
- 不同 border 模式
- gemm 的不同 `M/N/K` 组合（方阵、瘦高、胖宽）

## 4. 基准框架设计

### 4.1 执行模型

- 统一 benchmark runner。
- 两个实现后端：`impl=cvh` 与 `impl=opencv`。
- 同一份输入在同一进程顺序执行两种实现。
- 每条样本都做 output 校验（误差阈值按 dtype 设置）。

### 4.2 统计方法

- 每 case 执行：`warmup + repeats`
- 记录：`median_ms`, `p95_ms`, `stdev_ms`
- 对外主指标：`speedup = opencv_median_ms / cvh_median_ms`

### 4.3 输出格式（建议）

CSV 字段建议：

`impl,op,dtype,channels,shape,threads,warmup,repeats,median_ms,p95_ms,stdev_ms,ok`

对比汇总字段建议：

`op,dtype,channels,shape,threads,opencv_ms,cvh_ms,speedup,delta_pct,ok`

## 5. 对用户的直观展示

### 5.1 Scoreboard（建议自动生成）

- 每个算子一张表：`OpenCV(ms) / CVH(ms) / Speedup(x)`。
- 颜色标识：`speedup > 1.0` 绿色，`< 1.0` 红色。
- 增加“线程扩展曲线”（1/2/4/8）观察并行收益。

### 5.2 建议产物

- `benchmark/reports/scoreboard.md`
- `benchmark/reports/scoreboard.json`
- （可选）`benchmark/reports/scoreboard.html`

## 6. CI 策略

### 6.1 PR（强门禁）

- 跑 quick 子集（覆盖核心热点组合）。
- 固定单线程对比（`threads=1`）+ 必要的 `threads=4` 抽样。
- 强制 `base_ref(main)` vs `PR HEAD` 同机对比。
- 阈值策略：
  - 全局阈值（如 8%）
  - 按算子/数据类型的局部阈值覆盖

### 6.2 Nightly（观察+回归）

- 跑 full 矩阵（含更多 shape/thread/kernel/border）。
- 输出完整 scoreboard，作为趋势跟踪。

### 6.3 关键约束

- 不使用跨机器静态 baseline 做强门禁。
- 若运行指纹（线程、亲和性、采样预算）不一致，标记 `reduced confidence`。

## 7. 与当前仓库的落地映射

已存在能力（可复用）：

- `benchmark/imgproc_ops_benchmark.cpp`
- `scripts/check_imgproc_benchmark_regression.py`
- `scripts/ci_imgproc_quick_gate.sh`

建议新增：

- `benchmark/opencv_compare_benchmark.cpp`（或在现有 benchmark 增加 `--impl cvh|opencv`）
- `scripts/report_scoreboard.py`（将对比 CSV 生成 Markdown/JSON 报告）
- `benchmark/policy_compare.json`（集中管理对比阈值与矩阵）

## 8. 分阶段实施

### P0（1-2 天）

- 先打通 `add/sub/mul/gemm + GaussianBlur/Sobel` 的 cvh/opencv 同机对比。
- 产出最小 `scoreboard.md`。

### P1（2-4 天）

- 补 `erode/dilate`，补 channels/shape/thread 矩阵。
- 接入 PR quick 对比门禁。

### P2（持续）

- 补 ROI、border、kernel 深度覆盖。
- 建立 nightly 趋势报告。

## 9. 验收标准（DoD）

- 用户可在一份报告中直接看到每个关键算子的 `OpenCV vs CVH` 速度对比。
- PR 能对关键热点算子性能回退进行稳定拦截。
- 报告可复现（同机重复运行波动在可控范围）。

性能测试需要固定opencv版本和编译方式：
TODO

性能测试需要涉及到的算子列表：
| 优先级 | 算子                      | 原因                 |
| --- | ----------------------- | ------------------ |
| S   | `resize`                | 图像预处理最高频           |
| S   | `cvtColor`              | BGR/RGB/GRAY 转换极高频 |
| S   | `convertTo`             | 8U/32F 转换、AI 预处理常用 |
| S   | `GaussianBlur`          | 基础滤波代表             |
| S   | `threshold`             | 二值化、mask 生成常用      |
| A   | `Sobel`                 | 梯度/边缘基础算子          |
| A   | `add/subtract/multiply` | pixel-wise 基础能力    |
| A   | `absdiff`               | 图像差分常用             |
| A   | `blur/boxFilter`        | 基础平滑               |
| A   | `bitwise_*`             | mask 操作常用          |
| A   | `erode/dilate`          | 二值图形态学常用           |
| B   | `copyMakeBorder`        | border 基础设施        |
| B   | `split/merge`           | channel 操作         |
| B   | `LUT`                   | 快速像素映射             |
| B   | `warpAffine`            | 几何变换常用             |
| C   | `findContours`          | 传统 CV 常用但复杂        |
| C   | `connectedComponents`   | 二值区域分析             |
| C   | `equalizeHist`          | 图像增强               |
| C   | `medianBlur`            | 非线性滤波              |
| C   | `filter2D/sepFilter2D`  | 通用滤波框架             |
