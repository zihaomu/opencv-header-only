# `benchmark` Framework

这个目录的目标不是临时跑几个数字，而是给 `opencv-header-only` 建立一套可以长期演进的性能判断系统。框架围绕两个模块展开：

- `core`：以 `Mat` 和基础数组算子为中心。
- `imgproc`：以图像预处理/后处理常用算子为中心。

框架同时支持两种 benchmark 模式：

- **内部回归模式**：旧版本 header-only 和当前版本 header-only 对比，用来指导提速过程。
- **OpenCV 对比模式**：当前 header-only 实现和官方 OpenCV 主仓库实现对比，用来展示差距和选择优化优先级。

## Mode A: Internal Header-only Regression

内部回归模式回答的问题是：这次改动让 `cvh` 自己变快了，还是变慢了？

约束：

- 只比较 header-only 产物，不依赖 OpenCV，不依赖需要编译的 `.cpp` 扩展层。
- baseline 和 candidate 必须使用同一组 case、同一套输入生成规则、同一套编译参数和同一台机器。
- baseline 可以来自旧 commit、上一版发布产物、或同一二进制内强制 scalar fallback 的诊断行。
- candidate 默认使用当前 checkout 的 `cvh::headers` 或 `cvh::headers_fast`。
- 结果用于本项目优化决策，可以作为 CI gate。

推荐输出位置：

```text
benchmark/results/internal/<suite>/<profile>/baseline.csv
benchmark/results/internal/<suite>/<profile>/current.csv
benchmark/results/internal/<suite>/<profile>/report.md
benchmark/results/internal/<suite>/<profile>/meta.json
```

推荐实现名：

| Implementation | 含义 |
|---|---|
| `cvh_headers_baseline` | 旧版本 `cvh::headers` 或旧版本默认 header-only target。 |
| `cvh_headers_current` | 当前版本 `cvh::headers`。 |
| `cvh_headers_fast_current` | 当前版本 `cvh::headers_fast`。 |
| `scalar_fallback` | 同一二进制内强制 fallback 的诊断路径，只用于拆内核成本。 |
| `opencv_ui_fastpath` | 同一二进制内直接 OpenCV UI fast path 的诊断路径。 |

当前已可用的纯 header-only benchmark：

| Target | Scope | 状态 |
|---|---|---|
| `cvh_benchmark_cvtcolor_bgr2gray_header` | `CV_8UC3` `BGR2GRAY` / `RGB2GRAY`，含 scalar/public/direct UI/micro rows。 | 可用于 imgproc 内部诊断。 |
| `cvh_benchmark_resize_bilinear_header` | `CV_8UC1` `INTER_LINEAR` exact 2x downsample，含 scalar/public/direct UI/micro rows。 | 可用于 imgproc 内部诊断。 |

当前需要迁移的 legacy benchmark：

| Target | 当前问题 | 目标 |
|---|---|---|
| `cvh_benchmark_core_ops` | 仍链接旧 `cvh::native`。 | 拆成 `cvh_benchmark_core_mat_header`，只链接 `cvh::headers` / `cvh::headers_fast`。 |
| `cvh_benchmark_imgproc_ops` | 仍链接旧 `cvh::native`。 | 并入或替换为 `cvh_benchmark_imgproc_header`。 |
| `cvh_benchmark_imgproc_filter` | 仍链接旧 `cvh::native`，但已有 dispatch A/B 思路。 | 迁到 header-only filter suite，保留 forced fallback/fast-path 诊断能力。 |

最小运行示例：

```bash
cmake -S . -B build-bench \
  -DCMAKE_BUILD_TYPE=Release \
  -DCVH_BUILD_NATIVE_BACKEND=OFF \
  -DCVH_BUILD_TESTS=OFF \
  -DCVH_BUILD_BENCHMARKS=ON

cmake --build build-bench -j --target \
  cvh_benchmark_cvtcolor_bgr2gray_header \
  cvh_benchmark_resize_bilinear_header

mkdir -p benchmark/results/internal/imgproc/quick

./build-bench/cvh_benchmark_cvtcolor_bgr2gray_header \
  --profile quick \
  --warmup 3 \
  --iters 10 \
  --repeats 7 \
  --output benchmark/results/internal/imgproc/quick/cvtcolor_current.csv

./build-bench/cvh_benchmark_resize_bilinear_header \
  --profile quick \
  --warmup 3 \
  --iters 10 \
  --repeats 7 \
  --output benchmark/results/internal/imgproc/quick/resize_current.csv
```

后续 runner 应支持通过 `git worktree` 拉起旧版本，例如：

```text
benchmark/internal/run_header_regression.sh --baseline-ref <git-ref> --suite core_mat --profile quick
benchmark/internal/run_header_regression.sh --baseline-ref <git-ref> --suite imgproc --profile quick
```

## Mode B: OpenCV Upstream Compare

OpenCV 对比模式回答的问题是：当前 `cvh` header-only 实现和官方 OpenCV 主仓库还有多大差距？

约束：

- 对比对象只包括当前 `cvh::headers_fast` 和官方 OpenCV `core` / `imgproc`。
- `cvh::headers_fast` 在 Mode B 中代表 header-only 项目的最快实现，避免报告变成内部 profile 对比。
- 这个模式默认是 report/log-only，不作为每个 PR 的硬 gate。
- 每份结果必须记录 `cvh` commit、OpenCV commit、编译器、平台、CPU、线程数、profile 和 CMake 选项。
- 不使用 `cvh::headers`、`native` / `lite` 作为 OpenCV upstream compare 的 implementation。

本机 OpenCV 源码位置：

```text
/Users/zmu/work/my_project/ocvh/opencv
```

从当前仓库相对路径看，它等价于：

```text
../opencv
```

推荐输出位置：

```text
benchmark/results/opencv/<suite>/<profile>/compare.csv
benchmark/results/opencv/<suite>/<profile>/report.md
benchmark/results/opencv/<suite>/<profile>/meta.json
```

目标实现名：

| Implementation | 含义 |
|---|---|
| `cvh_headers_fast` | 当前 `cvh::headers_fast`，代表 header-only 最快实现。 |
| `opencv` | 同机同编译模式下的官方 OpenCV。 |

现状：

- `benchmark/opencv_compare/` 已经可以生成 `cvh vs OpenCV` 报告。
- 该目录已经裁剪为纯 header-only compare：只用 `cvh::headers_fast` 对比 OpenCV。
- `cvh::headers` 留在 Mode A 内部回归和默认 header-only 验证中，不进入 Mode B 报告。

## Suites

### `core_mat`

第一优先级是 `Mat` 基础成本，而不是泛化数学库。

候选 case：

| Area | Case |
|---|---|
| Allocation/lifetime | `Mat::create`, `release`, reuse create, reallocation create。 |
| Copy/layout | `clone`, `copyTo`, continuous copy, ROI/non-contiguous copy。 |
| Fill/convert | `setTo`, `convertTo`, saturating cast。 |
| Shape/view | `reshape`, ROI construction, step/stride traversal。 |
| Basic array ops | `add`, `subtract`, `multiply`, `divide`, `compare`, `merge`, `split`，等这些进入 header-only contract 后再纳入 gate。 |

核心指标：

- `ns/op`：适合小矩阵和元数据操作。
- `MElems/s`：适合元素级算子。
- `GB/s`：适合内存带宽型路径。
- allocation count 或 reuse/recreate 标记：用于拆 `Mat::create` 影响。

目标 target：

```text
cvh_benchmark_core_mat_header
```

### `imgproc`

第一优先级是 AI vision 预处理/后处理中高频、容易被 OpenCV 用户感知的算子。

候选 case：

| Area | Case |
|---|---|
| Resize | `INTER_NEAREST`, `INTER_NEAREST_EXACT`, `INTER_LINEAR`，覆盖 downsample/upsample/非整数缩放/非对齐宽度。 |
| Color | BGR/RGB/GRAY/BGRA/RGBA，YUV encode/decode 家族按已支持布局覆盖。 |
| Threshold/LUT | `threshold`, `LUT`。 |
| Border/filter | `copyMakeBorder`, `filter2D`, `sepFilter2D`, `boxFilter`, `blur`, `GaussianBlur`。 |
| Grad/morphology | `Sobel`, `Canny`, `erode`, `dilate`, `morphologyEx`。 |

核心指标：

- `ms/call`：用户最直观的延迟。
- `MPix/s`：图像算子主指标。
- `GB/s`：内存带宽型路径辅助指标。
- `tail_ratio`：SIMD 尾部比例，尤其关注非 16/32 对齐宽度。
- `allocation_mode`：`reuse` / `recreate`，用于拆公共入口分配成本。
- `dispatch_path`：`scalar_fallback` / `opencv_ui` / `platform_fastpath`。

目标 target：

```text
cvh_benchmark_imgproc_header
```

## Common CSV Schema

新 benchmark 尽量收敛到同一批字段。现有专项 benchmark 可以先保持原 schema，但新 runner/report 应能映射到下面的标准字段。

| Field | 含义 |
|---|---|
| `mode` | `internal` 或 `opencv_compare`。 |
| `suite` | `core_mat` 或 `imgproc`。 |
| `module` | `core` / `imgproc`。 |
| `op` | 算子名。 |
| `variant` | 插值、border、kernel size、color code 等变体。 |
| `depth` | `CV_8U` / `CV_32F` 等。 |
| `channels` | 通道数。 |
| `layout` | `continuous` / `roi` / YUV layout 等。 |
| `shape` | 人类可读尺寸。 |
| `pixels` | 输出像素数；core 非图像 case 可为元素数。 |
| `implementation` | Mode A 可使用 `cvh_headers`, `cvh_headers_fast`, `scalar_fallback` 等；Mode B 只使用 `cvh_headers_fast`, `opencv`。 |
| `dispatch_path` | 实际命中的内部路径。 |
| `allocation_mode` | `reuse` / `recreate` / `none`。 |
| `warmup`, `iters`, `repeats`, `threads` | 采样参数。 |
| `min_ms`, `median_ms` | 最小值和中位数。 |
| `mpix_per_sec`, `melems_per_sec`, `gb_per_sec` | 吞吐指标。 |
| `checksum` | 防止编译器消除和粗粒度结果一致性检查。 |
| `status`, `note` | 支持状态和跳过原因。 |

## Profiles And Gates

| Profile | 用途 | 建议采样 | Gate |
|---|---|---|---|
| `quick` | 本地开发和 PR 预检查。 | 小到中尺寸，短采样。 | internal regression 可 fail。 |
| `stable` | 合并前或阶段收口。 | 更多 repeats，固定线程，固定机器。 | internal regression 可 fail。 |
| `full` | 周期性扫描。 | 全尺寸矩阵和更多边界 case。 | 默认 log-only。 |
| `micro` | 拆内核成本。 | 单内核、单职责。 | 不直接作为产品性能 gate。 |

建议 gate：

- 内部回归 quick：默认允许最多 `8%` slowdown。
- 内部回归 stable：对已接受 fast path 应收紧到 `5%` 左右；噪声大的平台可 log-only。
- OpenCV 对比：默认不 fail，只输出 `OpenCV/CVH` 和 unsupported cases。
- 只有当某个 fast path 已经以 benchmark 证据进入支持表，才为它设置硬性性能门槛。

## Measurement Rules

- Release 构建，尽量固定编译器和 CMake 选项。
- 单线程优先；多线程只在明确测试并行路径时启用。
- 同一份输入在同一 case 内复用。
- 同时记录 `reuse` 和 `recreate`，避免把 `Mat::create` 成本误判为 kernel 成本。
- micro benchmark 只解释瓶颈，不代表用户可见 API 性能。
- 每个结果必须带 metadata；没有 metadata 的 CSV 只能作为临时诊断。

## Cleanup Rules

- 新生成结果放入 `benchmark/results/` 或 `benchmark/opencv_compare/results/`，不再放在 `benchmark/` 根目录。
- `benchmark/*.csv` 视为历史阶段产物，不再作为长期文档入口。
- OpenCV compare 的 Markdown 报告是生成产物，后续由 runner 写入 `benchmark/results/opencv/.../report.md`。
- 源码 benchmark 暂时保留在当前路径，等 `core_mat` / `imgproc` header-only target 成型后再移动目录和改 CMake。

## Implementation Plan

Detailed execution steps live in
[`doc/benchmark-refactor-implementation-plan.md`](../doc/benchmark-refactor-implementation-plan.md).

1. **P-Bench-0：目录和文档收口** - complete
   - 明确两种模式、两个 suite、输出目录和 schema。
   - 清理 tracked 的阶段性 CSV/Markdown 报告。
   - 增加 ignore 规则，避免新结果再次进入源码树。

2. **P-Bench-1：公共 benchmark helper** - complete
   - 新增 header-only benchmark 公共 helper。
   - 统一计时、CSV、metadata、checksum 和 profile 解析。

3. **P-Bench-2：内部回归 runner** - complete
   - 新增 `benchmark/internal/run_header_regression.sh`。
   - 支持 `--baseline-ref <git-ref>`，用临时 `git worktree` 构建旧版本。
   - 输出 baseline/current/report/meta。

4. **P-Bench-3：`core_mat` header-only target** - complete
   - 从 `cvh_benchmark_core_ops` 拆出 `cvh_benchmark_core_mat_header`。
   - 只链接 `cvh::headers` / `cvh::headers_fast`。
   - 覆盖 `Mat` create/copy/convert/layout 成本。

5. **P-Bench-4：`imgproc` header-only target** - complete
   - 合并 `cvtColor` 和 `resize` 专项 benchmark 的可复用测量代码。
   - 形成 `cvh_benchmark_imgproc_header`。
   - 保留 scalar/public/direct UI/micro 诊断维度。

6. **P-Bench-5：统一 report/gate** - complete
   - 统一 CSV to Markdown/JSON summary。
   - 让内部回归可按 suite/op/variant 设置阈值。
   - OpenCV compare 继续保持 log-only，但输出 gap 排序和 unsupported matrix。

7. **P-Bench-6：OpenCV 主仓库 compare** - complete
   - 支持本地 `../opencv` 源码和用户指定 OpenCV build dir。
   - 只用 `cvh::headers_fast` 对比 OpenCV，代表 header-only 最快实现。
   - 移除 compare 报告里的产品层 `native` / `lite` 叙事。

8. **P-Bench-7：CI integration** - complete
   - 内部回归进入 quick gate。
   - OpenCV compare 保持 on-demand/log-only。
