# 测试来源（优先复用 OpenCV 官方用例）

- 测试数据来源：`/Users/moo/workssd/my_project/opencv_test/opencv_extra-4.x/testdata`
- 测试代码来源：`/Users/moo/workssd/my_project/opencv_test/opencv-4.13.0/modules/*/test`
- 复用策略：
  - 在接口语义对齐场景优先移植官方 `test` case（先做 contract 子集）。
  - 在性能/回归场景保留本仓基准与 smoke 流程，和官方 case 并行使用。

# M1 执行计划：Morphology + Gradient（Sobel / Erode / Dilate）

更新时间：2026-04-18  
状态：`Completed (W1~W5 完成)`

## 1. 目标

将 `Sobel / erode / dilate / morphologyEx` 从“对比报告中的 `UNSUPPORTED_CVH` 占位”升级为“可调用 + 可测试 + 可对比 + 可验收”的真实能力。

本轮先做最小可用闭环（M1）：

- `erode/dilate`：`CV_8U`，`C1/C3/C4`，`3x3`，`BORDER_DEFAULT/BORDER_REPLICATE`。
- `Sobel`：`CV_8U/CV_16S/CV_32F -> CV_16S/CV_32F`，覆盖 `dx/dy` 一阶组合与 `ksize=3/5`。
- `morphologyEx`：覆盖 `MORPH_ERODE/MORPH_DILATE/MORPH_OPEN/MORPH_CLOSE/MORPH_GRADIENT/MORPH_TOPHAT/MORPH_BLACKHAT/MORPH_HITMISS`。
- compare quick 结果中 `SOBEL/ERODE/DILATE` 不再出现 `UNSUPPORTED_CVH`。

## 2. 范围边界（本轮明确不做）

- 不做 `ksize>5` 的 Sobel。
- 不做 morphology 的任意 kernel 优化（先 fallback 正确性）。
- 不做 SIMD/并行优化（先正确、可测、可对比）。
- 不把 compare 性能阈值作为硬门禁（先做状态门禁）。

## 3. 工作拆分

### W1. API 与头文件接入

目标：对外可见 API，Lite/Full 都可编译。

改动文件：

- `include/cvh/imgproc/sobel.h`（新增）
- `include/cvh/imgproc/morphology.h`（新增，含 `erode/dilate`）
- `include/cvh/imgproc/imgproc.h`（聚合入口增加 include）
- `test/imgproc/imgproc_header_layout_test.cpp`（头文件可单独包含）

DoD：

- `#include <cvh/imgproc/imgproc.h>` 后可调用 `cvh::Sobel/erode/dilate`。
- `cvh_header_compile_smoke`、`cvh_test_imgproc` 可编译。

### W2. Full backend 注册接入

目标：在 Full 模式统一走 backend 注册路径，与现有 `resize/cvtColor/threshold/box/gaussian` 风格一致。

改动文件：

- `src/imgproc/resize_backend.cpp`

DoD：

- `register_all_backends()` 注册 `sobel/erode/dilate` backend。
- Full 模式 `is_*_backend_registered()==true`；Lite 模式为 false。

### W3. 合同测试（Correctness）

目标：先把行为锁住，避免后续优化回归。

改动文件：

- `test/imgproc/imgproc_morph_gradient_contract_test.cpp`（新增）
- `CMakeLists.txt`（纳入 `cvh_test_imgproc`）
- `test/smoke/cvh_resize_dispatch_smoke.cpp`（扩展注册态与最小功能烟测）

DoD：

- 覆盖 `C1/C3/C4` 基础 case。
- 覆盖 `ROI/non-contiguous` 至少 1 组。
- 覆盖非法参数（空输入/不支持 depth/不支持参数组合）。
- `./build-m1/cvh_test_imgproc` 与 `./build-m1/cvh_resize_dispatch_full_smoke` 通过。

### W4. Compare 基准接入（去占位）

目标：把 compare 的 `SOBEL/ERODE/DILATE` 从 OpenCV-only 占位改为真实 cvh 对比。

改动文件：

- `benchmark/opencv_compare_benchmark.cpp`

DoD：

- `SOBEL/ERODE/DILATE` 行 `status=OK`。
- 不再出现 `note=cvh_api_not_available_yet`。
- `opencv_compare/csv_to_markdown.py` 产出的报告可显示真实对比行。

### W5. 文档收口

目标：确保对外文档与实际能力一致。

改动文件：

- `include/cvh/imgproc/readme.md`
- `opencv_compare/README.md`
- `doc/current-gap-audit-*.md`（新增一版）

DoD：

- 文档中不再将 `SOBEL/ERODE/DILATE` 归类为仅占位。
- 明确本轮“已支持范围”和“未支持范围”。

## 4. 验收标准（量化）

### A. 功能验收

1. API 可见性：
   - `imgproc.h` 聚合包含 `Sobel/erode/dilate`。
2. 正确性：
   - 新增合同测试通过，且覆盖 `C1/C3/C4 + ROI + 异常输入`。
3. 模式一致性：
   - Lite 构建可编译运行 fallback；Full 构建注册状态正确。

### B. 对比验收

1. `cvh_benchmark_compare --profile quick` 输出 CSV 中：
   - `op in {SOBEL,ERODE,DILATE}` 的 `status` 全部为 `OK`。
2. Markdown 报告能展示这三类算子的 CVH/OpenCV 时间列。

### C. 工程验收

1. 不破坏既有 gate（`ci_smoke`, `ci_core_basic`, `imgproc_quick_gate`）。
2. `cvh_test_imgproc` 全量通过。

## 5. 验收命令（执行清单）

```bash
cmake -S . -B build-m1 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCVH_BUILD_FULL_BACKEND=ON \
  -DCVH_BUILD_TESTS=ON \
  -DCVH_BUILD_BENCHMARKS=ON \
  -DCVH_ENABLE_OPENCV_COMPARE=ON

cmake --build build-m1 --target \
  cvh_test_imgproc \
  cvh_resize_dispatch_full_smoke \
  cvh_benchmark_compare -j

./build-m1/cvh_test_imgproc
./build-m1/cvh_resize_dispatch_full_smoke

./build-m1/cvh_benchmark_compare \
  --profile quick --warmup 1 --iters 5 --repeats 1 \
  --output opencv_compare/results/m1_compare_quick.csv

python3 opencv_compare/csv_to_markdown.py \
  --input opencv_compare/results/m1_compare_quick.csv \
  --output doc/opencv_compare_m1_quick.md \
  --title "cvh vs OpenCV M1 (Morphology+Gradient)"
```

## 6. 里程碑状态

- [x] 方案文档落地
- [x] W1 API 与头文件接入
- [x] W2 Full backend 注册接入
- [x] W3 合同测试
- [x] W4 Compare 去占位
- [x] W5 文档收口

## 7. 本轮执行结果（2026-04-18）

- 构建：`build-m1` 配置与目标编译通过。
- 测试：
  - `./build-m1/cvh_test_imgproc`：`103/103 PASS`
  - `./build-m1/cvh_resize_dispatch_full_smoke`：PASS
- 对比：
  - `./build-m1/cvh_benchmark_compare --profile quick --warmup 1 --iters 3 --repeats 1`
  - 输出：`opencv_compare/results/m1_compare_quick.csv`
  - 汇总：`rows=58, supported=58, unsupported=0`
  - `SOBEL/ERODE/DILATE` 均为 `status=OK`
- 报告：
  - `doc/opencv_compare_m1_quick.md` 已生成。
- 官方用例移植：
  - 已移植 `modules/imgproc/test/test_filter.cpp` 中：
    - `Imgproc_Morphology.iterated`（PASS_NOW）
    - `Imgproc.filter_empty_src_16857`（PASS_NOW，已实现算子的空输入覆盖）
    - `Imgproc.morphologyEx_small_input_22893`（PASS_NOW，`MORPH_DILATE` 小输入回归）
    - `Imgproc_MorphEx.hitmiss_regression_8957`（PASS_NOW，`MORPH_HITMISS` 回归）
    - `Imgproc_MorphEx.hitmiss_zero_kernel`（PASS_NOW，`MORPH_HITMISS` 零核行为）
    - `Imgproc_Sobel.borderTypes`（PASS_NOW，已对齐 ROI + `BORDER_ISOLATED` 语义）
    - `Imgproc_Sobel.s16_regression_13506`（PASS_NOW，`CV_16S` + `ksize=5`）
    - `Imgproc_GaussianBlur.regression_11303`（PASS_NOW，`CV_32F` 常量图 + `sigma` 自动核路径）
  - 新增 contract 用例：
    - `morphologyEx_hitmiss_signed_kernel_semantics`（PASS，`CV_8SC1` kernel：`1` 前景、`-1` 背景、`0` 忽略）
