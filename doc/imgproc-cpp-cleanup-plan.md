# Imgproc C++ Cleanup And Header-only Fast-path Migration Plan

## 背景

`opencv-header-only` 的公共产品定位是纯 header-only。当前 `imgproc` 的公开
header 已经包含可工作的 fallback，但 `src/imgproc` 仍保留约 6,400 行 `.cpp`
实现，并通过 `backend_registry.cpp` 在 `CVH_NATIVE` 模式下把运行时函数指针
替换为编译型 fast-path。

这形成了两套并行实现：

- `cvh::headers` / `cvh::headers_fast` 使用
  `include/cvh/imgproc/*.h` 中的 header-only 实现。
- legacy native 开发构建通过 `register_all_backends()` 注入
  `src/imgproc/*.cpp` 中的另一套实现。

Mode B benchmark 只测 `cvh::headers_fast`，因此现有 `.cpp` 即使更快，也不代表
项目的公开性能。继续维护这套双轨结构会带来行为漂移、重复修复和错误性能归因。

本计划的目标不是直接删除所有 `.cpp`。每个 fast-path 必须先经过正确性和性能
判断：值得保留的迁入 ODR-safe header，不值得保留或只是调用 fallback 的实现
直接删除。最终移除 imgproc 的运行时 backend registry。

## 当前事实

截至 2026-07-23，公开 header 已支持 README 中列出的 resize、cvtColor、
threshold、LUT、border、filter、blur、Sobel、Canny、morphology 和
warpAffine 子集。`src/imgproc` 主要是 legacy fast-path，而不是这些 API 的唯一
实现来源。

| 文件组 | 约行数 | 当前职责 | 初步方向 |
|---|---:|---|---|
| `resize.cpp` | 409 | U8 nearest/linear、并行行循环 | benchmark 后迁移有收益路径 |
| `cvtcolor*.cpp` | 1848 | RGB/GRAY 与 YUV420/422/444 fast-path | 分颜色族迁入 detail headers |
| `threshold.cpp`、`lut.cpp`、`copy_make_border.cpp` | 406 | 点算子和复制类 fast-path | 作为首批低风险迁移 |
| `box_filter.cpp`、`gaussian_blur.cpp` | 1442 | 3x3/通用 separable filter fast-path | 抽公共 filter 基础后迁移 |
| `filter2d.cpp`、`sep_filter2d.cpp`、`sobel.cpp` | 1330 | 卷积和导数 fast-path | 按共享 row/column kernel 迁移 |
| `canny_image.cpp`、`canny_deriv.cpp` | 501 | Canny derivative/image fast-path | Sobel 稳定后迁移 |
| `erode.cpp`、`dilate.cpp` | 358 | 两份高度相似的 3x3 morphology 路径 | 合并为一个 header helper |
| `warp_affine.cpp` | 20 | 仅调用现有 header fallback | 审计后直接删除 |
| `backend_registry.cpp` | 79 | 注册全部 compiled backends | 所有迁移完成后删除 |
| `fastpath_common.h`、`cvtcolor_internal.h` | 256 | `.cpp` 共享 helper 和声明 | 迁入 `include/cvh/imgproc/detail` 或删除 |

当前架构还包含：

- 每个公共头中的 `*Fn` 函数指针、`register_*_backend()` 和
  `is_*_backend_registered()`。
- `detail::ensure_backends_registered_once()` 对 `CVH_NATIVE` 的条件分支。
- native smoke 对“backend 是否注册”的断言。
- `last_boxfilter_dispatch_path()` 和
  `last_gaussianblur_dispatch_path()` 的 compiled-only telemetry。
- CMake 中 20 个 `src/imgproc` `.cpp` source entries。

## 清理目标

1. `cvh::headers` 提供稳定 scalar/header baseline。
2. `cvh::headers_fast` 在相同公共入口中启用经过验证的 UI 或平台 fast-path。
3. accepted imgproc API 不依赖 `src/imgproc/*.cpp`、`CVH_NATIVE` 或运行时注册。
4. 同一算子只保留一个公共 dispatch 图，不维护 header/native 两套实现。
5. fast-path 使用 OpenCV Universal Intrinsics，当前只接纳 NEON 和
   SSE/AVX 系列；RVV 保留为后续 TODO。
6. 不重新引入 xsimd，也不把 compiled imgproc 重新包装成产品层。
7. benchmark 只调用 `cvh::headers_fast` 公共 API，并如实记录
   `opencv_ui`、平台特化或 `headers_baseline`。

## 非目标

- 本轮不从零重写所有 imgproc 算法。
- 本轮不扩大 README 尚未承诺的 OpenCV API 范围。
- 本轮不要求所有迁移后的算子立即达到 upstream OpenCV 性能。
- 本轮不启用 RVV。
- 本轮不处理 highgui 的平台 `.cpp`。
- 本轮不保留“以后可能有用”但没有 benchmark 证据的 compiled fast-path。

## 迁移决策规则

每个 `.cpp` fast-path 只能得到以下两种结论：

### Migrate

同时满足：

- 与当前 header fallback 或 upstream OpenCV 的正确性 contract 一致。
- 在目标输入矩阵上有稳定收益，或明显缩小与 upstream OpenCV 的差距。
- 能以 `inline`、模板、`constexpr` 或 inline variable 形式满足 ODR。
- 不依赖全局 backend registry、compiled static initialization 或
  repository `src/` 路径。

### Delete

满足任一条件：

- 只转发到现有 fallback，例如当前 `warp_affine.cpp`。
- 与 header 实现重复且没有稳定收益。
- 只对 legacy native benchmark 有意义。
- 行为与公开 contract 不一致，且修正成本高于重新基于现有 header 优化。
- 依赖 xsimd、已放弃 ISA 或不能在 header-only 包中独立编译。

本轮不接受长期 `keep-dev-only`。需要保留实验快照时依靠 git history，而不是继续
留在活跃 source list。

## 目标结构

建议按算子族组织实现，避免把现有 `.cpp` 机械拼进公共 API 头：

```text
include/cvh/imgproc/
  resize.h
  cvtcolor.h
  threshold.h
  ...
  detail/
    common.h
    fastpath_common.hpp
    resize_impl.hpp
    cvtcolor_rgb_gray.hpp
    cvtcolor_yuv420.hpp
    cvtcolor_yuv422.hpp
    cvtcolor_yuv444.hpp
    filter_common.hpp
    box_filter_impl.hpp
    gaussian_blur_impl.hpp
    morphology_impl.hpp
    canny_impl.hpp
```

文件名不是硬性要求。只有在能降低 include 体积、减少重复或明确 ownership 时才
拆分。禁止采用：

```text
include/cvh/imgproc/resize.h -> #include "../../../src/imgproc/resize.cpp"
```

也禁止保留公共函数指针 dispatch，只为了兼容即将删除的 native backend。

## 实施阶段

### Imgproc-Clean-0：建立 ownership 和性能基线

状态：完成（2026-07-23）。

任务：

- 为每个 `src/imgproc` 文件记录：
  - CMake target ownership
  - 注册入口
  - 对应 header fallback
  - 测试覆盖
  - benchmark case
  - 迁移或删除候选
- 记录 `.cpp` fast-path 支持的 depth、channel、kernel、border、ROI 和 in-place
  矩阵。
- 在删除前保存 legacy native 路径的诊断 benchmark，仅作为迁移判断依据。
- 记录当前 `cvh::headers_fast` 和 upstream OpenCV stable 单线程结果。
- 确认公开 Mode B 不链接 `cvh_native_backend`。

完成条件：

- 所有 `.cpp` 和两个私有 `.h` 都有明确 ownership。
- 每个 fast-path 都能回答“比 header baseline 快多少”和“覆盖什么 contract”。
- 没有 benchmark 数据的代码不得直接判定为性能实现。

完成记录：

- 20 个 `src/imgproc/*.cpp` 均只属于未安装、未导出的
  `cvh_native_backend`；公开 Mode B 只链接 `cvh::headers_fast`。
- `backend_registry.cpp` 是所有 compiled imgproc 的唯一注册入口，公开 header
  fallback 是 accepted API 的完整实现来源。
- legacy native quick profile 已保存于本轮终端记录。720p U8 代表值为：
  resize nearest/linear `0.085/0.154 ms`，BGR2GRAY `0.146 ms`，
  box 3x3 `0.696 ms`，Gaussian 5x5 `0.807 ms`。这些数据仅用于决定是否保留
  算法，不作为 Mode B 项目成绩。
- ownership 决策：
  - resize、cvtColor、3x3 box/Gaussian 与共享并行 helper：迁移候选；
  - threshold、LUT、copyMakeBorder：按 public benchmark 和代码重复度决定，
    无独立收益则删除 compiled 副本；
  - filter2D、sepFilter2D、Sobel：复用统一 filter 基础后迁移；
  - Canny image/derivative：先删除重复 derivative 实现，再迁移一个版本；
  - erode/dilate：先合并重复 morphology kernel，再迁移；
  - warpAffine：仅转发 header fallback，直接删除。
- depth/channel/kernel/border/ROI/in-place contract 由对应
  `test/imgproc/*_contract_test.cpp` 作为迁移 gate；缺失组合不从 compiled
  行为反向扩大 public contract。

### Imgproc-Clean-1：冻结单一 dispatch 架构

状态：完成（2026-07-23）。

任务：

- 确立公共调用顺序：
  1. 已验证的 OpenCV UI 或平台 fast-path
  2. 通用 header fast-path
  3. scalar/header fallback
- 禁止新增 `register_*_backend()` 和新的 `CVH_NATIVE` 分支。
- 审计并删除 `warp_affine.cpp` 这类纯 fallback 转发。
- 为 dispatch telemetry 定义 header-only 方案：
  - 优先复用统一 `DispatchTag`
  - 如需字符串诊断，使用 ODR-safe `inline thread_local`
  - telemetry 仅服务测试和 benchmark，不影响公共结果
- 明确迁移期间 registry 只为尚未处理的算子临时存在。

完成条件：

- 新迁移算子不再经过运行时函数指针。
- `warpAffine` 等无独立 compiled 行为的 API 不再注册 backend。
- 文档和 CMake 注释不再把 native imgproc 描述为公开性能层。

完成记录：

- 新的目标 dispatch 固定为“header fast-path -> header fallback”，迁移完成的算子
  不再经过可变函数指针。
- `warpAffine` 的 compiled 实现确认只转发 fallback，已从公共 registry、native
  source list 和 `src/imgproc` 删除。
- registry 在中间态只保留尚未迁移的算子；禁止新增 backend 注册点。
- box/Gaussian telemetry 在 P5 迁移时改为 ODR-safe header 状态。

### Imgproc-Clean-2：迁移共享 fast-path 基础

状态：完成（2026-07-23）。

任务：

- 审计 `fastpath_common.h`：
  - parallel row/count helper
  - border/index helper
  - input/type/shape 检查
  - in-place source clone helper
- 与 `include/cvh/imgproc/detail/common.h` 去重。
- 保留项迁入 `include/cvh/imgproc/detail/fastpath_common.hpp` 或更小的领域头。
- 将所有定义改为模板、`inline` 或 `constexpr`。
- 审计 `cvtcolor_internal.h`，把颜色族内部声明改为直接 header include 关系。
- 避免大 lookup table 和可变状态在每个 translation unit 复制。

完成条件：

- 后续迁移不再 include `src/imgproc/fastpath_common.h`。
- shared helper staged install 后可独立编译。
- 两个 translation unit 同时包含 `cvh/imgproc/imgproc.h` 时无重复符号。

完成记录：

- `fastpath_common.h` 已迁到安装树
  `include/cvh/imgproc/detail/fastpath_common.hpp`。
- 新 helper 只包含 `detail/common.h` 和 `core/parallel.h`，不再通过
  `imgproc.h` 隐式引入全部算子。
- 所有中间态 `.cpp` 已改为显式包含所属公共头；旧
  `src/imgproc/fastpath_common.h` 路径不再存在。
- `cvtcolor_internal.h` 的声明仅服务尚未迁移的颜色族，确定在 P4 以 detail
  header include 顺序替代并删除。

### Imgproc-Clean-3：首批低风险算子

状态：完成（2026-07-23）。

范围：

- `threshold.cpp`
- `lut.cpp`
- `copy_make_border.cpp`

步骤：

#### Imgproc-Clean-3.0：threshold

- 对比 `.cpp` FP32 fast-path 与 header threshold 实现。
- 保留固定 threshold 的有效行并行或 UI 路径。
- OTSU/TRIANGLE 继续使用唯一 header fallback，不复制 histogram 逻辑。
- 覆盖 continuous、ROI、C1/C3、in-place 和所有 accepted threshold type。

#### Imgproc-Clean-3.1：LUT

- 迁移或删除 U8 table fast-path。
- 覆盖单通道 LUT 和 per-channel LUT。
- 保持 in-place、ROI 和 `lut.total()==256` contract。

#### Imgproc-Clean-3.2：copyMakeBorder

- 迁移有收益的 `BORDER_REPLICATE` 行复制路径。
- 其他 border mode 继续走唯一 header fallback。
- 覆盖 C1/C3/C4、U8/F32、非连续输入和大 border。

完成条件：

- 三个 `.cpp` 从 source list 删除。
- 公共入口不再调用对应 backend function pointer。
- header-only tests 和 Mode B case 均通过。

完成记录：

- threshold FP32 行并行路径迁入 `detail/threshold_impl.hpp`；U8、OTSU 和
  TRIANGLE 继续复用唯一 fallback。
- U8 LUT 与 `BORDER_REPLICATE` copyMakeBorder 路径分别迁入
  `detail/lut_impl.hpp` 和 `detail/copy_make_border_impl.hpp`。
- 三个公共入口均直接调用 `inline *_fast_impl`，对应 typedef、可变函数指针和
  registry 项已删除。
- 三个 `.cpp` 已从 `src/imgproc` 和 CMake source list 删除；141 个 imgproc
  contract tests 全部通过。

### Imgproc-Clean-4：resize 和 cvtColor

状态：完成（2026-07-23）。

这组优先保护已经验证的 OpenCV UI 路径，不能因迁移 generic fast-path 而改变
dispatch 优先级。

#### Imgproc-Clean-4.0：resize

- 对比 `.cpp` U8 nearest/linear 与 header fallback。
- 保留现有 `CV_8UC1` exact 2x `INTER_LINEAR` OpenCV UI 路径。
- 对有收益的 C1/C3/C4 nearest/linear 路径迁入 `resize_impl.hpp`。
- 坐标/权重预计算与像素 kernel 分离，避免每次调用重复构建无关状态。
- 覆盖奇数尺寸、非整数缩放、ROI、in-place 防护和尾部像素。

#### Imgproc-Clean-4.1：RGB/GRAY

- 以现有直接 OpenCV UI `BGR/RGB2GRAY` 为唯一 SIMD 方言。
- 对 `.cpp` 的 GRAY/BGR/BGRA/RGBA、swap-RB、alpha fill/drop 和 F32 路径逐项
  判断 migrate/delete。
- 不再创建第二层 SIMD adapter。
- 确认 `RGB2GRAY` 与 `BGR2GRAY` 共享 kernel 时无额外分支成本。

#### Imgproc-Clean-4.2：YUV family

- 分别迁移 YUV420、YUV422、YUV444，禁止合成一个超大头。
- 每个 family 单独覆盖 planar、semi-planar、packed、UV/VU 顺序和 BGR/RGB
  输出。
- 对奇偶尺寸、stride、ROI 限制和 rounding 建立明确 contract。

#### Imgproc-Clean-4.3：收尾

- 删除 `resize.cpp`、`cvtcolor.cpp`、`cvtcolor_rgb_gray.cpp` 和三个
  `cvtcolor_yuv*.cpp`。
- 删除 `cvtcolor_internal.h`。
- 更新 BGR/RGB2GRAY 和 resize 专项 benchmark，使其只测公共 header 路径。

完成条件：

- resize/cvtColor 的 accepted matrix 在 native OFF 时完整。
- ARM NEON 和 x86 SSE/AVX compile smoke 通过。
- 现有 UI fast-path 性能不发生不可解释的回退。

完成记录：

- resize U8 C1/C3/C4 nearest/nearest-exact/linear 路径迁入
  `detail/resize_impl.hpp`，坐标表仍按调用预计算，行 kernel 通过统一 parallel
  helper 执行。
- exact 2x `CV_8UC1 INTER_LINEAR` 继续优先进入现有 OpenCV UI；generic U8
  fast-path 不遮蔽该路径。公共入口增加同 buffer in-place clone 防护。
- RGB/GRAY、YUV420、YUV422、YUV444 已拆为四个 ODR-safe detail header，
  `cvtcolor_impl.hpp` 只负责 dispatch 顺序。
- `BGR2GRAY`/`RGB2GRAY` 保持直接 OpenCV UI 优先，不引入第二层 SIMD adapter。
- 六个 `.cpp`、`cvtcolor_internal.h`、对应 registry 项和 CMake source entries
  已删除；141 个 imgproc contract tests 全部通过。

### Imgproc-Clean-5：过滤器和 Sobel

状态：完成（2026-07-23）。

该阶段代码量最大，必须按共享基础、具体算子和 gate 分步，不能一次搬运 2,700
行实现。

#### Imgproc-Clean-5.0：filter common

- 抽取 anchor、border index、kernel coefficient、row buffer 和并行阈值。
- 明确 U8 accumulation、F32 accumulation、rounding 和 saturation 规则。
- 统一 continuous/ROI 和 in-place source handling。

#### Imgproc-Clean-5.1：boxFilter / GaussianBlur

- 先迁移 3x3 专用路径，再迁移通用 separable 路径。
- `last_*_dispatch_path()` 改为 header-only telemetry。
- 记录 3x3、5x5、通用 kernel 的 public API 性能。
- `blur` 保持为 `boxFilter` 语义 wrapper，不增加第三份实现。

#### Imgproc-Clean-5.2：filter2D / sepFilter2D

- 复用 `filter_common`，禁止各自复制 border 和行遍历逻辑。
- 区分 direct 2D convolution 与 separable kernel，benchmark 不混合语义。
- 覆盖 U8/F32 source、U8/F32 destination、C1/C3、delta 和 accepted border。

#### Imgproc-Clean-5.3：Sobel

- 复用 sepFilter/common 基础，保留 3x3/5x5 和 dx/dy contract。
- Canny 依赖的 `CV_16S` derivative 路径必须先稳定。

完成条件：

- 五个 filter `.cpp` 从 source list 删除。
- filter tests 只链接 `cvh::headers`。
- dispatch telemetry 与 benchmark 均不依赖 native symbol。

完成记录：

- filter 共享的 border map、row parallel、kernel 与 saturation helper 统一复用
  `fastpath_common.hpp` 和公共 `detail/common.h`。
- boxFilter、GaussianBlur、filter2D、sepFilter2D、Sobel 的 compiled fast-path
  已分别迁入 ODR-safe detail header，公共入口直接调用。
- box/Gaussian telemetry 已改为 `inline thread_local`，不再由 native library
  导出符号。
- 五个 filter `.cpp`、registry 项和 CMake source entries 已删除；141 个
  imgproc contract tests 全部通过。

### Imgproc-Clean-6：morphology 和 Canny

状态：完成（2026-07-23）。

#### Imgproc-Clean-6.0：erode / dilate

- 合并两份高度相似的 3x3 rect kernel，以 min/max policy 参数化。
- 保留通用 kernel、iterations、border value 和 in-place contract。
- `morphologyEx` 继续复用 erode/dilate，不复制基础 kernel。

#### Imgproc-Clean-6.1：Canny derivatives

- 先迁移 `canny_deriv.cpp` 的 magnitude、NMS 和 hysteresis。
- 限定临时 buffer ownership，避免大型每 TU 静态数据。
- 对 L1/L2、threshold 顺序和边缘连接规则建立 upstream contract。

#### Imgproc-Clean-6.2：Canny image

- 复用迁移后的 Sobel 和 derivative Canny。
- 覆盖 aperture 3/5、ROI、in-place 和空图错误路径。

完成条件：

- `erode.cpp`、`dilate.cpp`、`canny_deriv.cpp` 和 `canny_image.cpp` 删除。
- morphology/Canny accepted API 不再需要 backend registration。
- public API benchmark 明确记录临时 Mat 分配成本。

完成记录：

- erode/dilate 的两份相同 3x3 rect 实现已合并到
  `detail/morphology_impl.hpp`，通过 `is_erode` policy 选择 min/max。
- Canny image/derivative 中重复的 magnitude、NMS 与 hysteresis 已合并到
  `detail/canny_impl.hpp`；image 路径复用 header Sobel。
- 四个 `.cpp`、registry 项和 CMake source entries 已删除；公共 API 直接调用
  header fast-path，141 个 imgproc contract tests 全部通过。

### Imgproc-Clean-7：删除 registry 和 legacy targets

状态：完成（2026-07-23）。

任务：

- 从所有公共头删除：
  - `*Fn` backend typedef
  - `*_dispatch()` 可变函数指针
  - `register_*_backend()`
  - `is_*_backend_registered()`
  - `ensure_backends_registered_once()`
- 删除 `backend_registry.cpp` 和 `register_all_backends()`。
- 从 CMake 移除所有 `src/imgproc/*.cpp`。
- 更新或删除 `cvh_resize_dispatch_native_smoke`；header-only smoke 改为验证实际
  dispatch tag 和结果，不验证“是否注册”。
- 删除或迁移仍链接 `cvh::native` 的 legacy imgproc benchmark：
  - `cvh_benchmark_imgproc_ops`
  - `cvh_benchmark_imgproc_filter`
- 保留 native development target 时，其 source list 只能包含仍有明确理由的
  highgui/platform 文件，不得包含 imgproc。
- 增加静态检查，阻止 benchmark 和安装包引用 `src/imgproc`。

完成条件：

- `grep` 在公共 imgproc headers 中找不到 `CVH_NATIVE` 和 backend registry。
- `CVH_BUILD_NATIVE_BACKEND=OFF` 时全部 imgproc tests/benchmarks 可构建。
- staged install 不引用 `src/`，也不导出 compiled imgproc target。

完成记录：

- `backend_registry.cpp`、`register_all_backends()`、
  `ensure_backends_registered_once()` 和全部 imgproc backend typedef/注册函数已
  删除。
- `src/imgproc` 已无活跃文件，CMake native source list 只剩 highgui/platform
  source。
- 原 native-only imgproc benchmark 已移到通用 benchmark 区并改为只链接
  `cvh::headers_fast`。
- native dispatch smoke 已删除；现有 header dispatch smoke 改为验证真实
  fast-path tag 和结果，不再验证函数指针注册状态。
- 静态 grep 已确认公共 imgproc headers 不含 `CVH_NATIVE`、registry 或
  `src/imgproc` 引用。

### Imgproc-Clean-8：最终正确性、ODR 和性能 gate

状态：完成（2026-07-23）。

顺序：

1. public API compile/link smoke。
2. 多 translation unit imgproc ODR test。
3. header-only contract 和 staged install consumer。
4. 全量 `test/imgproc`。
5. ARM NEON 和 x86 SSE/AVX compile smoke。
6. Mode A old/current header-only regression。
7. Mode B `cvh::headers_fast` / upstream OpenCV stable 单线程对比。

Mode B 至少保留：

- resize：U8/F32、C1/C3/C4、nearest/linear、常见和非对齐尺寸。
- cvtColor：RGB/BGR/GRAY/BGRA/RGBA 与已接纳 YUV family。
- pointwise：threshold、LUT、copyMakeBorder。
- filter：box/Gaussian/filter2D/sepFilter2D/Sobel。
- edge/morphology：Canny、erode、dilate。
- 连续和 ROI case；不支持的组合明确标为 `UNSUPPORTED`。

完成条件：

- 报告中的 CVH 实现只有 `cvh_headers_fast`。
- 没有专用 fast-path 的算子自动执行 `headers_baseline`，不跳过。
- 所有计时从公共 API 进入，direct kernel 只可作为诊断行。
- 更新日期命名的 upstream performance Markdown。
- README operator status 与实际 header-only contract 一致。

完成记录：

- 新增 `cvh_imgproc_header_odr_smoke`，两个 translation unit 同时包含并调用
  resize、cvtColor、filter 和 Canny；已纳入 staged header-only contract。
- 全新 `CVH_BUILD_NATIVE_BACKEND=OFF` 构建与 CTest：`16/16` 通过。
- highgui-only `CVH_BUILD_NATIVE_BACKEND=ON` 开发构建与 CTest：`19/19`
  通过；native target 不再编译 imgproc。
- staged install/public dependency contract：`5/5` smoke 通过。
- Apple ARM64 `headers_fast` 构建验证 NEON 路径；AppleClang
  `-arch x86_64 -mavx2` 对 OpenCV UI 和 imgproc 聚合头语法编译通过。x86 本轮
  未做运行时性能测试。
- quick profile 显示迁移后 720p U8 C1：
  - exact-2x linear resize `0.0153 ms`，legacy native 基线 `0.1540 ms`；
  - BGR2GRAY `0.1057 ms`，legacy native 基线 `0.1461 ms`；
  - box 3x3 `0.6869 ms`，legacy native 基线 `0.6957 ms`；
  - Gaussian 5x5 `0.7906 ms`，legacy native 基线 `0.8068 ms`。
- filter Mode A 诊断中，720p C1 replicate 的 box 3x3 从 forced fallback
  `13.23 ms` 降到 `1.32 ms`，Gaussian 3x3 从 `7.34 ms` 降到
  `0.60 ms`。
- 日期报告
  `benchmark/opencv_compare/results/2026-07-23-opencv-upstream-performance.md`
  已重跑。Mode B imgproc 几何平均 `OpenCV/CVH` 从 `0.0320` 提升到
  `0.2913`，约为迁移前的 `9.1x`；全部 42 个 imgproc case 仍从公共 API
  进入。
- README operator status、benchmark target ownership 和 compare
  `dispatch_path` 已同步；无 fast-path 的 case 继续标记
  `headers_baseline`。

## 每阶段验证

每个迁移阶段至少执行：

```bash
cmake -S . -B build-imgproc-clean \
  -DCMAKE_BUILD_TYPE=Release \
  -DCVH_BUILD_NATIVE_BACKEND=OFF \
  -DCVH_BUILD_TESTS=ON \
  -DCVH_BUILD_BENCHMARKS=ON

cmake --build build-imgproc-clean -j
ctest --test-dir build-imgproc-clean --output-on-failure
./scripts/check_header_only_contract.sh
```

额外验证：

- direct include `cvh/imgproc/<operator>.h`。
- 两个 translation unit 同时 include `cvh/imgproc/imgproc.h`。
- staged install 后 downstream CMake consumer。
- ARM NEON 与 x86 AVX2 syntax/compile smoke。
- `git diff --check`。
- CMake、benchmark 和测试中没有新增 `src/imgproc` include。

## 风险与约束

### Header 体积和编译时间

`cvtcolor` 和 filter family 合计超过 4,000 行。必须按算子族拆分 detail header，
并避免 include 一个小算子时实例化所有 YUV、Canny 和 filter kernel。

### ODR 和 telemetry

匿名 namespace helper 会在每个 TU 复制。大型 table、可变 dispatch state 和
thread-local telemetry 必须使用合适的 inline variable、函数内静态或模板，
不能产生重复符号或每 TU 不一致状态。

### 行为漂移

`.cpp` fast-path 与 header fallback 可能在 rounding、border、stride、in-place
和错误处理上不同。迁移优先保持公开 contract；不能用 native 历史行为覆盖已经
通过测试的 header 行为。

### 性能归因

并行阈值、临时 Mat、坐标表和 kernel 系数构建必须计入 public API benchmark。
direct kernel microbenchmark 只能解释成本，不能替代 Mode B 结果。

### SIMD 边界

OpenCV Universal Intrinsics 是首选 SIMD 方言。当前只处理 NEON 和 SSE/AVX。
如果 UI 指令序列在 ARM 上不够好，可增加 header-only direct NEON 特化，但必须
由 benchmark 证明，且不能改变公共 dispatch 架构。

## 最终完成定义

本轮 imgproc C++ cleanup 完成需要同时满足：

- `src/imgproc` 不再包含活跃 `.cpp` 或私有实现头。
- CMake native source list 不再包含 imgproc。
- accepted imgproc API 只有一套 header dispatch 和 fallback。
- backend registry、`CVH_NATIVE` imgproc 分支和注册 smoke 已删除。
- 正确性、ODR、安装包、ARM/x86 compile gate 全部通过。
- Mode A 和 Mode B benchmark 只测 header-only 公共入口。
- 有收益的 legacy fast-path 已迁移，无收益或重复实现已删除。
- README、benchmark 文档和本计划状态与实际代码一致。

状态：全部完成（2026-07-23）。
