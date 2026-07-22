# opencv-header-only 设计文档

## 当前定位

`opencv-header-only` 是一个纯 header-only 的 OpenCV-style C++ 子集库。项目目标不是完整替代 OpenCV，而是在不引入必需库构建步骤的前提下，提供常用 `Mat`、`imgproc`、`imgcodecs` 能力，并为 AI vision preprocessing/postprocessing 的热点路径提供可验证的 header-only 加速。

公开产品面只包含两个 CMake target：

```cmake
cvh::headers
cvh::headers_fast
```

- `cvh::headers`：默认 header-only baseline，优先保证 API 可用性、可移植性和标量 fallback 正确性。
- `cvh::headers_fast`：header-only 加速 profile，只启用已经通过 correctness gate 和 benchmark gate 的 SIMD fast-path。

历史 `.cpp` 实现只能作为 legacy/experimental 代码存在，不进入公开产品结构，也不作为公开 API 可用性的前提。

## 核心目标

- 提供接近 OpenCV 风格的基础数据结构和常用算子。
- 保证用户仅包含 headers 或链接 interface target 即可使用。
- 保持公开依赖、宏开关、安装导出和文档叙事一致。
- 先建立 correctness contract，再引入 SIMD fast-path。
- 用 benchmark 决定 fast-path 是否进入 `cvh::headers_fast`。

## 非目标

- 不追求完整复现 OpenCV 全模块。
- 不承诺所有算子、所有类型、所有 flag 与 OpenCV 完全一致。
- 不把任何 `.cpp` 实现作为公开 API 的唯一来源。
- 不把 xsimd 作为 `cvh::headers_fast` 的默认或推荐性能路线。
- 不承诺所有路径都快于 OpenCV；只承诺 benchmark-gated 的热点优化。

## Header-only Contract

公开 API 必须满足：

- `include/` 内可独立完成编译。
- `cvh::headers` 不依赖 `src/`。
- `cvh::headers_fast` 不依赖 `src/`，只通过 interface include 和 compile definitions 启用已验证 fast-path。
- 每个标记为 Supported 的算子必须有 header-only correctness test。
- 没有 header-only 实现或链接不过的 API 必须标记为 WIP 或移出公开入口。

## Public Targets

### `cvh::headers`

默认入口，适合所有需要稳定 header-only 行为的用户。

要求：

- 不默认启用 OpenCV Universal Intrinsics。
- 不默认启用 xsimd。
- 不要求 OpenCV 库或其它二进制依赖。
- 以 scalar fallback 和标准 C++ 实现为基线。

### `cvh::headers_fast`

加速入口，适合愿意接受平台 SIMD 宏和已验证 fast-path 的用户。

要求：

- 继承 `cvh::headers`。
- 传播 `CVH_ENABLE_OPENCV_INTRIN=1`。
- 传播 `CVH_ENABLE_PLATFORM_INTRINSICS=1`。
- 不传播 `CVH_ENABLE_XSIMD=1`。
- 不编译或链接 `.cpp`。

当前 accepted fast-path：

- `cvtColor`：`CV_8UC3 BGR2GRAY/RGB2GRAY`
- `resize`：exact 2x `CV_8UC1 INTER_LINEAR`

## Module Responsibilities

### `core`

负责 `Mat`、基础类型、错误处理、类型/channel 宏、ROI、copy/clone/convert 等基础能力。

`core` 的首要职责是支撑 header-only correctness，而不是承接所有 AI kernel 或历史算子。

### `imgproc`

负责 OpenCV-style 图像处理算子，例如：

- `resize`
- `cvtColor`
- `threshold`
- `LUT`
- `copyMakeBorder`
- `filter2D`
- `sepFilter2D`
- `boxFilter` / `blur`
- `GaussianBlur`
- `Sobel`
- `Canny`
- `erode` / `dilate` / `morphologyEx`
- `warpAffine`

每个算子必须明确支持的 depth、channel、flag、border 和错误行为。

### `imgcodecs`

提供最小读写闭环：

- `imread`
- `imwrite`

当前读写能力基于 vendored stb，目标是满足“读图 -> 处理 -> 写图”的端到端 header-only 使用链路。

### `highgui`

显示和事件循环不属于纯 header-only 产品承诺。`imshow` / `waitKey` 在 header-only fallback 下应明确报错，并建议用户使用 `imwrite` 或应用自己的 UI/event loop。

## SIMD Strategy

项目采用三条规则：

- scalar fallback 是所有公开算子的 correctness 基线。
- OpenCV Universal Intrinsics 是 `cvh::headers_fast` 的主要 portable SIMD 路径。
- direct platform intrinsics 只能在 benchmark 证明 OpenCV Universal Intrinsics 不足时进入候选。

xsimd 不再作为图像 kernel 的主性能路线。P5.2 起它只能通过 legacy/experimental 显式 opt-in 参与内部验证；默认 header-only target、`cvh::headers_fast`、安装导出和 header-only CI 都不能依赖它。

## Documentation Rules

公开文档必须保持一致：

- 第一屏定位必须是 pure header-only。
- 推荐用法只写 `cvh::headers` 和 `cvh::headers_fast`。
- `.cpp` 历史代码只能被描述为 legacy/experimental。
- 算子支持状态必须区分 Supported、Supported + fast path、WIP、Out of scope。
- 性能描述必须绑定 benchmark，不写泛化的“整体快于 OpenCV”。

## Completion Criteria

一个公开算子进入 Supported 状态，至少需要：

- header-only 实现存在。
- header-only target 可编译。
- correctness test 覆盖正常路径和关键边界。
- 文档明确输入约束和未支持范围。
- README 支持矩阵能追溯到 `scripts/ci_headers_all.sh` 中的 header-only test/gate。

一个 fast-path 进入 `cvh::headers_fast`，至少需要：

- scalar fallback 已稳定。
- fast-path 正确性与 scalar 对齐。
- benchmark 证明收益存在。
- 不满足 fast-path 条件时能回退到 scalar fallback。

## 当前结论

项目价值来自“真实可用的纯 header-only OpenCV-style 子集”，而不是 header 和 `.cpp` 扩展的混合叙事。后续工作应优先收口公开面、补齐 header-only contract，再按 benchmark 扩展 `cvh::headers_fast`。
