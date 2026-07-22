# OpenCV Universal Intrinsics Adapter 引入计划

## 背景

`opencv-header-only` 的长期方向是纯 header-only。当前 header-only 层的 CPU 加速主要考虑：

- scalar fallback
- xsimd portable SIMD
- header-only `std::thread` 并行
- 少量平台 intrinsic

项目原本希望用 xsimd 接管大部分 kernel 的 SIMD 优化，但在 ARM 场景下，xsimd 与手写 NEON 的性能差距可能达到数倍。这说明 xsimd 适合作为通用 SIMD 基线，但不适合作为所有图像 kernel 的唯一高性能抽象。

OpenCV 的 Universal Intrinsics 是 OpenCV 内部用于跨平台 SIMD 的轻量抽象层，覆盖 NEON、SSE、AVX、AVX512、RVV、VSX、WASM SIMD 等平台。它比 xsimd 更贴近图像处理 kernel，尤其是：

- `uint8` / `uint16` / `int16` / `int32` 图像像素处理
- widen / narrow / pack / saturate
- interleave / deinterleave
- fixed-point resize / filter
- color conversion
- 以 OpenCV kernel 形态组织的 SIMD 算子

因此，本计划评估并规划将 OpenCV Universal Intrinsics 以 adapter 形式引入 `opencv-header-only` 的 header-only 加速层。

## 核心判断

本项目不应直接复用 OpenCV 的完整 HAL 层。

OpenCV 中的 HAL 可以粗略分为两类：

- HAL 函数接口层：`cv_hal_*`、`CALL_HAL`、`custom_hal.hpp`、模块内部替换机制等。
- Universal Intrinsics 层：`opencv2/core/hal/intrin*.hpp` 中的 `v_load`、`v_store`、`v_uint8x16`、`v_pack`、`vx_*` 等 SIMD 抽象。

前者和 OpenCV 的模块源码、CMake 配置、CPU dispatch、错误处理、内部 ABI 有较强耦合，不适合作为本项目 header-only 主线。

后者可以作为本项目的一个可选 SIMD adapter，但不能直接暴露为项目 kernel 的唯一依赖。

目标一句话：

> 引入 OpenCV Universal Intrinsics 作为 `cvh` header-only SIMD facade 的一种后端实现，而不是把 OpenCV HAL 整层搬入项目。

## 目标

- 保持 `cvh::headers` 纯 header-only，不引入 `.cpp` 编译层。
- 引入一个内部 `cvh` SIMD facade，避免业务 kernel 直接依赖 `cv::v_*`。
- 将 OpenCV Universal Intrinsics 作为 facade 的可选 adapter。
- 保留 scalar fallback，任意 SIMD adapter 缺失时 API 仍可运行。
- 保留 xsimd，但不再要求 xsimd 接管所有 kernel。
- 为 ARM 热点 kernel 提供比 xsimd 更接近手写 NEON 的实现路径。
- 建立可重复的 OpenCV upstream 同步机制。

## 非目标

- 不复用 OpenCV 完整 HAL 函数接口层。
- 不引入 OpenCV binary 依赖。
- 不要求用户安装 OpenCV。
- 不依赖 native backend。
- 不在第一阶段追求 OpenCV 完整 CPU dispatch 体系。
- 不承诺跟随 OpenCV master。
- 不把 OpenCV Universal Intrinsics 的原始 API 作为本项目公开 API。

## 设计原则

### 1. facade 优先

业务 kernel 不直接写：

```cpp
cv::v_uint8x16
cv::v_load
cv::v_pack
```

而是写：

```cpp
cvh::detail::simd::u8x16
cvh::detail::simd::load_u8
cvh::detail::simd::pack_u16_to_u8
```

这样 OpenCV Universal Intrinsics 发生 API 变化时，只需要修 adapter，不需要大面积修改 imgproc kernel。

### 2. header-only 不等于无平台特化

只要实现位于头文件中，且不要求额外 `.cpp` 编译或链接，就仍然属于 header-only 路径。

允许在 header-only 层存在：

- scalar implementation
- xsimd implementation
- OpenCV Universal Intrinsics implementation
- direct NEON / AVX implementation

### 3. 默认纯净

Lite 默认仍应保持最小依赖。

建议默认：

```cpp
CVH_ENABLE_XSIMD=0
CVH_ENABLE_OPENCV_INTRIN=0
CVH_ENABLE_PLATFORM_INTRINSICS=0
```

用户显式启用 header-only 加速 profile 时再打开这些能力。

### 4. adapter 是内部实现细节

OpenCV Universal Intrinsics adapter 不进入公开 API 承诺。

推荐命名空间：

```cpp
cvh::detail::simd
```

避免使用：

```cpp
cvh::simd
```

除非后续决定把 SIMD facade 变成稳定公开接口。

### 5. 以 benchmark 决定升降级

OpenCV Universal Intrinsics 不应凭感觉替代 xsimd。

每个 kernel 的采用顺序应由 benchmark 决定：

```text
scalar
vs xsimd
vs OpenCV Universal Intrinsics
vs direct NEON / AVX
```

如果 OpenCV Universal Intrinsics 在某个 kernel 上接近 direct NEON，就可作为该 kernel 的主 SIMD 路径。

如果仍明显落后，则保留为 portable SIMD 层，热点继续使用 direct platform intrinsics。

## 推荐目录结构

第三方源码区域：

```text
include/cvh/3rdparty/opencv_intrin/
  UPSTREAM.md
  LICENSE.opencv
  intrin.hpp
  intrin_cpp.hpp
  intrin_forward.hpp
  intrin_neon.hpp
  intrin_sse.hpp
  intrin_avx.hpp
  intrin_avx512.hpp
  ...
```

`cvh` adapter 区域：

```text
include/cvh/core/simd/
  config.h
  traits.h
  scalar_adapter.h
  xsimd_adapter.h
  opencv_intrin_adapter.h
  platform_intrin_adapter.h
  simd.h
```

业务 kernel 只 include：

```cpp
#include "cvh/core/simd/simd.h"
```

不直接 include：

```cpp
#include "cvh/3rdparty/opencv_intrin/intrin.hpp"
```

## 能力开关

保留已有开关：

```cpp
CVH_ENABLE_XSIMD
CVH_ENABLE_THREADS
CVH_ENABLE_FAST_MATH
```

新增建议：

```cpp
CVH_ENABLE_OPENCV_INTRIN
CVH_ENABLE_PLATFORM_INTRINSICS
```

含义：

- `CVH_ENABLE_OPENCV_INTRIN=1`：允许使用 vendored OpenCV Universal Intrinsics adapter。
- `CVH_ENABLE_PLATFORM_INTRINSICS=1`：允许使用项目自写 NEON / AVX 等平台专项 header 实现。

选择顺序建议：

```cpp
#if CVH_ENABLE_PLATFORM_INTRINSICS && CVH_CAN_USE_NEON
    direct_neon_impl(...);
#elif CVH_ENABLE_OPENCV_INTRIN && CVH_CAN_USE_OPENCV_INTRIN
    opencv_intrin_impl(...);
#elif CVH_ENABLE_XSIMD
    xsimd_impl(...);
#else
    scalar_impl(...);
#endif
```

注意：这里是编译期选择，不是 OpenCV 那种多目标 runtime dispatch。

## 平台检测

不要要求用户直接定义 `CVH_ARCH_NEON`、`CVH_ARCH_AVX2` 等平台宏。

用户只控制能力：

```cpp
CVH_ENABLE_OPENCV_INTRIN=1
CVH_ENABLE_PLATFORM_INTRINSICS=1
```

项目内部根据编译器宏推断平台：

```cpp
#if defined(__aarch64__) || defined(_M_ARM64)
#define CVH_ARCH_ARM64 1
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
#define CVH_ARCH_NEON 1
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define CVH_ARCH_X86_64 1
#endif

#if defined(__AVX2__)
#define CVH_ARCH_AVX2 1
#endif
```

再组合出内部能力宏：

```cpp
#if CVH_ENABLE_OPENCV_INTRIN && CVH_ARCH_NEON
#define CVH_CAN_USE_OPENCV_INTRIN 1
#endif
```

## adapter API 第一阶段范围

第一阶段不要试图完整复刻 OpenCV Universal Intrinsics。

先覆盖高频 kernel 所需的最小集合：

- lane traits
- aligned / unaligned load
- store
- setzero / setall
- add / sub / mul
- min / max
- compare
- bitwise and / or / xor
- shift
- widen
- narrow
- saturating pack
- deinterleave / interleave
- reduce sum

优先类型：

```text
u8
s16
u16
s32
f32
```

暂不优先：

```text
f64
i64
complex
mask-heavy generic algorithms
```

## 试点 kernel

建议第一阶段只选择 3 个试点，不要大面积迁移。

### P1：cvtColor BGR/RGB to Gray

原因：

- ARM 上通道 deinterleave 很关键。
- OpenCV Universal Intrinsics 对这类图像 kernel 表达能力较强。
- 很容易和 xsimd、direct NEON 做对比。

比较对象：

```text
scalar
xsimd
opencv_intrin
direct_neon
OpenCV official
```

### P2：resize bilinear u8

原因：

- 预处理高频热点。
- fixed-point、pack、saturate、边界处理都很典型。
- 能直接检验 OpenCV Universal Intrinsics 是否适合复杂图像 kernel。

### P3：normalize + HWC to CHW

原因：

- AI preprocessing 高频。
- 可以验证 load/deinterleave/convert/store 的组合性能。
- 对项目“快于 OpenCV 的预处理链路”目标价值高。

如果当前 API 还没有完整 normalize / layout conversion，可以先作为 benchmark-only kernel 验证。

## OpenCV upstream 同步策略

OpenCV Universal Intrinsics 不是完全稳定的第三方 API。项目必须采用固定版本加同步脚本的策略。

### 版本锁定

不要跟随 OpenCV master。

每次引入时固定：

```text
OpenCV release: 4.x.y
OpenCV commit: <sha>
Imported files: <whitelist>
Local patches: <patch list>
```

记录到：

```text
include/cvh/3rdparty/opencv_intrin/UPSTREAM.md
```

### 白名单导入

只导入需要的头文件，不导入整个 OpenCV core。

同步脚本建议：

```text
scripts/sync_opencv_intrin.py
```

职责：

- 从指定 OpenCV source tree 复制白名单文件。
- 校验文件 hash。
- 应用本地 namespace/include/macro patch。
- 更新 `UPSTREAM.md`。
- 拒绝未列入白名单的 OpenCV 文件进入仓库。

### 本地 patch 管理

所有本地修改必须可重复。

不要手动改 vendored 文件后不记录。

推荐维护：

```text
include/cvh/3rdparty/opencv_intrin/patches/
  0001-namespace-and-include-rewrite.patch
  0002-cvh-cpu-macro-bridge.patch
```

如果不想引入 patch 文件，也必须在 `UPSTREAM.md` 中记录等价修改说明。

### API 兼容策略

OpenCV 4.11 曾出现 universal intrinsics 运算符变更，第三方直接使用 `x + y`、`x * y` 的代码会受影响。

本项目 adapter 中应尽量使用函数形式封装，不让业务 kernel 依赖 OpenCV 原始语法：

```cpp
cvh::detail::simd::add(a, b)
cvh::detail::simd::mul(a, b)
```

adapter 内部根据 vendored OpenCV 版本适配：

```cpp
#if CVH_OPENCV_INTRIN_HAS_OPERATORS
    return a + b;
#else
    return cv::v_add(a, b);
#endif
```

业务 kernel 不关心这类差异。

## 许可证要求

本项目当前使用 Apache License 2.0。

OpenCV 4.5+ 使用 Apache License 2.0；更早版本使用 BSD 3-Clause。两者都允许源码复用，但必须保留对应版权、license、notice。

引入时必须：

- 保留 OpenCV 原始文件头部版权声明。
- 添加 `LICENSE.opencv`。
- 如果有 NOTICE 要求，按 OpenCV 版本保留对应 notice。
- 在 `UPSTREAM.md` 中记录文件来源与本地修改。
- README 或 third-party 文档中说明 vendored OpenCV Universal Intrinsics 的来源。

工程侧建议优先选择 OpenCV 4.5+ 的 Apache 2.0 版本，降低许可证组合复杂度。

## 验证矩阵

### 编译矩阵

至少覆盖：

```text
macOS AppleClang ARM64
Linux GCC x86_64
Linux Clang x86_64
```

有条件时增加：

```text
Android NDK ARM64
Jetson ARM64
Windows MSVC x64
```

### 配置矩阵

```text
scalar only
CVH_ENABLE_XSIMD=1
CVH_ENABLE_OPENCV_INTRIN=1
CVH_ENABLE_PLATFORM_INTRINSICS=1
```

### 正确性验证

每个试点 kernel 需要比较：

```text
scalar vs xsimd
scalar vs opencv_intrin
scalar vs direct_neon
cvh vs OpenCV official
```

允许误差由 kernel 单独定义。

对 `uint8` fixed-point kernel，优先追求 bit-exact 或接近 OpenCV 的整数舍入行为。

### 性能验证

每个试点 kernel 输出：

```text
impl
platform
compiler
flags
shape
channels
dtype
ms_per_iter
gb_per_sec
speedup_vs_scalar
speedup_vs_xsimd
speedup_vs_opencv
```

在 ARM 上重点关注：

- xsimd vs opencv_intrin
- opencv_intrin vs direct_neon
- cvh chain vs OpenCV official preprocessing chain

## 风险

### 风险 1：复制范围膨胀

如果直接搬 OpenCV HAL，很容易拖入大量 `cvdef.h`、dispatch、模块内部 helper。

控制方式：

- 只导入 Universal Intrinsics 白名单。
- 业务 kernel 只依赖 `cvh::detail::simd`。
- 不允许直接 include OpenCV core 其他头文件。

### 风险 2：OpenCV upstream API 变化

OpenCV Universal Intrinsics 曾发生过破坏第三方代码的 API 调整。

控制方式：

- 固定 release/commit。
- adapter 隔离原始 API。
- 升级只通过同步脚本和 benchmark gate。

### 风险 3：header-only 编译时间增加

SIMD 头文件可能显著增加模板和宏展开成本。

控制方式：

- 默认关闭。
- `cvh::headers_fast` 或用户宏显式开启。
- kernel 按需 include `core/simd/simd.h`，避免全局重头文件污染。

### 风险 4：和用户 OpenCV include 冲突

如果 vendored OpenCV intrinsics 仍使用 `cv` namespace 或 `CV_*` 宏，可能和用户工程里的 OpenCV 冲突。

控制方式：

- 尽量把 adapter 封装在 `cvh::detail`。
- 不在公开头里暴露 `cv::v_*` 类型。
- 评估是否需要 namespace rewrite。
- 如果 namespace rewrite 成本过高，必须限定 include 边界并做冲突测试。

### 风险 5：性能仍不如手写 NEON

OpenCV Universal Intrinsics 仍是抽象层，不保证所有 ARM kernel 接近手写 NEON。

控制方式：

- benchmark 决策。
- 极热点允许 direct NEON header 实现覆盖。
- OpenCV Universal Intrinsics 作为 portable SIMD 层，而不是唯一性能层。

## 分阶段计划

### P0：决策冻结

状态：已冻结。

- 明确不引入 OpenCV 完整 HAL。
- 明确只评估 Universal Intrinsics adapter。
- 明确 adapter 只属于 header-only CPU 加速层。
- 新增本计划文档。

冻结结果：

- `cvh` 不 vendor OpenCV 完整 HAL 函数接口层。
- `cvh` 只评估并引入 OpenCV Universal Intrinsics 的白名单头文件子集。
- OpenCV Universal Intrinsics adapter 只能位于 header-only CPU 加速层。
- 业务 kernel 只能通过 `cvh::detail::simd` facade 使用该 adapter。
- 后续实现从 P1 开始，不应绕过本决策直接引入 `cv_hal_*`、`CALL_HAL` 或 OpenCV 模块内部 dispatch。

### P1：第三方源码导入试验

状态：已完成。

- 选定一个 OpenCV release/commit。
- 建立 `include/cvh/3rdparty/opencv_intrin/`。
- 建立 `UPSTREAM.md` 和 license 文件。
- 写最小同步脚本。
- 完成空 adapter 编译 smoke。

落地结果：

- OpenCV upstream 固定为本地源码树 `/Users/zmu/work/my_project/ocvh/opencv`。
- Upstream commit 固定为 `d48bf69f65444a13f8a34b8982b083c1b78fa0e8`，describe 为 `4.13.0-457-gd48bf69f65`。
- 白名单导入 `intrin.hpp`、`intrin_cpp.hpp`、`intrin_forward.hpp`、`simd_utils.impl.hpp` 和 `LICENSE`。
- 新增最小 OpenCV core shim：`cvdef.h`、`utility.hpp`、`saturate.hpp`。
- 新增 `scripts/sync_opencv_intrin.py --check` 用于校验白名单文件是否和本地 OpenCV upstream 一致。
- 新增 `cvh_opencv_intrin_smoke`，以 `CVH_ENABLE_OPENCV_INTRIN=1` 和 `CV_FORCE_SIMD128_CPP=1` 验证 header-only 编译路径。

验收：

- 默认 Lite 不 include OpenCV intrinsics。
- `CVH_ENABLE_OPENCV_INTRIN=1` 时可以独立编译一个 smoke。
- 不需要链接 OpenCV。
- `scripts/sync_opencv_intrin.py --check` 通过。
- `cvh_opencv_intrin_smoke` 通过。

### P2：facade 最小 API

状态：已完成。

- 增加 `include/cvh/core/simd/`。
- 实现 scalar adapter。
- 实现 OpenCV Universal Intrinsics adapter 的最小操作集合。
- 保持 xsimd adapter 可并存。

落地结果：

- 新增 `cvh::detail::simd` facade 入口 `include/cvh/core/simd/simd.h`。
- 新增 scalar adapter：`include/cvh/core/simd/scalar_adapter.h`。
- 新增 xsimd adapter：`include/cvh/core/simd/xsimd_adapter.h`。
- 扩展 OpenCV Universal Intrinsics adapter：`include/cvh/core/simd/opencv_intrin_adapter.h`。
- 第一阶段 facade API 聚焦 `f32` 向量，覆盖 `load_f32`、`store_f32`、`setzero_f32`、`setall_f32`、`add`、`sub`、`mul`、`min`、`max`、`reduce_sum`、`f32_lanes`、`backend_name`。
- 新增同源 smoke `cvh_simd_facade_smoke.cpp`，分别编译 scalar / xsimd / opencv_intrin 三个目标。

边界：

- P2 不直接迁移 imgproc kernel。
- P2 不暴露 `cv::v_*` 给业务 kernel。
- `u8`、deinterleave、pack、widen/narrow 等图像专用 API 留到 P3 试点 kernel 根据真实需求扩展。

验收：

- 同一份测试能跑 scalar / xsimd / opencv_intrin。
- kernel 不直接使用 `cv::v_*`。
- `cvh_simd_facade_scalar_smoke` 通过。
- `cvh_simd_facade_xsimd_smoke` 通过。
- `cvh_simd_facade_opencv_intrin_smoke` 通过。

### P3：试点 kernel

状态：进行中，首个试点已落地。

P3 不直接跳到 `cvh::headers_fast`。它先分成若干可验收的小阶段，只有性能和正确性都成立，才进入 P4。

#### P3.1：`BGR2GRAY u8` 正确性试点

状态：已完成。

- 扩展 `cvh::detail::simd` facade 的图像专用最小 API：`u8/u16/u32`、`load_deinterleave3_u8`、`expand_u8`、`expand_low_u16`、`expand_high_u16`、`setall_u16`、`setall_u32`、`mul_expand_u16`、`u32 add/mul`、`rshr_pack_u32_to_u16`、`pack_u16_to_u8`、`store_u8`、`u8_lanes`。
- `include/cvh/imgproc/cvtcolor.h` 在 `CVH_ENABLE_OPENCV_INTRIN=1` 时，为 `COLOR_BGR2GRAY + CV_8UC3` 启用 header-only SIMD 路径。
- 该路径通过 facade 调用 OpenCV Universal Intrinsics，不在业务 kernel 中直接暴露 `cv::v_*`。
- xsimd adapter 先复用 scalar `u8` facade 实现，保证接口兼容；不把它声明为 P3 的性能路径。
- 新增 `cvh_cvtcolor_opencv_intrin_smoke` 覆盖 16-lane 主循环和尾部像素，并对齐 scalar 定点公式。
- 同源 `cvh_simd_facade_smoke.cpp` 已扩展到 `u8` facade，覆盖 scalar / xsimd / opencv_intrin 三个 adapter 的接口编译和基础正确性。

验证：

- `cvh_simd_facade_scalar_smoke` 通过。
- `cvh_simd_facade_xsimd_smoke` 通过。
- `cvh_simd_facade_opencv_intrin_smoke` 通过。
- `cvh_cvtcolor_opencv_intrin_smoke` 通过。

#### P3.2：`BGR2GRAY u8` 性能验证 gate

状态：已完成首轮本机 ARM quick 验证，P4 gate 暂未通过。

目标：

- 加一个 header-only benchmark，同一输入分别跑 scalar fallback 和 `CVH_ENABLE_OPENCV_INTRIN=1`。
- benchmark 不链接 OpenCV，不启用 native backend。
- 输出每个尺寸的耗时和吞吐量，至少包含 `ms/call`、`MPix/s`、speedup。
- 尺寸矩阵至少覆盖 `640x480`、`1280x720`、`1920x1080`、`3840x2160`。
- 每个 case 需要 warmup 和多次 repeat，记录 min/median，避免单次计时噪声。
- 在 ARM 机器上跑，确认 OpenCV Universal Intrinsics 是否真的值得进入 P4。

对比范围：

- scalar fallback：默认 `cvh::headers`，不打开 `CVH_ENABLE_OPENCV_INTRIN`。
- opencv_intrin：打开 `CVH_ENABLE_OPENCV_INTRIN=1`。
- xsimd 暂不作为 `u8 BGR2GRAY` 性能候选，因为当前 xsimd 的 `u8` facade 只是 scalar 兼容实现。

进入 P4 的建议门槛：

- ARM 上至少一个主流尺寸达到明显收益，例如 `>= 1.5x` scalar。
- 不能在常见尺寸上出现系统性退化；如有退化，需要记录原因并暂停 P4。
- benchmark 结果写入本计划或单独 benchmark 结果文档，作为 P4 的输入证据。

落地结果：

- 新增 header-only benchmark target：`cvh_benchmark_cvtcolor_bgr2gray_header`。
- benchmark 只链接 `cvh::headers`，不链接 OpenCV，不启用 native backend。
- benchmark 在同一个可执行文件里用同一份输入对比 `cvh::detail::cvtcolor_bgr2gray_u8_scalar_impl` 和 `CVH_ENABLE_OPENCV_INTRIN=1` 的 `cvh::cvtColor(..., COLOR_BGR2GRAY)`。
- 为避免不同宏 translation unit 带来的 inline ODR 风险，benchmark 不采用双 TU 方案。
- ARM 构建为该 target 显式打开 `CV_NEON=1`，当前结果记录为 `opencv_intrin_neon`。
- 为测真实 NEON 路径，vendor 白名单新增 `intrin_neon.hpp` 和直接依赖的 `intrin_math.hpp`；本地 `cvdef.h` shim 补 `CV_DECL_ALIGNED` 和 `<arm_neon.h>` 包含。
- 结果 CSV：`benchmark/cvtcolor_bgr2gray_header_current.csv`。

本机 quick 结果：

环境：

- 机器：`arm64`，`Apple M5`。
- 编译器：`AppleClang 21.0.0.21000099`。
- 构建：`Release`，`CVH_BUILD_NATIVE_BACKEND=OFF`，`CVH_BUILD_BENCHMARKS=ON`。
- 命令：`./build-opencv-intrin-p3-bench/cvh_benchmark_cvtcolor_bgr2gray_header --profile quick --warmup 3 --iters 10 --repeats 7 --output benchmark/cvtcolor_bgr2gray_header_current.csv`。

| shape | scalar median ms | opencv_intrin_neon median ms | scalar MPix/s | opencv_intrin_neon MPix/s | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `640x480` | 0.079671 | 0.064283 | 3855.866892 | 4778.846139 | 1.239370 |
| `1280x720` | 0.243383 | 0.176900 | 3786.619706 | 5209.723007 | 1.375824 |
| `1920x1080` | 0.302575 | 0.207238 | 6853.176898 | 10005.911092 | 1.460040 |
| `3840x2160` | 0.965204 | 0.817946 | 8593.414741 | 10140.525204 | 1.180034 |

结论：

- `opencv_intrin_neon` 与 scalar 输出 checksum 完全一致。
- 当前 BGR2GRAY NEON UI 路径有收益，但没有达到 `>= 1.5x` 的 P4 建议门槛。
- 不建议基于当前结果进入 P4；下一步进入 P3.4，先拆解性能成本并解释 `3840x2160` speedup 下降到 `1.18x` 的原因。

#### P3.3：补齐 `RGB2GRAY`

状态：已完成。

P3.1 当时未直接落地 `RGB2GRAY` 的原因：

- 当时项目的公开 `ColorConversionCodes` 里只有 `COLOR_BGR2GRAY`，没有 `COLOR_RGB2GRAY`。
- 如果在 P3.1 里直接补 `COLOR_RGB2GRAY`，会同时改变公开 API、枚举兼容性、fallback 分发和 contract tests，范围超过本次“验证 OpenCV Universal Intrinsics 是否能接入真实 kernel”的目标。
- BGR 和 RGB 的 SIMD 算法本身可以共用同一条路径，只需要把 R/B 通道选择参数化；阻塞点不是 SIMD 能力，而是先补公开 color code 和标量语义。

落地结果：

- `include/cvh/imgproc/detail/common.h` 追加 `COLOR_RGB2GRAY = 70`，不插入原有枚举中间，避免改变既有 code 的数值。
- `include/cvh/imgproc/cvtcolor.h` 将 3 通道转灰的 scalar fallback 和 OpenCV UI `u8` SIMD 路径统一参数化为编译期 `rgb_order`。
- 保留 `cvtcolor_bgr2gray_u8_scalar_impl` 作为 benchmark 的显式 scalar baseline 入口，新增 `cvtcolor_rgb2gray_u8_scalar_impl`。
- `cvtColor(..., COLOR_RGB2GRAY)` 支持 `CV_8UC3 -> CV_8UC1` 和 `CV_32FC3 -> CV_32FC1`。
- `CVH_ENABLE_OPENCV_INTRIN=1` 时，`COLOR_RGB2GRAY + CV_8UC3` 走同一条 header-only OpenCV Universal Intrinsics SIMD 路径。
- `cvh_cvtcolor_opencv_intrin_smoke` 覆盖 `BGR2GRAY` 和 `RGB2GRAY`。
- `imgproc_cvtcolor_contract_test` 补充 `RGB2GRAY` 的 `u8`、`f32`、非连续 ROI 和非法输入覆盖。

#### P3.4：`BGR/RGB2GRAY` 性能诊断与优化决策

状态：待做。

目标：

- 不直接进入 P4，先解释 `BGR2GRAY u8` 在 ARM 上未稳定达到 `>= 1.5x` 的原因。
- 明确 `3840x2160` speedup 从 `1920x1080` 的 `1.460040x` 下降到 `1.180034x` 是由外层调用、`Mat` 分配、内核指令序列、尾部处理、cache/内存带宽还是写回压力导致。
- 用同一套 benchmark 覆盖 `BGR2GRAY` 和 `RGB2GRAY`，确认 P3.3 的共享路径没有引入额外性能成本。

步骤 1：拆 benchmark

- 对比 `cvh::cvtColor` 公共入口和 `cvh::detail::*_simd_impl` 直接入口。
- 对比每次调用内部 `dst.create` 和预分配 `dst` 后复用输出 buffer。
- 将 `RGB2GRAY` 加入同一个 header-only benchmark，记录 BGR/RGB 两条编译期 `rgb_order` 路径的耗时差异。
- 输出继续保留 `ms/call`、`MPix/s`、speedup、checksum、shape、backend。

步骤 2：拆内核成本

- 单独测 `v_load_deinterleave` 成本。
- 单独测 widen / multiply / accumulate / pack 成本。
- 记录每个 shape 的主循环像素数、尾部像素数和尾部比例，重点覆盖非 16 对齐宽度。
- 增加足够大的输入矩阵观察 4K 行为，判断是否已经转为内存带宽、cache 或写回主导。

步骤 3：给决策

- 如果主要成本在 `cvh::cvtColor` 公共入口或 `dst.create`，优先优化 header-only 公共路径。
- 如果主要成本在 OpenCV Universal Intrinsics 生成的 ARM 指令序列，考虑增加 header-only direct NEON 特化；该特化仍属于 header-only 加速层，不进入 native。
- 如果 `BGR/RGB2GRAY` 仍无法稳定达到 `>= 1.5x`，暂停 P4，不继续扩大 kernel 迁移面，进入 P3.5 选择第二个热点验证 OpenCV UI 是否仍有引入价值。

验收：

- benchmark 能在同一输入上输出公共入口、直接 SIMD 入口、预分配输出和 `BGR/RGB2GRAY` 的对比结果。
- 文档记录 `3840x2160` speedup 下降的主要原因或最可能原因。
- 给出明确结论：优化当前 kernel、增加 direct NEON header-only 特化、进入 P3.5，或暂停 P4。

#### P3.5：选择第二个热点 kernel

状态：待 P3.4 诊断结果决定。

候选：

- `resize bilinear u8`
- `normalize + HWC to CHW` benchmark-only 或正式 API

选择原则：

- 如果 P3.4 证明 OpenCV Universal Intrinsics 对 `BGR/RGB2GRAY u8` 的收益仍有明确优化空间，优先优化当前 kernel。
- 如果 P3.4 证明收益不足但原因是该 kernel 特性不适合 UI，选择第二个热点继续验证，优先迁移 `resize bilinear u8`。
- 如果 P3.4 证明 UI 路径在 ARM 上没有足够性价比，暂停 P4，不继续扩大 kernel 迁移面。
- direct NEON 仍作为后续平台特化备选，不进入当前 header-only OpenCV UI adapter 的默认路径。

P3 总体验收：

- 正确性对齐 scalar。
- ARM 上给出 scalar / opencv_intrin 的性能对比。
- xsimd 只有在实现真实 `u8` vector path 后才进入该 kernel 的性能矩阵。
- 至少一个热点证明 OpenCV Universal Intrinsics 有引入价值。

### P4：headers_fast profile

如果 P3 结果成立，增加纯 header-only 加速 profile：

```cmake
cvh::headers_fast
```

该 target 只传播宏和 include，不编译 `.cpp`：

```cmake
CVH_ENABLE_XSIMD=1
CVH_ENABLE_OPENCV_INTRIN=1
CVH_ENABLE_PLATFORM_INTRINSICS=1
```

验收：

- `cvh::headers` 保持默认纯净。
- `cvh::headers_fast` 仍然不构建 native backend。

### P5：常态维护

- 固定 OpenCV upstream 版本。
- 每次升级通过同步脚本。
- 每次升级必须跑正确性和性能矩阵。
- benchmark 退化超过阈值则不升级。

## 成功标准

短期成功：

- 可以在 header-only 模式下启用 OpenCV Universal Intrinsics adapter。
- 默认 Lite 不受影响。
- 不引入 native 编译层。
- 至少一个 ARM 热点 kernel 明显优于 xsimd。

中期成功：

- `cvh` 形成自己的 `cvh::detail::simd` 方言。
- xsimd、OpenCV Universal Intrinsics、direct platform intrinsics 可以共存。
- 试点 kernel 的实现选择由 benchmark 驱动。

长期成功：

- OpenCV Universal Intrinsics 成为 `cvh` 图像 kernel 的重要 portable SIMD 层。
- ARM 关键预处理链路能接近或超过 OpenCV official。
- 项目仍保持 header-only 默认体验和可审计第三方依赖边界。

## 结论

OpenCV Universal Intrinsics 兼容进 `opencv-header-only` 的难度是中等；难点不在能否编译，而在长期维护、命名隔离、宏配置、API 变化和性能验证。

建议采用：

```text
固定 upstream 版本
白名单 vendor
cvh::detail::simd facade
adapter 隔离
benchmark gate
小范围试点
```

不建议采用：

```text
复制 OpenCV 完整 HAL
业务 kernel 直接写 cv::v_*
跟随 OpenCV master
默认开启所有 SIMD 头文件
```

这条路线可以在不触碰 native backend 的前提下，增强 header-only 层的 CPU SIMD 能力，并给 ARM 热点 kernel 留出比 xsimd 更高性能的实现空间。
