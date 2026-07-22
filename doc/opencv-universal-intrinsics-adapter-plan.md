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
- 不依赖 `.cpp` 实验路径。
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
- benchmark 不链接 OpenCV，不启用 `.cpp` 实验路径。
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
- benchmark 只链接 `cvh::headers`，不链接 OpenCV，不启用 `.cpp` 实验路径。
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

状态：已完成，结论是不进入 P4，进入 P3.5 选择第二个热点 kernel。

目标：

- 不直接进入 P4，先解释 `BGR2GRAY u8` 在 ARM 上未稳定达到 `>= 1.5x` 的原因。
- 明确 `3840x2160` speedup 从 `1920x1080` 的 `1.460040x` 下降到 `1.180034x` 是由外层调用、`Mat` 分配、内核指令序列、尾部处理、cache/内存带宽还是写回压力导致。
- 用同一套 benchmark 覆盖 `BGR2GRAY` 和 `RGB2GRAY`，确认 P3.3 的共享路径没有引入额外性能成本。

步骤 1：拆 benchmark

- 对比 `cvh::cvtColor` 公共入口和 `cvh::detail::*_simd_impl` 直接入口。
- 对比每次调用内部 `dst.create` 和预分配 `dst` 后复用输出 buffer。
- 将 `RGB2GRAY` 加入同一个 header-only benchmark，记录 BGR/RGB 两条编译期 `rgb_order` 路径的耗时差异。
- 输出继续保留 `ms/call`、`MPix/s`、speedup、checksum、shape、backend。

P3.4.1 落地结果：

- `benchmark/cvtcolor_bgr2gray_header_benchmark.cpp` 扩展为 `BGR2GRAY/RGB2GRAY` 同测。
- CSV schema 新增 `entry`、`allocation_mode`、`simd_lanes`、`tail_pixels`、`tail_ratio`。
- `entry` 包含 `public_cvtColor` 和 `direct_detail`，用于区分公共 API 入口和直接 SIMD helper。
- `allocation_mode` 包含 `reuse` 和 `recreate`，其中 `recreate` 每次调用前 `dst.release()`，用于强制暴露输出重建成本。
- quick 结果写入 `benchmark/cvtcolor_bgr_rgb_gray_header_p341.csv`。

P3.4.1 本机 quick 观察：

| op | shape | scalar reuse median ms | public reuse median ms | direct SIMD reuse median ms | public/direct |
| --- | --- | ---: | ---: | ---: | ---: |
| `BGR2GRAY` | `1920x1080` | 0.240967 | 0.203238 | 0.202137 | 1.0054 |
| `BGR2GRAY` | `3840x2160` | 0.962138 | 0.817408 | 0.819667 | 0.9972 |
| `RGB2GRAY` | `1920x1080` | 0.241138 | 0.201179 | 0.203350 | 0.9893 |
| `RGB2GRAY` | `3840x2160` | 0.970117 | 0.818017 | 0.818233 | 0.9997 |

阶段判断：

- 1080p 和 4K 下 `public_cvtColor` 与 `direct_detail` 基本等价，公共入口不是 4K speedup 下降的主因。
- 1080p 和 4K 下 `reuse` 与 `recreate` 基本等价，输出 `Mat` 重建不是 4K speedup 下降的主因。
- 4K 下 `BGR2GRAY` 和 `RGB2GRAY` 的 direct SIMD median 分别为 `0.819667 ms` 和 `0.818233 ms`，P3.3 共享路径没有暴露额外成本。
- quick 尺寸宽度均可被 16 整除，`tail_pixels=0`；尾部成本需要在后续 full/非对齐尺寸中继续确认。
- 当前证据更指向内核指令序列或内存层面，需要进入步骤 2 拆 `v_load_deinterleave`、widen/mul/pack 和读写带宽。

步骤 2：拆内核成本

- 单独测 `v_load_deinterleave` 成本。
- 单独测 widen / multiply / accumulate / pack 成本。
- 记录每个 shape 的主循环像素数、尾部像素数和尾部比例，重点覆盖非 16 对齐宽度。
- 增加足够大的输入矩阵观察 4K 行为，判断是否已经转为内存带宽、cache 或写回主导。

P3.4.2 落地结果：

- `cvh::detail::simd` facade 新增 `load_u8`，同步实现 scalar / xsimd / opencv_intrin adapter，并加入 `cvh_simd_facade_smoke` 覆盖。
- header-only benchmark 增加 micro rows：`MICRO_STORE_U8`、`MICRO_SCALAR_READ3_WRITE1_U8`、`MICRO_LOAD_DEINTERLEAVE_STORE_U8`、`MICRO_PLANAR_LOAD3_WIDEN_MUL_PACK_STORE_U8`。
- `full` profile 增加尾部压力尺寸：`63x480`、`1919x1080`、`3839x2160`。
- full 结果写入 `benchmark/cvtcolor_bgr_rgb_gray_header_p342.csv`。

P3.4.2 本机 full 观察：

| case | shape | tail ratio | median ms | MPix/s | speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| `BGR2GRAY direct SIMD` | `63x480` | 0.238095 | 0.007896 | 3829.88 | 0.965703 |
| `BGR2GRAY direct SIMD` | `1919x1080` | 0.007817 | 0.204683 | 10125.50 | 1.192371 |
| `BGR2GRAY direct SIMD` | `3839x2160` | 0.003907 | 0.822471 | 10082.11 | 1.186724 |
| `BGR2GRAY direct SIMD` | `3840x2160` | 0.000000 | 0.824250 | 10062.97 | 1.178769 |
| `BGR2GRAY direct SIMD` | `4096x2160` | 0.000000 | 0.879483 | 10059.72 | 1.180603 |

4K micro cost：

| micro row | `3840x2160` median ms | MPix/s | observation |
| --- | ---: | ---: | --- |
| `MICRO_STORE_U8` | 0.140338 | 59103.23 | write-only 成本明显小于完整 kernel |
| `MICRO_LOAD_DEINTERLEAVE_STORE_U8` | 0.306550 | 27057.25 | `v_load_deinterleave + store` 不是唯一瓶颈 |
| `MICRO_PLANAR_LOAD3_WIDEN_MUL_PACK_STORE_U8` | 0.728500 | 11385.59 | widen/mul/pack 相关路径占主要成本 |
| `BGR2GRAY direct SIMD` | 0.824250 | 10062.97 | 与 planar arithmetic micro 同量级 |

阶段判断：

- 4K 附近 `3839x2160`、`3840x2160`、`4096x2160` 的 SIMD 吞吐都约 `10.0` GPix/s，`3839x2160` 非 16 对齐并没有明显退化；常规尾部不是 4K speedup 下降主因。
- `63x480` 的尾部比例达到 `23.8%`，SIMD speedup 掉到 `0.965703`，说明高尾部比例的小宽度场景不适合当前 16-lane 主循环。
- `MICRO_STORE_U8` 远快于完整 kernel，写回不是主因。
- `MICRO_LOAD_DEINTERLEAVE_STORE_U8` 明显快于完整 kernel，单独的 `v_load_deinterleave` 也不是主因。
- `MICRO_PLANAR_LOAD3_WIDEN_MUL_PACK_STORE_U8` 与完整 SIMD kernel 同量级，当前瓶颈更偏向 widen / multiply / accumulate / pack 指令序列，而不是外层入口、`Mat` 分配、尾部或纯内存写回。
- 本轮 full 中 1080p 与 4K 的 direct SIMD speedup 都约 `1.18x`，P3.2 中 `1920x1080` 的 `1.46x` 更像 benchmark 运行波动或 scalar baseline 状态差异，不应作为 P4 gate 的唯一依据。

步骤 3：给决策

- 如果主要成本在 `cvh::cvtColor` 公共入口或 `dst.create`，优先优化 header-only 公共路径。
- 如果主要成本在 OpenCV Universal Intrinsics 生成的 ARM 指令序列，考虑增加 header-only direct NEON 特化；该特化仍属于 header-only 加速层，不进入 native。
- 如果 `BGR/RGB2GRAY` 仍无法稳定达到 `>= 1.5x`，暂停 P4，不继续扩大 kernel 迁移面，进入 P3.5 选择第二个热点验证 OpenCV UI 是否仍有引入价值。

P3.4 决策结果：

- `BGR/RGB2GRAY u8` 继续保留当前 OpenCV Universal Intrinsics header-only 路径，但不足以作为进入 P4 的依据。
- 4K speedup 下降的主要原因不是公共入口、`Mat` 分配、常规尾部或单纯写回，更接近当前 widen / multiply / accumulate / pack 指令序列的成本。
- 暂不为 `BGR/RGB2GRAY u8` 追加 direct NEON header-only 特化；该方向保留为后续平台专项优化备选。
- 进入 P3.5，用第二个热点 kernel 验证 OpenCV Universal Intrinsics 是否在另一类图像处理模式中仍有价值。

验收：

- benchmark 能在同一输入上输出公共入口、直接 SIMD 入口、预分配输出和 `BGR/RGB2GRAY` 的对比结果。
- 文档记录 `3840x2160` speedup 下降的主要原因或最可能原因。
- 给出明确结论：优化当前 kernel、增加 direct NEON header-only 特化、进入 P3.5，或暂停 P4。

#### P3.5：选择第二个热点 kernel

状态：已完成，选择 `resize bilinear u8`。

候选：

- `resize bilinear u8`
- `normalize + HWC to CHW` benchmark-only 或正式 API

选择原则：

- 如果 P3.4 证明 OpenCV Universal Intrinsics 对 `BGR/RGB2GRAY u8` 的收益仍有明确优化空间，优先优化当前 kernel。
- 如果 P3.4 证明收益不足但原因是该 kernel 特性不适合 UI，选择第二个热点继续验证，优先迁移 `resize bilinear u8`。
- 如果 P3.4 证明 UI 路径在 ARM 上没有足够性价比，暂停 P4，不继续扩大 kernel 迁移面。
- direct NEON 仍作为后续平台特化备选，不进入当前 header-only OpenCV UI adapter 的默认路径。

选择结果：

- 第二个热点选择 `cvh::resize(..., INTER_LINEAR)` 的 `CV_8U` 路径，首轮只做 `CV_8UC1`。
- 不选择 `normalize + HWC to CHW` 作为当前第二热点，因为项目当前没有稳定的 normalize/layout-conversion 公共 API；先做 benchmark-only kernel 会绕开现有 `cvh` 兼容 surface，不能直接回答 OpenCV Universal Intrinsics 是否值得进入正式 header-only imgproc 路径。
- 不继续扩大 `BGR/RGB2GRAY` 面，避免在单个低算术强度 kernel 上过度优化后误判 OpenCV Universal Intrinsics 的整体价值。

选择 `resize bilinear u8` 的理由：

- 它已经是公开 API，位于 `include/cvh/imgproc/resize.h`，并已有 `test/imgproc/imgproc_resize_contract_test.cpp` 覆盖。
- 当前 header-only fallback 对 `INTER_LINEAR` 使用逐像素浮点坐标计算和插值，存在明确的算法层与 SIMD 层拆解空间。
- 它能验证不同于 `BGR/RGB2GRAY` 的模式：坐标/权重表、4-tap 插值、边界处理、saturate/pack，以及 C1/C3/C4 通道扩展。
- 旧有 imgproc benchmark 已覆盖 `RESIZE_LINEAR CV_8U`，但该 benchmark 链接 native；P3.5 后续必须新增只链接 `cvh::headers` 的专用 benchmark，避免混入 native 后端结果。

边界：

- 本轮仍只关注 CPU header-only，不启用 `.cpp` 实验路径。
- P3.5 的结论只决定第二热点，不承诺进入 P4。
- `resize` 的 direct NEON 特化仍是后续备选，不能抢在 OpenCV Universal Intrinsics 试点之前进入默认路径。

#### P3.6：`resize bilinear u8` OpenCV UI 试点

状态：已完成，P3.6.1/P3.6.2/P3.6.3/P3.6.4 已完成本机诊断。

P3.6.1：建立 header-only resize benchmark

状态：已完成。

- 新增专用 benchmark，对比当前 scalar fallback、公共 `cvh::resize` 入口和后续 direct detail 入口。
- 输出 `ms/call`、`MPix/s`、输入尺寸、输出尺寸、通道数、dtype、allocation mode、checksum。
- 尺寸至少覆盖 `640x480->320x240`、`1280x720->640x360`、`1920x1080->960x540`、`3840x2160->1920x1080`。
- 额外覆盖非整数缩放和非 16 对齐宽度，避免只验证 2x downsample 的理想路径。
- benchmark target 必须只链接 `cvh::headers`，不链接 OpenCV，也不链接 `.cpp` 实验 target。

落地结果：

- 新增 target：`cvh_benchmark_resize_bilinear_header`。
- 新增源码：`benchmark/resize_bilinear_header_benchmark.cpp`。
- target 只链接 `cvh::headers`，并显式打开 `CVH_ENABLE_OPENCV_INTRIN=1`、`CVH_ENABLE_PLATFORM_INTRINSICS=1`；ARM 构建打开 `CV_NEON=1`。
- P3.6.1/P3.6.2 阶段尚未有 OpenCV Universal Intrinsics fast-path，因此当时 CSV 中公共入口 backend 标记为 `opencv_intrin_neon_no_resize_fastpath`，避免误读。
- quick 结果写入 `benchmark/resize_bilinear_header_p361.csv`。
- full 结果写入 `benchmark/resize_bilinear_header_p362.csv`。

P3.6.2：拆算法层成本

状态：已完成。

- 单独记录坐标/权重计算成本，避免把当前 fallback 的逐像素 `floor/clamp` 成本误计为 SIMD 收益。
- 将坐标/权重预计算和像素插值内核分开计时。
- 先用 `CV_8UC1` 做基线；只有 C1 路径收益明确，再扩展 C3/C4。

落地结果：

- benchmark 增加 `MICRO_RESIZE_LINEAR_TABLES`，只测 `x/y` 坐标和 `wx/wy` 权重表构建。
- benchmark 增加 `MICRO_RESIZE_LINEAR_U8_C1_PRECOMPUTED_TABLES`，只测复用预计算表后的 `CV_8UC1` scalar bilinear 像素内核。
- 每个 case 启动前做 correctness self-check：`detail::resize_fallback`、`cvh::resize`、预计算表像素内核三者输出必须一致。

P3.6.1/P3.6.2 本机 full 观察：

| shape | direct scalar ms | public resize ms | precomputed scalar pixel ms | table build ms | precomputed speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `640x480->320x240` | 0.123125 | 0.113563 | 0.081450 | 0.002004 | 1.511664 |
| `1280x720->640x360` | 0.280046 | 0.281296 | 0.217896 | 0.003450 | 1.285229 |
| `1920x1080->960x540` | 0.626088 | 0.630283 | 0.462958 | 0.005162 | 1.352363 |
| `3840x2160->1920x1080` | 2.526583 | 2.531487 | 2.003046 | 0.010621 | 1.261371 |
| `641x479->321x239` | 0.096400 | 0.096462 | 0.069792 | 0.002075 | 1.381253 |
| `1920x1080->853x480` | 0.511246 | 0.513825 | 0.374946 | 0.004712 | 1.363519 |
| `3839x2160->1917x1079` | 2.600571 | 2.598050 | 1.958304 | 0.010625 | 1.327971 |

阶段判断：

- `public_resize` 与 `detail::resize_fallback` 在主要尺寸上基本等价，公共入口不是 resize 当前成本主因。
- 坐标/权重表构建成本很小，4K half resize 中约 `0.010621 ms`，远小于完整 scalar 的 `2.526583 ms`。
- 预计算表后的 scalar 像素内核已经有 `1.24x` 到 `1.51x` 收益，说明当前 fallback 的逐像素坐标/权重计算是明确可优化点。
- 后续评估 OpenCV Universal Intrinsics resize 时，不能只拿当前 naive scalar fallback 当唯一 baseline；应同时对比预计算表 scalar pixel kernel，否则会高估 SIMD adapter 的贡献。

P3.6.3：实现最小 OpenCV Universal Intrinsics resize 路径

状态：已完成。

- 只覆盖 `INTER_LINEAR + CV_8UC1`。
- 优先复用 `cvh::detail::simd` facade，业务 kernel 不直接写 `cv::v_*`。
- 如需新增 facade API，限制在 resize 必需的 load/convert/mul/pack/store 操作。
- scalar fallback 保持可用；不满足 fast-path 条件时回退到现有实现。

落地结果：

- `cvh::detail::simd` facade 新增 `load_deinterleave2_u8`、`add(u16)`、`rshr_pack_u16_to_u8<shift>`，同步实现 scalar / xsimd / opencv_intrin adapter。
- `include/cvh/imgproc/resize.h` 新增 `resize_linear_u8c1_downsample2_opencv_intrin_impl`。
- fast-path 条件严格限定为 `INTER_LINEAR + CV_8UC1 + exact 2x downsample`，即 `src_cols == dst_cols * 2` 且 `src_rows == dst_rows * 2`。
- 2x half-resize 下现有 scalar 浮点公式等价为四个源像素平均，OpenCV UI 路径使用 `(p00 + p01 + p10 + p11 + 2) >> 2`，与当前 `saturate_cast<uchar>(round(...))` 语义 bit-exact。
- 不满足 exact 2x 的 `CV_8UC1 INTER_LINEAR` 仍回退到原有 scalar fallback，避免用不完整 SIMD 路径覆盖通用 resize。
- 新增 `cvh_resize_opencv_intrin_smoke`，覆盖 16-lane 主循环和尾部像素。
- `benchmark/cvtcolor_bgr2gray_header_benchmark.cpp` 不受影响；resize 专用 benchmark 更新 scalar baseline，避免 fast-path 接入后污染 scalar direct 行。
- quick 结果写入 `benchmark/resize_bilinear_header_p363.csv`。
- full 结果写入 `benchmark/resize_bilinear_header_p364.csv`。

P3.6.3 本机 full 观察：

| shape | scalar ms | public OpenCV UI ms | direct OpenCV UI ms | public speedup | direct speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `640x480->320x240` | 0.123079 | 0.005750 | 0.005733 | 21.405078 | 21.467427 |
| `1280x720->640x360` | 0.294254 | 0.013792 | 0.013421 | 21.335755 | 21.925235 |
| `1920x1080->960x540` | 0.624621 | 0.029467 | 0.029771 | 21.197518 | 20.980991 |
| `3840x2160->1920x1080` | 2.521854 | 0.120996 | 0.122037 | 20.842476 | 20.664583 |

非 fast-path case 观察：

- `641x479->321x239`、`1919x1080->961x541`、`1920x1080->853x480`、`3839x2160->1917x1079`、`3840x2160->1280x720` 都标记为 `opencv_intrin_neon_no_resize_fastpath`。
- 这些 case 的 `public_resize` 与 scalar direct 基本等价，说明 fast-path 条件没有误覆盖非 2x resize。

阶段判断：

- OpenCV Universal Intrinsics 对 exact 2x `CV_8UC1` bilinear downsample 明确有效，远超 P4 gate 的 `>= 1.5x` 门槛。
- 该结论不能外推到任意比例 bilinear resize；通用 resize 仍是 gather/权重表问题，后续需要单独设计。
- public 入口与 direct detail 基本等价，当前 fast-path 接入点没有明显额外开销。

P3.6.4：正确性和性能 gate

状态：已完成。

- 正确性先对齐现有 scalar fallback，ROI/边界尺寸继续由 resize contract test 覆盖。
- ARM 上记录 scalar vs OpenCV Universal Intrinsics 的 `CV_8UC1` 结果。
- 若 `resize bilinear u8` 仍无法稳定达到 `>= 1.5x`，暂停 P4，转向 direct NEON header-only 或重新评估 OpenCV Universal Intrinsics adapter 的投入边界。

正确性 gate 补强结果：

- `cvh_resize_opencv_intrin_smoke` 扩展为 exact 2x `CV_8UC1 INTER_LINEAR` fast-path gate，同时验证公共 `cvh::resize` 入口和 direct detail 实现。
- 正例覆盖纯尾部、小尺寸、刚好一个 SIMD lane、多 lane、lane 后尾部，以及非连续 ROI 输入。
- 复用错误形状的既有 `dst`，验证 fast-path 会重新创建为正确的 2D 输出。
- 负例覆盖非 2x `CV_8UC1`、exact 2x `CV_8UC3`、exact 2x `INTER_NEAREST`，要求支持谓词不误命中，公共入口结果继续等于 scalar fallback。
- 支持谓词补充 `src.dims == 2`，与公共 resize contract 对齐，避免 detail gate 对非 2D Mat 表达错误支持。

性能 gate 补强结果：

- ARM 机器：`arm64 / Apple M5`。
- benchmark 参数：`--profile full --warmup 5 --iters 30 --repeats 9`。
- 结果写入：`benchmark/resize_bilinear_header_p364_arm_gate.csv`。

| shape | scalar ms | public OpenCV UI ms | direct OpenCV UI ms | public speedup | direct speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `640x480->320x240` | 0.173210 | 0.007254 | 0.006651 | 23.877275 | 26.041225 |
| `1280x720->640x360` | 0.280864 | 0.013954 | 0.014176 | 20.127601 | 19.812075 |
| `1920x1080->960x540` | 0.620665 | 0.030501 | 0.030526 | 20.348748 | 20.332083 |
| `3840x2160->1920x1080` | 2.476376 | 0.125667 | 0.125619 | 19.705913 | 19.713322 |

非 fast-path case 观察：

- `641x479->321x239`、`1919x1080->961x541`、`1920x1080->853x480`、`3839x2160->1917x1079`、`3840x2160->1280x720` 仍标记为 `opencv_intrin_neon_no_resize_fastpath`。
- 这些 case 的 public 入口继续走 scalar fallback；reuse 下 speedup 约 `0.91x` 到 `1.00x`，只说明未误覆盖 SIMD，不作为 OpenCV UI 收益样本。

阶段判断：

- exact 2x `CV_8UC1 INTER_LINEAR` 在 ARM 上稳定超过 `>=1.5x` gate，允许作为 P4 的首个受限 fast-path 候选。
- 该结论不外推到通用比例 resize；通用 resize 仍需要独立设计坐标/权重表与 gather/pack 路径。
- P4 若启动，应先只把已验证的 exact 2x `CV_8UC1` 路径纳入 `headers_fast`，不扩大到 C3/C4 或任意比例。

P3 总体验收：

- 正确性对齐 scalar：通过。
- ARM 上给出 scalar / opencv_intrin 的性能对比：通过。
- xsimd 只有在实现真实 `u8` vector path 后才进入该 kernel 的性能矩阵。
- 至少一个热点证明 OpenCV Universal Intrinsics 有引入价值：exact 2x `CV_8UC1 INTER_LINEAR` 通过。

### P4：headers_fast profile

状态：已完成，P4.0/P4.1/P4.2/P4.3/P4.4/P4.5 已完成。

P4 目标是在不改变默认 `cvh::headers` 行为的前提下，增加纯 header-only 加速 profile：

```cmake
cvh::headers_fast
```

项目对外确立公开 header-only 使用结构：

- `cvh::headers`：默认 header-only baseline，最小依赖、最少平台分支，不默认启用 SIMD 加速实验路径。
- `cvh::headers_fast`：header-only + 已验证 SIMD fast-path，只传播宏和 include，不编译或链接 `.cpp`。

说明：历史 `.cpp` 实现只作为 legacy/experimental 代码存在，不进入 `opencv-header-only` 的公开产品叙事。

P4 明确不让 `cvh::headers_fast` 启用 xsimd。xsimd 暂时只作为 legacy/experimental 手动开关存在；除非某个 kernel 有独立 benchmark 证明稳定收益，否则不进入 header-only fast profile。P5 将进一步推进隔离和移除。

`cvh::headers_fast` 首轮只传播：

```cmake
CVH_ENABLE_OPENCV_INTRIN=1
CVH_ENABLE_PLATFORM_INTRINSICS=1
```

首批纳入范围只限已通过 gate 的 header-only SIMD fast-path：

- `BGR2GRAY/RGB2GRAY` 的 OpenCV Universal Intrinsics 路径。
- exact 2x `CV_8UC1 INTER_LINEAR` resize OpenCV Universal Intrinsics 路径。

不纳入首轮 P4：

- xsimd adapter。
- 通用比例 resize。
- `resize` 的 C3/C4 路径。
- direct NEON / AVX 专项实现。

P4.0：封存 P3.6.4

状态：已完成。

- commit P3.6.4 的正确性 gate、ARM benchmark CSV 和文档结论。
- commit 后再开始 P4 代码修改，避免把 gate 结果和 profile 实现混在同一个变更里。

落地结果：

- P3.6.4 已提交：`11bd27a Complete resize intrinsics gate`。

验收：

- 工作区只剩 P4 相关修改，或干净后再进入 P4.1：通过。
- P3.6.4 commit 信息能明确说明 exact 2x `CV_8UC1 INTER_LINEAR` 已通过 gate。

P4.1：建立 `cvh::headers_fast` target

状态：已完成。

- 新增 `cvh_headers_fast` `INTERFACE` target 和 `cvh::headers_fast` alias。
- `cvh_headers_fast` 继承 `cvh::headers` 的 include 和 C++17 要求。
- `cvh_headers_fast` 只传播 `CVH_ENABLE_OPENCV_INTRIN=1`、`CVH_ENABLE_PLATFORM_INTRINSICS=1`。
- ARM 构建时传播 OpenCV UI 需要的 NEON 宏；x86 后续按 OpenCV UI adapter 的可用宏扩展。
- 不传播 `CVH_ENABLE_XSIMD=1`，不链接 `.cpp` 实验 target。

落地结果：

- `CMakeLists.txt` 新增 `cvh_headers_fast` `INTERFACE` target。
- build-tree alias 为 `cvh::headers_fast`。
- `EXPORT_NAME` 设置为 `headers_fast`，为 P4.4 的 install/export 验收预留稳定名称。
- build-tree 下额外传播 vendored OpenCV UI include root：`include/cvh/3rdparty/opencv_intrin`。
- install-tree include root 暂按 `$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cvh/3rdparty/opencv_intrin>` 写入 target；P4.4 继续验证外部项目解析。

验收：

- 用户可以用 `target_link_libraries(app PRIVATE cvh::headers_fast)` 打开已验证 header-only fast-path：build-tree target 已建立。
- `cvh::headers` 默认行为不变：未修改 `cvh_headers` target。
- `cvh::headers_fast` 不引入任何 `.cpp` 编译单元：通过，target 类型为 `INTERFACE`。

P4.2：增加 `headers_fast` smoke

状态：已完成。

- 新增或扩展 smoke，专门只链接 `cvh::headers_fast`。
- 验证 `CVH_ENABLE_OPENCV_INTRIN == 1`。
- 验证 `CVH_ENABLE_PLATFORM_INTRINSICS == 1`。
- 验证 `CVH_ENABLE_XSIMD == 0`。
- 验证不需要 `.cpp` 实验路径。

落地结果：

- 新增 `test/smoke/cvh_headers_fast_smoke.cpp`。
- 新增 CTest：`cvh_headers_fast_smoke`。
- smoke 只链接 `cvh::headers_fast`，不手写 `CVH_ENABLE_OPENCV_INTRIN`、`CVH_ENABLE_PLATFORM_INTRINSICS` 或 OpenCV UI include 路径。
- 编译期验证 `CVH_LITE`、`CVH_ENABLE_OPENCV_INTRIN == 1`、`CVH_ENABLE_PLATFORM_INTRINSICS == 1`、`CVH_ENABLE_XSIMD == 0`，并禁止 `CVH_NATIVE`。
- 运行期验证 `cvh::detail::simd::backend_name()` 为 `opencv_intrin`，并跑一个最小 exact 2x `CV_8UC1 INTER_LINEAR` resize。

验收：

- CTest 增加 `headers_fast` smoke：通过。
- smoke 在不构建 `.cpp` 实验 target 的配置下通过：通过。

P4.3：迁移现有 OpenCV UI smoke/benchmark 到 profile 入口

状态：已完成。

- 将适合代表用户入口的 OpenCV UI smoke 改为链接 `cvh::headers_fast`，减少手写宏。
- resize/cvtColor header-only benchmark 保留可观测 backend 字段，但优先通过 `cvh::headers_fast` 启用 OpenCV UI。
- 保留少量显式宏测试，用于验证 adapter 在独立宏开启时仍可编译。

落地结果：

- `cvh_cvtcolor_opencv_intrin_smoke` 改为只链接 `cvh::headers_fast`，不再手写 OpenCV UI 宏或 include 路径。
- `cvh_resize_opencv_intrin_smoke` 改为只链接 `cvh::headers_fast`，不再手写 OpenCV UI 宏或 include 路径。
- `cvh_benchmark_cvtcolor_bgr2gray_header` 改为链接 `cvh::headers_fast`，由 profile 传播 OpenCV UI 宏、平台 intrinsic 宏和 vendored include root。
- `cvh_benchmark_resize_bilinear_header` 改为链接 `cvh::headers_fast`，由 profile 传播 OpenCV UI 宏、平台 intrinsic 宏和 vendored include root。
- `cvh_opencv_intrin_smoke` 和 `cvh_simd_facade_opencv_intrin_smoke` 保留显式宏路径，用于继续验证 adapter 独立开启时可编译。
- benchmark CSV 的 public 入口名称改为 `public_headers_fast_cvtColor` 和 `public_headers_fast_resize`，继续与 `scalar/direct_detail` 和 OpenCV UI `direct_detail` 区分。

验收：

- `cvh_cvtcolor_opencv_intrin_smoke` 和 `cvh_resize_opencv_intrin_smoke` 至少有一个通过 `cvh::headers_fast` 路径覆盖：通过，两个 smoke 都已迁移。
- benchmark 输出仍能区分 scalar baseline、public `headers_fast` 入口、direct detail 入口：通过，entry 字段已显式标记 `public_headers_fast_*`。

P4.4：安装导出与用户文档

状态：已完成。

- 确认 install/export 后外部项目能引用 `cvh::headers_fast`。
- 更新用户文档，明确 `cvh::headers` / `cvh::headers_fast` 两个公开 header-only target 的使用方式。
- 明确 `headers_fast` 是 opt-in，不是默认 `headers`。

落地结果：

- `install(TARGETS ...)` 同时导出 `cvh_headers` 和 `cvh_headers_fast`。
- install/export 后外部项目可以通过 `find_package(opencv_header_only CONFIG REQUIRED)` 引用 `cvh::headers` 和 `cvh::headers_fast`。
- `README.md` 明确 `cvh::headers` 和 `cvh::headers_fast` 是两个公开 header-only target。
- `README.md` 明确 `cvh::headers_fast` 是 opt-in header-only fast profile，不启用 xsimd，不编译或链接 `.cpp`。
- 用户文档不再建议手动组合 `CVH_ENABLE_OPENCV_INTRIN`、`CVH_ENABLE_PLATFORM_INTRINSICS` 或 vendored OpenCV UI include 路径；推荐链接 `cvh::headers_fast`。

验收：

- build-tree 和 install-tree 都能解析 `cvh::headers_fast`：通过。
- 文档中示例不再建议用户手动组合 `CVH_ENABLE_OPENCV_INTRIN`、`CVH_ENABLE_PLATFORM_INTRINSICS`：通过。

P4.5：P4 验收矩阵

状态：已完成。

- 跑默认 `cvh::headers` smoke，确认 baseline 未变。
- 跑 `cvh::headers_fast` smoke，确认 profile 宏正确。
- 跑 OpenCV UI correctness smoke，确认 fast-path 正确。
- 跑 resize/cvtColor benchmark quick，确认 profile 入口没有性能退化。

落地结果：

- 构建配置：`CVH_BUILD_NATIVE_BACKEND=OFF`、`CVH_BUILD_TESTS=ON`、`CVH_BUILD_BENCHMARKS=ON`。
- `ctest --test-dir build-opencv-intrin-p3-bench --output-on-failure`：`16/16` 通过。
- `scripts/sync_opencv_intrin.py --check`：通过，当前 vendored OpenCV UI 对齐 `4.13.0-457-gd48bf69f65 d48bf69f65444a13f8a34b8982b083c1b78fa0e8`。
- 默认 `cvh::headers` smoke 的编译 flags 无 `CVH_ENABLE_OPENCV_INTRIN`、`CVH_ENABLE_PLATFORM_INTRINSICS`、`CVH_ENABLE_XSIMD` 或 `CVH_NATIVE`。
- `cvh::headers_fast` smoke 的编译 flags 只包含 `CVH_ENABLE_OPENCV_INTRIN=1`、`CVH_ENABLE_PLATFORM_INTRINSICS=1` 和 ARM 下的 `CV_NEON=1`；无 `CVH_ENABLE_XSIMD`，无 `CVH_NATIVE`。
- 新增 P4.5 quick benchmark CSV：`benchmark/cvtcolor_bgr_rgb_gray_header_p45.csv`。
- 新增 P4.5 quick benchmark CSV：`benchmark/resize_bilinear_header_p45.csv`。
- `public_headers_fast_cvtColor` 共 16 行，最小 `speedup_vs_scalar=1.175953`，无低于 scalar 的行。
- `public_headers_fast_resize` 共 8 行，最小 `speedup_vs_scalar=19.605797`，无低于 scalar 的行。
- `cvh_resize_opencv_intrin_smoke` 继续覆盖非 exact-2x、C3、`INTER_NEAREST`、tensor-like 输入等非 fast-path 条件的 public fallback，并通过。
- `benchmark/readme.md` 已同步为 `cvh::headers_fast` profile 语义，不再把 header benchmark 描述为手写宏或只链接 `cvh::headers`。

验收：

- `cvh::headers` 保持默认纯净：通过。
- `cvh::headers_fast` 仍然不构建 `.cpp` 实验 target：通过。
- `cvh::headers_fast` 不传播 `CVH_ENABLE_XSIMD=1`：通过。
- 不满足已验证 fast-path 条件的 API 继续回退到 scalar fallback：通过。

### P5：公开面收口与 xsimd 退场

P5 的目标不是继续扩 kernel 面，而是让项目定位、公开 CMake target、宏开关、测试和文档全部对齐到“纯 header-only OpenCV-style subset”。

P5 不应一次性硬删所有历史代码。先收口 public surface，再建立 gate，最后移除没有进入 accepted fast path 的 xsimd 路线。

P5.0：Public Surface Cleanup

状态：已完成。

- 将公开叙事固定为纯 header-only：`cvh::headers` 和 `cvh::headers_fast` 是唯一推荐入口。
- 清理 README 之外其它文档中的 native 主叙事；历史 `.cpp` 路径只能作为 legacy/experimental 代码描述。
- 修正 `highgui` header-only fallback 的报错文案，不再要求用户理解 native/backend 术语。
- 收口 package/export 叙事，用户安装后应只看到并使用 `cvh::headers` / `cvh::headers_fast`。
- 保留源码内必要兼容开关时，必须标为 internal/development-only，不能成为公开产品路径。

落地结果：

- `README.md` 已固定为 pure header-only 叙事，只推荐 `cvh::headers` 和 `cvh::headers_fast`。
- `CMakeLists.txt` 将历史 `.cpp` 构建选项标为 internal/development-only。
- 安装导出只包含 `cvh_headers` 和 `cvh_headers_fast`；历史 `.cpp` target 不再进入 installed package export。
- `cmake/opencv_header_onlyConfig.cmake.in` 移除 installed package 中的旧兼容 alias。
- `highgui` fallback 报错改为 pure header-only unsupported，不再引导用户理解 backend 术语。
- `doc/design.md` 已重写为当前 header-only 设计文档。
- 旧 compiled extension 迁移文档已改为历史归档，并明确被 P5 路线取代。

验收：

- 公开文档不再把 native/backend 作为项目层次结构的一部分。
- `README.md`、主计划文档和用户入口说明对 `cvh::headers` / `cvh::headers_fast` 的定义一致。
- `highgui` 在 header-only 下的错误信息表达为“not supported in pure header-only”，而不是引导到 `.cpp` 扩展路径。

验收结果：

- `rg` 检查 README、doc、cmake config、benchmark/test readme 和 highgui 后，公开文档层面不再保留 native/backend 作为推荐产品结构；旧 target 字面量仅保留在 `CMakeLists.txt` 的 build-tree legacy target 内。
- 默认 header-only 配置：`cmake -S . -B build-p50 -DCVH_BUILD_NATIVE_BACKEND=OFF -DCVH_BUILD_BACKEND_KERNEL_SOURCES=OFF -DCVH_BUILD_TESTS=ON -DCVH_BUILD_BENCHMARKS=OFF` 通过。
- P5.0 相关测试：`cvh_header_compile_smoke`、`cvh_include_only_smoke`、`cvh_headers_fast_smoke`、`cvh_test_highgui` 通过。
- `./scripts/ci_lite_all.sh` 通过，`cvh_test_core_lite` 25 tests 和 `cvh_test_imgproc` 141 tests 全部通过。
- 默认安装包和打开 legacy `.cpp` 构建开关后的安装包都只导出 `cvh::headers` 与 `cvh::headers_fast`。

P5.1：Header-only Contract Gate

状态：已完成。

- 增加或强化 gate：`cvh::headers` 和 `cvh::headers_fast` 不能依赖 `src/`。
- install/export 验证只围绕两个 header-only public targets。
- `cvh::headers_fast` 必须持续验证不传播 `CVH_ENABLE_XSIMD=1`。
- 将 CI/test 命名中残留的 `lite` 语义逐步收口到 `headers`，避免“lite vs full”误导项目定位。
- 对 README 中 `Supported` 的算子建立可追溯 header-only test 映射；没有 header-only 实现或链接不过的 API 必须保持 WIP。

落地结果：

- 新增 `scripts/check_header_only_contract.sh`。
  - 先运行 public header include 检查。
  - 构建 `cvh_header_compile_smoke`、`cvh_include_only_smoke`、`cvh_headers_fast_smoke`。
  - 安装 package 到临时目录，检查 installed CMake targets 只包含 `cvh::headers` / `cvh::headers_fast`。
  - 用外部最小 CMake 工程分别消费 `cvh::headers` 和 `cvh::headers_fast`。
  - `cvh::headers` consumer 验证默认不启用 OpenCV UI、platform intrinsics、xsimd 或 legacy `.cpp` 模式。
  - `cvh::headers_fast` consumer 验证启用 OpenCV UI/platform intrinsics，且不启用 xsimd 或 legacy `.cpp` 模式。
  - 即使 `CVH_BUILD_NATIVE_BACKEND=ON`，installed package 仍不能导出 legacy `.cpp` target。
- 新增 `scripts/ci_headers_all.sh` 作为新的 header-only CI 入口。
- `scripts/ci_lite_all.sh` 改为 deprecated compatibility wrapper，转发到 `scripts/ci_headers_all.sh`。
- `scripts/ci_headers_all.sh` 纳入 `cvh_test_core_lite`、`cvh_test_imgproc`、`cvh_test_imgcodecs`、`cvh_test_highgui`，让 README Supported / Out of scope 状态有 header-only 测试映射。
- `README.md` 新增 Header-only Contract Tests 映射表，并将开发命令改为 `./scripts/ci_headers_all.sh`。
- `test/smoke/readme.md` 增加 `cvh_headers_fast_smoke` 说明。

验收：

- 外部最小工程可以只通过 `find_package(...); target_link_libraries(app PRIVATE cvh::headers)` 编译通过。
- 外部最小工程可以只通过 `cvh::headers_fast` 编译并验证 OpenCV UI 宏状态。
- `CVH_ENABLE_XSIMD` 不会由任何公开 target 自动传播。

验收结果：

- `./scripts/check_header_only_contract.sh` 通过；默认 header-only install/export 和打开 legacy `.cpp` build option 后的 install/export 都只暴露 `cvh::headers` / `cvh::headers_fast`。
- 外部 `cvh::headers` consumer 验证 `CVH_ENABLE_OPENCV_INTRIN=0`、`CVH_ENABLE_PLATFORM_INTRINSICS=0`、`CVH_ENABLE_XSIMD=0`，且能执行最小 `cvh::resize`。
- 外部 `cvh::headers_fast` consumer 验证 `CVH_ENABLE_OPENCV_INTRIN=1`、`CVH_ENABLE_PLATFORM_INTRINSICS=1`、`CVH_ENABLE_XSIMD=0`，且能执行 exact 2x `cvh::resize`。
- `./scripts/ci_headers_all.sh` 通过；`cvh_test_core_lite` 25 tests、`cvh_test_imgproc` 141 tests、`cvh_test_imgcodecs` 7 tests + 1 skipped、`cvh_test_highgui` 4 tests。
- `./scripts/ci_lite_all.sh` 通过 deprecated wrapper 验证，会提示迁移到 `scripts/ci_headers_all.sh` 并转发执行同一套 header-only gate。

P5.2：xsimd Quarantine

状态：待开始。

- 从公开文档和推荐用法中移除 `CVH_ENABLE_XSIMD`。
- `simd.h` 不再把 xsimd 作为 accepted adapter 路线描述；如需保留，必须明确为 legacy/experimental。
- 删除或改名 `cvh_simd_facade_xsimd_smoke`，避免 CI 给出“xsimd 是官方支持加速后端”的信号。
- 整理 `doc/design.md`、`doc/mat-rollout-history.md`、transpose todo 和本计划文档中的 xsimd 旧判断。
- 历史 xsimd kernel 只能停留在非公开实验路径；不能影响 header-only public target、安装包和默认测试。

验收：

- 用户文档中不再推荐 `CVH_ENABLE_XSIMD`。
- header-only fast profile 只剩 OpenCV Universal Intrinsics + scalar fallback 两条 accepted 路径。
- xsimd 相关测试不再作为 header-only public contract 的必跑项。

P5.3：xsimd Removal

状态：待开始。

- 删除 `include/cvh/core/simd/xsimd_adapter.h`。
- 删除 `include/cvh/3rdparty/xsimd/` vendor 目录。
- 删除 CMake 中 xsimd include、宏、测试 target 和相关源码引用。
- 删除或迁移 `src/core/kernel/*_xsimd.cpp` 及依赖 `DispatchMode::XSimdOnly` / `DispatchTag::XSimd` 的历史路径。
- 删除 benchmark 参数中的 `xsimd-only`，或把对应 benchmark 移到归档文档。
- 更新 license/third-party 说明，确保不再列出已经移除的 xsimd vendor。

验收：

- `rg -n "xsimd|XSIMD|CVH_ENABLE_XSIMD|XSimd" CMakeLists.txt include src test benchmark README.md doc` 只允许出现在历史归档说明中，或完全无结果。
- `cvh::headers` 和 `cvh::headers_fast` 全量 smoke/test 通过。
- `cvh::headers_fast` 的 accepted fast path 仍覆盖 `BGR2GRAY/RGB2GRAY` 和 exact 2x `CV_8UC1 INTER_LINEAR resize`。

P5.4：OpenCV UI 常态维护

状态：待开始。

- 固定 OpenCV upstream 版本。
- 每次升级通过同步脚本。
- 每次升级必须跑正确性和性能矩阵。
- benchmark 退化超过阈值则不升级。

## 成功标准

短期成功：

- 可以在 header-only 模式下启用 OpenCV Universal Intrinsics adapter。
- 默认 `cvh::headers` 不受影响。
- 不引入 `.cpp` 编译层作为公开依赖。
- 至少一个 ARM 热点 kernel 明显优于 xsimd。

中期成功：

- `cvh` 形成自己的 `cvh::detail::simd` 方言。
- OpenCV Universal Intrinsics 和 scalar fallback 成为 accepted header-only SIMD/基线路径。
- xsimd 从公开路径隔离，并在 P5.3 后移除。
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

这条路线可以在不引入 `.cpp` 编译层的前提下，增强 header-only 层的 CPU SIMD 能力，并让 ARM 热点 kernel 优先走 OpenCV Universal Intrinsics 或必要的 direct platform intrinsics，而不是继续依赖 xsimd。
