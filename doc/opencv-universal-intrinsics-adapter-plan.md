# OpenCV Universal Intrinsics Adapter 引入计划

P5.3.5 更新：xsimd 已从 public adapter、legacy runtime、测试入口和 vendor 目录中移除。当前 accepted header-only fast profile 只包含 scalar fallback + OpenCV Universal Intrinsics；direct platform intrinsics 仍是 benchmark-gated 候选。下文早期阶段中的 xsimd 内容只作为历史上下文记录，不再代表公开推荐用法。

P5.4.1b 更新：`CVH_ENABLE_OPENCV_INTRIN` 改为默认开启，不再作为用户打开加速的选项。`cvh::headers` 默认使用 OpenCV Universal Intrinsics facade，同时继续保留 scalar fallback 作为 correctness/benchmark 对照；`cvh::headers_fast` 只负责额外平台 fast-profile toggles。

P6 更新：项目后续不再维护 `cvh::detail::simd` 二次 SIMD facade。OpenCV Universal Intrinsics 本身就是跨 NEON/SSE/AVX 的 adapter，P6 的目标是移除 `opencv_intrin_adapter.h`，让 header-only 业务 kernel 直接使用 `cv::v_*` / `cv::VTraits` / `vx_*` 这一套 OpenCV UI 写法，以降低迁移 OpenCV 原始 kernel 的成本。

平台范围设定：当前 SIMD 工作只处理 ARM NEON 和 x86 AVX 系列。RVV 支持放入后续 TODO，不进入 P6 的 vendor、实现、benchmark 或 accepted backend；SSE header/宏只作为 x86 OpenCV UI/AVX 编译链路的基础条件，不作为当前独立优化路线。

## 背景

`opencv-header-only` 的长期方向是纯 header-only。当前 header-only 层的 CPU 加速主要考虑：

- scalar fallback
- OpenCV Universal Intrinsics
- header-only `std::thread` 并行
- benchmark-gated direct platform intrinsics

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
- 将 xsimd 从公开 fast profile 和可执行实现路径移除；P5.3 后只允许在历史归档或旧宏 hard error 中出现。
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
- OpenCV Universal Intrinsics implementation
- direct NEON / AVX implementation

历史 xsimd adapter 已在 P5.3 移除，不作为 accepted header-only fast profile，也不作为内部 legacy/experimental 路径继续维护。

### 3. 默认纯净（历史判断，P5.4.1b 已更新）

早期判断中 Lite 默认保持最小依赖：

建议默认：

```cpp
CVH_ENABLE_OPENCV_INTRIN=0
CVH_ENABLE_PLATFORM_INTRINSICS=0
```

用户显式启用 header-only 加速 profile 时通过 `cvh::headers_fast` 打开已验证能力。

P5.4.1b 后当前策略改为：

```cpp
CVH_ENABLE_OPENCV_INTRIN=1
CVH_ENABLE_PLATFORM_INTRINSICS=0
```

OpenCV Universal Intrinsics 属于默认 header-only CPU SIMD facade；额外 direct platform toggles 仍由 `cvh::headers_fast` 控制。

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

OpenCV Universal Intrinsics 不应凭感觉进入 `cvh::headers_fast`。

每个 kernel 的采用顺序应由 benchmark 决定：

```text
scalar
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

accepted header-only fast profile 开关：

```cpp
CVH_ENABLE_OPENCV_INTRIN
CVH_ENABLE_PLATFORM_INTRINSICS
```

其它非 SIMD profile 能力开关仍可独立存在：

```cpp
CVH_ENABLE_THREADS
CVH_ENABLE_FAST_MATH
```

含义：

- `CVH_ENABLE_OPENCV_INTRIN=1`：允许使用 vendored OpenCV Universal Intrinsics adapter。
- `CVH_ENABLE_PLATFORM_INTRINSICS=1`：允许使用项目自写 NEON / AVX 等平台专项 header 实现。

xsimd 相关开关已移除，不再作为公开推荐用法；继续手动定义旧宏时应被 header-only SIMD facade 明确拒绝。

选择顺序建议：

```cpp
#if CVH_ENABLE_PLATFORM_INTRINSICS && CVH_CAN_USE_NEON
    direct_neon_impl(...);
#elif CVH_ENABLE_OPENCV_INTRIN && CVH_CAN_USE_OPENCV_INTRIN
    opencv_intrin_impl(...);
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
CVH_ENABLE_OPENCV_INTRIN=1
CVH_ENABLE_PLATFORM_INTRINSICS=1
```

legacy xsimd 只在隔离验证时单独打开，不进入默认配置矩阵。

### 正确性验证

每个试点 kernel 需要比较：

```text
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
speedup_vs_opencv
```

在 ARM 上重点关注：

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
- 历史冻结判断曾要求业务 kernel 只能通过 `cvh::detail::simd` facade 使用该 adapter；P6 已废弃该二次 facade 路线，改为 direct OpenCV UI internal dialect。
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
- P2 阶段曾保持 xsimd adapter 可并存；P5.2 后该 adapter 只作为 legacy/quarantined 路径保留。

落地结果：

- 新增 `cvh::detail::simd` facade 入口 `include/cvh/core/simd/simd.h`。
- 新增 scalar adapter：`include/cvh/core/simd/scalar_adapter.h`。
- 新增 xsimd adapter：`include/cvh/core/simd/xsimd_adapter.h`。
- 扩展 OpenCV Universal Intrinsics adapter：`include/cvh/core/simd/opencv_intrin_adapter.h`。
- 第一阶段 facade API 聚焦 `f32` 向量，覆盖 `load_f32`、`store_f32`、`setzero_f32`、`setall_f32`、`add`、`sub`、`mul`、`min`、`max`、`reduce_sum`、`f32_lanes`、`backend_name`。
- 新增同源 smoke `cvh_simd_facade_smoke.cpp`，最初分别编译 scalar / xsimd / opencv_intrin 三个目标；P5.2 后 xsimd 目标已改为 legacy 显式选项。

边界：

- P2 不直接迁移 imgproc kernel。
- P2 不暴露 `cv::v_*` 给业务 kernel。
- `u8`、deinterleave、pack、widen/narrow 等图像专用 API 留到 P3 试点 kernel 根据真实需求扩展。

验收：

- 同一份测试能跑 scalar / xsimd / opencv_intrin。
- kernel 不直接使用 `cv::v_*`。
- `cvh_simd_facade_scalar_smoke` 通过。
- 历史 `cvh_simd_facade_xsimd_smoke` 当时通过；P5.2 后改名为 `cvh_legacy_xsimd_facade_smoke` 并默认不构建。
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
- 同源 `cvh_simd_facade_smoke.cpp` 已扩展到 `u8` facade，覆盖 scalar / opencv_intrin 的接口编译和基础正确性；xsimd adapter 覆盖在 P5.2 后只通过 legacy 显式选项保留。

验证：

- `cvh_simd_facade_scalar_smoke` 通过。
- 历史 `cvh_simd_facade_xsimd_smoke` 当时通过；P5.2 后改名为 `cvh_legacy_xsimd_facade_smoke` 并默认不构建。
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

P4 当时的目标是在不改变默认 `cvh::headers` 行为的前提下，增加纯 header-only 加速 profile；P5.4.1b 已将 OpenCV Universal Intrinsics 改为默认开启，因此本节保留为历史记录：

```cmake
cvh::headers_fast
```

项目对外确立公开 header-only 使用结构：

- `cvh::headers`：P5.4.1b 后默认启用 OpenCV Universal Intrinsics facade，保留 scalar fallback 作为 correctness/benchmark 对照。
- `cvh::headers_fast`：P5.4.1b 后只传播额外平台 fast-profile toggles，不编译或链接 `.cpp`。

说明：历史 `.cpp` 实现只作为 legacy/experimental 代码存在，不进入 `opencv-header-only` 的公开产品叙事。

P4 明确不让 `cvh::headers_fast` 启用 xsimd；P5.3 已进一步完成 public adapter、legacy runtime、测试入口和 vendor 目录移除。xsimd 不再作为 header-only fast profile 或 legacy `.cpp` 实验路径维护。

P4 首轮 `cvh::headers_fast` 曾传播：

```cmake
CVH_ENABLE_OPENCV_INTRIN=1
CVH_ENABLE_PLATFORM_INTRINSICS=1
```

P5.4.1b 后 `CVH_ENABLE_OPENCV_INTRIN=1` 成为 `cvh::headers` 默认值，`cvh::headers_fast` 不再需要传播该宏。

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
  - `cvh::headers` consumer 验证默认启用 OpenCV UI，不启用 platform intrinsics、xsimd 或 legacy `.cpp` 模式。
  - `cvh::headers_fast` consumer 验证继承默认 OpenCV UI、启用 platform intrinsics，且不启用 xsimd 或 legacy `.cpp` 模式。
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
- 外部 `cvh::headers` consumer 验证 `CVH_ENABLE_OPENCV_INTRIN=1`、`CVH_ENABLE_PLATFORM_INTRINSICS=0`、`CVH_ENABLE_XSIMD=0`，且能执行最小 `cvh::resize`。
- 外部 `cvh::headers_fast` consumer 验证 `CVH_ENABLE_OPENCV_INTRIN=1`、`CVH_ENABLE_PLATFORM_INTRINSICS=1`、`CVH_ENABLE_XSIMD=0`，且能执行 exact 2x `cvh::resize`。
- `./scripts/ci_headers_all.sh` 通过；`cvh_test_core_lite` 25 tests、`cvh_test_imgproc` 141 tests、`cvh_test_imgcodecs` 7 tests + 1 skipped、`cvh_test_highgui` 4 tests。
- `./scripts/ci_lite_all.sh` 通过 deprecated wrapper 验证，会提示迁移到 `scripts/ci_headers_all.sh` 并转发执行同一套 header-only gate。

P5.2：xsimd Quarantine

状态：已完成。

- 从公开文档和推荐用法中移除 `CVH_ENABLE_XSIMD`。
- `simd.h` 不再把 xsimd 作为 accepted adapter 路线描述；如需保留，必须明确为 legacy/experimental。
- 删除或改名 `cvh_simd_facade_xsimd_smoke`，避免 CI 给出“xsimd 是官方支持加速后端”的信号。
- 整理 `doc/design.md`、`doc/mat-rollout-history.md`、transpose todo 和本计划文档中的 xsimd 旧判断。
- 历史 xsimd kernel 只能停留在非公开实验路径；不能影响 header-only public target、安装包和默认测试。

落地结果：

- `simd.h` 不再仅凭 `CVH_ENABLE_XSIMD=1` 接受 xsimd adapter；内部验证必须同时显式打开 legacy opt-in。
- `xsimd_adapter.h` 直接 include 也要求 legacy opt-in，避免绕过 `simd.h` 重新变成公开 adapter。
- 新增 `CVH_ENABLE_LEGACY_XSIMD` 默认值，`cvh::headers` / `cvh::headers_fast` contract gate 明确验证该宏不传播。
- `scripts/check_header_only_contract.sh` 新增负向 consumer，验证用户只定义 `CVH_ENABLE_XSIMD=1` 会被编译期错误拦截。
- `cvh_simd_facade_xsimd_smoke` 改为 `cvh_legacy_xsimd_facade_smoke`，且只在 `CVH_BUILD_LEGACY_XSIMD_TESTS=ON` 时构建和注册 CTest。
- `README.md`、`doc/design.md`、`include/cvh/core/readme.md`、`doc/mat-rollout-history.md`、transpose handoff 和本计划文档都已明确 xsimd 不是 accepted header-only fast profile。

验收：

- 用户文档中不再推荐 `CVH_ENABLE_XSIMD`。
- header-only fast profile 只剩 OpenCV Universal Intrinsics + scalar fallback 两条 accepted 路径。
- xsimd 相关测试不再作为 header-only public contract 的必跑项。

验收结果：

- `./scripts/check_header_only_contract.sh` 通过；其中负向 consumer 验证 `CVH_ENABLE_XSIMD=1` 缺少 legacy opt-in 时会被编译期错误拦截。
- `./scripts/ci_headers_all.sh` 通过，cache 中 `CVH_BUILD_LEGACY_XSIMD_TESTS=OFF`；默认 header-only CI 不构建或运行 xsimd smoke。
- 显式 legacy 验证通过：`cmake -S . -B build-p52-legacy-xsimd -DCVH_BUILD_NATIVE_BACKEND=OFF -DCVH_BUILD_BACKEND_KERNEL_SOURCES=OFF -DCVH_BUILD_TESTS=ON -DCVH_BUILD_BENCHMARKS=OFF -DCVH_BUILD_LEGACY_XSIMD_TESTS=ON`，随后 `cvh_legacy_xsimd_facade_smoke` build + CTest 通过。

P5.3：xsimd Removal

状态：已完成。

P5.3 不一次性硬删。先通过依赖扫描确认 xsimd 暴露面，再按 public header-only、CMake/test、legacy `.cpp`、benchmark/vendor 的顺序移除，避免误伤仍需要保留的 scalar/OpenCV UI 路径。

P5.3.0：依赖扫描

状态：已完成。

- 扫描 `include/`、`src/`、`test/`、`benchmark/`、`CMakeLists.txt`、`scripts/`、`README.md`、`doc/` 中所有 `xsimd`、`XSIMD`、`CVH_ENABLE_XSIMD`、`XSimd`。
- 按 public header-only、legacy `.cpp`、test/gate、benchmark、文档、vendor 分组。
- 明确每组在 P5.3.1-P5.3.5 的处理顺序。

扫描结果：

- 非 vendor 扫描命令：`rg --count-matches "xsimd|XSIMD|CVH_ENABLE_XSIMD|XSimd" CMakeLists.txt include src test benchmark README.md doc scripts -g '!include/cvh/3rdparty/xsimd/**'`。
- 非 vendor 命中：33 个文件，916 处命中。
- 目录分布：`include/` 7 个文件，`src/` 12 个文件，`test/` 4 个文件，`benchmark/` 1 个文件，`doc/` 5 个文件，`scripts/` 2 个文件，另有 `CMakeLists.txt` 和 `README.md`。
- vendor 目录：`include/cvh/3rdparty/xsimd/` 共 96 个文件，约 1.8M。
- root `LICENSE` 未发现 xsimd 字面量；第三方说明主要由 vendor 自带 `include/cvh/3rdparty/xsimd/LICENSE` / `README.md` 承载，P5.3.4 删除 vendor 时一并移除。

分组清单：

| 分组 | 文件 | 处理阶段 |
|---|---|---|
| public SIMD adapter / config | `include/cvh/core/simd/simd.h`、`include/cvh/core/simd/xsimd_adapter.h`、`include/cvh/detail/config.h` | P5.3.1 |
| public detail dispatch / helper 残留 | `include/cvh/core/detail/dispatch_control.h`、`include/cvh/core/detail/xsimd_kernel_utils.h`、`include/cvh/core/define.h` | P5.3.3 |
| CMake / smoke / contract gate | `CMakeLists.txt`、`scripts/check_header_only_contract.sh`、`scripts/ci_headers_all.sh`、`test/smoke/cvh_headers_fast_smoke.cpp`、`test/smoke/readme.md` | P5.3.2 |
| legacy `.cpp` core kernels | `src/core/kernel/binary_kernel_xsimd.*`、`src/core/kernel/gemm_kernel_xsimd.*`、`src/core/kernel/normalization_kernel_xsimd.*`、`src/core/kernel/transpose_kernel.cpp` | P5.3.3 |
| legacy `.cpp` callers | `src/core/basic_op.cpp`、`src/core/basic_op_scalar.cpp`、`src/core/mat_gemm.cpp`、`src/core/readme.md`、`src/core/kernel/readme.md` | P5.3.3 |
| xsimd-only tests | `test/core/binary_op_contract_test.cpp`、`test/core/mat_contract_test.cpp` | P5.3.3 |
| benchmark dispatch option | `benchmark/core_ops_benchmark.cpp` | P5.3.3 |
| user / design docs | `README.md`、`doc/design.md`、`doc/native-backend-migration-plan.md`、`doc/mat-rollout-history.md`、`doc/transpose-x86-todo-2026-04-19.md`、本计划文档 | P5.3.5 |
| vendor | `include/cvh/3rdparty/xsimd/` | P5.3.4 |

P5.3.0 额外发现：

- `src/core/kernel/normalization_kernel_xsimd.cpp` 存在，但当前 `CMakeLists.txt` 的 `CVH_NATIVE_BACKEND_SOURCES` 未纳入该 `.cpp`；同时 `src/core/basic_op.cpp` include 了 `normalization_kernel_xsimd.h` 并调用相关函数。
- `src/core/basic_op.cpp` 引用 `silu_kernel_xsimd`，扫描未找到对应定义；P5.3.3 处理 legacy `.cpp` 路径时应优先删除或降级这些旧 AI-kernel 调用，而不是继续维护 xsimd 实现。
- `DispatchMode::XSimdOnly` / `DispatchTag::XSimd` 同时影响 benchmark、legacy tests 和 legacy `.cpp` dispatcher；需要在 P5.3.3 与 P5.3.4 之间连续处理，避免留下不可达枚举或失效 CLI 参数。

P5.3.1：删除 public header-only xsimd adapter

状态：已完成。

- 删除 `include/cvh/core/simd/xsimd_adapter.h`。
- 删除 `include/cvh/detail/config.h` 中 `CVH_ENABLE_XSIMD` / `CVH_ENABLE_LEGACY_XSIMD` 的公开默认入口。
- `include/cvh/core/simd/simd.h` 只保留 scalar fallback 和 OpenCV Universal Intrinsics 选择。

落地结果：

- 已删除 `include/cvh/core/simd/xsimd_adapter.h`。
- `include/cvh/detail/config.h` 不再为 `CVH_ENABLE_XSIMD` / `CVH_ENABLE_LEGACY_XSIMD` 提供默认值。
- `include/cvh/core/simd/simd.h` 不再 include 或 using xsimd adapter；只在 `CVH_ENABLE_OPENCV_INTRIN=1` 时选择 OpenCV UI，否则选择 scalar。
- `simd.h` 对外部继续手动定义 `CVH_ENABLE_XSIMD` 或 `CVH_ENABLE_LEGACY_XSIMD` 的情况保留明确编译期错误，避免旧宏静默失效。
- `scripts/check_header_only_contract.sh` 的负向 consumer 已同步为“旧 xsimd 宏已移除”错误文本；P5.3.2 会继续把 CMake/test 中的 legacy xsimd smoke 和负向 gate 清成无残留检查。

验收结果：

- `rg -n "xsimd|XSIMD|CVH_ENABLE_XSIMD|CVH_ENABLE_LEGACY_XSIMD|XSimd" include/cvh/core/simd include/cvh/detail/config.h -g '!include/cvh/3rdparty/xsimd/**'` 只剩 `simd.h` 中旧宏的 compile-time hard error。
- `./scripts/check_header_only_contract.sh` 通过。
- `./scripts/ci_headers_all.sh` 通过；默认 header-only CI 仍为 `CVH_BUILD_LEGACY_XSIMD_TESTS=OFF`。

P5.3.2：删除 CMake/test 中 xsimd smoke

状态：已完成。

- 删除 `CVH_BUILD_LEGACY_XSIMD_TESTS`。
- 删除 `cvh_legacy_xsimd_facade_smoke` target 和 CTest 注册。
- 更新 `scripts/check_header_only_contract.sh`，从“xsimd 负向 gate”转为“公开 target/package 无 xsimd 宏、target、include 残留”。

落地结果：

- `CMakeLists.txt` 已删除 `CVH_BUILD_LEGACY_XSIMD_TESTS` option、`cvh_legacy_xsimd_facade_smoke` target 和对应 CTest 注册。
- `scripts/ci_headers_all.sh` 不再展示 legacy xsimd cache key，并通过 `-U 'CVH_BUILD_LEGACY_*'` 清理复用 build dir 中的旧 legacy cache。
- `scripts/check_header_only_contract.sh` 删除旧 xsimd 负向 consumer，改为检查 installed CMake package surface 不含 `xsimd`、`XSIMD`、`CVH_ENABLE_XSIMD`、`CVH_ENABLE_LEGACY_XSIMD` 或 `XSimd`。
- `test/smoke/cvh_headers_fast_smoke.cpp` 不再检查已移除的 xsimd 宏；`simd.h` 本身仍会对旧宏做 compile-time hard error。
- `test/smoke/readme.md` 删除 legacy xsimd smoke 说明。

验收结果：

- `./scripts/check_header_only_contract.sh` 通过。
- `./scripts/ci_headers_all.sh` 通过；cache 输出不再包含 `CVH_BUILD_LEGACY_XSIMD_TESTS`。
- `build-ci-headers-all/CMakeCache.txt` 不含 `CVH_BUILD_LEGACY_XSIMD_TESTS`。
- `ctest -N --test-dir build-ci-headers-all` 和 `cmake --build build-ci-headers-all --target help` 不含 `legacy_xsimd` / `xsimd_facade`。
- 即使显式传入已删除的 `-DCVH_BUILD_LEGACY_XSIMD_TESTS=ON`，`build-p532-old-xsimd-opt` 也不会生成 legacy xsimd target 或 CTest。

P5.3.3：处理 legacy `.cpp` xsimd kernel

状态：已完成。

- 删除或迁移 `src/core/kernel/*_xsimd.*`。
- 处理 `src/core/basic_op.cpp`、`src/core/basic_op_scalar.cpp`、`src/core/mat_gemm.cpp`、`src/core/kernel/transpose_kernel.cpp` 对 xsimd kernel、`DispatchMode::XSimdOnly`、`DispatchTag::XSimd` 的依赖。
- 如果 legacy `.cpp` build 仍保留最小可编译性，则把相关路径降级到 scalar fallback；否则同步从 legacy source list 移除对应能力。

落地结果：

- `CMakeLists.txt` 已从 legacy native source list 移除 `binary_kernel_xsimd.cpp`、`gemm_kernel_xsimd.cpp`，并移除 native target 对 xsimd vendor include dir 的依赖。
- 删除 `src/core/kernel/binary_kernel_xsimd.*`、`src/core/kernel/gemm_kernel_xsimd.*`、`src/core/kernel/normalization_kernel_xsimd.*`、`include/cvh/core/detail/xsimd_kernel_utils.h`。
- 删除未进入当前 CMake source list、且仍引用 normalization/xsimd kernel 的历史 `src/core/basic_op.cpp`。
- `src/core/basic_op_scalar.cpp` 不再尝试 xsimd fast dispatch；mat-mat、mat-scalar、compare 路径统一走 scalar fallback，并继续设置 `DispatchTag::Scalar`。
- `src/core/mat_gemm.cpp` 保留 `gemm_pack_b` / packed GEMM API，但 packed B 改为 contiguous copy，NN/NT GEMM 使用 scalar multiply-accumulate fallback。
- `src/core/kernel/transpose_kernel.cpp` 删除 xsimd transpose、probe cache 和 `CVH_TRANSPOSE_XSIMD_PROBE_LOG` 路径，保留 tiled/memcpy scalar fallback。
- `include/cvh/core/detail/dispatch_control.h` 删除 `DispatchMode::XSimdOnly` 和 `DispatchTag::XSimd`。
- `test/core/binary_op_contract_test.cpp`、`test/core/mat_contract_test.cpp` 的 xsimd-only 正确性用例改为 scalar-only 正确性用例。
- 因 `XSimdOnly` enum 已删除，`benchmark/core_ops_benchmark.cpp` 同步移除 `--dispatch xsimd-only`，P5.3.4 不再需要单独处理该参数。
- `README.md`、`doc/design.md`、`include/cvh/core/readme.md`、`include/cvh/core/define.h`、`src/core/readme.md`、`src/core/kernel/readme.md` 已同步当前代码面的 xsimd 状态。

验证结果：

- `./scripts/check_header_only_contract.sh` 通过。
- `./scripts/ci_headers_all.sh` 通过。
- `cmake -S . -B build-p533-native -DCVH_BUILD_NATIVE_BACKEND=ON -DCVH_BUILD_TESTS=ON -DCVH_BUILD_BENCHMARKS=ON` 通过。
- `cmake --build build-p533-native --target cvh_native_backend cvh_test_core_native cvh_benchmark_core_ops -j` 通过。
- `ctest --test-dir build-p533-native --output-on-failure -R 'cvh_test_core_native|cvh_test_core$'` 通过。
- `cvh_benchmark_core_ops --help` 不再输出 `xsimd`；`--dispatch xsimd-only` 会按预期报错 `expected auto/scalar-only`。
- `rg -n "DispatchMode::XSimdOnly|DispatchTag::XSimd|xsimd-only|CVH_TRANSPOSE_XSIMD_PROBE_LOG|binary_kernel_xsimd|gemm_kernel_xsimd|normalization_kernel_xsimd|xsimd_kernel_utils" CMakeLists.txt include src test benchmark scripts README.md -g '!include/cvh/3rdparty/xsimd/**'` 无结果。

P5.3.4：删除 vendor 和许可说明

状态：已完成。

- 删除 `include/cvh/3rdparty/xsimd/` vendor 目录。
- 更新 third-party/license 说明，确保不再列出已经移除的 xsimd vendor。

落地结果：

- 已删除 `include/cvh/3rdparty/xsimd/` vendor 目录；P5.3.0 扫描时该目录共 96 个文件。
- root `LICENSE` 未包含 xsimd 条目；原 xsimd 许可说明只随 vendor 自带 `LICENSE` / `README.md` 存在，随目录删除一并移除。
- `README.md`、`doc/design.md` 和历史 compiled-extension 归档文档已同步为 P5.3 已移除 xsimd vendor 的状态。
- OpenCV Universal Intrinsics 的第三方来源和许可仍由 `include/cvh/3rdparty/opencv_intrin/UPSTREAM.md` 与 `LICENSE.opencv` 记录。

验证结果：

- `test ! -e include/cvh/3rdparty/xsimd` 通过。
- `rg --files include src test benchmark | rg "xsimd|XSimd"` 无结果。
- 源码和构建面扫描只剩 deliberate old-macro hard error、contract/package gate，以及 README/文档中的历史说明。
- `./scripts/check_header_only_contract.sh` 通过。
- `./scripts/ci_headers_all.sh` 通过。
- `cmake -S . -B build-p534-native -DCVH_BUILD_NATIVE_BACKEND=ON -DCVH_BUILD_TESTS=ON -DCVH_BUILD_BENCHMARKS=ON` 通过。
- `cmake --build build-p534-native --target cvh_native_backend cvh_test_core_native cvh_benchmark_core_ops -j` 通过。
- `ctest --test-dir build-p534-native --output-on-failure -R 'cvh_test_core_native|cvh_test_core$'` 通过。
- `cvh_benchmark_core_ops --help` 不再输出 `xsimd`，dispatch 参数只剩 `auto|scalar-only`。
- `git diff --check` 通过。

P5.3.5：验证

状态：已完成。

- `rg -n "xsimd|XSIMD|CVH_ENABLE_XSIMD|XSimd" CMakeLists.txt include src test benchmark README.md doc scripts` 只允许以下残留：
  - `include/cvh/core/simd/simd.h` 中对旧宏的 compile-time hard error。
  - `scripts/check_header_only_contract.sh` 中对 installed package 的负向扫描。
  - `README.md` / `doc/design.md` 中“xsimd 不属于 accepted fast path”的当前策略说明。
  - `doc/` 中明确标记为历史归档或历史阶段记录的上下文。
- `./scripts/check_header_only_contract.sh` 通过。
- `./scripts/ci_headers_all.sh` 通过。
- 如 legacy `.cpp` build 仍声明可构建，补跑 `CVH_BUILD_NATIVE_BACKEND=ON` 的最小配置验证。

落地结果：

- `doc/mat-rollout-history.md` 更新为 P5.3.5 历史归档口径，明确 xsimd public adapter、legacy runtime、测试入口和 vendor 目录已在 P5.3 移除。
- `doc/transpose-x86-todo-2026-04-19.md` 从可执行 handoff 改为历史归档，明确旧 xsimd transpose/probe 路径已删除，不再作为后续 TODO。
- P5.3.5 明确了允许残留清单：旧宏 hard error、installed package gate、README/design 当前策略说明和历史归档上下文。

验证结果：

- `test ! -e include/cvh/3rdparty/xsimd` 通过。
- `rg --files include src test benchmark | rg "xsimd|XSimd"` 无结果。
- `rg -n "DispatchMode::XSimdOnly|DispatchTag::XSimd|xsimd-only|CVH_TRANSPOSE_XSIMD_PROBE_LOG|binary_kernel_xsimd|gemm_kernel_xsimd|normalization_kernel_xsimd|xsimd_kernel_utils|CVH_BUILD_LEGACY_XSIMD_TESTS|cvh_legacy_xsimd_facade_smoke|cvh_simd_facade_xsimd_smoke" CMakeLists.txt include src test benchmark scripts README.md` 无结果。
- `rg -n "xsimd|XSimd|XSIMD|CVH_ENABLE_XSIMD|CVH_ENABLE_LEGACY_XSIMD" CMakeLists.txt include src test benchmark README.md scripts -g '!doc/**'` 只剩 README 策略说明、`simd.h` 旧宏 hard error、`scripts/check_header_only_contract.sh` installed package gate，以及 `include/cvh/core/readme.md` / `src/core/readme.md` / `src/core/kernel/readme.md` 历史说明。
- `./scripts/check_header_only_contract.sh` 通过。
- `./scripts/ci_headers_all.sh` 通过。
- `cmake -S . -B build-p535-native -DCVH_BUILD_NATIVE_BACKEND=ON -DCVH_BUILD_TESTS=ON -DCVH_BUILD_BENCHMARKS=ON` 通过。
- `cmake --build build-p535-native --target cvh_native_backend cvh_test_core_native cvh_benchmark_core_ops -j` 通过。
- `ctest --test-dir build-p535-native --output-on-failure -R 'cvh_test_core_native|cvh_test_core$'` 通过。
- `cmake --build build-p535-native --target cvh_headers_fast_smoke cvh_cvtcolor_opencv_intrin_smoke cvh_resize_opencv_intrin_smoke -j` 通过。
- `ctest --test-dir build-p535-native --output-on-failure -R 'cvh_headers_fast_smoke|cvh_cvtcolor_opencv_intrin_smoke|cvh_resize_opencv_intrin_smoke'` 通过。
- `cvh_benchmark_core_ops --help` 不再输出 `xsimd`，dispatch 参数只剩 `auto|scalar-only`。
- `git diff --check` 通过。

验收：

- `include/`、`src/`、`test/`、`benchmark/` 中不再有 xsimd 文件名。
- 源码残留只允许 old-macro hard error；构建脚本残留只允许 installed package gate。
- 文档残留必须是当前策略说明或历史归档，不得继续描述 xsimd 待办或可执行路线。
- `cvh::headers` 和 `cvh::headers_fast` 全量 smoke/test 通过。
- `cvh::headers_fast` 的 accepted fast path 仍覆盖 `BGR2GRAY/RGB2GRAY` 和 exact 2x `CV_8UC1 INTER_LINEAR resize`。

P5.4：OpenCV UI 常态维护

状态：进行中，P5.4.1b 已完成；下一步 P5.4.2。

- 固定 OpenCV upstream 版本。
- 每次升级通过同步脚本。
- 每次升级必须跑正确性和性能矩阵。
- benchmark 退化超过阈值则不升级。

P5.4 不继续扩 kernel 面，目标是把 OpenCV Universal Intrinsics vendor 树变成可长期升级、可审计、可拒绝退化的维护对象。

P5.4.0：审计当前 OpenCV UI vendor 状态

状态：已完成。

- 当前 OpenCV 源树：`/Users/zmu/work/my_project/ocvh/opencv`。
- 当前 OpenCV describe：`4.13.0-457-gd48bf69f65`。
- 当前 OpenCV commit：`d48bf69f65444a13f8a34b8982b083c1b78fa0e8`。
- 当前 OpenCV version header：`4.14.0-pre`。
- 当前 vendor 文件仅包含 OpenCV UI 白名单、`LICENSE.opencv`、`UPSTREAM.md` 和本地 shim。
- `scripts/sync_opencv_intrin.py --check` 在 P5.4.0 起作为 OpenCV UI vendor 维护 gate。

P5.4.1：强化 sync 脚本和 UPSTREAM 元数据 gate

状态：已完成。

- `scripts/sync_opencv_intrin.py --check` 必须同时验证：
  - vendored OpenCV whitelist 文件与本地 OpenCV 源树一致。
  - `include/cvh/3rdparty/opencv_intrin/UPSTREAM.md` 中的 repository、describe、commit、version header 与本地 OpenCV 源树一致。
  - `include/cvh/3rdparty/opencv_intrin/` 中不存在白名单之外的 OpenCV 文件。
  - 本地 shim 全部存在，且不被 sync 脚本从 OpenCV 源树覆盖。
- 非 `--check` 模式同步 whitelist 文件，并重新生成 `UPSTREAM.md`。
- `UPSTREAM.md` 明确当前 accepted fast path 只包含 `BGR2GRAY/RGB2GRAY` 和 exact 2x `CV_8UC1 INTER_LINEAR resize`。

落地结果：

- `scripts/sync_opencv_intrin.py` 新增 OpenCV version header 解析，并用本地 OpenCV 源树生成 `UPSTREAM.md`。
- `scripts/sync_opencv_intrin.py --check` 现在同时检查：
  - whitelist 文件内容是否与本地 OpenCV 源树一致。
  - `UPSTREAM.md` 是否与当前 repository / describe / commit / OpenCV version header 一致。
  - `include/cvh/3rdparty/opencv_intrin/` 是否存在 whitelist 外的意外文件。
  - `opencv2/core/cvdef.h`、`opencv2/core/saturate.hpp`、`opencv2/core/utility.hpp` 三个本地 shim 是否存在。
- `include/cvh/3rdparty/opencv_intrin/UPSTREAM.md` 已由同步脚本重新生成，删除早期 P1/P3.2 临时叙事，改为维护合同和当前 accepted fast path。

验证结果：

- `scripts/sync_opencv_intrin.py` 通过，重新生成 whitelist vendor 文件和 `UPSTREAM.md`。
- `scripts/sync_opencv_intrin.py --check` 通过，当前锁定 `4.13.0-457-gd48bf69f65 d48bf69f65444a13f8a34b8982b083c1b78fa0e8`。
- `python3 -m py_compile scripts/sync_opencv_intrin.py` 通过。
- `./scripts/check_header_only_contract.sh` 通过。
- `./scripts/ci_headers_all.sh` 通过。
- `cmake -S . -B build-p54-opencv-intrin -DCVH_BUILD_NATIVE_BACKEND=OFF -DCVH_BUILD_TESTS=ON -DCVH_BUILD_BENCHMARKS=ON` 通过。
- `cmake --build build-p54-opencv-intrin --target cvh_headers_fast_smoke cvh_opencv_intrin_smoke cvh_simd_facade_opencv_intrin_smoke cvh_cvtcolor_opencv_intrin_smoke cvh_resize_opencv_intrin_smoke cvh_benchmark_cvtcolor_bgr2gray_header cvh_benchmark_resize_bilinear_header -j` 通过。
- `ctest --test-dir build-p54-opencv-intrin --output-on-failure -R 'cvh_headers_fast_smoke|cvh_opencv_intrin_smoke|cvh_simd_facade_opencv_intrin_smoke|cvh_cvtcolor_opencv_intrin_smoke|cvh_resize_opencv_intrin_smoke'` 通过。
- `git diff --check` 通过。

P5.4.1a：平台 UI header 白名单扩展（NEON/AVX，SSE 作为 x86 依赖）

状态：已完成。

目标：

- 让 `include/cvh/3rdparty/opencv_intrin/` 覆盖 portable fallback、NEON 和 OpenCV UI 的 x86 header。
- x86 AVX 系列可以作为 `cvh::headers_fast` 在 x86 上的实际候选 backend；SSE header/宏只作为 OpenCV UI/AVX 编译链路的基础条件。
- RVV 暂不进入 vendor 和 accepted backend；当前阶段只关注 NEON 与 AVX。

落地策略：

- 新增同步白名单：
  - `intrin_sse_em.hpp`
  - `intrin_sse.hpp`
  - `intrin_avx.hpp`
  - `intrin_avx512.hpp`
- `opencv2/core/cvdef.h` 按编译器 feature macro 自动打开 `CV_SSE2`、`CV_AVX2`、`CV_AVX512_SKX` 和 `CV_NEON`。
- `CV_RVV` 和 `CV_RVV071` 默认保持 `0`，且手动启用会 hard error。原因是当前 `cvh::detail::simd` facade 使用固定 lane 类型（如 `v_uint8x16` / `v_uint16x8`），不能表达 RVV scalable lane 语义；RVV 需要后续单独设计 scalable facade 后再评估。
- 新增 x86-only smoke：`cvh_opencv_intrin_x86_smoke`，只在 x86 CMake processor 上构建，并用 `-mavx2` / `/arch:AVX2` 验证 AVX2 UI header 能编译和跑通；SSE2 是该 x86 编译链路的基础条件。

验收：

- `scripts/sync_opencv_intrin.py --check` 必须接受新增 x86 白名单，并拒绝 RVV header 或其它白名单外文件。
- ARM 当前机器不会构建 x86 smoke；x86 CI 或 x86 本机应构建并运行 `cvh_opencv_intrin_x86_smoke`。
- RVV 在 scalable facade 没完成前不进入 vendor、不进入 `cvh::headers_fast` accepted backend。

验证结果：

- `scripts/sync_opencv_intrin.py --check` 通过，新增 x86 AVX-family whitelist 在维护边界内；SSE 文件只作为 x86 OpenCV UI/AVX 依赖，RVV header 不在 vendor 边界内。
- `python3 -m py_compile scripts/sync_opencv_intrin.py` 通过。
- `/usr/bin/c++ -std=c++17 -arch x86_64 -mavx2 ... cvh_opencv_intrin_x86_smoke.cpp` compile-only 通过；这验证 x86 AVX2 UI header 和当前 fixed-lane facade 能在 x86_64 AVX2 目标下编译。
- 当前 ARM64 机器上 `cvh_opencv_intrin_x86_smoke` 不生成，符合 x86-only 条件 target 设计。
- `cmake -S . -B build-p54-platform-intrin -DCVH_BUILD_NATIVE_BACKEND=OFF -DCVH_BUILD_TESTS=ON -DCVH_BUILD_BENCHMARKS=ON` 通过。
- `cmake --build build-p54-platform-intrin --target cvh_headers_fast_smoke cvh_opencv_intrin_smoke cvh_simd_facade_opencv_intrin_smoke cvh_cvtcolor_opencv_intrin_smoke cvh_resize_opencv_intrin_smoke cvh_benchmark_cvtcolor_bgr2gray_header cvh_benchmark_resize_bilinear_header -j` 通过。
- `ctest --test-dir build-p54-platform-intrin --output-on-failure -R 'cvh_headers_fast_smoke|cvh_opencv_intrin_smoke|cvh_simd_facade_opencv_intrin_smoke|cvh_cvtcolor_opencv_intrin_smoke|cvh_resize_opencv_intrin_smoke'` 通过。
- `./scripts/check_header_only_contract.sh` 通过。
- `./scripts/ci_headers_all.sh` 通过。
- `git diff --check` 通过。

P5.4.1b：OpenCV UI 默认开启

状态：已完成。

目标：

- `CVH_ENABLE_OPENCV_INTRIN` 默认值从 `0` 改为 `1`，不再要求用户通过宏或 `cvh::headers_fast` 打开 OpenCV Universal Intrinsics。
- `cvh::headers` 默认传播 vendored OpenCV UI include root，保证 CMake consumer 只链接 `cvh::headers` 即可编译。
- `cvh::headers_fast` 不再传播 `CVH_ENABLE_OPENCV_INTRIN=1`，只保留 `CVH_ENABLE_PLATFORM_INTRINSICS=1` 等额外 fast-profile toggles。
- scalar fallback 仍保留，并通过内部 smoke/benchmark 显式 `CVH_ENABLE_OPENCV_INTRIN=0` 作为正确性和性能对照。

落地结果：

- `include/cvh/detail/config.h` 中 `CVH_ENABLE_OPENCV_INTRIN` 默认改为 `1`。
- `cvh_headers` target 传播 `include/cvh/3rdparty/opencv_intrin` build/install include root。
- `cvh_headers_fast` 删除冗余的 `CVH_ENABLE_OPENCV_INTRIN=1` compile definition。
- `cvh_opencv_intrin_smoke` / `cvh_simd_facade_opencv_intrin_smoke` 不再手写开启宏，只保留 `CV_FORCE_SIMD128_CPP=1` 覆盖 portable UI 编译路径。
- `cvh_simd_facade_scalar_smoke` 显式 `CVH_ENABLE_OPENCV_INTRIN=0`，继续保留 scalar facade gate。
- `scripts/check_header_only_contract.sh` 的 `cvh::headers` external consumer 改为验证 OpenCV UI 默认开启，且 platform intrinsics 仍默认关闭。
- 直接 include smoke 增加 vendored OpenCV UI include root；非 CMake 直接使用时同样需要提供该 include root。

验收：

- `cvh::headers` consumer 可以不手写 `CVH_ENABLE_OPENCV_INTRIN` 即获得 `opencv_intrin` backend。
- `cvh::headers_fast` consumer 仍验证 `CVH_ENABLE_PLATFORM_INTRINSICS=1`，且不传播 xsimd 或 legacy `.cpp` 模式。
- scalar fallback 只能作为内部显式 opt-out 对照路径，不再作为用户默认路径。

验证结果：

- `scripts/sync_opencv_intrin.py --check` 通过。
- `python3 -m py_compile scripts/sync_opencv_intrin.py` 通过。
- `cmake -S . -B build-p54-default-opencv-intrin -DCVH_BUILD_NATIVE_BACKEND=OFF -DCVH_BUILD_TESTS=ON -DCVH_BUILD_BENCHMARKS=ON` 通过。
- `cmake --build build-p54-default-opencv-intrin --target cvh_header_compile_smoke cvh_include_only_smoke cvh_headers_fast_smoke cvh_opencv_intrin_smoke cvh_simd_facade_scalar_smoke cvh_simd_facade_opencv_intrin_smoke cvh_cvtcolor_opencv_intrin_smoke cvh_resize_opencv_intrin_smoke cvh_benchmark_cvtcolor_bgr2gray_header cvh_benchmark_resize_bilinear_header -j` 通过。
- `ctest --test-dir build-p54-default-opencv-intrin --output-on-failure -R 'cvh_header_compile_smoke|cvh_include_only_smoke|cvh_headers_fast_smoke|cvh_opencv_intrin_smoke|cvh_simd_facade_scalar_smoke|cvh_simd_facade_opencv_intrin_smoke|cvh_cvtcolor_opencv_intrin_smoke|cvh_resize_opencv_intrin_smoke'` 通过。
- `./scripts/check_header_only_contract.sh` 通过；external `cvh::headers` consumer 默认获得 `opencv_intrin` backend。
- `./scripts/ci_headers_all.sh` 通过。
- `/usr/bin/c++ -std=c++17 -arch x86_64 -mavx2 ... cvh_opencv_intrin_x86_smoke.cpp` 在不手写 `CVH_ENABLE_OPENCV_INTRIN=1` 的情况下 compile-only 通过。
- `CV_RVV=1` 负向 compile 检查仍按预期报错 `RVV is deferred`。
- `git diff --check` 通过。

P5.4.2：正确性 gate 固化

状态：并入 P6。P6 会重建 direct OpenCV UI 形态下的 correctness gate。

- 固化 OpenCV UI adapter smoke：
  - `cvh_opencv_intrin_smoke`
  - `cvh_simd_facade_opencv_intrin_smoke`
  - `cvh_cvtcolor_opencv_intrin_smoke`
  - `cvh_resize_opencv_intrin_smoke`
  - `cvh_headers_fast_smoke`
- 升级 OpenCV UI 后必须跑 `./scripts/check_header_only_contract.sh` 和 `./scripts/ci_headers_all.sh`。
- 若 local shim 或 whitelist 变化，需要补充 smoke 覆盖。

P5.4.3：性能 gate 和阈值

状态：并入 P6。P6 会在删除二次 facade 后重新跑 accepted fast path 的 benchmark gate。

- 固化 header-only benchmark 输入矩阵：
  - `cvh_benchmark_cvtcolor_bgr2gray_header`
  - `cvh_benchmark_resize_bilinear_header`
- 对 accepted fast path 建立最小性能阈值：
  - `BGR2GRAY/RGB2GRAY`：相对 scalar baseline 不应退化到无收益区间。
  - exact 2x `CV_8UC1 INTER_LINEAR resize`：必须保持显著快于 scalar fallback。
- 非 accepted fast path fallback case 不作为性能通过条件，但必须保持正确性。
- benchmark gate 默认作为维护/升级 gate，不塞进普通 header-only CI。

P5.4.4：OpenCV UI 升级 runbook

状态：并入 P6。P6 会把升级 runbook 改成 direct OpenCV UI dialect 的维护流程。

- 升级流程必须是：
  1. 切换或更新本地 OpenCV 源树到目标 commit。
  2. 运行 `scripts/sync_opencv_intrin.py` 更新 vendor 和 `UPSTREAM.md`。
  3. 运行 `scripts/sync_opencv_intrin.py --check`。
  4. 运行 P5.4.2 correctness gate。
  5. 运行 P5.4.3 benchmark gate。
  6. 只有 correctness 通过且 benchmark 未超过退化阈值，才允许提交 upstream commit 更新。

### P6：Direct OpenCV UI SIMD Dialect

状态：进行中，P6.6 已完成；下一步 P6.7。

P6 的目标不是继续给 OpenCV UI 包一层 `cvh::detail::simd`，而是把 OpenCV UI 本身确立为 `cvh` header-only 内部 SIMD 方言。

核心判断：

- OpenCV Universal Intrinsics 已经是对 NEON/SSE/AVX/AVX512 的 adapter，`opencv_intrin_adapter.h` 的二次封装收益不足。
- 直接使用 `cv::v_*`、`cv::VTraits<T>`、`CV_SIMD`、`CV_SIMD_WIDTH`、`vx_*`，可以最大化复用 OpenCV 原始 kernel 片段，例如 `box_filter.simd.hpp` 中的 `v_float32x4` / `v_uint16x8` / `v_pack_store` 代码形态。
- `cvh` 不把 OpenCV UI 暴露为用户 API 承诺；它只是 `include/cvh/**` 内部 SIMD implementation dialect。
- scalar fallback 不再通过 SIMD facade 模拟。scalar 对照应保持为显式 `*_scalar_impl` 或 benchmark direct helper。
- RVV 支持放入后续 TODO，因为 scalable SIMD 和当前 fixed-lane kernel porting 目标不同；P6 只关注 ARM NEON 和 x86 AVX 系列。SSE header/宏只作为 x86 OpenCV UI/AVX 编译链路的基础条件，不作为当前独立优化路线。

P6.0：决策固化与命名边界

状态：已完成。

目标：

- 在文档中明确废弃 `cvh::detail::simd` 作为项目自有 SIMD 方言的路线。
- 明确 OpenCV UI 是内部 SIMD dialect，不是 public API。
- 明确 `CVH_ENABLE_OPENCV_INTRIN=1` 是默认路径，`CVH_ENABLE_OPENCV_INTRIN=0` 只用于 scalar fallback 对照。

落地项：

- 更新 `doc/design.md`、`README.md`、`include/cvh/core/readme.md` 中的 SIMD strategy。
- 更新 `include/cvh/3rdparty/opencv_intrin/UPSTREAM.md` 生成模板，不再要求 “business kernels behind `cvh::detail::simd`”。
- 在计划文档中标注早期 `cvh::detail::simd` 相关内容为历史路径。

验收：

- 文档不再把 `cvh::detail::simd` 描述为未来主路线。
- 文档明确 direct OpenCV UI 只属于内部实现，不影响用户公开 API。

落地结果：

- `README.md` 已改为 `cvh::headers` 默认启用 OpenCV Universal Intrinsics headers 作为 internal SIMD dialect。
- `doc/design.md` 已明确业务 SIMD kernel 可以直接使用 `cv::v_*`、`cv::VTraits`、`CV_SIMD`、`CV_SIMD_WIDTH` 和 `vx_*`，但这些类型不构成 `cvh` 用户公开 API。
- `include/cvh/core/readme.md` 已明确不再把 `cvh::detail::simd` 二次 facade 作为未来主路线。
- `scripts/sync_opencv_intrin.py` 的 `UPSTREAM.md` 模板已改为 direct OpenCV UI 边界：允许 `include/cvh/**` 内部实现使用 OpenCV UI，但不把 `cv::v_*` 暴露为 `cvh` public API。
- `include/cvh/3rdparty/opencv_intrin/UPSTREAM.md` 已由同步脚本重新生成。

验证结果：

- `scripts/sync_opencv_intrin.py` 通过并重新生成 `UPSTREAM.md`。
- P6.0 属于文档和 vendor metadata 模板收口，未修改业务 kernel。

P6.1：建立极薄 include/config gateway

状态：已完成。

目标：

- 保留一个项目级 include 入口，避免业务文件直接写 vendor 长路径。
- gateway 只负责 include、宏策略和负向 gate，不再封装 SIMD operations。

建议形态：

```cpp
#include "cvh/core/simd/opencv_ui.h"
```

该 header 只做：

- include `cvh/detail/config.h`。
- 处理已移除的 xsimd 宏 hard error。
- 当 `CVH_ENABLE_OPENCV_INTRIN=1` 时 include `cvh/3rdparty/opencv_intrin/opencv2/core/hal/intrin.hpp`。
- 保持 `CV_RVV` / `CV_RVV071` deferred hard error。
- 提供最小 backend 诊断 helper，例如 `cvh::detail::opencv_ui_backend_name()`，如果仍需要 smoke/benchmark 字段。

明确不做：

- 不定义 `cvh::detail::simd::u8` / `u16` / `u32`。
- 不包装 `v_load` / `v_store` / `v_pack` / `v_add`。
- 不提供 scalar SIMD facade。

验收：

- `cvh/core/simd/opencv_ui.h` 可独立 include。
- `cvh::headers` consumer 不需要手写 OpenCV UI include root 或宏。
- RVV 手动启用仍明确失败。

落地结果：

- 新增 `include/cvh/core/simd/opencv_ui.h` 作为 direct OpenCV UI 的唯一项目级 include gateway。
- gateway 只处理 `cvh/detail/config.h`、xsimd 已移除宏 hard error、RVV deferred hard error，以及 `CVH_ENABLE_OPENCV_INTRIN=1` 时 include vendored `opencv2/core/hal/intrin.hpp`。
- gateway 提供 `cvh::detail::opencv_ui_backend_name()` 作为 smoke/benchmark 诊断字段，不定义 `cvh::detail::simd::*` 类型或操作。
- `include/cvh/core/simd/simd.h` 暂时兼容保留，但先 include `opencv_ui.h`，后续 P6.2/P6.3 迁移业务 kernel 后在 P6.4 删除或转为兼容 header。
- `cvh_opencv_intrin_smoke` 改为 include `cvh/core/simd/opencv_ui.h`，并直接使用 `cv::v_uint8x16`、`cv::v_setzero_u8`、`cv::v_store`、`cv::VTraits`。
- `cvh_opencv_intrin_x86_smoke` 不再使用 `cvh::detail::simd` facade，改为直接验证 `cv::v_uint8x16` 和 `cv::v_uint8x32` / `cv::v256_load` / `cv::v256_setzero_u8`。
- `test/smoke/readme.md` 已记录 `opencv_ui.h` gateway 和 x86 AVX2 direct UI smoke 的职责。

验证结果：

- `cmake -S . -B build-p61-opencv-ui-gateway -DCVH_BUILD_NATIVE_BACKEND=OFF -DCVH_BUILD_TESTS=ON -DCVH_BUILD_BENCHMARKS=ON` 通过。
- `cmake --build build-p61-opencv-ui-gateway --target cvh_header_compile_smoke cvh_include_only_smoke cvh_opencv_intrin_smoke cvh_headers_fast_smoke cvh_simd_facade_scalar_smoke cvh_simd_facade_opencv_intrin_smoke cvh_cvtcolor_opencv_intrin_smoke cvh_resize_opencv_intrin_smoke -j` 通过。
- `ctest --test-dir build-p61-opencv-ui-gateway --output-on-failure -R 'cvh_header_compile_smoke|cvh_include_only_smoke|cvh_opencv_intrin_smoke|cvh_headers_fast_smoke|cvh_simd_facade_scalar_smoke|cvh_simd_facade_opencv_intrin_smoke|cvh_cvtcolor_opencv_intrin_smoke|cvh_resize_opencv_intrin_smoke'` 通过，8/8 tests passed。
- `/usr/bin/c++ -std=c++17 -arch x86_64 -mavx2 -Iinclude -Iinclude/cvh/3rdparty/opencv_intrin -DCVH_ENABLE_PLATFORM_INTRINSICS=1 -c test/smoke/cvh_opencv_intrin_x86_smoke.cpp -o /tmp/cvh_opencv_intrin_x86_smoke_p61.o` 通过，用于在 ARM 主机上 compile-only 验证 x86 AVX2 direct UI header。
- `./scripts/check_header_only_contract.sh` 通过。
- `./scripts/ci_headers_all.sh` 通过，core 25/25、imgproc 141/141、imgcodecs 7 passed / 1 optional skipped、highgui 4/4。
- `scripts/sync_opencv_intrin.py --check && python3 -m py_compile scripts/sync_opencv_intrin.py` 通过。
- `CV_RVV=1` 负向 compile gate 仍按预期失败，并输出 RVV deferred hard error。
- `git diff --check` 通过。

P6.2：迁移 `BGR/RGB2GRAY` 到 direct OpenCV UI

状态：已完成。

目标：

- 将 `include/cvh/imgproc/cvtcolor.h` 中当前依赖 `cvh::detail::simd` 的 `CV_8UC3 BGR2GRAY/RGB2GRAY` fast path 改成直接 OpenCV UI 写法。

迁移原则：

- `simd::u8` -> `cv::v_uint8x16` 或 `cv::v_uint8`，优先 fixed 128-bit 形态以保持当前行为稳定。
- `simd::u16` -> `cv::v_uint16x8`。
- `simd::u32` -> `cv::v_uint32x4`。
- `simd::load_deinterleave3_u8` -> `cv::v_load_deinterleave`。
- `simd::mul_expand_u16` -> `cv::v_mul_expand`。
- `simd::rshr_pack_u32_to_u16<16>` -> `cv::v_rshr_pack<16>`。
- `simd::pack_u16_to_u8` -> `cv::v_pack`。
- `simd::store_u8` -> `cv::v_store`。
- `simd::u8_lanes()` -> `cv::VTraits<cv::v_uint8x16>::vlanes()`。

验收：

- `cvh_cvtcolor_opencv_intrin_smoke` 通过。
- `cvh_test_imgproc` 中 cvtColor 相关测试通过。
- header benchmark 中 `BGR2GRAY/RGB2GRAY` checksum 不变。
- 迁移后 `include/cvh/imgproc/cvtcolor.h` 不再引用 `cvh::detail::simd`。

落地结果：

- `include/cvh/imgproc/cvtcolor.h` 的 `CV_8UC3 BGR2GRAY/RGB2GRAY` fast path 已从 `../core/simd/simd.h` 切换到 `../core/simd/opencv_ui.h`。
- fast path 内部直接使用 fixed 128-bit OpenCV UI 类型和函数：`cv::v_uint8x16`、`cv::v_uint16x8`、`cv::v_uint32x4`、`cv::v_load_deinterleave`、`cv::v_expand`、`cv::v_mul_expand`、`cv::v_rshr_pack<16>`、`cv::v_pack`、`cv::v_store`、`cv::VTraits<cv::v_uint8x16>::vlanes()`。
- scalar fallback 和尾部处理逻辑保持不变。
- `benchmark/cvtcolor_bgr2gray_header_benchmark.cpp` 中该 kernel 的 lanes 和 micro helpers 也同步改为 direct OpenCV UI，避免 P6.2 后 benchmark 仍通过旧 facade 做成本拆解。
- `rg -n "cvh::detail::simd|namespace simd|simd::" include/cvh/imgproc/cvtcolor.h benchmark/cvtcolor_bgr2gray_header_benchmark.cpp` 无结果。

验证结果：

- `cmake --build build-p61-opencv-ui-gateway --target cvh_cvtcolor_opencv_intrin_smoke cvh_benchmark_cvtcolor_bgr2gray_header -j` 通过。
- `ctest --test-dir build-p61-opencv-ui-gateway --output-on-failure -R 'cvh_cvtcolor_opencv_intrin_smoke'` 通过。
- `./build-p61-opencv-ui-gateway/cvh_benchmark_cvtcolor_bgr2gray_header --profile quick --warmup 1 --iters 1 --repeats 1` 通过；benchmark 内置 scalar/public/direct correctness check 未发现 mismatch，`BGR2GRAY/RGB2GRAY` checksum 对齐。代表 checksum：`BGR2GRAY 640x480 = 11546163921466850888`、`RGB2GRAY 640x480 = 17809929532755976443`、`BGR2GRAY 3840x2160 = 450430794711396147`、`RGB2GRAY 3840x2160 = 2135016106324211590`。
- `cmake --build build-p61-opencv-ui-gateway --target cvh_test_imgproc cvh_header_compile_smoke cvh_include_only_smoke cvh_opencv_intrin_smoke cvh_headers_fast_smoke cvh_simd_facade_scalar_smoke cvh_simd_facade_opencv_intrin_smoke cvh_cvtcolor_opencv_intrin_smoke -j` 通过。
- `/usr/bin/c++ -std=c++17 -arch x86_64 -mavx2 -Iinclude -Iinclude/cvh/3rdparty/opencv_intrin -DCVH_ENABLE_PLATFORM_INTRINSICS=1 -c test/smoke/cvh_cvtcolor_opencv_intrin_smoke.cpp -o /tmp/cvh_cvtcolor_opencv_intrin_x86_p62.o` 通过，用于 ARM 主机上的 x86 AVX2 compile-only 验证。
- `./build-p61-opencv-ui-gateway/cvh_test_imgproc --gtest_filter='ImgprocCvtColor_TEST.*'` 通过，58/58 tests passed。
- `ctest --test-dir build-p61-opencv-ui-gateway --output-on-failure -R 'cvh_header_compile_smoke|cvh_include_only_smoke|cvh_opencv_intrin_smoke|cvh_headers_fast_smoke|cvh_simd_facade_scalar_smoke|cvh_simd_facade_opencv_intrin_smoke|cvh_cvtcolor_opencv_intrin_smoke'` 通过，7/7 tests passed。
- `./scripts/check_header_only_contract.sh` 通过。
- `./scripts/ci_headers_all.sh` 通过，core 25/25、imgproc 141/141、imgcodecs 7 passed / 1 optional skipped、highgui 4/4。
- `scripts/sync_opencv_intrin.py --check && python3 -m py_compile scripts/sync_opencv_intrin.py` 通过。
- `CV_RVV=1` 负向 compile gate 仍按预期失败，并输出 RVV deferred hard error。
- `git diff --check` 通过。

P6.3：迁移 exact 2x resize 到 direct OpenCV UI

状态：已完成。

目标：

- 将 `include/cvh/imgproc/resize.h` 中 exact 2x `CV_8UC1 INTER_LINEAR` fast path 改成 direct OpenCV UI。

迁移原则：

- `simd::load_deinterleave2_u8` -> `cv::v_load_deinterleave`。
- `simd::expand_u8` -> `cv::v_expand`。
- `simd::add` -> `cv::v_add`。
- `simd::rshr_pack_u16_to_u8<2>` -> `cv::v_rshr_pack<2>`。
- `simd::store_u8` -> `cv::v_store`。
- lane 数从 `cv::VTraits<cv::v_uint8x16>::vlanes()` 获取。

验收：

- `cvh_resize_opencv_intrin_smoke` 通过。
- `cvh_test_imgproc` 中 resize 相关测试通过。
- header benchmark 中 exact 2x resize checksum 不变，性能不得明显退化。
- 迁移后 `include/cvh/imgproc/resize.h` 不再引用 `cvh::detail::simd`。

落地结果：

- `include/cvh/imgproc/resize.h` 的 exact 2x `CV_8UC1 INTER_LINEAR` fast path 已从 `../core/simd/simd.h` 切换到 `../core/simd/opencv_ui.h`。
- fast path 内部直接使用 fixed 128-bit OpenCV UI 类型和函数：`cv::v_uint8x16`、`cv::v_uint16x8`、`cv::v_load_deinterleave`、`cv::v_expand`、`cv::v_add`、`cv::v_rshr_pack<2>`、`cv::v_store`、`cv::VTraits<cv::v_uint8x16>::vlanes()`。
- scalar fallback、fast path 选择条件和尾部处理逻辑保持不变。
- `benchmark/resize_bilinear_header_benchmark.cpp` 的 lane 查询同步改为 direct OpenCV UI，避免 P6.3 后 benchmark 仍通过旧 facade 读取 lanes。
- `rg -n "cvh::detail::simd|namespace simd|simd::|core/simd/simd.h" include/cvh/imgproc/resize.h benchmark/resize_bilinear_header_benchmark.cpp` 无结果。
- RVV 负向 gate 的 hard error 文案同步收紧为 `NEON or AVX`，与当前平台范围设定一致。

验证结果：

- `cmake --build build-p61-opencv-ui-gateway --target cvh_resize_opencv_intrin_smoke cvh_benchmark_resize_bilinear_header -j` 通过。
- `ctest --test-dir build-p61-opencv-ui-gateway --output-on-failure -R 'cvh_resize_opencv_intrin_smoke'` 通过。
- `./build-p61-opencv-ui-gateway/cvh_benchmark_resize_bilinear_header --profile quick --warmup 1 --iters 1 --repeats 1` 通过；benchmark 内置 scalar/public/direct correctness check 未发现 mismatch，exact 2x resize checksum 对齐。代表 checksum：`640x480_to_320x240 = 2059971736401517523`、`1280x720_to_640x360 = 8392939373037756272`、`1920x1080_to_960x540 = 13721900255821653829`、`3840x2160_to_1920x1080 = 14534879005442455564`。
- `/usr/bin/c++ -std=c++17 -arch x86_64 -mavx2 -Iinclude -Iinclude/cvh/3rdparty/opencv_intrin -DCVH_ENABLE_PLATFORM_INTRINSICS=1 -c test/smoke/cvh_resize_opencv_intrin_smoke.cpp -o /tmp/cvh_resize_opencv_intrin_x86_p63.o` 通过，用于 ARM 主机上的 x86 AVX2 compile-only 验证。
- `cmake --build build-p61-opencv-ui-gateway --target cvh_test_imgproc cvh_header_compile_smoke cvh_include_only_smoke cvh_opencv_intrin_smoke cvh_headers_fast_smoke cvh_simd_facade_scalar_smoke cvh_simd_facade_opencv_intrin_smoke cvh_resize_opencv_intrin_smoke -j` 通过。
- `./build-p61-opencv-ui-gateway/cvh_test_imgproc --gtest_filter='ImgprocResize_TEST.*'` 通过，11/11 tests passed。
- `ctest --test-dir build-p61-opencv-ui-gateway --output-on-failure -R 'cvh_header_compile_smoke|cvh_include_only_smoke|cvh_opencv_intrin_smoke|cvh_headers_fast_smoke|cvh_simd_facade_scalar_smoke|cvh_simd_facade_opencv_intrin_smoke|cvh_resize_opencv_intrin_smoke'` 通过，7/7 tests passed。
- `./scripts/check_header_only_contract.sh` 通过。
- `./scripts/ci_headers_all.sh` 通过，core 25/25、imgproc 141/141、imgcodecs 7 passed / 1 optional skipped、highgui 4/4。
- `scripts/sync_opencv_intrin.py --check && python3 -m py_compile scripts/sync_opencv_intrin.py` 通过。
- `CV_RVV=1` 负向 compile gate 仍按预期失败，并输出 `RVV is deferred; use NEON or AVX paths until a scalable design exists`。
- `git diff --check` 通过。

P6.4：删除 `opencv_intrin_adapter.h` 和 `scalar_adapter.h`

状态：已完成。

目标：

- 完全移除 `opencv_intrin_adapter.h`，同时处理 `scalar_adapter.h` 和 `simd.h` 是否还有存在价值。

删除/调整范围：

- 删除 `include/cvh/core/simd/opencv_intrin_adapter.h`。
- 删除或重命名 `include/cvh/core/simd/scalar_adapter.h`，除非某个 benchmark 仍临时需要。
- 将 `include/cvh/core/simd/simd.h` 改成兼容转发 header，或直接删除：
  - 如果保留，内容只 include `opencv_ui.h` 并标注 deprecated internal header。
  - 如果删除，必须同步所有 include 和 smoke。
- 删除 `cvh_simd_facade_scalar_smoke`。
- 删除或改造 `cvh_simd_facade_opencv_intrin_smoke`，改成 `cvh_opencv_ui_smoke`。
- 更新 `scripts/check_header_only_contract.sh`，不再依赖 `cvh::detail::simd::backend_name()`。

验收：

- `rg -n "opencv_intrin_adapter|scalar_adapter|cvh::detail::simd|namespace simd|simd::" include benchmark test scripts CMakeLists.txt -g '!include/cvh/3rdparty/opencv_intrin/**'` 无业务依赖残留。
- 允许的残留只能是历史文档归档。
- `cvh::headers` 和 `cvh::headers_fast` external consumer 都能通过。

落地结果：

- 删除 `include/cvh/core/simd/opencv_intrin_adapter.h`。
- 删除 `include/cvh/core/simd/scalar_adapter.h`。
- 删除 `test/smoke/cvh_simd_facade_smoke.cpp`。
- `include/cvh/core/simd/simd.h` 保留为 deprecated internal compatibility header，内容只 include `opencv_ui.h`，不再定义任何二次 SIMD facade namespace、类型或操作。
- `CMakeLists.txt` 删除 `cvh_simd_facade_scalar_smoke` 和 `cvh_simd_facade_opencv_intrin_smoke` target/test。
- `test/smoke/cvh_headers_fast_smoke.cpp` 改为 include `cvh/core/simd/opencv_ui.h`，并用 `cvh::detail::opencv_ui_backend_name()` 做 backend 诊断。
- `scripts/check_header_only_contract.sh` 的 external consumer 检查改为 include `cvh/core/simd/opencv_ui.h`，不再依赖旧 backend helper。
- `include/cvh/core/readme.md` 去掉旧二次 facade 的字面 API 名称，避免 include 文档路径在业务残留扫描中误命中。

验证结果：

- `cmake -S . -B build-p64-direct-ui-cleanup -DCVH_BUILD_NATIVE_BACKEND=OFF -DCVH_BUILD_TESTS=ON -DCVH_BUILD_BENCHMARKS=ON` 通过。
- `cmake --build build-p64-direct-ui-cleanup --target cvh_header_compile_smoke cvh_include_only_smoke cvh_headers_fast_smoke cvh_opencv_intrin_smoke cvh_cvtcolor_opencv_intrin_smoke cvh_resize_opencv_intrin_smoke cvh_benchmark_cvtcolor_bgr2gray_header cvh_benchmark_resize_bilinear_header -j` 通过。
- `ctest --test-dir build-p64-direct-ui-cleanup --output-on-failure -R 'cvh_header_compile_smoke|cvh_include_only_smoke|cvh_headers_fast_smoke|cvh_opencv_intrin_smoke|cvh_cvtcolor_opencv_intrin_smoke|cvh_resize_opencv_intrin_smoke'` 通过，6/6 tests passed。
- `./build-p64-direct-ui-cleanup/cvh_benchmark_cvtcolor_bgr2gray_header --profile quick --warmup 1 --iters 1 --repeats 1` 通过；代表 checksum 仍包括 `BGR2GRAY 3840x2160 = 450430794711396147`、`RGB2GRAY 3840x2160 = 2135016106324211590`。
- `./build-p64-direct-ui-cleanup/cvh_benchmark_resize_bilinear_header --profile quick --warmup 1 --iters 1 --repeats 1` 通过；代表 checksum `3840x2160_to_1920x1080 = 14534879005442455564`。
- `/usr/bin/c++ -std=c++17 -arch x86_64 -mavx2 -Iinclude -Iinclude/cvh/3rdparty/opencv_intrin -DCVH_ENABLE_PLATFORM_INTRINSICS=1 -c test/smoke/cvh_opencv_intrin_x86_smoke.cpp -o /tmp/cvh_opencv_intrin_x86_p64.o` 通过。
- `./scripts/check_header_only_contract.sh` 通过。
- `./scripts/ci_headers_all.sh` 通过，core 25/25、imgproc 141/141、imgcodecs 7 passed / 1 optional skipped、highgui 4/4。
- `CV_RVV=1` 负向 compile gate 仍按预期失败，并输出 `RVV is deferred; use NEON or AVX paths until a scalable design exists`。
- `scripts/sync_opencv_intrin.py --check && python3 -m py_compile scripts/sync_opencv_intrin.py` 通过。
- `cmake --build build-p64-direct-ui-cleanup --target help | rg "cvh_simd_facade|simd_facade"` 无结果，确认旧 facade smoke target 不再生成。
- `rg -n "opencv_intrin_adapter|scalar_adapter|cvh::detail::simd|namespace simd|simd::|cvh_simd_facade|simd_facade" include benchmark test scripts CMakeLists.txt -g '!include/cvh/3rdparty/opencv_intrin/**'` 无结果。
- `git diff --check` 通过。

P6.5：建立 OpenCV kernel 迁移规范

状态：已完成。

目标：

- 让后续从 OpenCV 移植 kernel 时尽量保留原始 UI 风格，减少重写。

规范：

- 优先保留 OpenCV UI 表达：
  - `v_*` / `vx_*`
  - `VTraits<T>::vlanes()`
  - `CV_SIMD` / `CV_SIMD_WIDTH`
  - `v_pack_store`
  - `v_load_deinterleave`
- 仅替换 OpenCV module/runtime 依赖：
  - `Mat` / `Size` / `Range` 等类型适配到 `cvh` 自有类型。
  - `CV_Assert` / `CV_Error_` 使用现有 `cvh` header-only 错误机制。
  - 去掉 OpenCV dispatch table、IPP/OCL/HAL function layer。
- 禁止为了迁移而重新包一层 `cvh::detail::simd`。
- 每个迁移 kernel 必须保留 scalar fallback，fast path 条件不满足时必须回退。

首个候选：

- `boxFilter` / `blur` 的局部 UI 代码片段可以作为 P6 后第一个迁移练习，但 P6 本身不扩大 accepted kernel 面；P6 先完成架构清理。

验收：

- 文档中给出一段 OpenCV UI 代码迁移 checklist。
- 后续新 fast path PR 可以按 checklist 评审，不再讨论是否要包 `cvh::detail::simd`。

落地结果：

- 新增 `doc/opencv-ui-kernel-migration-checklist.md`，作为 P6 后从 OpenCV `*.simd.hpp` 迁移 kernel 片段的执行 checklist。
- checklist 固化当前 SIMD 范围：ARM NEON 和 x86 AVX family 通过 OpenCV UI；SSE 只作为 x86 OpenCV UI/AVX 编译链路基础；RVV 保持 future TODO。
- checklist 明确迁移时保留 `cv::v_*`、`cv::vx_*`、`cv::VTraits<T>::vlanes()`、`CV_SIMD`、`CV_SIMD_WIDTH`、`cv::v_pack_store`、`cv::v_load_deinterleave` 等 OpenCV UI 原始表达。
- checklist 明确只替换 OpenCV runtime/module 依赖，移除 dispatch table、IPP/OCL/HAL C dispatch glue、parallel runtime、global/module registration 等非 header-only 依赖。
- checklist 明确禁止重新引入 `cvh::detail::simd` 二次 facade，禁止让 `cvh::headers_fast` 依赖 `.cpp`、native target 或 xsimd。
- `README.md` 的 Performance/Compare workspace 增加 checklist 入口。
- `doc/design.md` 的 SIMD strategy 增加 checklist 入口。

验证结果：

- checklist 已覆盖 source selection、required shape、OpenCV UI style、runtime dependency removal、correctness gate、benchmark gate、required commands 和 review checklist。
- 文档入口已从 README 和 design 文档挂接，后续新增 fast path 可以按该 checklist 评审。

P6.6：验证矩阵

状态：已完成。

必须通过：

- `scripts/sync_opencv_intrin.py --check`
- `python3 -m py_compile scripts/sync_opencv_intrin.py`
- `cmake -S . -B build-p6-direct-opencv-ui -DCVH_BUILD_NATIVE_BACKEND=OFF -DCVH_BUILD_TESTS=ON -DCVH_BUILD_BENCHMARKS=ON`
- `cmake --build build-p6-direct-opencv-ui --target cvh_header_compile_smoke cvh_include_only_smoke cvh_headers_fast_smoke cvh_opencv_intrin_smoke cvh_cvtcolor_opencv_intrin_smoke cvh_resize_opencv_intrin_smoke cvh_benchmark_cvtcolor_bgr2gray_header cvh_benchmark_resize_bilinear_header -j`
- `ctest --test-dir build-p6-direct-opencv-ui --output-on-failure -R 'cvh_header_compile_smoke|cvh_include_only_smoke|cvh_headers_fast_smoke|cvh_opencv_intrin_smoke|cvh_cvtcolor_opencv_intrin_smoke|cvh_resize_opencv_intrin_smoke'`
- `./scripts/check_header_only_contract.sh`
- `./scripts/ci_headers_all.sh`
- x86_64 `-mavx2` compile-only smoke，不手写 `CVH_ENABLE_OPENCV_INTRIN=1`
- `CV_RVV=1` 负向 compile 检查
- `git diff --check`

benchmark gate：

- `BGR/RGB2GRAY` direct OpenCV UI 迁移后，当前 ARM benchmark 不应比 P5.4 默认 OpenCV UI 版本明显退化。
- exact 2x resize direct OpenCV UI 迁移后，当前 ARM benchmark 应保持显著快于 scalar fallback。
- 如果性能退化来自单纯改写错误，修 direct UI 实现；如果来自 OpenCV UI 原始指令选择，则记录为后续 direct NEON/AVX candidate，而不是恢复 `cvh::detail::simd` facade。

验证结果：

- `scripts/sync_opencv_intrin.py --check` 通过，当前 vendor 仍对应 OpenCV `4.13.0-457-gd48bf69f65` / `d48bf69f65444a13f8a34b8982b083c1b78fa0e8`。
- `python3 -m py_compile scripts/sync_opencv_intrin.py` 通过。
- `git diff --check` 通过。
- 代码面残留扫描通过：`include/`、`benchmark/`、`test/`、`scripts/`、`CMakeLists.txt` 中无 `opencv_intrin_adapter`、`scalar_adapter`、`cvh::detail::simd`、`simd_facade` 业务依赖残留。
- `cmake -S . -B build-p6-direct-opencv-ui -DCVH_BUILD_NATIVE_BACKEND=OFF -DCVH_BUILD_TESTS=ON -DCVH_BUILD_BENCHMARKS=ON` 通过。
- `cmake --build build-p6-direct-opencv-ui --target cvh_header_compile_smoke cvh_include_only_smoke cvh_headers_fast_smoke cvh_opencv_intrin_smoke cvh_cvtcolor_opencv_intrin_smoke cvh_resize_opencv_intrin_smoke cvh_benchmark_cvtcolor_bgr2gray_header cvh_benchmark_resize_bilinear_header -j` 通过。
- `ctest --test-dir build-p6-direct-opencv-ui --output-on-failure -R 'cvh_header_compile_smoke|cvh_include_only_smoke|cvh_headers_fast_smoke|cvh_opencv_intrin_smoke|cvh_cvtcolor_opencv_intrin_smoke|cvh_resize_opencv_intrin_smoke'` 通过，6/6 tests passed。
- `./scripts/check_header_only_contract.sh` 通过，external consumer 的 `cvh::headers` 和 `cvh::headers_fast` 都能在 pure header-only 场景下编译运行。
- `./scripts/ci_headers_all.sh` 通过：core 25/25、imgproc 141/141、imgcodecs 7 passed / 1 optional skipped、highgui 4/4。
- x86_64 AVX2 compile-only gate 通过：`/usr/bin/c++ -std=c++17 -arch x86_64 -mavx2 -Iinclude -Iinclude/cvh/3rdparty/opencv_intrin -DCVH_ENABLE_PLATFORM_INTRINSICS=1 -c test/smoke/cvh_opencv_intrin_x86_smoke.cpp -o /tmp/cvh_opencv_intrin_x86_p66.o`。该命令没有手动定义 `CVH_ENABLE_OPENCV_INTRIN=1`，验证默认开启策略生效。
- `CV_RVV=1` 负向 compile gate 按预期失败，并输出 `RVV is deferred; use NEON or AVX paths until a scalable design exists`。
- `cmake --build build-p6-direct-opencv-ui --target help | rg 'cvh_simd_facade|simd_facade'` 无结果，确认旧 facade smoke target 不再生成。

benchmark quick gate：

- `./build-p6-direct-opencv-ui/cvh_benchmark_cvtcolor_bgr2gray_header --profile quick --warmup 1 --iters 1 --repeats 1` 通过，checksum 对齐。
- 代表 4K public reuse 结果：`BGR2GRAY` scalar `1.119500 ms`，OpenCV UI NEON `0.871083 ms`，`9521.939930 MPix/s`，speedup `1.285182`，checksum `450430794711396147`。
- 代表 4K public reuse 结果：`RGB2GRAY` scalar `0.983750 ms`，OpenCV UI NEON `0.828292 ms`，`10013.859847 MPix/s`，speedup `1.187685`，checksum `2135016106324211590`。
- `./build-p6-direct-opencv-ui/cvh_benchmark_resize_bilinear_header --profile quick --warmup 1 --iters 1 --repeats 1` 通过，checksum 对齐。
- 代表 4K exact 2x public reuse 结果：scalar `2.996625 ms`，OpenCV UI NEON `0.140709 ms`，`14736.797220 MPix/s`，speedup `21.296612`，checksum `14534879005442455564`。
- 结论：direct OpenCV UI 改造没有引入 correctness regression；`cvtColor` 仍是 modest speedup，`resize exact 2x` 仍是强收益 fast path。若后续要求 `cvtColor >=1.5x` 稳定门槛，应作为 direct NEON/AVX 特化候选，而不是恢复二次 SIMD facade。

P6.7：最终清理与提交边界

状态：已完成，P6.7.3 已完成；P6 direct OpenCV UI 收口完成。

目标：

- P6 提交后，代码面和文档面都不再把 `opencv_intrin_adapter.h` 作为当前路线。

最终状态要求：

- `include/cvh/core/simd/opencv_intrin_adapter.h` 不存在。
- `include/cvh/core/simd/scalar_adapter.h` 不存在，或只剩明确 legacy/benchmark-only 文件且不被 public target 引用。
- 业务 fast path 直接 include `cvh/core/simd/opencv_ui.h` 并使用 `cv::v_*`。
- `UPSTREAM.md`、README、design、benchmark readme、smoke readme 口径一致。
- `cvh::headers` 仍是默认 header-only target，OpenCV UI 默认开启。
- `cvh::headers_fast` 仍只表示额外 platform fast-profile toggles，不重新承担 “打开 OpenCV UI” 的含义。

推荐提交拆分：

1. P6.0/P6.1：文档和 include gateway。
2. P6.2：迁移 cvtColor。
3. P6.3：迁移 resize。
4. P6.4/P6.5：删除 adapter/facade，更新 smoke 和迁移规范。
5. P6.6/P6.7：验证结果和最终文档收口。

P6.7.0：文档一致性检查

状态：已完成。

目标：

- README、`doc/design.md`、`include/cvh/3rdparty/opencv_intrin/UPSTREAM.md`、`test/smoke/readme.md`、`benchmark/readme.md` 与 P6 direct OpenCV UI 路线一致。
- `benchmark/opencv_compare/README.md` 虽然保留 legacy/internal `native` / `lite` compare mode，但必须明确这些名字不是 public CMake targets。
- 当前路线文档不再把 `opencv_intrin_adapter.h`、`scalar_adapter.h` 或 `cvh::detail::simd` 描述为未来主线。
- 当前路线文档明确 `cvh::headers` 默认启用 OpenCV UI，`cvh::headers_fast` 只表示额外 platform fast-profile toggles。

检查项：

- 搜索当前文档中的 `adapter`、`facade`、`cvh::detail::simd`、`xsimd`、`native`、`.cpp`、`headers_fast`、`CVH_ENABLE_OPENCV_INTRIN`。
- 区分历史记录和当前路线；历史阶段可以保留旧判断，但 README/design/vendor/smoke/benchmark 当前说明必须收口。
- benchmark 文档中的 direct detail 入口必须指向 OpenCV UI direct implementation，而不是旧 SIMD facade。
- smoke 文档必须指向 `cvh/core/simd/opencv_ui.h` gateway，而不是旧 `simd.h` facade。

验收：

- 当前路线文档口径一致。
- 允许旧词只出现在历史计划记录或“已移除/禁止/legacy”语境中。

落地结果：

- `README.md` 的 Compare workspace 链接已标注 OpenCV compare 是 legacy/internal compare mode，不是 public CMake target。
- `benchmark/readme.md` 中 `cvh_benchmark_cvtcolor_bgr2gray_header` 的对比入口已从旧 `detail::*_simd_impl` 表述改为 direct OpenCV UI detail 入口。
- `benchmark/readme.md` 中阶段 CSV 保留规则已从 “OpenCV Universal Intrinsics adapter 计划” 改为 “OpenCV UI direct migration 计划”。
- `benchmark/opencv_compare/README.md` 已明确 `native` / `lite` 是 compare 脚本和 CSV schema 的内部实现模式，不是 `cvh::headers` / `cvh::headers_fast` 的替代命名。
- `include/cvh/3rdparty/opencv_intrin/UPSTREAM.md` 中当前 policy 的 “adapter tree” 表述已改成 “vendor tree”，避免和已移除的二次 adapter 路线混淆。

审计结果：

- README、design、vendor upstream、smoke readme、benchmark readme 当前路线口径一致：`cvh::headers` 默认 OpenCV UI，`cvh::headers_fast` 只表示额外 platform fast-profile toggles。
- `cvh::detail::simd` / SIMD facade 的当前文档命中只剩 “不再作为未来路线” 语境。
- `xsimd` 的当前文档命中只剩 “已移除 / 不启用 / 不依赖” 语境。
- `native` 的公开文档命中只剩 legacy/internal compare mode 或 legacy `.cpp` 实验链路语境。

P6.7.1：代码面最终残留检查

状态：已完成。

目标：

- 确认删除文件、include 入口和业务 fast path 都符合 P6 最终状态。

检查项：

- `include/cvh/core/simd/opencv_intrin_adapter.h` 不存在。
- `include/cvh/core/simd/scalar_adapter.h` 不存在。
- `include/cvh/core/simd/simd.h` 只作为 deprecated compatibility header 转发到 `opencv_ui.h`。
- `include/cvh/imgproc/cvtcolor.h` 和 `include/cvh/imgproc/resize.h` 直接 include `cvh/core/simd/opencv_ui.h`。
- `CMakeLists.txt`、`test/`、`benchmark/`、`scripts/` 中无旧 `simd_facade` target 或 `cvh::detail::simd` 业务依赖。

验收：

- `rg -n "opencv_intrin_adapter|scalar_adapter|cvh::detail::simd|namespace simd|simd::|cvh_simd_facade|simd_facade" include benchmark test scripts CMakeLists.txt -g '!include/cvh/3rdparty/opencv_intrin/**' -S` 无结果。
- `cmake --build build-p6-direct-opencv-ui --target help | rg 'cvh_simd_facade|simd_facade'` 无结果。

落地结果：

- `include/cvh/core/simd/opencv_intrin_adapter.h` 不存在。
- `include/cvh/core/simd/scalar_adapter.h` 不存在。
- `include/cvh/core/simd/simd.h` 只作为 deprecated internal compatibility header 保留，内容仅 include `opencv_ui.h`，不定义 namespace、类型或操作。
- `include/cvh/imgproc/cvtcolor.h` 和 `include/cvh/imgproc/resize.h` 直接 include `cvh/core/simd/opencv_ui.h`，业务 fast path 使用 `cv::v_*` / `cv::VTraits` / `cv::v_load_deinterleave` / `cv::v_pack` / `cv::v_rshr_pack` / `cv::v_store`。
- `cvtColor` direct OpenCV UI detail 函数从旧泛化 `*_simd_impl` 命名收口为 `*_opencv_intrin_impl`，避免和已删除的二次 SIMD facade 路线混淆。
- `resize` direct detail 入口已保持 `resize_linear_u8c1_downsample2_opencv_intrin_impl` 命名。
- `CMakeLists.txt`、`test/`、`benchmark/`、`scripts/` 中无旧 `simd_facade` target 或 `cvh::detail::simd` 业务依赖。

验证结果：

- `test ! -e include/cvh/core/simd/opencv_intrin_adapter.h && test ! -e include/cvh/core/simd/scalar_adapter.h` 通过。
- `rg -n "cvtcolor_.*simd_impl|wide_simd|simd_impl" include/cvh/imgproc/cvtcolor.h benchmark/cvtcolor_bgr2gray_header_benchmark.cpp test benchmark include scripts CMakeLists.txt -g '!include/cvh/3rdparty/opencv_intrin/**' -S` 无结果。
- `rg -n "opencv_intrin_adapter|scalar_adapter|cvh::detail::simd|namespace simd|simd::|cvh_simd_facade|simd_facade" include benchmark test scripts CMakeLists.txt -g '!include/cvh/3rdparty/opencv_intrin/**' -S` 无结果。
- `cmake --build build-p6-direct-opencv-ui --target help | rg 'cvh_simd_facade|simd_facade'` 无结果。
- `cmake --build build-p6-direct-opencv-ui --target cvh_cvtcolor_opencv_intrin_smoke cvh_benchmark_cvtcolor_bgr2gray_header -j` 通过。
- `ctest --test-dir build-p6-direct-opencv-ui --output-on-failure -R 'cvh_cvtcolor_opencv_intrin_smoke'` 通过。
- `./build-p6-direct-opencv-ui/cvh_benchmark_cvtcolor_bgr2gray_header --profile quick --warmup 1 --iters 1 --repeats 1` 通过，direct detail 和 public path checksum 仍对齐；4K 代表 checksum：`BGR2GRAY = 450430794711396147`、`RGB2GRAY = 2135016106324211590`。
- `git diff --check` 通过。

P6.7.2：最终验证

状态：已完成。

目标：

- 在 P6.7.0/P6.7.1 收口后复跑关键 gate，确认文档和代码最终状态没有破坏 P6.6 验证面。

必须通过：

- `git diff --check`
- `scripts/sync_opencv_intrin.py --check`
- `python3 -m py_compile scripts/sync_opencv_intrin.py`
- `cmake -S . -B build-p6-direct-opencv-ui -DCVH_BUILD_NATIVE_BACKEND=OFF -DCVH_BUILD_TESTS=ON -DCVH_BUILD_BENCHMARKS=ON`
- `cmake --build build-p6-direct-opencv-ui --target cvh_header_compile_smoke cvh_include_only_smoke cvh_headers_fast_smoke cvh_opencv_intrin_smoke cvh_cvtcolor_opencv_intrin_smoke cvh_resize_opencv_intrin_smoke cvh_benchmark_cvtcolor_bgr2gray_header cvh_benchmark_resize_bilinear_header -j`
- `ctest --test-dir build-p6-direct-opencv-ui --output-on-failure -R 'cvh_header_compile_smoke|cvh_include_only_smoke|cvh_headers_fast_smoke|cvh_opencv_intrin_smoke|cvh_cvtcolor_opencv_intrin_smoke|cvh_resize_opencv_intrin_smoke'`
- `./scripts/check_header_only_contract.sh`
- `./scripts/ci_headers_all.sh`
- x86_64 AVX2 compile-only gate，不手动定义 `CVH_ENABLE_OPENCV_INTRIN=1`。
- `CV_RVV=1` 负向 compile gate。

验证结果：

- `git diff --check` 通过。
- `scripts/sync_opencv_intrin.py --check` 通过，vendor 仍对应 OpenCV `4.13.0-457-gd48bf69f65` / `d48bf69f65444a13f8a34b8982b083c1b78fa0e8`。
- `python3 -m py_compile scripts/sync_opencv_intrin.py` 通过。
- 代码面残留预检查通过：`opencv_intrin_adapter`、`scalar_adapter`、`cvh::detail::simd`、`simd_facade`、旧 `cvtcolor_*_simd_impl` 命名均无业务命中。
- `cmake -S . -B build-p6-direct-opencv-ui -DCVH_BUILD_NATIVE_BACKEND=OFF -DCVH_BUILD_TESTS=ON -DCVH_BUILD_BENCHMARKS=ON` 通过。
- `cmake --build build-p6-direct-opencv-ui --target cvh_header_compile_smoke cvh_include_only_smoke cvh_headers_fast_smoke cvh_opencv_intrin_smoke cvh_cvtcolor_opencv_intrin_smoke cvh_resize_opencv_intrin_smoke cvh_benchmark_cvtcolor_bgr2gray_header cvh_benchmark_resize_bilinear_header -j` 通过。
- `ctest --test-dir build-p6-direct-opencv-ui --output-on-failure -R 'cvh_header_compile_smoke|cvh_include_only_smoke|cvh_headers_fast_smoke|cvh_opencv_intrin_smoke|cvh_cvtcolor_opencv_intrin_smoke|cvh_resize_opencv_intrin_smoke'` 通过，6/6 tests passed。
- `./scripts/check_header_only_contract.sh` 通过，external consumer 的 `cvh::headers` 和 `cvh::headers_fast` 都能编译运行。
- `./scripts/ci_headers_all.sh` 通过：core 25/25、imgproc 141/141、imgcodecs 7 passed / 1 optional skipped、highgui 4/4。
- x86_64 AVX2 compile-only gate 通过：`/usr/bin/c++ -std=c++17 -arch x86_64 -mavx2 -Iinclude -Iinclude/cvh/3rdparty/opencv_intrin -DCVH_ENABLE_PLATFORM_INTRINSICS=1 -c test/smoke/cvh_opencv_intrin_x86_smoke.cpp -o /tmp/cvh_opencv_intrin_x86_p672.o`。该命令没有手动定义 `CVH_ENABLE_OPENCV_INTRIN=1`。
- `CV_RVV=1` 负向 compile gate 按预期失败，并输出 `RVV is deferred; use NEON or AVX paths until a scalable design exists`。
- `cmake --build build-p6-direct-opencv-ui --target help | rg 'cvh_simd_facade|simd_facade'` 无结果，确认旧 facade target 不再生成。

benchmark quick 复核：

- `./build-p6-direct-opencv-ui/cvh_benchmark_cvtcolor_bgr2gray_header --profile quick --warmup 1 --iters 1 --repeats 1` 通过，checksum 对齐。
- 代表 4K public reuse 结果：`BGR2GRAY` scalar `1.334125 ms`，OpenCV UI NEON `0.950500 ms`，speedup `1.403603`，checksum `450430794711396147`。
- 代表 4K public reuse 结果：`RGB2GRAY` scalar `0.976292 ms`，OpenCV UI NEON `0.822834 ms`，speedup `1.186499`，checksum `2135016106324211590`。
- `./build-p6-direct-opencv-ui/cvh_benchmark_resize_bilinear_header --profile quick --warmup 1 --iters 1 --repeats 1` 通过，checksum 对齐。
- 代表 4K exact 2x public reuse 结果：scalar `2.588791 ms`，OpenCV UI NEON `0.136791 ms`，speedup `18.925156`，checksum `14534879005442455564`。

P6.7.3：提交边界

状态：已完成。

目标：

- 将 P6 direct OpenCV UI 改造作为一组可审查的提交收口。

建议：

- 如果当前工作区只包含 P6 相关修改，P6.7 完成后可以直接提交。
- commit message 建议描述为 direct OpenCV UI dialect、删除二次 facade、默认 OpenCV UI、迁移 checklist 和验证矩阵。
- 提交前再次确认 `git status --short --branch`，避免夹带非 P6 修改。

落地结果：

- P6 direct OpenCV UI 改造已提交为 `87c743f Use direct OpenCV UI for header SIMD paths`。
- 该提交包含 direct OpenCV UI gateway、`cvtColor` / `resize` direct UI 迁移、旧 adapter/facade 删除、smoke/contract 更新、迁移 checklist 和 P6.6/P6.7 验证记录。
- 提交前 `git diff --check` 通过，且待提交范围均为 P6 direct OpenCV UI 相关修改。
- 提交后 `git status --short --branch` 显示 `main...origin/main [ahead 5]`，无未提交修改。
- P6.7.3 的本段记录作为 doc-only 收口提交单独提交，避免修改实现提交后再次混入代码变更。

## 成功标准

短期成功：

- `cvh::headers` 默认启用 OpenCV Universal Intrinsics。
- 已接受 fast path 直接使用 OpenCV UI 写法，不依赖 `opencv_intrin_adapter.h`。
- 不引入 `.cpp` 编译层作为公开依赖。
- RVV 仍保持 deferred，不进入当前 NEON/AVX 路线；SSE 仅作为 x86 OpenCV UI/AVX 编译依赖。

中期成功：

- OpenCV UI 成为 `cvh` 内部 SIMD dialect。
- scalar fallback 成为显式 correctness/benchmark 对照路径，而不是 SIMD facade backend。
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
direct OpenCV UI internal dialect
thin include/config gateway
benchmark gate
小范围试点
```

不建议采用：

```text
复制 OpenCV 完整 HAL
重新包一套 cvh::detail::simd 二次 facade
跟随 OpenCV master
默认开启所有 SIMD 头文件
```

这条路线可以在不引入 `.cpp` 编译层的前提下，增强 header-only 层的 CPU SIMD 能力，并让后续 OpenCV kernel 迁移更接近原始 `cv::v_*` / `VTraits` / `vx_*` 写法；必要的 direct NEON/AVX 特化仍必须通过 benchmark gate。
