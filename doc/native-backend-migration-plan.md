# Native Backend 改造计划

## 背景

项目名称 `opencv-header-only` 对用户有很强的默认暗示：只包含头文件即可使用，不需要编译或链接额外库。

当前仓库同时包含：

- `include/`：公开头文件与 Lite fallback 实现。
- `src/`：需要编译的后端实现。

这种结构本身可以成立，但需要把语义讲清楚：项目的默认体验必须是 header-only；`src/` 中的 C++ 实现是可选的 native backend，而不是 header-only 主路径的一部分。

## 目标

将项目公开模型调整为：

- `cvh::headers` 是主路径，纯 header-only，默认可用。
- `cvh::native` 是可选 native backend，需要编译和链接。
- 默认构建不启用 native backend。
- 所有 `src/*.cpp`、平台显示实现、SIMD/dispatch 增强等编译型能力统一归入 native backend。
- 不再把编译型后端称为 `Full`，避免暗示 Lite 是残缺版本。

目标一句话：

> `opencv-header-only` 默认就是 header-only；如果用户显式选择 `native`，才会编译并链接额外后端能力。

## 命名约定

### CMake targets

目标命名：

```cmake
cvh::headers
cvh::native
```

内部 target 命名：

```cmake
cvh_headers
cvh_native_backend
```

废弃命名：

```cmake
cvh::full
cvh::full_backend
cvh_full_backend
```

### CMake options

目标选项：

```cmake
CVH_BUILD_NATIVE_BACKEND=OFF
```

废弃选项：

```cmake
CVH_BUILD_FULL_BACKEND
CVH_BUILD_BACKEND_KERNEL_SOURCES
```

其中 `CVH_BUILD_BACKEND_KERNEL_SOURCES` 如果仍有必要保留，应改为 native backend 的内部细分开关，而不是公开主模式开关。

### 编译宏

目标宏：

```cpp
CVH_LITE=1
CVH_NATIVE=1
```

语义：

- 未显式启用 native 时，默认 `CVH_LITE=1`。
- 链接 `cvh::native` 时，target 对外传播 `CVH_NATIVE=1`。
- `CVH_LITE` 与 `CVH_NATIVE` 不应同时定义。

废弃宏：

```cpp
CVH_FULL
```

兼容期内可以让 `CVH_FULL` 映射到 `CVH_NATIVE`，但必须给出 deprecated 说明。

## 默认构建策略

当前默认行为应调整为：

```cmake
option(CVH_BUILD_NATIVE_BACKEND "Build optional native backend" OFF)
```

默认配置：

```bash
cmake -S . -B build
cmake --build build -j
```

默认只保证：

- `cvh::headers` 可用。
- Lite smoke/core/imgproc/imgcodecs/highgui fallback 测试可运行。
- 不编译 `src/`。
- 不创建 `cvh::native` target。

显式启用 native：

```bash
cmake -S . -B build-native -DCVH_BUILD_NATIVE_BACKEND=ON
cmake --build build-native -j
```

此时才构建：

- `src/core/*.cpp`
- `src/imgproc/*.cpp`
- `src/highgui/*.cpp`
- macOS Cocoa / Linux X11 framebuffer / Windows Win32 平台实现
- xsimd/OpenMP/dispatch 相关增强路径

## 目录语义

保持当前物理目录也可以，但需要在文档中固定语义：

```text
include/
  header-only public API and Lite fallback implementation

src/
  optional native backend sources, built only when CVH_BUILD_NATIVE_BACKEND=ON
```

如果后续要进一步强化语义，可以考虑把 `src/` 拆成：

```text
native/
  core/
  imgproc/
  highgui/
```

但第一阶段不建议移动目录，避免一次性改动过大。先完成命名和默认行为改造即可。

## 迁移步骤

### P0：文档与语义冻结

- README 第一屏明确：
  - Lite/header-only 是默认路径。
  - native backend 是可选编译后端。
  - 项目不是 OpenCV 全量替代。
- `doc/design.md` 中把 `Full` 改为 `Native backend`。
- 新增本文件作为迁移计划。

### P1：CMake target 重命名

- 新增 `CVH_BUILD_NATIVE_BACKEND`，默认 `OFF`。
- 新增 `cvh_native_backend` target。
- 新增 alias：

```cmake
add_library(cvh::native ALIAS cvh_native_backend)
```

- `cvh::headers` 保持不变。
- 暂时保留兼容 alias：

```cmake
add_library(cvh::full ALIAS cvh_native_backend)
add_library(cvh::full_backend ALIAS cvh_native_backend)
```

- 兼容 alias 只保留一个过渡期，文档标注 deprecated。

### P2：宏与模式检查重命名

- `CVH_FULL` 改为 `CVH_NATIVE`。
- `include/cvh/detail/config.h` 中模式检查改为：

```cpp
#if defined(CVH_LITE) && defined(CVH_NATIVE)
#error "CVH_LITE and CVH_NATIVE cannot be enabled at the same time"
#endif

#if !defined(CVH_LITE) && !defined(CVH_NATIVE)
#define CVH_LITE 1
#endif
```

- 兼容期内允许：

```cpp
#if defined(CVH_FULL) && !defined(CVH_NATIVE)
#define CVH_NATIVE 1
#endif
```

兼容映射应有明确废弃注释。

### P3：测试目标改名

当前测试命名里的 `full` 应改为 `native`：

```text
cvh_full_backend_smoke      -> cvh_native_backend_smoke
cvh_mode_full_smoke         -> cvh_mode_native_smoke
cvh_resize_dispatch_full_smoke -> cvh_resize_dispatch_native_smoke
cvh_test_core               -> cvh_test_core_native
```

Lite 测试保持：

```text
cvh_test_core_lite
cvh_test_imgproc
cvh_lite_pipeline_smoke
```

需要注意：`cvh_test_imgproc` 当前链接 `cvh::headers`，它本质是 Lite/header-only 测试，应在文档或 target 名中明确这一点。后续可改名为：

```text
cvh_test_imgproc_lite
cvh_test_imgproc_native
```

### P4：脚本改名

建议新增脚本：

```text
scripts/ci_native_all.sh
```

保留兼容脚本：

```text
scripts/ci_full_all.sh
```

兼容脚本只调用 `ci_native_all.sh`，并输出 deprecated 提示。

Lite 脚本保持：

```text
scripts/ci_lite_all.sh
```

### P5：安装与包导出

安装导出目标应包含：

```cmake
cvh::headers
cvh::native
```

当 `CVH_BUILD_NATIVE_BACKEND=OFF` 时，安装包只包含 `cvh::headers`。

当 `CVH_BUILD_NATIVE_BACKEND=ON` 时，安装包额外包含 `cvh::native`。

兼容导出 `cvh::full` 可短期保留，但应标注 deprecated。

## README 推荐表述

建议 README 顶部使用以下语义：

```text
opencv-header-only (cvh) is an OpenCV-style C++ vision library.

The default Lite path is header-only: include headers from include/ and no library build is required.
This repository also provides an optional native backend for compiled platform integration,
dispatch, and selected optimized implementations.
```

模式说明建议改为：

```text
Modes:

- Lite: default header-only mode.
- Native backend: optional compiled backend, enabled with CVH_BUILD_NATIVE_BACKEND=ON.
```

避免继续使用：

```text
Full mode
legacy transition path
```

如果仍需表达历史来源，可以放到迁移说明中，而不是 README 第一屏主叙述。

## 兼容策略

建议兼容期为 1 到 2 个小版本。

兼容期内：

- `CVH_BUILD_FULL_BACKEND` 仍可用，但会转发到 `CVH_BUILD_NATIVE_BACKEND`。
- `cvh::full` 仍可用，但 alias 到 `cvh::native`。
- `CVH_FULL` 仍可用，但映射到 `CVH_NATIVE`。
- 文档与 CMake configure 输出中标注 deprecated。

兼容期结束后：

- 移除 `CVH_BUILD_FULL_BACKEND`。
- 移除 `cvh::full` / `cvh::full_backend`。
- 移除 `CVH_FULL`。

## 验收标准

### Lite 默认验收

在默认配置下：

```bash
cmake -S . -B build-lite-default
cmake --build build-lite-default -j
ctest --test-dir build-lite-default --output-on-failure
```

必须满足：

- 不编译 `src/` 中的任何 `.cpp/.mm`。
- 不生成 `cvh_native_backend`。
- `cvh::headers` 可用。
- include-only smoke 通过。
- Lite pipeline smoke 通过。
- core-lite/imgproc/imgcodecs/highgui fallback 合约通过。

### Native 显式验收

在显式启用 native backend 时：

```bash
cmake -S . -B build-native -DCVH_BUILD_NATIVE_BACKEND=ON
cmake --build build-native -j
ctest --test-dir build-native --output-on-failure
```

必须满足：

- 构建 `cvh_native_backend`。
- 提供 `cvh::native`。
- native dispatch smoke 通过。
- native core/imgproc/highgui 后端测试通过。
- 兼容期内 `cvh::full` alias 仍可链接。

## 风险与边界

- `native` 不是性能承诺词，不暗示所有 case 都比 OpenCV 快。
- `native` 表示需要针对当前平台编译和链接的本地后端。
- 项目仍不承诺完整替代 OpenCV。
- Lite 中未支持的能力应明确报错，不应静默依赖 native。
- native backend 可以增强覆盖度、平台能力、dispatch 和性能，但不能成为公开 API 可用性的唯一来源。

## 最终状态

最终希望用户形成稳定认知：

```text
include only -> cvh::headers -> header-only Lite
need compiled platform/dispatch/optimization -> opt in cvh::native
```

这能最大限度消除项目名 `opencv-header-only` 与仓库内部 `src/` 实现之间的认知不对称。
