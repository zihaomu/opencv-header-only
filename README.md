# opencv-header-only

A lightweight OpenCV-like subset library with dual mode:

- `CVH_LITE`: header-first fallback mode (default)
- `CVH_FULL`: linked backend enhancement mode (`src/`)

本库旨在提供高频 OpenCV 风格能力的轻量子集。项目目标是：

- 默认 Lite 模式可运行，满足基础图像处理链路
- 用户尽量仅将命名空间从 `cv` 改为 `cvh` 即可迁移常见用法
- 需要更高性能/更广覆盖时，链接 Full backend 获取增强实现

## Compatibility

- 对外承诺：`API/行为兼容`（优先覆盖高频 OpenCV 用法）
- 不承诺：`ABI/内存布局兼容`（例如 `cvh::Mat` 与 `cv::Mat` 对象内部实现可不同）
- 当前 v1 路线：`Mat-only` 接口优先，不引入 `OutputArray/InputArray` 兼容层
- 因此凡依赖 `OutputArray` 的 upstream 用例，按“设计性不对齐”台账跟踪，而非实现缺陷

## Build (current temporary layout)

当前仓库是过渡布局：

- `include/`：公开 API 与 fallback 入口
- `src/`：Full backend 过渡实现

构建命令：

```bash
cmake -S . -B build
cmake --build build -j
cmake --build build --target test
```

推荐脚本：

```bash
./scripts/ci_core_basic.sh
./scripts/ci_imgproc_quick_gate.sh
```

可通过环境变量设置 warning 预算（默认 `0`）：

```bash
CVH_WARNING_BUDGET=0 ./scripts/ci_core_basic.sh
```

Trust gate policy 单一配置源位于 `benchmark/gate_policy.json`。

## Build Options

- `CVH_BUILD_FULL_BACKEND=ON/OFF`：是否构建 Full backend（默认 `ON`）
- `CVH_BUILD_BACKEND_KERNEL_SOURCES=ON/OFF`：兼容保留开关（默认 `ON`）
- `CVH_BUILD_TESTS=ON/OFF`：是否构建测试目标（默认 `ON`）

## Install & Package

安装（默认前缀可通过 `CMAKE_INSTALL_PREFIX` 覆盖）：

```bash
cmake -S . -B build-release -DCVH_BUILD_TESTS=OFF -DCVH_BUILD_BENCHMARKS=OFF
cmake --build build-release -j
cmake --install build-release
```

安装后可通过 CMake package 使用：

```cmake
find_package(opencv_header_only CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE cvh::headers)
# 若构建并安装了 Full backend，可链接：
# target_link_libraries(your_target PRIVATE cvh::full_backend)
# 或
# target_link_libraries(your_target PRIVATE cvh::full)
```

## Mode Semantics

- 未显式定义模式宏时，头文件默认进入 `CVH_LITE`
- 链接 Full backend 目标时，会通过编译定义启用 `CVH_FULL`
- `CVH_LITE` 与 `CVH_FULL` 互斥

## Current Test Targets

- `cvh_header_compile_smoke`
- `cvh_include_only_smoke`
- `cvh_lite_pipeline_smoke`（纯头文件链路，验证：`imread -> resize -> cvtColor -> threshold -> imwrite`）
- `cvh_resize_dispatch_lite_smoke`（Lite 下 `resize/cvtColor/threshold` dispatch 入口保持 fallback）
- `cvh_mode_lite_smoke`
- `cvh_mode_full_smoke`（仅 Full backend 构建时）
- `cvh_resize_dispatch_full_smoke`（仅 Full backend 构建时，验证 `resize/cvtColor/threshold` backend 注册生效）
- `cvh_full_backend_smoke`（仅 Full backend 构建时）
- `cvh_test_core`（仅 Full backend 构建时）


## 目前存在的问题
1. xsimd处理3通道图片的问题
由于选择的xsimd这种跨平台指令集，其对连续缓冲区的 aligned/unaligned load/store 和 batch 运算有着明显的加速，但是对于3通道图片的滤波算法支持能力有限，因为3通道图片的滤波或者rgb2bgr会有很多interleaved的操作，这种操作是xsimd不擅长的部分，主要原因是各家对于interleaved的支持情况不同导致的。 而3通道图片又是最常用的图片格式，针对这部分问题，目前有待解决。

## Release Docs

- 版本号来源：`VERSION.txt`
- 变更日志：`CHANGELOG.md`
- 发布流程：`doc/release.md`
