# opencv-header-only
A lightweight header-only wrapper for OpenCV, enabling easy integration without building or linking against OpenCV binaries.


本库旨在提供一个最轻量的opencv库，能够通过头文件引入，解决用户80%的需求。
主要目标：用户不需要修改原始opencv代码，而只需要将命名空间从cv改成cvh就能兼容目前80%的opencv代码需求，并且能够达到80%的性能。

包含模块：
- core
    - 实现基本的Mat内存和计算。
- imgproc
    - 实现主要的图像处理库
- imgcodecs
    - 仅实现部分图像格式的读写操作

实现计划在doc中，主要api对齐opencv，一步步实现中，详细实施计划见：doc/header-only-opencv-plan.md

## Build (current temporary layout)

当前仓库使用 `include/` + `src/core/` 的临时过渡布局：

- `include/`：对外头文件与兼容头。
- `src/core/`：从旧工程迁移过来的 `.cpp` 临时实现。

构建命令：

```bash
cmake -S . -B build
cmake --build build -j
```

默认会构建：

- `cvh_headers`（接口头目标）
- `cvh_legacy_core`（`src/core` 临时静态库）
- `cvh_header_compile_smoke`、`cvh_legacy_core_smoke`（最小 smoke 程序）

可选开关：

- `CVH_BUILD_LEGACY_CORE=ON/OFF`：是否构建 `src/core` 临时静态库（默认 `ON`）
- `CVH_BUILD_BACKEND_KERNEL_SOURCES=ON/OFF`：是否编译依赖旧 backend 的源文件（默认 `OFF`）
- `CVH_BUILD_SMOKE_TESTS=ON/OFF`：是否构建 smoke 程序（默认 `ON`）
- `CVH_BUILD_LEGACY_TESTS=ON/OFF`：旧测试入口预留开关（当前未接线，默认 `OFF`）
