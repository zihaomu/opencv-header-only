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
cmake --build build --target test
```

推荐使用统一脚本（本地与 CI 一致）：

```bash
./scripts/ci_smoke.sh
```

可选 core 基础测试脚本（含 warning 计数）：

```bash
./scripts/ci_core_basic.sh
```

可通过环境变量设置 warning 预算（默认 `0`）：

```bash
CVH_WARNING_BUDGET=0 ./scripts/ci_core_basic.sh
```

默认会构建：

- `cvh_headers`（接口头目标）
- `cvh_header_compile_smoke`（最小 smoke 程序）
- `cvh_include_only_smoke`（仅 `-Iinclude` 的 smoke 程序）

公共头依赖检查：

```bash
./scripts/check_public_headers.sh
```

可选开关：

- `CVH_BUILD_LEGACY_CORE=ON/OFF`：是否启用 legacy core 预留开关（当前未接线，默认 `OFF`）
- `CVH_BUILD_BACKEND_KERNEL_SOURCES=ON/OFF`：是否编译依赖旧 backend 的源文件（默认 `OFF`）
- `CVH_BUILD_SMOKE_TESTS=ON/OFF`：是否构建 smoke 程序（默认 `ON`）
- `CVH_BUILD_LEGACY_TESTS=ON/OFF`：是否构建迁移中的 core 基础测试子集（默认 `OFF`）

构建 core 基础测试子集：

```bash
cmake -S . -B build-core -DCVH_BUILD_LEGACY_TESTS=ON -DCVH_BUILD_SMOKE_TESTS=OFF
cmake --build build-core -j
cmake --build build-core --target test
```
