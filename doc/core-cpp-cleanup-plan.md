# Core C++ Cleanup And Header-only Migration Plan

## 背景

`opencv-header-only` 的公共产品目标是纯 header-only，但 `core` 中仍有一批
公开 API 只有声明，实际实现位于 `src/core/*.cpp`。当前最直接的例子是：

- `add` / `subtract` / `multiply` / `divide`
- `transpose` / `transposeND`
- `gemm` / `gemm_pack_b`

旧 benchmark 可以通过 `cvh::native` 或直接链接 legacy 静态库得到结果，但
这种结果不能代表 `cvh::headers_fast`。在继续扩展 OpenCV upstream compare
之前，必须先清理这些 `.cpp` 的职责，并把被接受的实现迁入真正的 header-only
路径。

本计划不是从零重写 core kernel。原则是复用现有实现、减少重复来源，并在迁移
过程中收窄历史代码的职责。

## 当前事实

截至 2026-07-23，`src/core` 的主要实现规模如下：

| 文件 | 行数约值 | 当前职责 | 初步处理方向 |
|---|---:|---|---|
| `basic_op_scalar.cpp` | 1428 | Mat-Mat/Mat-Scalar 算术、比较、归约和 transpose 公共入口 | 拆分并迁移已接受算子，不整文件复制 |
| `mat_gemm.cpp` | 659 | GEMM shape、packing、FP32/FP16/INT8 kernel 和公共入口 | 按 shape/pack/kernel/API 分层迁移 |
| `kernel/transpose_kernel.cpp` | 143 | blocked 2D transpose kernel | 迁为 inline detail kernel |
| `mat_expr.cpp` | 977 | `MatExpr` 运算符和求值 | 在直接算术 API 稳定后单独处理 |
| `mat.cpp` | 1148 | 历史 Mat 实现 | 与 `mat_lite_impl.h` 对照审计，不直接迁移 |
| `mat_convert.cpp` | 332 | 历史 convert 实现 | 与 header-only `Mat::convertTo` 对照审计 |
| `system.cpp` | 246 | error/system 实现 | 与 `system.inl.h` 对照审计 |
| `utils.cpp` | 309 | shape/string/utility 实现 | 分类为公共 header helper 或仅开发代码 |
| `memory_utils.cpp` | 52 | 内存辅助函数 | 当前不在主要 source list，检查引用后决定删除或迁移 |

已经存在的 header-only 基础包括：

- `include/cvh/core/mat_lite_impl.h`
- `include/cvh/core/system.inl.h`
- `include/cvh/core/detail/parallel_runtime.h`
- `include/cvh/core/detail/dispatch_control.h`
- `include/cvh/core/simd/opencv_ui.h`

因此不能把 `mat.cpp`、`system.cpp` 等文件机械地搬进 `include/`；必须先识别与
现有 header 实现重复、冲突或已经失效的代码。

## 清理目标

1. `cvh::headers` 和 `cvh::headers_fast` 能独立提供被接受的 core 计算 API。
2. 同一算子只保留一个实现来源，不长期维护 `.cpp` 和 `.hpp` 两套副本。
3. header 中所有非模板定义满足 ODR，使用 `inline`、模板或 `constexpr`。
4. benchmark 不 include `.cpp`，不链接 legacy core 静态库。
5. 未完成迁移的 API 保持明确 WIP，不伪装成 header-only supported。
6. 清理完成后再扩展 Mode B 的 Mat 计算性能报告。

## 非目标

- 本轮不同时重写所有 core 算法。
- 本轮不追求 add/sub/mul/div/GEMM/transpose 的最终 SIMD 性能。
- 本轮不重新引入 xsimd。
- 本轮不启用 RVV。
- 本轮不把历史 `native` 层重新定义为项目产品层。
- 本轮不顺带迁移 softmax、SiLU、RMSNorm、RoPE 等全部 AI kernel。

## 目标目录结构

建议将被接受的实现拆到 `include/cvh/core/detail/`，公共头只保留 API 和最终
include：

```text
include/cvh/core/
  basic_op.h
  mat.h
  detail/
    binary_op_common.hpp
    binary_op_scalar.hpp
    transpose_kernel.hpp
    transpose_impl.hpp
    gemm_shape.hpp
    gemm_pack.hpp
    gemm_kernel_scalar.hpp
    gemm_impl.hpp
```

这只是职责结构，不要求一次创建全部文件。文件只有在确实减少复杂度或避免
重复时才拆分。

禁止采用以下结构：

```text
include/cvh/core/basic_op.h -> #include "../../../src/core/basic_op_scalar.cpp"
```

也禁止 benchmark 自己复制一份简化算术或 GEMM kernel，因为那测到的不是公共
`cvh::headers_fast` API。

## 实施阶段

### Cpp-Clean-0：建立 source ownership 清单

状态：TODO。

任务：

- 逐个记录 `src/core` 文件被哪些 CMake target、测试和 benchmark 使用。
- 识别没有进入 source list 或没有引用者的文件。
- 对照 `mat_lite_impl.h`、`system.inl.h`，标出重复定义。
- 为公开 API 建立“声明位置、当前定义位置、目标定义位置”表。
- 禁止在清理期间新增对 `cvh::native` 的 benchmark 依赖。

完成条件：

- 每个 `src/core/*.cpp` 都有明确的 `migrate`、`keep-dev-only` 或 `delete`
  分类。
- add/sub/mul/div/gemm/transpose 的依赖链完整可见。

### Cpp-Clean-1：清理 Mat/system 重复实现边界

状态：TODO。

任务：

- 对比 `mat.cpp` 与 `mat_lite_impl.h` 的公开方法覆盖范围。
- 对比 `mat_convert.cpp` 与 header-only `Mat::convertTo`。
- 对比 `system.cpp` 与 `system.inl.h`。
- 对 `utils.cpp`、`memory_utils.cpp`、`define.impl.h`、`mat_utils.h` 做引用审计。
- 删除确认无引用且无独立价值的 dead code；仍需保留的开发代码明确注释边界。

完成条件：

- 不存在同一 header-only Mat/system API 的两套活跃实现来源。
- CMake source list 不再包含已经由 header-only 实现完全取代的 core 文件。

### Cpp-Clean-2：迁移常用逐元素算术

状态：TODO。

首批范围：

- `add(Mat, Mat, Mat&)`
- `subtract(Mat, Mat, Mat&)`
- `multiply(Mat, Mat, Mat&)`
- `divide(Mat, Mat, Mat&)`

任务：

- 从 `basic_op_scalar.cpp` 抽取已有 shape/type 检查和 typed loop。
- 抽取公共 destination 创建、连续/非连续遍历和 saturating cast 逻辑。
- 将非模板公共入口改为 `inline`，由 `basic_op.h` 引入。
- 保留 scalar fallback；后续 SIMD 必须在相同公共入口内 dispatch。
- 第一批只接纳已经有明确 OpenCV 兼容语义的 depth/channel 组合。
- Mat-Scalar、compare、merge/split 等后续按相同模式逐批接纳，不阻塞首批。

完成条件：

- 只链接 `cvh::headers` 的程序可调用首批四个 API。
- 两个 translation unit 同时包含 `cvh.h` 时无重复符号。
- 不再需要 `basic_op_scalar.cpp` 提供这四个公开符号。

### Cpp-Clean-3：迁移 transpose

状态：TODO。

任务：

- 将 `transpose2d_kernel_blocked` 迁到 inline detail header。
- 复用现有 tiled kernel，不重新实现另一套 transpose。
- 移除对 `src/core/kernel/transpose_kernel.h` 私有 include 路径的依赖。
- 迁移 `transpose` / `transposeND` 公共入口和 shape/order 校验。
- 保持 scalar blocked fallback；NEON/AVX fast-path 后续单独 benchmark 后接入。

完成条件：

- `transpose` / `transposeND` 只链接 `cvh::headers` 可用。
- 连续、ROI/non-contiguous、C1/C3/C4 和非方阵正确性通过。
- `transpose_kernel.cpp` 不再是公共功能必需项。

### Cpp-Clean-4：拆分并迁移 GEMM

状态：TODO。

首批接纳范围：

- `CV_32F`、二维、单 batch 的 NN GEMM。
- 明确是否包含 `gemm_pack_b`；如果包含，报告必须区分 pack-once 和
  end-to-end。

任务：

- 从 `mat_gemm.cpp` 复用现有 shape/broadcast、packing 和 scalar kernel。
- 将 shape 校验、packing、kernel、公共 API 分离，避免一个 600+ 行 header。
- 所有 header 定义满足 ODR。
- FP16、INT8、broadcast 和 prepack 在首批 FP32 NN 稳定后逐项接纳。
- 不因迁移顺便改变现有 GEMM 数值语义。

完成条件：

- 首批 GEMM 只链接 `cvh::headers` 可用。
- naive、packed 和 OpenCV `cv::gemm` 的输入语义在测试中明确区分。
- `mat_gemm.cpp` 不再是已接纳 GEMM 路径的必需链接项。

### Cpp-Clean-5：处理 MatExpr 和运算符

状态：TODO。

任务：

- 直接 API 稳定后，再审计 `mat_expr.cpp` 的 977 行表达式实现。
- 避免 MatExpr 再复制一套 add/sub/mul/div 算术 kernel。
- 运算符求值统一调用已经接纳的公共算术入口。
- 未接纳的表达式继续保持 WIP，不通过 legacy 链接补齐。

完成条件：

- `a + b`、`a - b`、`a * b`、`a / b` 与直接 API 共用实现。
- expression test 只链接 `cvh::headers`。

### Cpp-Clean-6：删除 legacy core 链接逃生口

状态：TODO。

任务：

- 从 CMake 清理已被 header 实现取代的 `src/core/*.cpp`。
- 删除 `cvh_benchmark_compare_lite` 直接链接 native 静态库的历史做法。
- 增加检查，防止 header-only benchmark 引入 `cvh_native_backend`。
- staged install 后使用安装目录重新编译 core smoke。

完成条件：

- `CVH_BUILD_NATIVE_BACKEND=OFF` 时 core accepted API、测试和 benchmark 完整。
- 安装包不引用 `src/` 路径。
- `nm`/link map 中没有来自 legacy core 静态库的 accepted API 符号。

### Cpp-Clean-7：正确性接纳后恢复性能测试

状态：TODO。

顺序：

1. public API compile/link smoke。
2. 多 translation unit ODR test。
3. 与 upstream OpenCV 的正确性 contract test。
4. Mode A 旧版/current header-only 回归。
5. Mode B `cvh::headers_fast`/upstream OpenCV 性能对比。

Mode B 首批计算矩阵：

- add/sub/mul/div：`CV_8U`、`CV_32F`，C1/C3，VGA/720p/1080p。
- transpose：`CV_8U`、`CV_32F`，方阵/非方阵、连续/ROI。
- GEMM：FP32 `128^3`、`256^3`、`512^3`；分别记录 end-to-end 和
  pack-once（若已接纳）。

完成条件：

- 报告中的实现仍然只有 `cvh_headers_fast` 与 `opencv`。
- 没有 fast-path 的 accepted 算子记录为 `headers_baseline`，不跳过。
- 未迁移算子记录为 `UNSUPPORTED`，不得链接 `.cpp` 伪造结果。
- 更新日期命名的 OpenCV upstream performance Markdown。

## 每阶段验证

每个迁移阶段至少执行：

```bash
cmake -S . -B build-core-clean \
  -DCMAKE_BUILD_TYPE=Release \
  -DCVH_BUILD_NATIVE_BACKEND=OFF \
  -DCVH_BUILD_TESTS=ON \
  -DCVH_BUILD_BENCHMARKS=ON

cmake --build build-core-clean -j
ctest --test-dir build-core-clean --output-on-failure
./scripts/check_header_only_contract.sh
```

还需要单独验证：

- direct include 模式，不依赖仓库 `src/`。
- staged install 后的 downstream CMake consumer。
- 至少两个 translation unit 的 ODR/link test。
- ARM NEON 和 x86 SSE/AVX 编译 smoke。

## 风险与约束

### 编译时间和 header 体积

把 2000 行以上实现直接塞进公共头会显著增加编译成本。迁移必须按算子拆分，
并避免无关 API 自动包含全部 GEMM/AI kernel。

### ODR

匿名 namespace helper 在 header 中会产生每 TU 副本；大型 lookup table 或
全局状态必须使用 `inline` variable、函数内静态或模板，不能产生重复定义。

### 行为漂移

清理阶段优先保持现有语义。任何与 OpenCV rounding、saturation、zero divide、
reshape/transpose layout 不一致的问题，应作为独立 correctness 修复记录。

### benchmark 污染

benchmark 只能调用公共 API。direct kernel microbenchmark 可以用于诊断，但不能
替代公共入口结果，也不能作为 Mode B 的主报告数据。

## 最终完成定义

本轮 core C++ 清理完成需要同时满足：

- add/sub/mul/div/transpose 和首批 GEMM 具有真实 header-only 定义。
- accepted API 不再依赖 `src/core/*.cpp`。
- legacy 重复实现已删除或明确隔离。
- header-only、ODR、安装包和正确性测试通过。
- Mode B 能在不启用 legacy compiled layer 的情况下生成 Mat 计算结果。
- README 的 core operator 状态由 WIP 更新为与实际接纳范围一致。
