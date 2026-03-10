# `src/core` 迁移台账（P0-05）

- 更新时间：2026-03-10
- 目标：为 `src/core` 每个文件给出迁移状态、去向和删除条件，防止迁移长期失控。

## 状态定义

- `已迁移`：已迁到 `include/`，`src/` 对应文件已移除或仅保留转发。
- `待迁移`：仍在 `src/`，后续需迁移到 header-only 结构。
- `待重写`：现实现不适合直接迁移，需要先重构再迁移。
- `待删除`：占位或废弃文件，确认无引用后删除。

## 本次已完成的低风险迁移闭环

- `src/core/kernel/openmp_utils.h` -> `include/cvh/core/detail/openmp_utils.h`（已迁移）
- `src/core/kernel/xsimd_kernel_utils.h` -> `include/cvh/core/detail/xsimd_kernel_utils.h`（已迁移）

## 文件级台账

| 源文件 | 当前状态 | 目标去向 | 删除条件/下一步 |
|---|---|---|---|
| `src/core/basic_op.cpp` | 待重写 | `include/cvh/core/basic_op*.h` + `detail/` | 完成 API/行为清理（当前含历史算子与模板问题）后迁移 |
| `src/core/define.impl.h` | 待迁移 | `include/cvh/core/detail/define_impl.h` | `memory_utils.h` 完成迁移后可移除 `src` 版本 |
| `src/core/mat.cpp` | 待迁移 | `include/cvh/core/mat.inl.h` 或 `detail/mat_impl.h` | `Mat` 合同冻结后分批迁移 |
| `src/core/mat_convert.cpp` | 待迁移 | `include/cvh/core/detail/mat_convert.h` | `convertTo` 行为与类型表稳定后迁移 |
| `src/core/mat_expr.cpp` | 待重写 | `include/cvh/core/detail/mat_expr.h` | 先修复表达式分派问题，再迁移 |
| `src/core/mat_gemm.cpp` | 待重写 | `include/cvh/core/detail/mat_gemm.h` | 先收敛依赖与行为，再做 header-only 化 |
| `src/core/mat_utils.h` | 待删除 | N/A | 当前为空壳，确认无引用后删除 |
| `src/core/memory_utils.h` | 待迁移 | `include/cvh/core/detail/memory_utils.h` | 清理 `define.impl.h` 依赖后迁移 |
| `src/core/memory_utils.cpp` | 待迁移 | `include/cvh/core/detail/memory_utils.h` | `memory_utils.h` 迁移时一并内联或改为可选源 |
| `src/core/system.cpp` | 待迁移 | `include/cvh/core/detail/system_impl.h` | 错误消息与格式化逻辑收敛后迁移 |
| `src/core/utils.cpp` | 待迁移 | `include/cvh/core/detail/utils_impl.h` | 对齐 `core` API 边界后迁移 |
| `src/core/kernel/binary_kernel_xsimd.h` | 待迁移 | `include/cvh/core/detail/kernel/binary_kernel_xsimd.h` | 随 `.cpp` 一起迁移 |
| `src/core/kernel/binary_kernel_xsimd.cpp` | 待迁移 | `include/cvh/core/detail/kernel/binary_kernel_xsimd.h` | 完成模板/接口稳定后迁移 |
| `src/core/kernel/gemm_kernel_xsimd.h` | 待迁移 | `include/cvh/core/detail/kernel/gemm_kernel_xsimd.h` | 随 `.cpp` 一起迁移 |
| `src/core/kernel/gemm_kernel_xsimd.cpp` | 待迁移 | `include/cvh/core/detail/kernel/gemm_kernel_xsimd.h` | 与 `mat_gemm` 一起规划迁移 |
| `src/core/kernel/normalization_kernel_xsimd.h` | 待迁移 | `include/cvh/core/detail/kernel/normalization_kernel_xsimd.h` | 随 `.cpp` 一起迁移 |
| `src/core/kernel/normalization_kernel_xsimd.cpp` | 待迁移 | `include/cvh/core/detail/kernel/normalization_kernel_xsimd.h` | 与 `basic_op` 清理同步推进 |
| `src/core/kernel/transpose_kernel.h` | 待迁移 | `include/cvh/core/detail/kernel/transpose_kernel.h` | 随 `.cpp` 一起迁移 |
| `src/core/kernel/transpose_kernel.cpp` | 待迁移 | `include/cvh/core/detail/kernel/transpose_kernel.h` | 与 `transpose` API 收敛后迁移 |

## 阶段性验收记录（2026-03-10）

- 已完成“每文件状态可追踪”台账。
- 已完成 2 个低风险头文件迁移闭环。
- 迁移后验证：
  - `./scripts/ci_smoke.sh` 通过
  - `CVH_BUILD_LEGACY_TESTS=ON` 的 `cvh_core_basic_tests` 通过
