# transpose x86 TODO（历史归档）

日期：2026-04-19

P5.3.5 更新：本文件只保留历史排查背景，不再作为 TODO 或 handoff 执行。`transpose` 的 xsimd 分支、`CVH_TRANSPOSE_XSIMD_PROBE_LOG`、`DispatchMode::XSimdOnly` / `DispatchTag::XSimd` 已在 P5.3.3 删除；`include/cvh/3rdparty/xsimd/` vendor 目录已在 P5.3.4 删除。

当前公开性能路线以 `cvh::headers_fast` 的 OpenCV Universal Intrinsics fast-path 为准。新的 `transpose` 性能工作必须重新进入 header-only 层并通过 correctness gate 与 benchmark gate，不能复活旧 xsimd `.cpp` 实验路径。

## 历史问题

在 GitHub Actions（Ubuntu + x86_64）上曾出现两类失败：

1. `GemmPackContract_TEST.fp32_packed_transposed_input_matches_reference`
2. `MatContract_TEST.transpose2d_preserves_interleaved_bytes_for_multi_type_multi_channel`

两者当时都依赖 `transpose(b)` 路径，问题集中在 `transpose2d_kernel_blocked()` 的 xsimd 分支。

当时的临时策略是为不同 `elem_size` 做 probe/cache，probe 失败时回退到 fallback；后续又尝试修复 lane 读写和 `xsimd::transpose` 使用方式。这些实现均属于历史 `.cpp` 实验路径，已被 P5.3 removal 取代。

## 当前状态

- `src/core/kernel/transpose_kernel.cpp` 不再包含 xsimd transpose、probe cache 或 `CVH_TRANSPOSE_XSIMD_PROBE_LOG`。
- legacy `.cpp` transpose 路径保留 tiled/memcpy scalar fallback。
- benchmark dispatch 参数只剩 `auto|scalar-only`。
- `test/core/mat_contract_test.cpp` 的相关用例已经收口为 scalar-only 正确性验证。

## 保留价值

本文件只解释“为什么 xsimd transpose 不应继续维护”。后续不要按旧 P0-P3 TODO 继续投入；如果 `transpose` 重新成为热点，应另开 OpenCV Universal Intrinsics 或 direct platform intrinsics 的 header-only 计划。
