# transpose x86 TODO（handoff）

日期：2026-04-19  
适用场景：你切换到另一台机器（Ubuntu/x86_64）继续排查 `transpose` 的 xsimd 正确性问题。

## 1. 背景与当前状态

在 GitHub Actions（Ubuntu + x86_64）上曾出现两类失败：

1. `GemmPackContract_TEST.fp32_packed_transposed_input_matches_reference`
2. `MatContract_TEST.transpose2d_preserves_interleaved_bytes_for_multi_type_multi_channel`

两者都依赖 `transpose(b)` 路径，问题集中在 `transpose2d_kernel_blocked()` 的 xsimd 分支。

当前代码已加“自检 + 按 elem_size 熔断”保护（不是全局禁用）：

- 关键实现位置：`src/core/kernel/transpose_kernel.cpp`
- 自检与缓存状态：约 `188-257` 行
- 分发入口与 xsimd/fallback 决策：约 `261-313` 行

### 2026-04-19（本轮执行进展）

- 已完成 P0：新增 `CVH_TRANSPOSE_XSIMD_PROBE_LOG` 日志开关，可在 CI/本地输出 `elem_size` 的 `probe/cache pass/fail`，并在 probe 失败时输出 `shape + mismatch_byte` 细节。
- 已完成 P1：新增 kernel 级最小复现测试，直接调用 `transpose2d_kernel_blocked()`，覆盖 `elem_size=1/2/4/8` 与形状 `11x29 / 5x7 / 13x29 / 64x65`，并按字节对比 reference。
- 已推进 P2：`transpose2d_xsimd` 改为位级安全的 lane 读写（`memcpy`），且按 `elem_size` 使用更稳定的 lane 类型映射（`1->int8_t`, `2->int16_t`, `4->float`, `8->double`）。在当前 x86 机器上，扩展 probe 形状全部通过。
- 当前运行结果：`cvh_test_core` 全量通过（118 tests: 116 passed, 2 skipped），并且 xsimd probe 在 `elem_size=1/2/4/8` 均为 pass（本机验证）。

## 2. 当前保护逻辑（已落地）

`transpose2d_kernel_blocked()` 会按 `elem_size = elem_size1 * channels` 做一次 probe：

- 仅 `elem_size in {1,2,4,8}` 尝试 xsimd
- probe 通过：允许 xsimd
- probe 失败：该 elem_size 后续直接走 fallback
- `DispatchMode::XSimdOnly` 但 probe 失败时，仍按现有契约抛 `StsNotImplemented`

## 3. 你在新机器上的第一步

先确认当前行为稳定，不回归：

```bash
cmake -S . -B build-x86 -DCVH_BUILD_FULL_BACKEND=ON -DCVH_BUILD_TESTS=ON -DCVH_BUILD_BENCHMARKS=OFF
cmake --build build-x86 --target cvh_test_core -j
./build-x86/cvh_test_core --gtest_filter='GemmPackContract_TEST.*:MatContract_TEST.transpose2d_preserves_interleaved_bytes_for_multi_type_multi_channel:MatContract_TEST.transpose3d_last_two_swap_preserves_interleaved_bytes_for_multichannel_types'
```

再跑全量 core：

```bash
./build-x86/cvh_test_core --gtest_brief=1
```

## 4. 根因修复 TODO（按优先级）

### P0：补可观测性（建议先做）

目标：在 x86 上快速看出每个 `elem_size` 是 `probe-pass` 还是 `probe-fail`。  
建议：在 `transpose_kernel.cpp` 加可控日志（环境变量开关），或把状态写到 benchmark metadata。

验收：
- CI log 能看到 `elem_size=1/2/4/8` 的 probe 结果。

### P1：最小化复现 transpose xsimd 核心问题

目标：把失败从“业务用例”收敛到“kernel 级”。

建议新增测试：
- 直接调用 `transpose2d_kernel_blocked()`。
- 覆盖 `elem_size=1/2/4/8`，形状覆盖：
  - 非方阵：`11x29`
  - 不是 batch 宽度整数倍：`5x7`、`13x29`
  - 大小混合：`64x65`
- 对比 `transpose2d_memcpy_fallback()` 的字节级结果。

验收：
- 在 x86 CI 上能稳定复现出错组合（若仍存在）。

### P2：修复 xsimd transpose（候选方向）

候选方向：

1. 检查 `xsimd::transpose(matrix, matrix + N)` 的 in-place 语义在 x86 是否可靠。  
2. 若不可靠，改为 out-of-place 暂存再写回。  
3. 对 `N`、尾块处理和 store 顺序做逐步断言（尤其 AVX2 路径）。

验收：
- 不依赖 probe 熔断，x86 下目标 `elem_size` 可稳定走 xsimd 且结果正确。

### P3：逐步放开保护

目标：从“全靠 probe 防守”转到“默认正确、probe 仅兜底”。

建议：
- 先恢复 `elem_size=4`（通常覆盖 `CV_32F/CV_32S`，收益高）
- 通过后再放开 `1/2/8`

验收：
- `cvh_test_core` 全量绿
- `GemmPackContract_TEST` 全绿
- `transpose` 相关回归测试全绿
- quick benchmark 无明显回退

## 5. 最终验收标准

满足以下 4 条即可关闭该 TODO：

1. Ubuntu x86 CI 上 `cvh_test_core` 稳定通过（多次重跑）。  
2. `GemmPackContract_TEST.fp32_packed_transposed_input_matches_reference` 稳定通过。  
3. `MatContract_TEST.transpose2d_preserves_interleaved_bytes_for_multi_type_multi_channel` 稳定通过。  
4. `transpose2d` 的 xsimd 路径在目标 elem_size 上恢复（至少 `elem_size=4`）。

## 6. 相关文件

- `src/core/kernel/transpose_kernel.cpp`
- `src/core/basic_op.cpp`
- `src/core/basic_op_scalar.cpp`
- `test/core/mat_contract_test.cpp`
- `test/core/gemm_pack_contract_test.cpp`
