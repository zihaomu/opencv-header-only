# cvh 跨平台并行框架迁移计划（第二次审查版）

- Date: 2026-04-15
- Branch: `main`
- Goal: 引入 `cvh::parallel_for_` 跨平台并行抽象（默认 `std::thread`，可选 OpenMP backend），并替换现有业务代码中的 OpenMP 直接使用。

## 1. 已确认决策

1. 方案路线: 工程化主线（先抽象并行层，再分阶段替换调用点）。
2. 审查模式: HOLD SCOPE（不扩 scope，不加 TBB/work-stealing）。
3. 后端策略: 运行时可切换，默认 `std::thread`，可选 `openmp`。
4. 线程创建失败策略: 默认降级串行；`strict` 模式抛异常。
5. 线程数策略: `clamp(1..min(256, hw*4))`。
6. API 策略: OpenCV 风格主接口 + lambda 便捷封装。
7. 分片策略: 默认静态连续分片（contiguous static chunk）。
8. 测试策略: 并行专项测试 + 后端矩阵 + strict/fallback 行为验证。
9. 性能门禁: 全局回退 <=8%，热点（filter/gemm/transpose）回退 <=5%。
10. 发布策略: 三阶段（框架落地 -> 分模块替换 -> 清理遗留 OpenMP 直写）。

## 2. 第二次审查结论（Findings）

### Finding 1 (High): `CV_16F` compare 的 xsimd 路径被 `_OPENMP` 误绑定

- Evidence: [src/core/basic_op_scalar.cpp:565](/home/mzh/work/my_project/opencv-header-only/src/core/basic_op_scalar.cpp:565)
- Problem: `try_dispatch_mat_mat_compare_xsimd_fp16()` 在 `#ifndef _OPENMP` 下直接 `return false`，这把“xsimd 能力”错误地绑定到了 OpenMP 宏。
- Risk: 去 OpenMP 或切换 backend 后出现无声能力回退，行为和性能都可能漂移。
- Action: 迁移时必须把该 gating 从 `_OPENMP` 解耦，改为与并行 backend 无关的 capability 判断。

### Finding 2 (Medium): CMake 的 `CVH_WITH_OPENMP` 与代码中的 `_OPENMP` 语义不一致

- Evidence: [CMakeLists.txt:93](/home/mzh/work/my_project/opencv-header-only/CMakeLists.txt:93), [include/cvh/core/detail/openmp_utils.h:7](/home/mzh/work/my_project/opencv-header-only/include/cvh/core/detail/openmp_utils.h:7)
- Problem: 构建层定义了 `CVH_WITH_OPENMP=1`，但代码主要依赖 `_OPENMP`。
- Risk: 宏语义分裂，后续定位 backend 激活条件困难。
- Action: 新并行框架统一使用一套 runtime/backend 状态，不再让业务代码分散依赖 `_OPENMP`。

### Finding 3 (Medium): GEMM 代码当前依赖 OpenMP `schedule(static)`，替换后需显式保序分片

- Evidence: [src/core/kernel/gemm_kernel_xsimd.cpp:247](/home/mzh/work/my_project/opencv-header-only/src/core/kernel/gemm_kernel_xsimd.cpp:247)
- Problem: 目前分配策略隐式借助 OpenMP 调度语义。
- Risk: 迁移到 `std::thread` 后若分片策略变化，会影响局部性与性能。
- Action: `parallel_for_` 的默认分片明确为 contiguous static chunk，并在 GEMM 回归里单列热点门禁。

### Finding 4 (Medium): 目前缺少“并行 runtime 层”的独立测试面

- Evidence: 现有 smoke 仅覆盖图像算子 dispatch，不覆盖并行 runtime 行为（见 [test/smoke/cvh_resize_dispatch_smoke.cpp](/home/mzh/work/my_project/opencv-header-only/test/smoke/cvh_resize_dispatch_smoke.cpp:1)）。
- Problem: 无法直接验证 backend 切换、strict/fallback、异常传播、nested guard。
- Risk: 迁移后存在 silent failure。
- Action: 新增 `test/core/parallel_for_runtime_test.cpp`（或等价命名）作为框架级测试入口。

## 3. 目标架构

```text
Kernel callsite
  -> cvh::parallel_for_(Range, Body, nstripes)
     -> Partitioner (static contiguous chunks)
     -> Runtime Guard (nested, strict, thread cap)
     -> Backend Adapter
          - std::thread (default)
          - OpenMP (optional)
     -> Join + exception funnel (first exception wins)
```

### 3.1 Shadow Paths

1. Happy path: `work > threshold` -> parallel execute -> join -> return.
2. Nil/invalid path: 非法参数 -> fail-fast（断言/异常）。
3. Empty path: `range` 为空 -> 直接 return。
4. Error path: worker 抛异常 -> 记录首异常 -> join -> rethrow（strict）；非 strict 串行降级仅用于线程创建失败场景。

## 4. API 设计（对齐 OpenCV 风格）

新增文件建议:

- `include/cvh/core/parallel.h`
- `include/cvh/core/detail/parallel_runtime.h`
- `src/core/parallel_runtime.cpp`（如果需要非 header-only 状态管理）

接口建议:

```cpp
namespace cvh {

enum class ParallelBackend {
    Auto = 0,
    StdThread,
    OpenMP
};

class ParallelLoopBody {
public:
    virtual ~ParallelLoopBody() = default;
    virtual void operator()(const Range& range) const = 0;
};

void parallel_for_(const Range& range, const ParallelLoopBody& body, double nstripes = -1.0);

template <class F>
void parallel_for_(const Range& range, F&& fn, double nstripes = -1.0);

void setNumThreads(int nthreads);
int getNumThreads();

void setParallelBackend(ParallelBackend backend);
ParallelBackend getParallelBackend();

void setParallelStrict(bool strict);
bool getParallelStrict();

const char* last_parallel_backend();
int last_parallel_chunks();

}  // namespace cvh
```

## 5. 迁移分阶段计划

### Phase P0: 框架落地（不动业务并行调用点）

1. 新增 `parallel_for_` runtime（默认 `std::thread`）。
2. 引入 OpenMP adapter（`CVH_USE_OPENMP=ON` 且可用时可选）。
3. 迁移 `should_parallelize_1d_loop` 到新 runtime（保留原行为）。
4. 在 `include/cvh/cvh.h` 导出并行 API。
5. 先加 runtime 专项测试，确保框架行为正确。

### Phase P1: 分模块替换 OpenMP 直写

替换顺序（低风险到高风险）:

1. [src/core/kernel/transpose_kernel.cpp](/home/mzh/work/my_project/opencv-header-only/src/core/kernel/transpose_kernel.cpp)
2. [src/core/kernel/normalization_kernel_xsimd.cpp](/home/mzh/work/my_project/opencv-header-only/src/core/kernel/normalization_kernel_xsimd.cpp)
3. [src/core/kernel/binary_kernel_xsimd.cpp](/home/mzh/work/my_project/opencv-header-only/src/core/kernel/binary_kernel_xsimd.cpp)
4. [src/core/kernel/gemm_kernel_xsimd.cpp](/home/mzh/work/my_project/opencv-header-only/src/core/kernel/gemm_kernel_xsimd.cpp)
5. [src/core/mat_gemm.cpp](/home/mzh/work/my_project/opencv-header-only/src/core/mat_gemm.cpp)
6. [src/imgproc/resize_backend.cpp](/home/mzh/work/my_project/opencv-header-only/src/imgproc/resize_backend.cpp)

迁移要求:

1. 业务代码不再出现 `#pragma omp`。
2. 不再在业务代码里 include `<omp.h>`。
3. 所有并行路径统一走 `cvh::parallel_for_`。

### Phase P2: 清理与收口

1. 清理 `openmp_utils` 的旧职责，保留/重命名为 backend adapter。
2. 清理文档和注释里的“直接 OpenMP”描述，改为“parallel_for_ + optional OpenMP backend”。
3. `rg '#pragma omp' src include` 应为 0（adapter 文件例外，若保留则限定在 detail/runtime 层）。

## 6. 测试计划

### 6.1 新增并行 runtime 测试

建议新增:

- `test/core/parallel_for_runtime_test.cpp`

覆盖点:

1. backend 切换（`std::thread`/`openmp`/auto）。
2. `setNumThreads/getNumThreads` clamp 行为。
3. `strict` 与 fallback 的分支行为。
4. worker 异常传播（first exception wins）。
5. nested 并行防过度订阅。
6. 空 range、极小 workload、大 workload 分片边界。

### 6.2 现有算子回归

1. `cvh_test_core`
2. `cvh_test_imgproc`
3. `cvh_resize_dispatch_full_smoke`
4. 现有 filter/gemm/transpose 相关用例全量回归。

## 7. 性能门禁

1. 基线对比: 迁移前后同机同参数 A/B。
2. 全局阈值: `max slowdown <= 8%`。
3. 热点阈值: `filter/gemm/transpose <= 5%`。
4. 若超阈值:
   - 优先检查分片策略与线程上限；
   - 再检查 cache locality 与小任务并行开销；
   - 必要时对热点保留 tuned chunk 规则。

## 8. 部署与回滚策略

1. 每个 Phase 独立提交，避免一次性大 PR。
2. 运行时回滚开关:
   - 先切 backend；
   - 若仍异常，回退最近模块替换提交。
3. 任何性能或稳定性回退先“开关回滚”，再做代码修复。

## 9. Definition of Done

1. 业务层 OpenMP 直写清零（`#pragma omp` 不再出现在 kernel 业务文件）。
2. `cvh::parallel_for_` API 在 core 头文件稳定可用。
3. 后端可运行时切换，默认 `std::thread`。
4. strict/fallback、异常传播、线程数 clamp 均有测试覆盖。
5. benchmark 门禁接入并通过（8%/5% 双阈值）。
6. 文档更新完成，迁移路径与故障回滚说明完整。

## 10. 执行顺序建议（可直接开工）

1. 先实现 P0（框架 + runtime 测试）。
2. 然后按 P1 顺序逐模块替换并每步回归。
3. 最后执行 P2 清理并上性能门禁。

