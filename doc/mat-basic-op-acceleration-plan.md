# Mat Basic-Op Acceleration Plan

- 文档版本：v0.3
- 更新时间：2026-04-06
- 适用范围：`cvh::Mat` 的 basic-op 加速路径，重点覆盖 `Mat-Mat`、`Mat-Scalar`、`compare`、`transpose`
- 目标平台：优先 `arm64` / Apple Silicon，其次保持 `x86 + xsimd` 兼容

## 1. 目标

- 把 `Mat` 从“部分 case 恰好快”补到“主流 `shape/type/channel` 默认快”。
- 优先修复 dispatch 缺口和 benchmark 可见性，再补 kernel 覆盖。
- 保持现有 `xsimd` 路线，不在当前阶段引入新的 SIMD 抽象层。
- 让性能路径对 `type/channel/layout` 的退化边界可预测、可测量、可解释。

## 2. 执行原则

- 按顺序做，不要乱。
- 先补覆盖率，再做平台专项优化。
- 先修 dispatch 和 benchmark，再补 kernel。
- 任何优化都必须带 benchmark 和正确性回归。
- 不允许“优化后默认启用但 benchmark 无法证明收益”。

## 3. 下一步加速方案

### Step 1：先修分发层和 benchmark

- 去掉 `fp16 scalar compare` 对 `_OPENMP` 的依赖。
- benchmark 增加 `scalar-only` / `xsimd-only` 开关。
- benchmark `quick` profile 加入 `CV_32U`。
- benchmark 输出里标明是否命中 SIMD。

#### 目的

- 先把“到底有没有命中 SIMD”这件事讲清楚。
- 避免后续优化时把“算法慢”和“根本没进 SIMD”混在一起。
- 修掉当前 `fp16 compare` 在 macOS/Apple clang 环境下可能被错误退回标量的问题。

#### 完成定义

- `compare(Mat,Scalar)` / `compare(Scalar,Mat)` 的 `CV_16F` 路径不再依赖 `_OPENMP` 才能走 SIMD。
- `cvh_benchmark_core_ops` 可以明确区分 `scalar-only` 和 `xsimd-only`。
- `quick` profile 能覆盖 `CV_32U`。
- 每条 benchmark 记录都能看出当前 case 是否命中 SIMD。

### Step 2：补整数 basic-op 的通用 xsimd kernel

- `CV_8U/8S/16U/16S/32S/32U` 的 `ADD/SUB/MUL` 先补。
- `DIV` 对整数要单独定义语义和零除处理，别混着上。
- `compare` 的 `EQ/NE/GT/GE/LT/LE` 给整数补 `xsimd`。
- 优先补 `Mat-Mat`，然后复用到 `Mat-Scalar`。

#### 目的

- 解决当前整数 basic-op 只有 `MAX/MIN` 真正吃到 `xsimd` 的问题。
- 让整数路径不再因为 `type` 一切换就大面积掉回标量。
- 给 `CV_32U` 补齐当前最明显的空洞。

#### 执行顺序

1. 先补 `Mat-Mat` 的整数 binary kernel 和 compare kernel。
2. 再把同一批 kernel 复用到 `Mat-Scalar` 的 uniform / non-uniform scalar 路径。
3. 最后补 benchmark 和 test，确认每个 `depth + op` 组合的 dispatch 行为。

#### 完成定义

- `CV_8U/8S/16U/16S/32S/32U` 的 `ADD/SUB/MUL` 在连续内存主路径可进 SIMD。
- 整数 compare 的常用比较操作可进 SIMD。
- `Mat-Mat` 与 `Mat-Scalar` 结果一致，且 benchmark 能证明 SIMD 命中。

### Step 3：补 ARM 的 fp16 快路径

- 在 `include/cvh/core/detail/xsimd_kernel_utils.h` 加 `__aarch64__/NEON` 的 half load/store 快路径。
- 这一步会直接改善实测里最慢的 `matmat_add_f16_c1` 和 `matscalar_add_f16_c3_uniform`。

#### 目的

- 解决当前 `fp16` 在 ARM 上主要靠逐元素转 `float` 的问题。
- 把 `fp16` 从“功能可用但转换成本很高”提升到“在 M 系列机器上有实际吞吐收益”。

#### 重点注意

- 不要破坏现有 x86 `f16c` 快路径。
- ARM 和 x86 路径都要保留通用 fallback。
- benchmark 必须分别验证 `Mat-Mat` 和 `Mat-Scalar` 的 `CV_16F` 提升。

#### 完成定义

- `arm64` 下 `load_hfloat_batch` / `store_hfloat_batch` 有明确的 NEON 快路径。
- `CV_16F` 的核心 hot path benchmark 有可观提升。
- `x86` 和非特化平台的现有行为不回退。

### Step 4：明确“容器语义”的边界与基础支持

- 保持 `channels <= 4` 为主要优化合同。
- `channels > 4` 如果不影响现有实现，可复用现有逻辑；如果需要单独处理，只提供基础正确实现或稳定 fallback。
- 给连续内存和常见 interleaved channel 布局做边界梳理。
- 对 `ROI / submat` 至少做到“明确快慢边界”，别让性能随机。

#### 目的

- 让 `Mat` 作为多 `shape/type/channel` 容器时，性能行为更像“合同”，而不是“碰运气”。
- 明确 `channels <= 4` 和 `channels > 4` 的不同支持级别，避免边界含混。
- 避免子视图、布局变化导致性能不可预期。

#### 执行顺序

1. 明确哪些路径保证快，哪些路径允许退回标量。
2. 对 `channels > 4` 明确基础实现/fallback 规则，不把它升级成当前阶段主优化目标。
3. 对最常见连续布局和 interleaved 布局给专门快路径或明确回退边界。
4. 对 `ROI / submat` 建立最小明确规则，至少让 benchmark 和文档能说明白。

#### 完成定义

- `channels <= 4` 的主路径保持为优化重点。
- `channels > 4` 的行为有明确规则：
  - 能直接复用现有实现的 case 继续支持
  - 需要单独处理的 case 只要求基础正确实现或稳定 fallback
- 常见连续多通道 case 有稳定快路径或明确回退边界。
- `ROI / submat` 的快慢边界有文档、有 benchmark、有测试。

### Step 5：最后才考虑手写 NEON

- 只对 microbench 和真实 workload 都证明确实卡住的路径下手。
- 当前预判最有价值的候选只有 `fp16 add/mul/compare` 和少数 `f32 compare`。

#### 目的

- 避免过早平台分叉。
- 只在 `xsimd` 覆盖补齐后，针对剩余热点做定点突破。

#### 完成定义

- 只有在 `xsimd` 路线已经证明覆盖充分、但单个 hot path 仍明显落后时，才进入手写 NEON。
- 每个手写 NEON 路径都必须有独立 benchmark、fallback、正确性回归。

## 4. 验证与实现循环

这一节定义后续每一轮加速工作的固定节奏。目标不是“想到一个优化就直接写”，而是形成稳定闭环：

1. 先选热点。
2. 再证明当前慢在哪里。
3. 然后只改一个清晰边界。
4. 最后用 benchmark + correctness + dispatch 证据收口。

建议后续所有 basic-op 加速都按这个循环执行。

### 4.1 单轮循环模板

每一轮只处理一个明确主题，例如：

- `fp16 scalar compare dispatch`
- `u8/s16/s32 add-sub-mul xsimd`
- `integer compare xsimd`
- `arm64 fp16 load/store fast path`
- `Mat-Scalar channels > 4 boundary`

每轮固定分为 6 个阶段。

#### Phase A：选题与基线

- 明确这轮只解决一个主题，不混多个方向。
- 记录影响范围：
  - 哪些 `type`
  - 哪些 `channel`
  - 哪些 `shape/layout`
  - 哪些 API：`Mat-Mat`、`Mat-Scalar`、`compare`、`transpose`
- 先跑基线 benchmark，保留原始 CSV。
- 先确认当前 dispatch 是走 `scalar` 还是 `xsimd`。

#### Phase B：最小 correctness 保护

- 先补测试，再改实现。
- 每轮至少补 3 类 case：
  - happy path
  - 边界值
  - 退化路径
- 对整数路径，必须显式覆盖：
  - 零值
  - 最大/最小值
  - 符号边界
- 对 `fp16` 路径，必须显式覆盖：
  - 正常值
  - 精度敏感值
  - `0` / `-0`
- 对 view/ROI 路径，至少覆盖：
  - continuous
  - 非 continuous

#### Phase C：实现最小闭环

- 只改一个层级：
  - dispatch 层
  - kernel 层
  - benchmark 层
  - ARM 特化层
- 优先复用现有 `xsimd` kernel 结构，不新建平行框架。
- 如果需要新增平台特化，必须保留 fallback。
- 不允许在这一阶段顺手扩 unrelated 优化。

#### Phase D：局部验证

- 跑这一轮主题对应的 targeted benchmark。
- 跑对应 correctness test。
- 对 dispatch 做一次显式核查：
  - 命中 `scalar`
  - 命中 `xsimd`
  - fallback 原因是什么

#### Phase E：回归验证

- 跑 `quick` benchmark profile。
- 跑受影响的 core test。
- 检查是否把原本已快的路径变慢。
- 检查 `x86` 代码路径是否仍能编译通过。

#### Phase F：收口

- 更新文档：
  - 本轮做了什么
  - 哪些 case 已覆盖
  - 哪些 case 仍未覆盖
  - benchmark 提升多少
  - 是否引入新退化边界
- 决定下一轮只处理一个新主题。

### 4.2 单轮输入与输出

每轮开始前必须具备这些输入：

- 一个明确主题
- 一组基线 benchmark CSV
- 一组对应 correctness case
- 一份受影响文件列表

每轮结束后必须产出这些输出：

- 一组新的 benchmark CSV
- 一组通过的 correctness test
- 一份命中/退化说明
- 一条文档更新记录

### 4.3 每轮必须回答的问题

每轮结束必须能回答这 8 个问题：

1. 这轮到底优化了哪个 case？
2. 这个 case 之前为什么慢？
3. 现在是进了 `xsimd`，还是只是别的标量改进？
4. 提升发生在 `Mat-Mat`、`Mat-Scalar`，还是两者都有？
5. 有没有把别的路径变慢？
6. 有没有新增平台分叉？
7. fallback 还在不在，行为是否一致？
8. 下一轮最该补的单点缺口是什么？

### 4.4 优先级队列

后续建议严格按以下顺序推进，不跨级跳。

#### P0：可见性闭环

- `fp16 scalar compare` 去 `_OPENMP` 绑定
- benchmark 增加 `scalar-only` / `xsimd-only`
- benchmark 输出增加 `dispatch path`
- `quick` profile 补 `CV_32U`

#### P1：整数算术覆盖

- `CV_8U/8S/16U/16S/32S/32U`
- `ADD/SUB/MUL`
- 先 `Mat-Mat`
- 再 `Mat-Scalar`

#### P2：整数 compare 覆盖

- `EQ/NE/GT/GE/LT/LE`
- 先 uniform scalar
- 再 non-uniform scalar
- 再 `channels <= 4` 的现有通道模型

#### P3：ARM fp16 快路径

- `load/store`
- `Mat-Mat` hot path
- `Mat-Scalar` hot path

#### P4：容器语义边界

- `channels <= 4` 的优化合同固化
- `channels > 4` 的基础实现/fallback 规则
- interleaved 常见布局
- ROI / submat 快慢边界

#### P5：手写 NEON

- 仅处理 `xsimd` 覆盖后仍显著落后的热点

### 4.5 验证矩阵

每轮都不要全量瞎跑，按矩阵选最小必要集合。

| 维度 | 必跑集合 |
|---|---|
| `type` | 当前主题相关 type + 相邻 type 1 组 |
| `channels` | `1`、`3`、`4`，做边界验证时加 `8` 或更高 |
| `shape` | 大 2D、瘦长、带 outer 维的 ND |
| `layout` | continuous，必要时加 submat / ROI |
| API 形态 | `Mat-Mat`、`Mat-Scalar`、`Scalar-Mat` |
| 比较类 | `EQ/GT/GE/LT/LE/NE` 至少抽样 2 个，收口时全补 |

推荐基线形状：

- `720x1280`
- `1536x512`
- `8x256x384`

这些 shape 已在现有 benchmark 里出现过，便于和历史结果对齐。

### 4.6 benchmark 分层

后续 benchmark 建议分 3 层，不要混成一个模糊数字。

#### Layer 1：microbench

- 只测单一 hot path
- 用于判断某个 kernel 或 dispatch 改动是否有效
- 输出最小，跑得快

#### Layer 2：core quick

- 用于 PR 日常回归
- 覆盖常见 `type/channel/shape`
- 可以在本地和 CI 都稳定跑

#### Layer 3：core full

- 用于阶段收口
- 覆盖更多 `depth`、更多 compare、更多 scalar pattern
- 只在阶段结束或大变更时跑

### 4.7 退化判定规则

每轮都需要明确什么叫“可以接受”，什么叫“打回去重做”。

#### 允许通过

- 目标 case 明显变快
- 非目标 case 无明显退化
- fallback 仍正确
- benchmark 能解释结果

#### 不允许通过

- 目标 case 无提升但代码复杂度显著上升
- benchmark 看起来提升，但其实只是切了 shape 或减少了覆盖
- `xsimd` 路径提升，fallback 路径 silent wrong result
- ARM 提升了，x86 编译或行为回退
- ROI / submat 变成随机快慢，边界无法解释

### 4.8 单轮验收表

每轮收口时建议在提交说明或文档更新中附这一张表。

| 项目 | 结果 |
|---|---|
| 主题 | 例如 `CV_16F Mat-Scalar add on arm64` |
| 改动层级 | dispatch / kernel / benchmark / arch-specialization |
| 命中路径 | scalar / xsimd / neon-specialized |
| 基线 CSV | 路径 |
| 新结果 CSV | 路径 |
| correctness | pass / fail |
| quick profile | pass / fail |
| 已知退化 | 无 / 有（列出） |
| 下一轮候选 | 只写 1 到 2 个 |

### 4.9 推荐节奏

建议以后按“小步快跑，但每步都完整收口”的节奏推进：

- 一个主题，一次提交
- 一个主题，一套 benchmark 对照
- 一个主题，一套 correctness 补齐
- 一个主题，一次文档更新

不要把“整数算术 + compare + ARM fp16 + channels 扩张”混在一次大改里。那样最后没有人知道哪一步真的有效。

## 5. 实施状态

这一节用于记录已经落地的轮次结果。后续每完成一轮实现，都在这里追加状态，保证任务恢复时能直接看到：

- 做到了什么
- 怎么验证的
- 还有什么残留
- 下一轮接什么

### P0：可见性闭环

- 状态：已完成
- 完成日期：2026-04-05
- 范围：dispatch 层、benchmark 层、`fp16` scalar compare 可用性

#### 已完成

- 去掉了 `fp16` scalar compare 对 `_OPENMP` 的依赖，`compare(Mat,Scalar)` / `compare(Scalar,Mat)` 的 `CV_16F` 路径在无 OpenMP 的 Apple clang 环境下也能进入现有分发逻辑。
- 新增运行时 dispatch 控制：
  - `auto`
  - `scalar-only`
  - `xsimd-only`
- `Mat-Mat`、`Mat-Scalar`、`compare`、`transpose` 的主 dispatch 点现在都会记录本次命中路径：
  - `scalar`
  - `xsimd`
  - `unknown`
- benchmark `quick` profile 已加入 `CV_32U`。
- benchmark CSV 已增加 `dispatch` 列，可以直接看出每个 case 实际命中的路径。
- `xsimd-only` 模式下，如果目标 case 没有 SIMD 覆盖，会直接跳过该 case，不再把 fallback 混进结果。

#### 验证结果

- 构建：
  - `cmake -S . -B build-arm64-check -DCVH_BUILD_TESTS=ON -DCVH_BUILD_BENCHMARKS=ON -DCVH_USE_OPENMP=OFF`
  - `cmake --build build-arm64-check -j 8 --target cvh_benchmark_core_ops cvh_test_core`
- correctness：
  - 新增 `compare_fp16_scalar_paths_work_without_openmp_requirement`
  - 新增测试通过，验证了 `CV_16F` 的 `Mat-Scalar` / `Scalar-Mat compare` 在无 OpenMP 环境下结果正确
- benchmark：
  - `./build-arm64-check/cvh_benchmark_core_ops --profile quick --bench matscalar --dispatch auto --warmup 1 --iters 2 --repeats 1 --output /tmp/cvh_p0_auto.csv`
  - `./build-arm64-check/cvh_benchmark_core_ops --profile quick --bench matscalar --dispatch scalar-only --warmup 1 --iters 2 --repeats 1 --output /tmp/cvh_p0_scalar.csv`
  - `./build-arm64-check/cvh_benchmark_core_ops --profile quick --bench matscalar --dispatch xsimd-only --warmup 1 --iters 2 --repeats 1 --output /tmp/cvh_p0_xsimd.csv`

#### 关键观察

- `auto` 模式能同时记录 `scalar` 和 `xsimd`，说明当前 dispatch 记录链路已经打通。
- `scalar-only` 输出稳定只包含 `scalar`。
- `xsimd-only` 输出只保留已有 SIMD 覆盖的 case，当前主要集中在 `CV_32F` 和 `CV_16F` 的 `Mat-Scalar` 路径。
- `CV_32U` 已进入 `quick` profile，但目前仍主要落在 `scalar`，这也进一步确认了 P1/P2 的优先级没有变化。

#### 已知残留

- 现有 `cvh_test_core` 仍有一个与本轮无关的失败：
  - `MatContract_TEST.unsupported_depth_is_rejected_in_create`
- 该失败在本轮实现前就存在，不属于 P0 改动引入的问题。
- benchmark 现在已经能看见命中路径，但还没有把“为什么 fallback”细分成更具体的原因码；这可以在后续需要时再补。

#### 下一轮

- 进入 P1：整数算术覆盖
- 优先顺序：
  1. `Mat-Mat` 的 `CV_8U/8S/16U/16S/32S/32U ADD/SUB/MUL`
  2. 再复用到 `Mat-Scalar`

### P1：整数算术覆盖

- 状态：已完成
- 最近更新时间：2026-04-06
- 当前完成范围：
  - `Mat-Mat` 的 `CV_8U/8S/16U/16S ADD/SUB/MUL`
  - `Mat-Mat` 的 `CV_32S/CV_32U ADD/SUB/MUL`
  - `Mat-Scalar` 的 `CV_8U/8S/16U/16S ADD/SUB/MUL`
  - `Mat-Scalar` 的 `CV_32S/CV_32U ADD/SUB/MUL`

#### 本轮已完成

- 为 `CV_32U` 增加了独立的 `xsimd` binary kernel。
- 为 `Mat-Scalar` 的 `CV_32S/CV_32U` 增加了独立的 `xsimd` dispatch：
  - uniform scalar 走 broadcast kernel
  - non-uniform scalar 走 channel-phase xsimd helper
- 为 `Mat-Scalar` 的 `CV_8U/8S/16U/16S` 增加了独立的 `xsimd` dispatch：
  - uniform scalar 复用已有 broadcast kernel
  - non-uniform scalar 走新增的 smallint channel-phase xsimd helper
- `CV_8U/8S/16U/16S` 的 `MUL` 现已补齐饱和语义：
  - vector 主路径使用 widen 后乘法
  - 收窄时保持与现有 `saturate_cast` 一致
- `Mat-Mat` 的整数 binary dispatch 现在不再一律只放行 `MAX/MIN`：
  - `CV_8U/8S/16U/16S` 新增饱和语义 `ADD/SUB/MUL`
  - `CV_32S` 新增 `ADD/SUB/MUL`
  - `CV_32U` 新增 `ADD/SUB/MUL`
- `Mat-Scalar` 的整数 binary dispatch 现在已覆盖：
  - `CV_8U/8S/16U/16S ADD/SUB/MUL`
  - `CV_32S ADD/SUB/MUL`
  - `CV_32U ADD/SUB/MUL`
- 继续保持 `DIV` 不进入这一轮覆盖，避免把零除语义和本轮混在一起。
- `CV_8U/8S/16U/16S` 的 `ADD/SUB` 已改为使用 `xsimd::sadd/ssub`，并保持标量尾部也是饱和语义。

#### 验证结果

- correctness：
  - 新增 `BinaryOpContract_TEST.mat_mat_add_sub_mul_int32_and_uint32_support_xsimd_only_mode`
  - 新增 `BinaryOpContract_TEST.mat_mat_add_sub_u8_s8_u16_s16_support_xsimd_only_with_saturation`
  - 新增 `BinaryOpContract_TEST.mat_scalar_add_sub_mul_int32_and_uint32_support_xsimd_only_mode`
  - 新增 `BinaryOpContract_TEST.mat_scalar_add_sub_u8_s8_u16_s16_support_xsimd_only_with_saturation`
  - 新增 `BinaryOpContract_TEST.mat_mat_and_mat_scalar_mul_u8_s8_u16_s16_support_xsimd_only_with_saturation`
  - 在 `xsimd-only` 模式下验证 `CV_32S/CV_32U` 的 `ADD/SUB/MUL` 结果正确
  - 在 `xsimd-only` 模式下验证 `CV_8U/8S/16U/16S` 的 `ADD/SUB/MUL` 结果正确，且保持饱和语义
  - 在 `xsimd-only` 模式下验证 `Mat-Scalar` 的 `CV_32S/CV_32U ADD/SUB/MUL` 结果正确，覆盖 uniform 和 non-uniform scalar
  - 在 `xsimd-only` 模式下验证 `Mat-Scalar` 的 `CV_8U/8S/16U/16S ADD/SUB/MUL` 结果正确，覆盖 uniform 和 non-uniform scalar
- benchmark：
  - `./build-arm64-check/cvh_benchmark_core_ops --profile quick --bench matmat --dispatch xsimd-only --warmup 1 --iters 2 --repeats 1 --output /tmp/cvh_p1_matmat_xsimd.csv`
  - `./build-arm64-check/cvh_benchmark_core_ops --profile quick --bench matmat --dispatch xsimd-only --warmup 1 --iters 2 --repeats 1 --output /tmp/cvh_p1_matmat_xsimd_v2.csv`
  - `./build-arm64-check/cvh_benchmark_core_ops --profile quick --bench matmat --dispatch xsimd-only --warmup 1 --iters 1 --repeats 1 --output /tmp/cvh_p1_mul_matmat.csv`
  - `./build-arm64-check/cvh_benchmark_core_ops --profile quick --bench matscalar --dispatch xsimd-only --warmup 1 --iters 2 --repeats 1 --output /tmp/cvh_p1_matscalar_xsimd.csv`
  - `./build-arm64-check/cvh_benchmark_core_ops --profile quick --bench matscalar --dispatch xsimd-only --scalar-pattern nonuniform --scalar-order both --warmup 1 --iters 1 --repeats 1 --output /tmp/cvh_p1_matscalar_xsimd_nu.csv`
  - `./build-arm64-check/cvh_benchmark_core_ops --profile quick --bench matscalar --dispatch xsimd-only --scalar-pattern both --scalar-order both --warmup 1 --iters 1 --repeats 1 --output /tmp/cvh_p1_matscalar_xsimd_v2.csv`
  - `./build-arm64-check/cvh_benchmark_core_ops --profile quick --bench matscalar --dispatch xsimd-only --scalar-pattern both --scalar-order both --warmup 1 --iters 1 --repeats 1 --output /tmp/cvh_p1_mul_matscalar.csv`
  - 结果中已出现：
    - `ADD/SUB,xsimd,CV_8U`
    - `ADD/SUB,xsimd,CV_8S`
    - `ADD/SUB,xsimd,CV_16U`
    - `ADD/SUB,xsimd,CV_16S`
    - `MUL,xsimd,CV_8U`
    - `MUL,xsimd,CV_8S`
    - `MUL,xsimd,CV_16U`
    - `MUL,xsimd,CV_16S`
    - `ADD/SUB/MUL,xsimd,CV_32S`
    - `ADD/SUB/MUL,xsimd,CV_32U`
    - `ADD_MS_U/NU_(MF|SF),xsimd,CV_32S`
    - `SUB_MS_U/NU_(MF|SF),xsimd,CV_32S`
    - `MUL_MS_U/NU_(MF|SF),xsimd,CV_32S`
    - `ADD_MS_U/NU_(MF|SF),xsimd,CV_32U`
    - `SUB_MS_U/NU_(MF|SF),xsimd,CV_32U`
    - `MUL_MS_U/NU_(MF|SF),xsimd,CV_32U`
    - `ADD_MS_U/NU_(MF|SF),xsimd,CV_8U`
    - `SUB_MS_U/NU_(MF|SF),xsimd,CV_8U`
    - `ADD_MS_U/NU_(MF|SF),xsimd,CV_8S`
    - `SUB_MS_U/NU_(MF|SF),xsimd,CV_8S`
    - `ADD_MS_U/NU_(MF|SF),xsimd,CV_16U`
    - `SUB_MS_U/NU_(MF|SF),xsimd,CV_16U`
    - `ADD_MS_U/NU_(MF|SF),xsimd,CV_16S`
    - `SUB_MS_U/NU_(MF|SF),xsimd,CV_16S`
    - `MUL_MS_U/NU_(MF|SF),xsimd,CV_8U`
    - `MUL_MS_U/NU_(MF|SF),xsimd,CV_8S`
    - `MUL_MS_U/NU_(MF|SF),xsimd,CV_16U`
    - `MUL_MS_U/NU_(MF|SF),xsimd,CV_16S`

#### 关键观察

- 这一轮确认了可以拆成两段推进：
  - 32-bit 先补 `ADD/SUB/MUL`
  - 8/16-bit 先补语义更清晰的饱和 `ADD/SUB`
- `quick` `matmat` 的 `xsimd-only` 输出已经把 `CV_8/16/32` 里本轮覆盖到的整数算术标成 `dispatch=xsimd`，说明 benchmark 和实际覆盖已经对齐。
- `quick` `matscalar` 的 `xsimd-only` 输出已经把 `CV_32S/CV_32U` 的 uniform / non-uniform、`mat_first / scalar_first` 组合标成 `dispatch=xsimd`，说明 `Mat-Scalar` 的 32-bit 覆盖也已经对齐。
- `quick` `matscalar` 的 `xsimd-only` 输出已经把 `CV_8/16` 的 uniform / non-uniform、`mat_first / scalar_first` `ADD/SUB/MUL` 组合标成 `dispatch=xsimd`，说明 `Mat-Scalar` 的 8/16-bit 整数算术覆盖也已经对齐。
- `AND/XOR/MOD` 这类非本轮目标仍显示为 `unknown` 或非 SIMD，这符合预期。

#### 收口结论

- P1 的整数算术覆盖已经完成：
  - `Mat-Mat`：`CV_8U/8S/16U/16S/32S/32U ADD/SUB/MUL`
  - `Mat-Scalar`：`CV_8U/8S/16U/16S/32S/32U ADD/SUB/MUL`
- 当前剩余核心缺口已经从整数算术转移到整数 compare 和 ARM `fp16` 快路径。

### P2：整数 compare 覆盖

- 状态：已完成
- 最近更新时间：2026-04-06
- 当前完成范围：
  - `Mat-Mat` 的 `CV_8U/8S/16U/16S/32S/32U EQ/NE/GT/GE/LT/LE`
  - `Mat-Scalar` 的 `CV_8U/8S/16U/16S/32S/32U` compare xsimd dispatch
  - `Scalar-Mat` 复用 `Mat-Scalar` compare xsimd dispatch

#### 本轮已完成

- 为 `CV_8U/8S/16U/16S/32S/32U` 增加了整数 compare 的 `xsimd` kernel：
  - `compare_broadcast_xsimd_*`
  - `compare_scalar_channels_xsimd_*`
- `Mat-Mat compare` 现在已支持整数 `EQ/NE/GT/GE/LT/LE` 的 `xsimd` 分发。
- `Mat-Scalar compare` 现在已支持整数 compare 的 `xsimd` 分发：
  - uniform scalar 走 broadcast kernel
  - non-uniform scalar 走 channel-phase xsimd helper
- `Scalar-Mat compare` 复用同一套分发，不再单独维护一份整数 compare 逻辑。
- 补了 `xsimd-only` correctness 测试，覆盖：
  - `Mat-Mat` 的 `EQ/GT/LE`
  - `Mat-Scalar` / `Scalar-Mat` 的 `GT/NE/GE`

#### 验证结果

- correctness：
  - 新增 `BinaryOpContract_TEST.mat_mat_integer_compare_supports_xsimd_only_mode`
  - 新增 `BinaryOpContract_TEST.mat_scalar_integer_compare_supports_xsimd_only_mode`
  - `xsimd-only` 模式下验证通过：
    - `Mat-Mat`：`CV_8U EQ`、`CV_16S GT`、`CV_32U LE`
    - `Mat-Scalar`：`CV_8S GT`、`CV_32S GE`
    - `Scalar-Mat`：`CV_16U C3 NE`
- benchmark：
  - `./build-arm64-check/cvh_benchmark_core_ops --profile quick --bench matmat --dispatch xsimd-only --warmup 1 --iters 1 --repeats 1 --output /tmp/cvh_p1_compare_matmat.csv`
  - `./build-arm64-check/cvh_benchmark_core_ops --profile quick --bench matscalar --dispatch xsimd-only --scalar-pattern both --scalar-order both --warmup 1 --iters 1 --repeats 1 --output /tmp/cvh_p1_compare_matscalar.csv`
  - `matmat` 结果中已出现：
    - `CMP_EQ/CMP_GT/CMP_GE/CMP_LT/CMP_LE/CMP_NE,xsimd,CV_8U`
    - `CMP_EQ/CMP_GT/CMP_GE/CMP_LT/CMP_LE/CMP_NE,xsimd,CV_8S`
    - `CMP_EQ/CMP_GT/CMP_GE/CMP_LT/CMP_LE/CMP_NE,xsimd,CV_16U`
    - `CMP_EQ/CMP_GT/CMP_GE/CMP_LT/CMP_LE/CMP_NE,xsimd,CV_16S`
    - `CMP_EQ/CMP_GT/CMP_GE/CMP_LT/CMP_LE/CMP_NE,xsimd,CV_32S`
    - `CMP_EQ/CMP_GT/CMP_GE/CMP_LT/CMP_LE/CMP_NE,xsimd,CV_32U`
  - `matscalar` quick profile 当前只统计 `CMP_GT_MS_*`，但相关整数 case 已全部命中 `dispatch=xsimd`，覆盖：
    - `CV_8U/8S/16U/16S/32S/32U`
    - `uniform / non-uniform`
    - `mat_first / scalar_first`

#### 关键观察

- `Mat-Mat` 的整数 compare 六种关系已经在 quick profile 中全部可见，说明 kernel、dispatch、benchmark 三层已经打通。
- `Mat-Scalar` 的 benchmark 设计目前只对 compare 输出 `CMP_GT_MS_*` 指标，所以 quick CSV 不能直接看到 `EQ/NE/GE/LT/LE` 的分项；这属于 benchmark 观测粒度问题，不是 dispatch 缺失。
- `Scalar-Mat` 复用 `Mat-Scalar` 后，整数 compare 的代码路径没有额外分叉。

#### 已知残留

- `Mat-Scalar compare` 的 benchmark 仍只暴露 `CMP_GT_MS_*` 指标；如果后续要做 compare 分项调优，建议把 `EQ/NE/GE/LT/LE` 也加进 benchmark 输出。
- `channels > 4` 仍不在当前 compare SIMD 覆盖范围内；按当前设定，这属于 Step 4 / P4 的边界说明和基础支持问题，不是主优化缺口。

#### 下一步

- 进入 P3：ARM `fp16` 快路径
- 优先顺序：
  1. `include/cvh/core/detail/xsimd_kernel_utils.h` 的 `arm64` half load/store 快路径
  2. `Mat-Mat` 的 `CV_16F` hot path 基准对比
  3. `Mat-Scalar` 的 `CV_16F` hot path 基准对比

### P3：ARM `fp16` 快路径

- 状态：已完成（`arm64` 本地验证）
- 最近更新时间：2026-04-06
- 当前完成范围：
  - `include/cvh/core/detail/xsimd_kernel_utils.h` 的 `arm64 + FP16 NEON` half load/store 快路径
  - `Mat-Mat` 的 `CV_16F` `ADD` 热点基准对比
  - `Mat-Scalar` 的 `CV_16F` `ADD` 热点基准对比

#### 本轮已完成

- 在 `__aarch64__ && __ARM_FEATURE_FP16_VECTOR_ARITHMETIC` 下新增了 `hfloat` batch 的 NEON 快路径：
  - `load_hfloat_batch`
  - `store_hfloat_batch`
- 当前实现直接走 half 向量装载/回写，再用 NEON 做 `f16 <-> f32` 批量转换，避免 ARM 上逐元素 `hfloat -> float -> hfloat` 的回退成本。
- 保留了现有 x86 `f16c` 快路径和通用 fallback，没有改动原有分支结构。
- 补了两条 `CV_16F` 的 `xsimd-only` correctness test：
  - `BinaryOpContract_TEST.mat_mat_add_fp16_supports_xsimd_only_mode`
  - `BinaryOpContract_TEST.mat_scalar_add_fp16_supports_xsimd_only_mode`

#### 验证结果

- correctness：
  - `BinaryOpContract_TEST.mat_mat_add_fp16_supports_xsimd_only_mode`
  - `BinaryOpContract_TEST.mat_scalar_add_fp16_supports_xsimd_only_mode`
  - `MatScalarOps_TEST.compare_fp16_scalar_paths_work_without_openmp_requirement`
  - 上述 3 条定向测试已通过
- benchmark：
  - baseline：
    - `/tmp/cvh_p3_fp16_before_matmat.csv`
    - `/tmp/cvh_p3_fp16_before_matscalar.csv`
  - after：
    - `/tmp/cvh_p3_fp16_after_matmat.csv`
    - `/tmp/cvh_p3_fp16_after_matscalar.csv`

#### 关键结果

- `Mat-Mat ADD,xsimd,CV_16F`
  - `720x1280,C1`：`48.28 ms -> 29.95 ms`，`19.09 -> 30.77 Melems/s`
  - `720x1280,C3`：`144.59 ms -> 99.91 ms`，`19.12 -> 27.67 Melems/s`
  - `8x256x256,C1`：`34.36 ms -> 17.17 ms`，`15.26 -> 30.53 Melems/s`
  - `8x256x256,C3`：`86.00 ms -> 53.47 ms`，`18.29 -> 29.41 Melems/s`
- `Mat-Scalar ADD,xsimd,CV_16F`
  - `720x1280,C1,uniform,mat_first`：`41.61 ms -> 29.32 ms`，`22.15 -> 31.43 Melems/s`
  - `720x1280,C3,uniform,mat_first`：`121.27 ms -> 87.04 ms`，`22.80 -> 31.76 Melems/s`
  - `720x1280,C3,nonuniform,scalar_first`：`123.23 ms -> 87.47 ms`，`22.44 -> 31.61 Melems/s`
  - `8x256x256,C1,uniform,mat_first`：`22.96 ms -> 16.61 ms`，`22.84 -> 31.56 Melems/s`
  - `8x256x256,C3,uniform,scalar_first`：`69.79 ms -> 50.16 ms`，`22.54 -> 31.35 Melems/s`

#### 关键观察

- 这轮优化是“转换成本回收”，不是换 kernel 语义；收益主要来自 `hfloat` batch 装载和回写不再逐元素做标量转换。
- `Mat-Mat` 和 `Mat-Scalar` 的 `CV_16F ADD` 都得到稳定提升，说明快路径已经覆盖到当前 `xsimd` 基本算术主干。
- 当前 benchmark 主要对 `ADD` 做了 before/after 对照，但同一条 `load/store` 快路径也会惠及 `SUB/MUL/DIV/MAX/MIN/compare` 的 `CV_16F` kernel。

#### 已知残留

- 还没有在当前机器上做 `x86` 编译回归；本轮只验证了 `arm64` 本地构建和定向测试。
- benchmark 还没有把 `CV_16F compare` 单独做一组 before/after 收口。

#### 下一步

- 进入 P4：容器语义边界
- 优先顺序：
  1. 固化 `channels <= 4` 为主优化合同
  2. 梳理 `channels > 4` 的基础实现/fallback 规则
  3. 常见 interleaved 多通道连续布局的边界说明
  4. `ROI / submat` 的快慢边界说明和验证

### P4：容器语义边界

- 状态：进行中
- 最近更新时间：2026-04-06
- 当前目标：
  - 固化 `channels <= 4` 为主优化合同
  - 梳理 `channels > 4` 的基础实现/fallback 规则
  - 明确 layout / ROI 的边界

#### 当前边界结论

- `channels <= 4`
  - `Mat-Scalar` / `Scalar-Mat`：当前主优化范围
  - 允许继续做 `xsimd` / `arm64` 专项优化
- `channels > 4`
  - `Mat-Mat`：不受这条边界限制，继续按现有实现支持
  - `Mat-Scalar` / `Scalar-Mat`：当前继续保持拒绝，不扩成主优化目标
  - `Mat::setTo(Scalar)`：当前继续保持拒绝
- layout / ROI
  - continuous interleaved layout：`channels <= 4` 是当前主要优化合同
  - non-continuous `ROI / submat`：当前首先保证结果正确和边界稳定，不承诺统一命中 SIMD
  - `channels > 4` 的 `ROI / submat`：当前只要求 `Mat-Mat` 基础可用，不扩到 `Mat-Scalar`

#### 支持矩阵

| API | `channels <= 4` continuous | `channels <= 4` ROI/submat | `channels > 4` continuous | `channels > 4` ROI/submat |
|---|---|---|---|---|
| `Mat-Mat` binary / compare | 主路径，已做优化覆盖 | 基础正确，部分 case 已有合同测试 | 基础支持 | 基础支持 |
| `Mat-Scalar` / `Scalar-Mat` binary / compare | 主路径，已做优化覆盖 | 基础正确，不承诺统一命中 SIMD | 当前拒绝 | 当前拒绝 |
| `Mat::setTo(Scalar)` | 基础支持 | 基础支持 | 当前拒绝 | 当前拒绝 |

补充说明：

- “主路径”表示这是当前优先优化和持续观测的合同范围。
- “基础支持”表示要求结果正确、行为稳定，但不承诺优化路径。
- “当前拒绝”表示按现有设定明确报错，不把它视为当前缺陷。

#### 本轮已完成

- 固化了当前通道边界，不再把 `channels > 4` 作为默认优化扩张方向。
- 现有测试已经覆盖：
  - `Mat::setTo(Scalar)` 在 `channels > 4` 时拒绝
  - `Mat-Scalar` / `Scalar-Mat` 的 binary 和 compare 在 `channels > 4` 时拒绝
- 新增 `Mat-Mat` 合同测试，验证 `channels > 4` 时：
  - `add(Mat,Mat)` 继续可用
  - `compare(Mat,Mat)` 继续可用
- 新增 `Mat-Mat` 的 non-continuous ROI 合同测试，验证 `channels > 4` 时：
  - `add(Mat,Mat)` 在 ROI 上继续可用
  - `compare(Mat,Mat)` 在 ROI 上继续可用

#### 验证结果

- correctness：
  - `MatScalarOps_TEST.setto_scalar_rejects_more_than_four_channels`
  - `MatScalarOps_TEST.scalar_binary_and_compare_reject_more_than_four_channels`
  - `MatScalarOps_TEST.mat_mat_binary_and_compare_still_support_more_than_four_channels`
  - `MatScalarOps_TEST.mat_mat_non_continuous_roi_still_supports_more_than_four_channels`

#### 下一步

- 继续补 P4 的边界说明，而不是扩张 `channels > 4` 的优化覆盖。
- 优先顺序：
  1. 如有必要，为 `channels <= 4` 的 continuous interleaved 主路径补 dispatch 级别观测
  2. 继续补 `ROI / submat` 的边界合同测试
  3. 如有必要，再为 `channels > 4` 增加“基础 fallback 明确可用”的补充测试

## 6. 当前缺口摘要

- 整数算术和整数 compare 的主路径已补齐：
  - `Mat-Mat`：`CV_8/16/32` 的整数 `ADD/SUB/MUL + EQ/NE/GT/GE/LT/LE`
  - `Mat-Scalar` / `Scalar-Mat`：`CV_8/16/32` 的整数 `ADD/SUB/MUL + compare`
- `fp16` 在 ARM 上的 half load/store 快路径已补齐，并在 `CV_16F ADD` 热点上验证到明显提升。
- `fp16 compare` 的 benchmark 观测粒度还不够细，后续如果要继续调优，需要补单独对照。
- 当前通道边界需要按设定明确：
  - `channels <= 4` 是主要优化范围
  - `channels > 4` 只要求基础正确实现或稳定 fallback，不作为当前主优化目标
- 当前通道边界的已知现状：
  - `Mat-Mat` 在 `channels > 4` 下仍可继续使用现有实现
  - `Mat-Mat` 在 `channels > 4` 的 non-continuous `ROI / submat` 下也应继续保持基础可用
  - `Mat-Scalar` / `Scalar-Mat` / `Mat::setTo(Scalar)` 在 `channels > 4` 下继续拒绝
- `Mat-Scalar` 的布局边界仍需要进一步说明，尤其是 interleaved 连续布局与 `ROI / submat`。

## 7. 非目标

- 当前阶段不切换到新的 SIMD 抽象层。
- 当前阶段不把 basic-op 全部改写成纯 NEON intrinsics。
- 当前阶段不优先扩张到 `gemm/softmax/rmsnorm` 等非 basic-op 热点。
- 当前阶段不为了单一 benchmark 好看而牺牲容器语义一致性。

## 8. 验收标准

- 能明确回答每个 benchmark case 是 `scalar` 还是 `xsimd`。
- 主流 `depth/channel` 组合的 `Mat-Mat` 和 `Mat-Scalar` basic-op 不再大面积掉回标量。
- `arm64` 下 `CV_16F` 的关键路径有可测量提升。
- `ROI / submat` 的快慢边界清晰，不再表现随机。
- 所有新加速路径都有 correctness test 和 benchmark 佐证。

## 9. Dream State Delta

### 当前状态

- 有一批可工作的 `xsimd` kernel。
- 但验证手段还不足以快速说明“为什么快”或“为什么慢”。
- 覆盖面不完整，导致容器语义一扩张，性能就随机掉回标量。

### 这份计划完成后

- 每一轮优化都有固定闭环。
- 可以快速判断一个新想法值不值得做。
- 可以把“实现速度”和“验证严谨度”同时保住。

### 12 个月理想状态

- `Mat` 的 basic-op 加速像基础设施一样稳定。
- 新增一个 `type/channel/layout` 组合时，团队知道怎么补，不靠个人记忆。
- benchmark、dispatch、fallback、correctness 四件事始终同步推进。
