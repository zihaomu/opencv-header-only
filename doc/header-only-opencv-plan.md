# OpenCV Header-Only 项目研发计划

- 文档日期：2026-03-09
- 适用仓库：`opencv-header-only`
- 当前结论：项目已迁移进来一部分 `Mat` 与底层定义，但整体仍处于**迁移早期/基线未闭环**状态，当前第一优先级不是继续堆功能，而是先把 `header-only`、命名空间、公共头文件、测试入口和工程骨架收口。

## 1. 项目目标

本项目的目标不是完整复刻全部 OpenCV，而是交付一个**可直接 `include` 使用、接口风格接近 OpenCV、覆盖高频核心能力**的 header-only 子集。

首版目标能力如下：

- `Mat` 与基础几何/标量类型。
- `Mat` 上的逐元素计算、类型转换、规约和通道操作。
- 面向 `Mat` 的基础图像处理能力。
- 常用滤波器与卷积框架。
- `RGB/BGR/GRAY/YUV` 常见颜色空间与布局转换。
- 基于 `OpenMP` 的统一并行后端。
- 基于 `xsimd` 的 SIMD 加速路径。
- 最小可用的 `imgcodecs` 能力，仅覆盖有限图像读写。
- 完整的单元测试、差分测试、benchmark 和文档。

## 2. 本次仓库扫描结论

本节结论基于对当前仓库 `include/`、`test/`、`benchmark/`、`doc/`、`README.md`、`CMakeLists.txt` 的扫描，以及对公共头文件进行的一次最小编译验证。

### 2.1 当前成熟度判断

当前仓库更接近于：

- 从旧项目迁移中的原型；
- 有部分 `Mat` 和底层定义，但尚未形成可持续演进的工程基线；
- 尚不能被视为真正意义上的 header-only OpenCV 子集。

换句话说，**现在不是“从 0 到 1 完成功能开发”的阶段，而是“先把迁移债务压平，建立可迭代底座”的阶段。**

### 2.2 已确认的现状事实

#### 1) 公共入口头文件当前不可独立编译

对 `cvh/cvh.h` 做最小编译检查时，发现：

- 直接 `-Iinclude` 编译失败，首先阻塞在 `include/cvh/core/mat.h` 对 `libnpy/npy.hpp` 的依赖。
- 补上 `test/3rdparty` 后，继续阻塞在 `include/cvh/core/utils.h` 对 `context.h` 的依赖。

这说明：

- 当前公共 API 仍然依赖测试目录或旧项目残留头文件。
- 仓库还没有达到“用户只需 `-Iinclude` 即可使用”的基本要求。

#### 2) 命名空间和项目身份尚未完成从 `minfer` 到 `cvh` 的迁移

扫描结果显示：

- 当前核心头文件和测试代码仍大量使用 `namespace minfer`、`minfer::`、`MINFER_*` include guard。
- `namespace cvh` 目前未真正落地。

这会直接带来两个问题：

- 项目定位与实际 API 不一致。
- 后续与 OpenCV 风格对齐时，命名空间、宏、错误处理和兼容层会反复返工。

#### 3) 实际实现仍散落在 `include/cvh/core-cpp-trash/`

当前公开头文件中，大量接口只有声明；真正实现仍主要在：

- `include/cvh/core-cpp-trash/mat.cpp`
- `include/cvh/core-cpp-trash/basic_op.cpp`
- `include/cvh/core-cpp-trash/mat_expr.cpp`
- `include/cvh/core-cpp-trash/mat_convert.cpp`
- `include/cvh/core-cpp-trash/mat_gemm.cpp`
- `include/cvh/core-cpp-trash/system.cpp`
- `include/cvh/core-cpp-trash/utils.cpp`

这说明：

- 当前项目还不是 header-only 形态。
- 迁移路径必须先定义“哪些实现需要保留，哪些只是过渡参考实现，哪些必须剥离”。

#### 4) 测试代码仍绑定旧工程依赖，当前不能作为直接回归基线

当前测试文件中仍存在大量旧依赖，例如：

- `minfer.h`
- `backend/cpu/kernel/*`
- `backend/cpu/layer/runtime_weight.h`
- 旧项目中的 xsimd kernel 头文件

这说明当前测试集存在两个问题：

- 不能直接用于验证当前仓库的公开 API。
- 测试覆盖的对象并不是“header-only OpenCV 子集”，而是“旧 minfer 能力的一部分”。

#### 5) `Mat` 模型与 OpenCV 目标尚未完全对齐

从头文件、readme 和测试注释可以看出：

- 当前 `Mat` 仍以旧项目的数据模型为主。
- 通道信息、OpenCV 风格类型编码、步长/布局语义尚未完全收口。
- `Mat` 相关测试也明确提到“目前不支持 OpenCV mat 中的通道信息”。

这意味着：

- `imgproc`、滤波、颜色转换、SIMD 都还没有一个稳定的底座。

#### 6) `core` 中混入了明显偏旧项目/偏算子库的接口

例如 `basic_op.h` 中目前包含：

- `softmax`
- `silu`
- `rmsnorm`
- `rope`
- 带 scale 的 `gemm`

这些接口并不属于 OpenCV 首版核心目标，会带来两个风险：

- 污染 `core` 模块边界；
- 让项目在“OpenCV 子集”和“通用算子/推理库”之间摇摆，导致范围失控。

#### 7) 工程骨架尚未建立

当前 `CMakeLists.txt` 基本为空壳。一次轻量配置虽能生成默认工程，但会告警缺失：

- `project()`
- 合理的 `cmake_minimum_required()`
- 目标、测试、安装、导出配置

这意味着当前仓库还没有“构建、测试、安装、回归”的主干流程。

#### 8) `imgproc` / `imgcodecs` / `benchmark` 仍基本处于占位状态

当前目录中：

- `include/cvh/imgproc/` 基本还是 `readme` 占位；
- `include/cvh/imgcodecs/` 基本还是 `readme` 占位；
- `benchmark/` 还没有实际 benchmark 代码。

这说明后续开发顺序必须严格遵守依赖关系，不能跳过 `Mat` 和工程基线直接推进上层模块。

### 2.3 当前项目的阶段定位

基于扫描结果，当前项目建议划分为以下阶段定位：

- **当前阶段**：迁移清理 + 核心基线收敛。
- **下一个阶段**：`Mat` 对齐 + 标量 core 能力闭环。
- **再下一个阶段**：`imgproc` / 颜色转换 / 滤波。
- **最后阶段**：`OpenMP` / `xsimd` / 发布打磨。

## 3. 基于现状的主要风险预判

以下风险不是抽象风险，而是结合当前仓库状态后的实施风险。

### 风险 1：项目会长期停留在“伪 header-only”状态

#### 当前信号

- 实现仍主要在 `include/cvh/core-cpp-trash/*.cpp`。
- 公共头 `cvh/cvh.h` 不能只依赖 `include/` 独立编译。

#### 影响

- 用户无法真正只通过头文件集成。
- 每新增一个功能，都会继续放大旧 `.cpp` 迁移债务。

#### 风险等级

- 高

#### 应对策略

- 将“公共头可独立编译”设为 Phase 0 的第一验收目标。
- 所有需要保留的 `.cpp` 实现，分批迁移到 `.hpp/.inl/.detail.hpp`。
- `core-cpp-trash` 在迁移完成前只作为参考区，不允许继续新增功能。

### 风险 2：`minfer` 旧项目语义持续污染新项目边界

#### 当前信号

- 命名空间、宏、错误处理、测试依赖都明显来自旧项目。
- `softmax/rmsnorm/rope` 等非 OpenCV 核心接口还在 `core` 中。

#### 影响

- 团队会在“做 OpenCV 子集”与“继续延用旧算子库”之间来回摇摆。
- API 会越来越难稳定。

#### 风险等级

- 高

#### 应对策略

- 尽快统一到 `cvh` 命名空间。
- 将非 OpenCV 核心算子移出主线计划，可选放入 `experimental/` 或单独 backlog。
- 在计划和代码结构里明确“OpenCV 主线能力”和“历史兼容能力”的边界。

### 风险 3：`Mat` 底座若不先收口，后续所有上层模块都会返工

#### 当前信号

- 通道模型、类型编码、步长/布局、ROI 语义尚未完全对齐 OpenCV。
- 测试注释已指出通道能力缺失。

#### 影响

- `imgproc`、滤波、颜色转换、并行和 SIMD 都会建立在不稳定的数据模型上。
- 一旦 `Mat` 改 ABI/改类型编码，上层实现要整体返工。

#### 风险等级

- 高

#### 应对策略

- 在功能开发前冻结 `Mat` 的首版数据模型。
- 首版优先完成：类型编码、通道数、连续/非连续、ROI/view、clone/copyTo/convertTo。
- 禁止在 `Mat` 未收口前推进上层性能优化。

### 风险 4：测试体系不能直接反映项目真实进度

#### 当前信号

- 测试仍依赖不存在于本仓库中的旧 `backend` 和 `minfer.h`。
- 当前测试对象不是公开 header-only API。

#### 影响

- 看似“有测试”，实际并不能验证当前项目可交付性。
- 后续可能出现“代码可编译但测试不可迁移”或“测试可跑但不是测试当前库”的双重偏差。

#### 风险等级

- 高

#### 应对策略

- 先建立一套只依赖 `include/cvh/*` 的最小测试集。
- 所有旧测试只保留可迁移的数据集和参考用例，不直接照搬依赖链。
- 差分测试统一围绕 OpenCV 参考行为建立。

### 风险 5：公共头中混入测试/工具依赖，导致库边界失控

#### 当前信号

- `mat.h` 直接依赖 `libnpy/npy.hpp`。
- `utils.h` 直接依赖缺失的 `context.h`。

#### 影响

- 用户必须带上测试或旧工程文件才能 include 主库。
- 公共 API 与内部测试工具发生耦合，后期难以拆分。

#### 风险等级

- 高

#### 应对策略

- 将 `npy` 读写功能下沉到 `test/utils` 或 `io/experimental`。
- 将运行时精度、上下文等推理项目残留能力移出核心公开头。
- 公共头只允许依赖标准库、仓库内稳定头文件、受控三方头文件。

### 风险 6：过早做 `OpenMP` 与 `xsimd` 会掩盖基础设计问题

#### 当前信号

- 工程骨架、Mat 模型、标量参考实现都还不稳定。
- 旧项目中的 xsimd kernel 依赖仍出现在测试中。

#### 影响

- 性能路径会和功能路径搅在一起，难以定位正确性问题。
- SIMD/并行代码会固化错误的数据布局假设。

#### 风险等级

- 高

#### 应对策略

- 强制采用“scalar 先行，OpenMP 第二，xsimd 第三”的顺序。
- 所有并行与 SIMD 路径都必须以标量路径为唯一真值参考。
- 先建立内核抽象，再挂接并行和 SIMD。

### 风险 7：类型系统若继续沿用旧定义，会破坏 OpenCV 风格兼容性

#### 当前信号

- 目前已有 `CV_8U/CV_32F` 等基础定义，但通道编码和类型宏体系尚不完整。
- readme 中已明确写到“数据类型命名和 OpenCV 不一样，需要针对性修改”。

#### 影响

- 后续 `CV_8UC1/CV_8UC3`、`Mat::type()`、`convertTo()`、颜色转换全会受到影响。

#### 风险等级

- 中高

#### 应对策略

- 早期就建立完整的 OpenCV 风格类型宏与辅助函数。
- 用 `CV_MAKETYPE/CV_MAT_DEPTH/CV_MAT_CN/CV_ELEM_SIZE` 这类机制统一类型编码。

### 风险 8：跨平台基础类型定义存在隐患

#### 当前信号

- 当前 `int64/uint64` 定义依赖 `long int/unsigned long int`，不同平台宽度并不完全一致。

#### 影响

- 后续跨平台编译和 ABI 可能出现行为差异。

#### 风险等级

- 中

#### 应对策略

- 尽快统一基于 `<cstdint>` 的固定宽度类型。
- 所有对齐、元素大小、类型转换逻辑都以固定宽度类型为基础。

### 风险 9：范围扩张过快会导致项目没有可落地版本

#### 当前信号

- 目标既包含 `Mat`、`imgproc`、滤波、颜色转换、并行、SIMD，又残留旧项目算子需求。
- `imgcodecs` 也已出现在仓库结构中。

#### 影响

- 很容易形成“每个模块都开始了，但没有一个模块可发布”的状态。

#### 风险等级

- 中高

#### 应对策略

- 必须先交付标量版 MVP，再做性能增强。
- `imgcodecs` 仅做最小可用，不得反客为主。
- 非主线旧算子不进入首版承诺。

## 4. 更新后的实施原则

结合当前仓库状态，实施原则要从“功能蓝图优先”调整为“去风险优先”。

### 4.1 先收口，再扩张

- 先把公共头、命名空间、构建、测试和 `Mat` 底座收口。
- 再进入 `imgproc`、滤波、颜色转换。
- 最后才是 `OpenMP` 和 `xsimd`。

### 4.2 只保留 OpenCV 主线目标，旧项目能力做隔离

- `softmax/silu/rmsnorm/rope` 这类能力不进入 OpenCV 主线首版交付。
- 如果一定要保留，应迁移到 `experimental/` 或单独模块，不阻塞主线。

### 4.3 头文件自洽是第一硬门槛

- 所有公共头都必须可以在只给出 `-Iinclude` 的情况下独立编译。
- 公共头不得依赖测试目录、旧工程私有目录或不存在的头文件。

### 4.4 Scalar 真值优先

- 所有功能先有标量正确实现。
- 并行与 SIMD 只能建立在已验证的 scalar 路径之上。

### 4.5 测试按“公共 API → 差分 → 性能”三层推进

- 第一层：只验证公共 API 自身正确性。
- 第二层：与 OpenCV 对齐做差分。
- 第三层：在 benchmark 中衡量 `OpenMP/xsimd` 收益。

## 5. 目标架构与模块边界

建议将项目逐步收敛到如下结构：

```text
include/
  cvh/
    cvh.h
    core/
      define.h
      system.h
      types.h
      mat.h
      mat.inl.h
      mat_ops.h
      saturate.h
      parallel.h
      simd.h
      detail/
        allocator.inl.h
        mat_impl.inl.h
        expr_impl.inl.h
        convert_impl.inl.h
    imgproc/
      border.h
      resize.h
      transform.h
      threshold.h
      filter.h
      color.h
    imgcodecs/
      imgcodecs.h
    experimental/
      nn_ops.h
test/
  core/
  imgproc/
  color/
  imgcodecs/
  utils/
benchmark/
  core/
  imgproc/
doc/
example/
```

### 5.1 模块边界要求

- `core`：只保留 OpenCV 主线基础能力。
- `imgproc`：只依赖 `core`，不依赖历史 `backend`。
- `parallel` / `simd`：只提供抽象和优化实现，不拥有业务语义。
- `experimental`：承接历史迁移的非 OpenCV 主线能力，避免污染主线。

## 6. 详细实施计划

以下计划已根据当前仓库状态重新排序，目标是先去掉“当前无法持续开发”的风险，再逐步扩展功能。

### Phase 0：仓库清理与最小可编译闭环

#### 阶段目标

让仓库从“迁移中的代码堆”变成“可持续开发的工程项目”。

#### 主要工作

- 补全真正可用的 `CMakeLists.txt`，明确 `project()`、语言标准、选项、测试开关、安装规则。
- 增加最小构建目标：
  - 头文件自检 target；
  - 单元测试 target；
  - benchmark 骨架 target。
- 修复公共头文件自洽性：
  - 移除 `mat.h` 对 `libnpy/npy.hpp` 的公开依赖；
  - 移除 `utils.h` 对 `context.h` 的依赖或将其下沉到非公共区域；
  - 确保 `#include <cvh/cvh.h>` 在仅 `-Iinclude` 时可编译。
- 将 `include/cvh/core-cpp-trash/` 定义为“迁移参考区”，冻结新增代码。
- 梳理 `README.md` 与 `doc/`，统一项目定位。

#### 阶段产出

- 可用的顶层 `CMakeLists.txt`
- 头文件编译检查 target
- 最小测试与 benchmark 骨架
- 一份迁移清单，列出 `core-cpp-trash` 中每个文件的去向

#### 验收目标

- `cvh/cvh.h` 可在只提供 `-Iinclude` 的情况下独立编译。
- `cmake -S . -B build` 不再是空工程，不再依赖默认伪 `project()`。
- 仓库中不存在公共头直接包含测试目录头文件的情况。
- `core-cpp-trash` 被标记为只读参考区，不再作为正式 API 实现入口。

### Phase 1：命名空间、错误系统与类型系统收口

#### 阶段目标

完成从旧项目身份到 `cvh` 项目的基础迁移，冻结首版公共语义。

#### 主要工作

- 将公共命名空间统一到 `cvh`。
- 清理 `MINFER_*` include guard、错误宏、日志宏的项目命名。
- 定义兼容策略：
  - 是否保留 `minfer` 临时兼容别名；
  - 是否用宏开关控制兼容期。
- 完善类型系统：
  - `CV_8U`、`CV_16U`、`CV_32F` 等基础深度；
  - `CV_MAKETYPE`、`CV_MAT_DEPTH`、`CV_MAT_CN`；
  - `CV_8UC1/CV_8UC3/CV_32FC1` 等常见类型宏；
  - 元素大小与通道数辅助函数。
- 基于 `<cstdint>` 统一基础整数类型定义。

#### 阶段产出

- 项目级命名统一方案
- OpenCV 风格类型宏与辅助函数
- 更新后的错误处理与断言风格

#### 验收目标

- 公共头中不再暴露 `namespace minfer` 作为主命名空间。
- `Mat::type()` 和基础类型宏可表达 OpenCV 风格通道信息。
- 类型相关单元测试覆盖深度、通道数、元素大小和辅助宏。

### Phase 2：`Mat` 数据模型与内存语义冻结

#### 阶段目标

让 `Mat` 成为后续所有模块的稳定底座。

#### 主要工作

- 明确 `Mat` 首版支持的语义范围：
  - `rows` / `cols` / `dims`
  - `type`
  - `channels`
  - `step` / stride
  - 连续与非连续布局
  - ROI/view
  - 外部 buffer 包装
  - clone / shallow copy / move
- 设计 `MatData`、allocator、引用计数与对齐分配策略。
- 确定是否保留表达式模板 `MatExpr`，若保留需严格控制范围；若不保留则先简化。
- 明确哪些 OpenCV 行为首版支持，哪些显式不支持。
- 为后续 SIMD 预留对齐与尾处理所需的最小元信息。

#### 阶段产出

- 冻结版 `Mat` 公开接口
- `Mat` 内存模型说明文档
- `Mat` 行为差异表（相对 OpenCV）

#### 验收目标

- `Mat` 支持常见构造、外部数据包装、浅拷贝、深拷贝、ROI、clone。
- 连续与非连续 `Mat` 的 `copyTo/reshape/convertTo` 行为有测试覆盖。
- `Mat` 的类型、通道、步长和总元素数语义在文档中明确。

### Phase 3：将核心实现真正迁移为 header-only

#### 阶段目标

把当前散落在 `.cpp` 中的核心实现迁移到受控的头文件实现层。

#### 主要工作

- 逐步迁移 `core-cpp-trash` 中仍需要保留的实现：
  - `mat.cpp`
  - `mat_expr.cpp`
  - `mat_convert.cpp`
  - `basic_op.cpp`
  - `system.cpp`
  - `utils.cpp`
- 迁移方式统一为：
  - 公共声明放在稳定头；
  - 实现放在 `.inl.h` 或 `detail/*.inl.h`；
  - 避免把过多细节全部塞进单一头文件。
- 对不再符合主线目标的实现直接下线或移入 `experimental`。
- 每迁移一类实现，就增加对应的 include 自检和单元测试。

#### 阶段产出

- `core` 模块的 header-only 实现层
- 对应的迁移完成清单

#### 验收目标

- `core` 主线能力不再依赖任何 `.cpp` 编译单元。
- 发布时可以完全不安装 `core-cpp-trash`。
- 头文件自检覆盖 `define/system/mat/basic_op/utils` 等核心入口。

### Phase 4：测试体系重建与 OpenCV 差分框架

#### 阶段目标

让测试真正服务于当前项目，而不是继续绑定旧工程。

#### 主要工作

- 重写现有测试入口，仅依赖 `include/cvh/*`。
- 清理对以下旧依赖的直接引用：
  - `minfer.h`
  - `backend/cpu/kernel/*`
  - `backend/cpu/layer/*`
- 保留现有 `.npy` 数据集和脚本中仍有价值的样例，迁移为项目自有测试资产。
- 将 `libnpy` 限定在测试工具域，而不是公共 API 域。
- 建立 OpenCV 差分测试框架：
  - 相同输入；
  - 输出数值误差对比；
  - 边界模式和 ROI 场景对比。
- 增加 include-only smoke tests、API smoke tests、随机测试。

#### 阶段产出

- 新版 `test/core/*`
- `test/utils/*` 下独立的输入加载工具
- 可选的 OpenCV 差分测试工具

#### 验收目标

- 现有测试可以在不依赖旧 `backend` 的情况下编译运行。
- 至少建立一套围绕 `Mat` 与基础算子的稳定回归用例。
- 差分测试能够对接首批 OpenCV 对齐能力。

### Phase 5：`core` 标量算子闭环

#### 阶段目标

交付一个正确、稳定、可复用的标量 core 子集。

#### 主要工作

- 实现并验证：
  - `saturate_cast`
  - `add/subtract/multiply/divide`
  - `compare`
  - `setTo`
  - `convertTo`
  - `norm`
  - `transpose/transposeND`
  - `split/merge/extractChannel/insertChannel`
- 明确表达式模板是否保留到首版：
  - 若保留，只覆盖少量高频运算；
  - 若成本过高，首版可降级为显式函数接口。
- 将非 OpenCV 主线能力从 `basic_op` 中分离：
  - `softmax`
  - `silu`
  - `rmsnorm`
  - `rope`
  - 带 scale 的推理相关接口

#### 阶段产出

- 稳定的 `core` 标量实现
- 与 OpenCV 对齐的基础算子集
- 非主线算子迁移/下线方案

#### 验收目标

- `core` 常用算子在 `uint8/int16/float32` 上结果正确。
- 非连续 `Mat` 和 ROI 输入通过测试。
- 关键基础算子已有 OpenCV 差分结果。

### Phase 6：基础 `imgproc` 框架

#### 阶段目标

建立上层图像处理的统一执行框架。

#### 主要工作

- 设计边界策略：
  - `constant`
  - `replicate`
  - `reflect`
  - `reflect101`
- 设计插值策略：
  - nearest
  - bilinear
- 实现：
  - `copyMakeBorder`
  - `resize`
  - `flip`
  - `transpose`
  - `rotate90/180/270`
  - `threshold`
- 统一参数校验、kernel 入口、边界处理与迭代策略。

#### 阶段产出

- `imgproc` 基础框架
- 首批高频图像处理接口

#### 验收目标

- `copyMakeBorder/resize/flip/threshold` 有独立测试。
- 与 OpenCV 的典型案例差分结果稳定。
- 支持 `1/3/4` 通道输入和 ROI 输入。

### Phase 7：滤波框架与常用滤波器

#### 阶段目标

基于统一 `imgproc` 框架实现常用滤波能力。

#### 主要工作

- 建立通用卷积框架。
- 建立 separable filter 路径。
- 实现：
  - `filter2D`
  - `boxFilter/blur`
  - `GaussianBlur`
  - `medianBlur`
  - `Sobel`
  - `Scharr`
  - `Laplacian`

#### 阶段产出

- 统一滤波内核框架
- 常用滤波器实现

#### 验收目标

- 各滤波器在典型输入上通过 OpenCV 差分验证。
- 边界模式切换、核尺寸变化和多通道输入都有覆盖。
- 至少形成一组可复用的 benchmark 基线。

### Phase 8：颜色空间与像素格式转换

#### 阶段目标

交付算法与媒体处理常见的数据格式转换能力。

#### 主要工作

- 实现：
  - `BGR <-> RGB`
  - `BGR/RGB <-> GRAY`
  - `BGR/RGB <-> BGRA/RGBA`
  - `RGB/BGR <-> YUV444`
  - `RGB/BGR <-> I420/YV12`
  - `RGB/BGR <-> NV12/NV21`
- 明确色彩标准：
  - BT.601 / BT.709
  - full-range / limited-range
- 处理奇数宽高、多 plane、步长不对齐场景。

#### 阶段产出

- `imgproc/color.h`
- 格式与系数说明文档

#### 验收目标

- 常用 RGB/YUV 转换路径通过测试和 OpenCV 对比。
- 色偏、量化误差和边界条件有明确阈值定义。
- 格式限制在文档中明确列出。

### Phase 9：最小 `imgcodecs` 能力

#### 阶段目标

提供足够支撑示例和测试的最小图像读写能力，但不让 `imgcodecs` 反客为主。

#### 主要工作

- 基于现有 `stb_image/stb_image_write` 提供最小封装。
- 首版优先支持：
  - PNG
  - JPEG
  - BMP
- 读写接口与 `Mat` 数据布局对齐。
- 明确只支持有限格式，不承诺视频和复杂容器。

#### 阶段产出

- `imgcodecs/imgcodecs.h`
- 示例与读写测试

#### 验收目标

- 示例程序可以读图、处理、写图。
- 图像读写不会污染 `core` 公共边界。
- 所有第三方依赖均明确在文档中说明。

### Phase 10：`OpenMP` 并行抽象层

#### 阶段目标

在不破坏语义的前提下，为热点操作提供统一并行执行能力。

#### 主要工作

- 设计 `parallel_for` / `parallel_reduce` 抽象。
- 把并行调度和业务 kernel 解耦。
- 为以下模块接入并行：
  - 逐元素 core 运算
  - resize
  - 滤波
  - 颜色转换
- 建立小任务回退策略，避免线程开销反噬。

#### 阶段产出

- 统一并行接口
- `OpenMP` 后端实现

#### 验收目标

- 并行路径与标量路径数值一致。
- 中大尺寸图像上有稳定收益。
- 小图场景有回退逻辑，不出现明显退化。

### Phase 11：基于 `xsimd` 的 SIMD 路径

#### 阶段目标

为热点内核增加可维护、可回退的指令级并行实现。

#### 主要工作

- 设计统一 SIMD traits 与 dispatch。
- 优先向量化：
  - 逐元素 arithmetic
  - 颜色转换
  - 卷积内核
  - resize 内核
- 处理对齐、尾元素、非连续输入、通道打包。
- 将 xsimd 作为增强路径，而不是语义来源。

#### 阶段产出

- `simd.h`
- xsimd 特化实现
- SIMD benchmark 数据

#### 验收目标

- SIMD 路径对标 scalar 路径正确。
- 至少 3 类热点算子获得稳定收益。
- 关闭 `xsimd` 宏后能完整回退。

### Phase 12：集成、发布与长期维护基线

#### 阶段目标

将项目收敛为可发布、可评估、可维护的首版。

#### 主要工作

- 输出统一入口头与模块文档。
- 增加 example 和 quick start。
- 固化 benchmark 基线。
- 建立 CI：
  - GCC
  - Clang
  - 可选 OpenCV 差分 job
- 形成版本号、变更日志、兼容性说明。

#### 阶段产出

- 发布版文档
- 示例程序
- CI 配置
- `CHANGELOG`

#### 验收目标

- 用户可按文档直接 include 并运行示例。
- 核心模块都有测试、基线性能和限制说明。
- 发布边界清晰，已知限制可追踪。

## 7. 推荐里程碑切分

### Milestone A：基线闭环

包含：

- Phase 0
- Phase 1
- Phase 2
- Phase 3
- Phase 4

目标：

- 仓库成为真正可开发、可测试、可 include 的 header-only 项目基线。

### Milestone B：标量版 MVP

包含：

- Phase 5
- Phase 6
- Phase 7
- Phase 8
- Phase 9

目标：

- 交付一个不依赖 `OpenMP/xsimd` 的可用标量版 OpenCV 子集。

### Milestone C：性能增强版

包含：

- Phase 10
- Phase 11
- Phase 12

目标：

- 在保证正确性的前提下加入并行与 SIMD，并形成可发布版本。

## 8. 建议优先级排序

如果开发资源有限，必须严格按以下优先级推进：

1. 公共头文件自洽与 CMake 基线。
2. 命名空间、类型系统、错误系统统一。
3. `Mat` 数据模型冻结。
4. `core` 实现迁移为 header-only。
5. 测试与差分框架重建。
6. `core` 标量算子闭环。
7. `imgproc` 基础框架。
8. 滤波。
9. 颜色空间转换。
10. 最小 `imgcodecs`。
11. `OpenMP` 并行。
12. `xsimd` SIMD。
13. 发布和长期维护。

原因如下：

- 当前最大的风险不在“功能不够多”，而在“项目基线还没闭环”。
- `Mat` 未冻结前，所有上层模块都会返工。
- 测试未重建前，性能优化没有可信真值参考。

## 9. 阶段完成定义

只有同时满足以下条件，某阶段才算完成：

- 代码已经进入正式目录，而不是继续停留在 `core-cpp-trash`。
- 公共头可以独立编译。
- 对应单元测试通过。
- 必要的 OpenCV 差分测试通过。
- 文档已更新，限制已说明。
- 若该阶段涉及性能增强，则 benchmark 结果已记录。

## 10. 立刻执行的下一步

基于当前仓库状态，建议马上开始以下工作，而不是直接写 `imgproc` 或 SIMD：

### Next 1：完成 Phase 0 的闭环

- 先修 `cvh/cvh.h` 的 include 自洽问题。
- 把 `libnpy` 和 `context.h` 从公共头依赖链中移出。
- 搭好真正可用的顶层 CMake。

### Next 2：完成 Phase 1 的项目身份迁移

- 统一到 `cvh` 命名空间。
- 冻结首版类型系统和错误处理风格。

### Next 3：完成 Phase 2 的 `Mat` 语义冻结

- 不要急着加滤波、颜色转换或并行。
- 先把 `Mat` 的类型、通道、步长、ROI、clone/copyTo/convertTo 设计定住。

## 11. 一句话总结

这个项目现在最重要的不是“继续加更多 OpenCV 模块”，而是先把**迁移债务、公共头依赖、命名空间混乱、测试失真、Mat 模型未冻结**这五个根风险压平。只有这一步做好，后面的 `imgproc`、滤波、`RGB/YUV`、`OpenMP` 和 `xsimd` 才不会在实现中反复返工。
