# OpenCV Core / Imgproc 第一阶段落地计划

更新时间：2026-07-24

状态：完成（P1-S0 至 P1-S10）

## 1. 目标与范围

本文档是
[opencv-core-imgproc-three-phase-support-plan.md](opencv-core-imgproc-three-phase-support-plan.md)
中“第一阶段：高频基础算子与通用图像流水线”的具体落地方案。

第一阶段新增：

- Core：43 个操作族；
- Imgproc：36 个操作族；
- 合计：79 个操作族；
- 完成后累计覆盖：107 / 220，48.6%。

本阶段只承诺文档中明确列出的 cvh 类型、通道、参数和布局子集，不承诺一次性
复制 OpenCV 的全部重载。所有实现必须保持纯 header-only、CPU-only，并通过
公开 `cvh` 头文件使用。

## 2. 总体顺序

步骤必须按照依赖关系推进。后一步可以在前一步接口稳定后开始，但不能在依赖
步骤未通过验收时标记完成。

| Step | 内容 | Core | Imgproc | 依赖 | 状态 |
|---|---|---:|---:|---|---|
| P1-S0 | 基础类型与差分测试前置 | 0 | 0 | 当前基线 | 完成 |
| P1-S1 | Core 逐元素与逻辑运算 | 8 | 0 | P1-S0 | 完成 |
| P1-S2 | Core 转换、数学与校验 | 9 | 0 | P1-S0、P1-S1 公共循环 | 完成 |
| P1-S3 | Core 归约与统计 | 13 | 0 | P1-S0、P1-S1、P1-S2 | 完成 |
| P1-S4 | Core 布局、复制与通道操作 | 13 | 0 | P1-S0、P1-S1 | 完成 |
| P1-S5 | Imgproc 核生成、积分与导数 | 0 | 10 | P1-S0、P1-S3、P1-S4 | 完成 |
| P1-S6 | Imgproc 非线性滤波与强度处理 | 0 | 7 | P1-S3、P1-S5 | 完成 |
| P1-S7 | Imgproc 累积、金字塔与颜色输入 | 0 | 10 | P1-S3、P1-S4、P1-S5、P1-S6 | 完成 |
| P1-S8 | Imgproc 几何矩阵生成 | 0 | 5 | P1-S0、P1-S3 | 完成 |
| P1-S9 | Imgproc 几何采样与透视变换 | 0 | 4 | P1-S4、P1-S8 | 完成 |
| P1-S10 | 第一阶段整体收口 | 0 | 0 | P1-S1 至 P1-S9 | 完成 |
| **合计** |  | **43** | **36** |  |  |

推荐每个 Step 使用独立提交。单个 Step 过大时，可以拆成“接口与 scalar
correctness”“benchmark 与 fast path”“文档收口”多个提交，但必须在全部验收
条件满足后才能把 Step 标记为完成。

## 3. 通用落地规则

### 3.1 Upstream 基准

所有行为定义以本地 OpenCV 为准：

- 路径：`/Users/zmu/work/my_project/ocvh/opencv`
- Commit：`d48bf69f65`
- 版本：`4.14.0-pre`

迁移 OpenCV 代码时记录来源文件和 commit。只迁移完成算子所需的 CPU 算法与
kernel，不迁移 runtime dispatch、IPP、OpenCL、模块注册或 `.cpp` 依赖。

### 3.2 代码组织

- 公开 API 放在 `include/cvh/core/` 或 `include/cvh/imgproc/`。
- 较长实现放在同模块 `detail/*_impl.hpp`。
- 新公开头文件加入 `include/cvh/cvh.h` 或
  `include/cvh/imgproc/imgproc.h`。
- 所有定义必须 `inline`、模板化或具有其他 ODR-safe 形式。
- 不在业务 kernel 上增加新的 cvh SIMD adapter。
- Scalar 路径是完整正确性基线；OpenCV Universal Intrinsics 只用于
  benchmark 证明值得优化的热点。
- 不使用 xsimd，不增加 native `.cpp` 实现，不启用 RVV。

### 3.3 每个算子的最低验收规则

每个算子至少满足：

- 公开入口可以通过 `#include <cvh/cvh.h>` 调用。
- 支持矩阵写入对应模块 README。
- 不支持的类型、通道、flag 或布局必须明确报错。
- 连续 Mat 和支持范围内的 ROI/non-contiguous Mat 均有测试。
- OpenCV 支持原地执行时，cvh 必须测试原地或别名行为。
- 整数输出与 OpenCV 精确一致；浮点输出使用预先定义的容差。
- 新增头文件经过多翻译单元 ODR smoke。
- 性能热点加入 benchmark；没有稳定收益时保留 scalar 实现，不宣称 fast
  path。

### 3.4 差分测试边界

cvh 与 OpenCV 头文件不能在同一测试翻译单元中直接混用。差分测试应沿用
`benchmark/opencv_compare` 的隔离方式：

- cvh 测试端只包含 cvh；
- OpenCV backend 单独编译；
- 两侧通过基础类型参数、字节缓冲区和 POD 结果交换；
- 同一随机种子生成相同输入；
- 比较输出 shape、type、step、数值和错误边界。

## 4. P1-S0：基础类型与差分测试前置

### 目标

消除第一阶段中会阻塞几何矩阵、双精度计算和 OpenCV 差分测试的基础缺口。
本 Step 不增加操作族覆盖数。

### 如何落地

1. 将 `CV_64F` 纳入 `Mat` 的真实支持深度：
   - 增加 `CV_64FC(n)`、`CV_64FC1` 至 `CV_64FC4`；
   - 覆盖 element size、分配、clone/copy、setTo、at、convertTo、ROI 和
     scalar dispatch；
   - 将依赖连续深度区间的判断改为明确的支持集合，避免再次漏掉非连续编号
     depth。
2. 引入 OpenCV 风格浮点几何类型：
   - `Point_<T>`；
   - `Point2i`、`Point2f`、`Point2d`；
   - 保持现有 `Point` 源码兼容。
3. 根据几何 API 需要补齐 `Size_<T>`，并保持现有 `Size` 兼容。
4. 建立可复用的 OpenCV 差分 backend：
   - 支持二维连续 Mat 与 ROI 的输入描述；
   - 支持 U8、S16、S32、F32、F64 输出；
   - 支持精确比较和绝对/相对误差比较；
   - 不让 OpenCV 成为默认 header-only 测试的强制依赖。

### 验收标准

- [x] `CV_64FC1/C2/C3/C4` 能够 create、clone、copy、setTo、at 和
      convertTo。
- [x] `CV_32F <-> CV_64F`、`CV_8U <-> CV_64F` 转换结果通过手工值测试。
- [x] `CV_64F` ROI 的 `step`、`elemSize1`、`isContinuous` 与预期一致。
- [x] `Point` 现有测试无需修改即可通过。
- [x] `Point2f`、`Point2d` 的构造、比较和数值转换测试通过。
- [x] 差分 backend 至少完成一个 Core 和一个 Imgproc smoke case。
- [x] `cvh_header_compile_smoke`、`cvh_core_header_odr_smoke`、
      `cvh_imgproc_header_odr_smoke` 通过。
- [x] `./scripts/check_header_only_contract.sh` 通过。

完成记录（2026-07-24）：

- 默认测试构建中 Core 124 个 case、Imgproc 141 个 case 通过（2 个既有
  Core case 保持显式 skip）。
- OpenCV 隔离差分 smoke 的 Core `U8 -> F64` 和 Imgproc linear resize
  共 2 个 case 通过。
- header compile、Core/Imgproc ODR smoke 与 header-only contract 检查通过。

## 5. P1-S1：Core 逐元素与逻辑运算

### 算子

`absdiff`、`bitwise_and`、`bitwise_not`、`bitwise_or`、`bitwise_xor`、
`inRange`、`min`、`max`

### 如何落地

1. 在现有 `basic_op` typed dispatch 基础上提取可复用的逐元素遍历：
   - Mat/Mat；
   - Mat/Scalar；
   - mask 输出；
   - 连续与逐行 ROI。
2. `absdiff`、`min`、`max` 使用对应类型的数值语义和饱和规则。
3. bitwise 系列按 OpenCV 的元素原始位模式处理，不把浮点输入转换为整数值后
   再运算。
4. `inRange` 首批覆盖 Mat 下界/上界和 Scalar 下界/上界。
5. 首批支持现有 Mat 深度；无法与 OpenCV 明确对齐的组合必须显式拒绝，而
   不是静默转换。

### 验收标准

- [x] 8 个公开 API 均可从 `<cvh/cvh.h>` 调用。
- [x] Mat/Mat 与适用的 Mat/Scalar 路径均有测试。
- [x] C1、C3、C4 和宽度非 SIMD lane 整数倍的输入通过测试。
- [x] 连续、ROI/non-contiguous 和支持的原地路径通过测试。
- [x] 整数类型输出与 OpenCV byte-exact。
- [x] F32/F64 的 `absdiff`、`min`、`max` 在 NaN、Inf、正负零用例上与
      接受的 OpenCV 语义一致。
- [x] bitwise 浮点输入按位结果与 OpenCV byte-exact。
- [x] `inRange` 输出固定为 `CV_8UC1`，边界包含关系与 OpenCV 一致。
- [x] benchmark 至少包含 `absdiff`、`bitwise_and`、`inRange`、
      `min`、`max` 的 U8/F32 代表组合。
- [x] Core 测试、ODR smoke 和 header-only contract 全部通过。

完成记录（2026-07-24）：

- 新增 8 个公开 API 及其 Mat/Scalar、mask、raw-bit、C1/C3/C4、ROI 和
  原地基线；OpenCV 标准整数深度与 F32/F64 边界差分通过。
- Core 133 个 case 运行，131 个通过，2 个既有 case 显式 skip；
  OpenCV 隔离差分 5 个 case 通过。
- `core_mat` benchmark 已新增 5 个代表算子的 U8/F32 CSV 行；header
  compile、Core/Imgproc ODR 与 header-only contract 通过。

## 6. P1-S2：Core 转换、数学与校验

### 算子

`scaleAdd`、`convertScaleAbs`、`convertFp16`、`sqrt`、`pow`、`exp`、
`log`、`checkRange`、`patchNaNs`

### 如何落地

1. 建立 unary transform typed dispatch，复用 S1 的连续/逐行遍历。
2. `scaleAdd` 首批覆盖同 shape、同 type 的 Mat 输入。
3. `convertScaleAbs` 对齐 OpenCV 的缩放、偏移、绝对值和 U8 饱和顺序。
4. `convertFp16` 对齐 OpenCV 的 FP32 与半精度位表示转换；文档明确
   `CV_16S` 半精度存储与 cvh 内部 `CV_16F` 类型的关系。
5. `sqrt`、`pow`、`exp`、`log` 首批覆盖 F32/F64。
6. `checkRange` 返回首个非法位置；`patchNaNs` 首批覆盖 F32，并在
   CV_64F 是否接受上与 upstream contract 保持一致。

### 验收标准

- [x] 9 个 API 均为 header-defined 且从 umbrella header 可见。
- [x] `convertScaleAbs` 对负数、溢出、舍入边界与 OpenCV byte-exact。
- [x] `convertFp16` 覆盖普通值、subnormal、Inf、NaN 和正负零。
- [x] F32/F64 数学函数覆盖零、负输入、Inf、NaN 和大数边界。
- [x] 浮点误差标准在测试中固定，不能以扩大容差掩盖实现差异。
- [x] `checkRange` quiet/position 行为和多通道输入边界有测试。
- [x] `patchNaNs` 只替换 NaN，不修改有限值和 Inf。
- [x] 连续、ROI/non-contiguous 与支持的原地路径通过。
- [x] benchmark 至少包含 `convertScaleAbs`、`sqrt`、`exp`、`log`。
- [x] S1 全部回归测试保持通过。

完成记录（2026-07-24）：

- 新增独立 `core/math.h` 与 header-only 实现，9 个 API 覆盖连续、ROI 和
  支持的原地路径。
- `convertScaleAbs` 与 FP16 bits 通过 byte-exact 差分；F32/F64 数学函数
  使用固定误差界通过 upstream 差分。
- pinned OpenCV 的 `pow` 头注释与 CPU 实现不一致，cvh 按实际 CPU 行为在
  负底数非整数幂时返回 NaN；`patchNaNs` 同样按 pinned CPU 实现仅接受 F32。
- Core 142 个 case 运行，140 个通过，2 个既有 case 显式 skip；OpenCV
  隔离差分 7 个 case 通过，ODR 与 header-only contract 通过。

## 7. P1-S3：Core 归约与统计

### 算子

`norm`、`sum`、`mean`、`meanStdDev`、`countNonZero`、`hasNonZero`、
`findNonZero`、`minMaxIdx`、`minMaxLoc`、`reduce`、`reduceArgMax`、
`reduceArgMin`、`normalize`

### 如何落地

1. 将当前仅声明的 `norm` 变成完整 inline 实现，不保留链接期缺口。
2. 建立共享归约框架：
   - 每个线程或分块使用局部 accumulator；
   - 明确整数提升和 F32/F64 累积精度；
   - 固定合并顺序，避免线程数改变导致不可控漂移。
3. 首批 `norm` 支持 `NORM_INF`、`NORM_L1`、`NORM_L2` 及双输入形式。
4. `sum`、`mean`、`meanStdDev` 首批覆盖 C1 至 C4 和适用 mask。
5. `countNonZero`、`hasNonZero`、`findNonZero` 首批限定单通道。
6. `minMaxLoc/minMaxIdx` 明确多通道限制和并列值的索引选择规则。
7. `reduce/reduceArg*` 首批覆盖二维 axis 0/1。
8. `normalize` 首批覆盖 `NORM_INF/L1/L2/MINMAX`。

### 验收标准

- [x] `norm` 不再处于 Declared only 状态。
- [x] 13 个 API 均有公开 contract 和 header-only 定义。
- [x] 空 Mat、单元素、全零、全相同、NaN/Inf 和大尺寸输入有测试。
- [x] U8、S16、S32、F32、F64 代表组合与 OpenCV 对比通过。
- [x] C1/C3/C4、mask、ROI/non-contiguous 的适用组合通过。
- [x] 并列 min/max、argmin/argmax 的首/末索引语义有固定测试。
- [x] `reduce` axis 和 dtype 转换输出 shape/type 与 OpenCV 一致。
- [x] 单线程和多线程结果满足同一数值容差。
- [x] benchmark 包含 `sum`、`meanStdDev`、`norm`、`minMaxLoc`、
      `reduce`、`normalize`，并记录单线程及项目默认线程数。
- [x] coverage 文档将 `norm` 从 Declared only 移出。

完成记录（2026-07-24）：

- 新增 `core/reduce.h` 和 ODR-safe header 实现，13 个归约、统计及谓词
  API 均可从 `<cvh/cvh.h>` 调用。
- contract 覆盖 C1/C3/C4、mask、非连续 ROI、N-D extrema、空/单元素/
  全零/全相同/NaN/Inf、大输入、原地 normalize 和 reduce alias。
- OpenCV 隔离差分覆盖 U8/S16/S32/F32/F64；上游不支持的
  `CV_32S -> CV_64F reduce` 组合只保留 cvh contract，不伪造差分结果。
- `core_mat` benchmark 新增 6 类归约行，并记录单线程和项目默认线程数；
  当前实现保持确定性串行 header loop，不宣称并行 fast path。
- Core 154 个 case 运行，152 个通过，2 个既有 case 显式 skip；Imgproc
  141 个 case 和 OpenCV 隔离差分 10 个 case 全部通过。CTest 16/16、
  ODR smoke 与 header-only contract 均通过。

## 8. P1-S4：Core 布局、复制与通道操作

### 算子

`borderInterpolate`、`copyTo`、`extractChannel`、`insertChannel`、
`mixChannels`、`flip`、`flipND`、`rotate`、`repeat`、`hconcat`、
`vconcat`、`broadcast`、`swap`

### 如何落地

1. 把现有 imgproc border 坐标逻辑提升为可复用的公开
   `borderInterpolate`。
2. `copyTo` 复用 `Mat::copyTo`，增加 upstream free-function 的 mask
   语义。
3. `extractChannel/insertChannel/mixChannels` 使用统一 channel routing
   描述，避免三套复制循环。
4. `flip/rotate/repeat/hconcat/vconcat` 使用 step-aware block copy。
5. `flipND` 和 `broadcast` 单独建立 N-D shape/stride 验证，不用二维
   假设强行扩展。
6. `swap` 只交换 Mat header/ownership，不复制像素。

### 验收标准

- [x] 13 个 API 均从 Core public header 可见。
- [x] 所有 Mat depth 和通道数在纯复制类操作中保持原始字节。
- [x] `copyTo` 无 mask、全零 mask、全一 mask、部分 mask 与 OpenCV
      一致。
- [x] channel extract/insert/mix 覆盖 C1/C2/C3/C4 和通道重排。
- [x] 输入输出别名、同 Mat 原地、相交 ROI 的策略明确且测试覆盖。
- [x] `flip` 水平/垂直/双向及 `rotate` 90/180/270 度通过。
- [x] `hconcat/vconcat` 对类型或非拼接维不匹配显式报错。
- [x] `broadcast/flipND` 覆盖 1D、2D、3D、前导 1 维和非法 shape。
- [x] `swap` 的 data 指针、shape、step、refcount 交换行为通过测试。
- [x] benchmark 包含 `copyTo(mask)`、`mixChannels`、`flip`、
      `hconcat/vconcat` 和 `broadcast`。

完成记录（2026-07-24）：

- 新增 `core/array.h` 与 ODR-safe 实现，13 个布局、复制和通道 API 从
  `<cvh/cvh.h>` 可见；imgproc 的 border helper 改为委托公开
  `borderInterpolate`。
- mask copy 对齐新输出清零和预分配输出保留语义；共享存储、同 Mat、
  相交 ROI 和 concat 输出 alias 在写入前保留可靠 source snapshot。
- byte-preserving contract 覆盖当前 Mat 深度及 C1/C2/C3/C4；2D/N-D
  flip、三种 rotate、broadcast 尾维规则、channel routing 与 swap
  ownership 均有固定测试。
- OpenCV 隔离差分对 U8/S16/S32/F32/F64、C1/C3/C4 做 byte-exact
  比较，共 13 个 case 通过。
- `core_mat` benchmark 新增 mask copy、mixChannels、flip、hconcat、
  vconcat 和 broadcast；这些行是性能基线，不宣称 fast path。
- Core 164 个 case 运行，162 个通过，2 个既有 case 显式 skip；CTest
  16/16、ODR smoke 和 header-only contract 全部通过。

## 9. P1-S5：Imgproc 核生成、积分与导数

### 算子

`getStructuringElement`、`getGaussianKernel`、`getDerivKernels`、
`getGaborKernel`、`createHanningWindow`、`integral`、`Scharr`、
`Laplacian`、`spatialGradient`、`sqrBoxFilter`

### 如何落地

1. 新增共享 kernel generator：
   - 矩形、十字、椭圆 morphology kernel；
   - Gaussian、derivative、Gabor、Hanning 系数；
   - F32/F64 输出策略。
2. `integral` 独立实现行累计与跨行累计，支持 sum，并为 sqsum/tilted
   后续扩展保留清晰边界。
3. `Scharr` 复用现有 Sobel/sepFilter2D 路径。
4. `Laplacian` 复用二阶 derivative kernel 与 filter path。
5. `spatialGradient` 共享一次输入遍历或共享 Sobel 中间结果。
6. `sqrBoxFilter` 复用 boxFilter border/窗口逻辑，但使用平方后高精度
   累积。

### 首批支持矩阵

- Kernel generator：F32/F64；
- `integral`：U8 输入，S32/F64 sum 输出；
- `Scharr/Laplacian/spatialGradient`：沿用现有 Sobel 的 U8/S16/F32
  主矩阵；
- `sqrBoxFilter`：U8/F32，C1/C3/C4，常用 border。

### 验收标准

- [x] 10 个 API 均加入 `imgproc/imgproc.h`。
- [x] kernel 尺寸、对称性、归一化、anchor 和非法参数有测试。
- [x] Gaussian 系数和为 1；Hanning 按 upstream 的二维窗开方语义检查
      零边缘、对称性和峰值，F32/F64 使用固定容差。
- [x] `integral` 输出尺寸、首行首列零边界和溢出策略与 OpenCV 一致。
- [x] Scharr/Laplacian 对常量、斜坡、冲击图像及边界像素对比通过。
- [x] spatialGradient 的 dx/dy 与独立 Sobel/Scharr 结果一致。
- [x] sqrBoxFilter 对 U8 大窗口不发生错误的窄类型溢出。
- [x] C1/C3/C4、ROI、奇数尺寸和常用 border 通过。
- [x] benchmark 包含 `integral`、`Scharr`、`Laplacian`、
      `sqrBoxFilter`。

完成记录（2026-07-24）：

- 新增 `kernels.h`、`integral.h`、`derivatives.h` 和
  `sqr_box_filter.h`，10 个 API 均为公开 inline header 实现。
- generator 覆盖 morphology rect/cross/ellipse/diamond、Gaussian、
  Sobel/Scharr derivative、Gabor 和 Hanning 的 F32/F64 子集。
- 计划原先把 Hanning 写成“系数和为 1”，但 pinned OpenCV 实际生成
  `sqrt(row_window * column_window)` 且不做总和归一化；contract 已按实际
  upstream 行为修正。
- integral 首批覆盖 U8 C1/C3/C4 到 S32/F64；Scharr、Laplacian 和
  spatialGradient 复用现有 Sobel 采样边界；sqrBoxFilter 使用宽累积。
- OpenCV 隔离 differential 15/15；Core 164 个 case 运行（162 pass、
  2 个既有 skip），Imgproc 148/148，CTest 16/16 和 header-only
  contract 全部通过。
- `imgproc_header` benchmark 已新增 integral、Scharr、Laplacian 和
  sqrBoxFilter；当前均记录为 scalar/public-header 基线。

## 10. P1-S6：Imgproc 非线性滤波与强度处理

### 算子

`medianBlur`、`bilateralFilter`、`stackBlur`、`adaptiveThreshold`、
`thresholdWithMask`、`equalizeHist`、`applyColorMap`

### 如何落地

1. `medianBlur` 先建立通用正确性路径，再根据 kernel size 选择排序网络或
   histogram fast path。
2. `bilateralFilter` 将空间权重和颜色权重预计算，避免在像素内层重复计算。
3. `stackBlur` 使用稳定的滑动窗口累计，不与 GaussianBlur 混为同一语义。
4. `adaptiveThreshold` 复用 box/Gaussian filter 与第一阶段 integral。
5. `thresholdWithMask` 复用现有 threshold kernel，只对 mask 命中位置
   写入。
6. `equalizeHist` 使用固定 256-bin U8 histogram。
7. `applyColorMap` 先支持内置 colormap 和用户 256-entry LUT 的明确子集。

### 首批支持矩阵

- `medianBlur`：U8/F32，C1/C3/C4，常用奇数 kernel；
- `bilateralFilter`：U8/F32，C1/C3；
- `stackBlur`：U8，C1/C3/C4；
- `adaptiveThreshold/equalizeHist`：U8C1；
- `thresholdWithMask`：继承现有 threshold 的 U8/F32 范围；
- `applyColorMap`：U8C1 输入，U8C3 输出。

### 验收标准

- [x] 7 个 API 均有独立 public contract。
- [x] kernel size、sigma、block size、C 值、colormap id 的非法参数显式
      报错。
- [x] median/bilateral/stack 对边界、单行、单列和小于 kernel 的图像通过。
- [x] adaptiveThreshold 的 MEAN/Gaussian 与 BINARY/BINARY_INV 组合通过。
- [x] thresholdWithMask 未命中像素的保持/初始化语义与 upstream contract
      一致。
- [x] equalizeHist 对全常量、双峰和完整 0..255 ramp 通过。
- [x] applyColorMap 的通道顺序与 OpenCV BGR 输出一致。
- [x] ROI/non-contiguous 输入通过；支持的原地行为有明确测试。
- [x] benchmark 包含 median 3x3/5x5、bilateral、adaptiveThreshold、
      equalizeHist。
- [x] 只有稳定快于 scalar 的路径才记录为 fast path。

### 执行记录

- [x] P1-S6.0：核对 upstream 签名、边界、原地和 mask 语义。
- [x] P1-S6.1：落地 7 个 public header-only scalar contract。
- [x] P1-S6.2：补齐参数、边界、ROI/non-contiguous 和 alias 测试。
- [x] P1-S6.3：接入隔离 OpenCV 差分 gate。
- [x] P1-S6.4：接入 benchmark、同步支持矩阵并执行阶段 gate。

完成记录：新增 7 个 contract test，覆盖边界、小图、ROI、alias 和非法参数；
9 个隔离 upstream 代表路径通过。CTest `16/16`、完整 OpenCV 差分
`16/16`、header-only contract `5/5`。benchmark 新增 median 3x3/5x5、
bilateral、adaptiveThreshold 和 equalizeHist scalar baseline。首批内置
colormap 固定为
`AUTUMN/JET/WINTER/COOL/HOT`；用户 `CV_8UC1/CV_8UC3` 256-entry LUT
作为完整路径。其它内置 id 在本阶段显式报错。

## 11. P1-S7：Imgproc 累积、金字塔与颜色输入

### 算子

`accumulate`、`accumulateProduct`、`accumulateSquare`、
`accumulateWeighted`、`blendLinear`、`pyrDown`、`pyrUp`、
`buildPyramid`、`cvtColorTwoPlane`、`demosaicing`

### 如何落地

1. 累积系列共享输入转换、mask 和 F32/F64 destination 更新循环。
2. `blendLinear` 明确两幅输入和两张权重图的 shape/type 约束。
3. `pyrDown/pyrUp` 复用 Gaussian kernel 和 border 逻辑，不通过多次
   `resize` 近似。
4. `buildPyramid` 逐层调用同一 pyrDown contract。
5. `cvtColorTwoPlane` 复用现有 NV12/NV21 YUV kernel，公开 Y plane 和
   UV plane 两输入形式。
6. `demosaicing` 首批覆盖常见 Bayer BG/GB/RG/GR 到 BGR/RGB 的线性
   插值路径。

### 首批支持矩阵

- accumulate 系列：U8/F32 source，F32 destination，C1/C3/C4；
- blendLinear：U8/F32 image，F32C1 weights；
- pyramid：U8/F32，C1/C3/C4；
- cvtColorTwoPlane：U8 Y + interleaved UV，NV12/NV21 到 BGR/RGB；
- demosaicing：U8 Bayer 单通道到 U8 三通道。

### 验收标准

- [x] 10 个 API 均从 imgproc umbrella 可见。
- [x] accumulate 四个变体覆盖无 mask、部分 mask 和多次累计。
- [x] accumulateWeighted 的 alpha=0、1 和中间值与 OpenCV 容差一致。
- [x] blendLinear 对零权重、互补权重和非归一化权重通过。
- [x] pyrDown/pyrUp 输出尺寸取整、边界和奇数宽高与 OpenCV 一致。
- [x] buildPyramid 每一层等价于连续调用 pyrDown。
- [x] cvtColorTwoPlane 覆盖 NV12/NV21、奇偶尺寸约束和 ROI step。
- [x] demosaicing 覆盖四种 Bayer pattern、边界和通道顺序。
- [x] benchmark 包含 accumulateWeighted、blendLinear、pyrDown、
      pyrUp、NV12/NV21 decode 和 Bayer decode。
- [x] 现有单 Mat YUV `cvtColor` 回归保持通过。

### 执行记录

- [x] P1-S7.0：核对 upstream 签名、目标初始化、尺寸和 plane contract。
- [x] P1-S7.1：落地 accumulate 系列与 blendLinear。
- [x] P1-S7.2：落地 pyrDown/pyrUp/buildPyramid。
- [x] P1-S7.3：落地 cvtColorTwoPlane 与线性 demosaicing。
- [x] P1-S7.4：补 contract、隔离 OpenCV 差分和 benchmark。
- [x] P1-S7.5：同步覆盖矩阵并执行阶段 gate。

完成记录：新增 6 组 contract 和 13 条隔离 upstream 代表路径。修正
`pyrUp` 在插零 2x 网格上的边界相位，并按 upstream 复制 Bayer 一像素
边界。CTest `16/16`、OpenCV 差分 `17/17`、header-only contract
`5/5`；benchmark 新增 7 条 scalar baseline，覆盖统计达到 Imgproc
`41/123`。

## 12. P1-S8：Imgproc 几何矩阵生成

### 算子

`getAffineTransform`、`getPerspectiveTransform`、
`getRotationMatrix2D`、`getRotationMatrix2D_`、
`invertAffineTransform`

### 如何落地

1. 使用 P1-S0 的 Point2f/Point2d 和 CV_64F。
2. 对固定 2x3/3x3 小矩阵使用局部求解，不提前依赖第二阶段通用
   `solve/invert`。
3. `getRotationMatrix2D` 返回 OpenCV 风格 2x3 CV_64F Mat。
4. `getRotationMatrix2D_` 返回项目中明确记录的固定矩阵值类型；若 cvh
   暂无 Matx，则先提供不引入 ABI 模糊的窄类型。
5. 对退化点集和不可逆 affine matrix 明确行为。

### 验收标准

- [x] 5 个 API 均公开可用。
- [x] identity、translation、scale、rotation 和 shear 手工用例通过。
- [x] 三点 affine、四点 perspective 的结果与 OpenCV F64 容差一致。
- [x] getRotationMatrix2D 保持旋转中心不变。
- [x] invertAffineTransform 正向与逆向复合后接近 identity。
- [x] F32/F64 输入点及大坐标、小尺度数值用例通过。
- [x] 重复点、共线点和奇异矩阵行为有固定测试。
- [x] 输出 type、shape 和矩阵元素布局与文档一致。

### 执行记录

- [x] P1-S8.0：固定 `AffineMatrix2x3d` 返回类型与退化输入语义。
- [x] P1-S8.1：实现 rotation、affine、perspective 和 affine inverse。
- [x] P1-S8.2：补手工 contract 与隔离 OpenCV 差分。
- [x] P1-S8.3：同步覆盖矩阵并执行阶段 gate。

完成记录：`getRotationMatrix2D_` 返回 `AffineMatrix2x3d`；固定小矩阵
求解器使用列尺度归一化，避免大坐标有效点集被误判为退化。新增 5 组手工
contract 和 1 组隔离 upstream 矩阵差分；默认 CTest `16/16`、OpenCV
差分 `18/18`、header-only contract `5/5`。覆盖统计达到 Imgproc
`46/123`。

## 13. P1-S9：Imgproc 几何采样与透视变换

### 算子

`remap`、`convertMaps`、`warpPerspective`、`getRectSubPix`

### 如何落地

1. 以 `remap` 为唯一通用采样内核：
   - 坐标读取；
   - nearest/linear 插值；
   - border 处理；
   - step-aware 行访问。
2. 将现有 `warpAffine` 可复用的采样逻辑下沉到共享 detail 层，但不得改变
   现有公开行为。
3. `convertMaps` 首批支持：
   - `CV_32FC1 + CV_32FC1`；
   - `CV_32FC2`；
   - `CV_16SC2 + CV_16UC1` 固定点表示。
4. `warpPerspective` 只负责生成透视坐标并调用共享采样内核。
5. `getRectSubPix` 使用同一 bilinear 采样与 border 规则。

### 首批支持矩阵

- Source：U8/F32，C1/C3/C4；
- Interpolation：`INTER_NEAREST`、`INTER_LINEAR`；
- Border：CONSTANT、REPLICATE、REFLECT、REFLECT_101；
- Transform matrix：F32/F64；
- Map：F32 pair/F32C2/固定点 pair。

### 验收标准

- [x] 4 个 API 均公开可用。
- [x] remap identity、整数平移、亚像素平移和越界坐标与 OpenCV 对比通过。
- [x] 三种 map 表示对同一坐标场产生一致结果。
- [x] convertMaps 正向转换后的 remap 输出与浮点 map 在接受容差内一致。
- [x] warpPerspective identity、平移、缩放、旋转组合和真实透视用例通过。
- [x] `WARP_INVERSE_MAP` 行为与 OpenCV 一致。
- [x] getRectSubPix 覆盖整数中心、半像素中心、边缘和输出类型转换。
- [x] 连续、ROI/non-contiguous、奇数尺寸和宽度尾部通过。
- [x] 支持的 in-place/alias 行为明确并通过测试。
- [x] 现有 warpAffine 全部回归通过。
- [x] benchmark 包含 remap float/fixed map、warpPerspective 和
      getRectSubPix，并区分 public/scalar/fast dispatch。

### 执行记录

- [x] P1-S9.0：固定三种 map 表示、插值量化、边界和 alias contract。
- [x] P1-S9.1：提取共享 geometric sampler，并让 warpAffine 回归复用。
- [x] P1-S9.2：实现 remap 与 convertMaps。
- [x] P1-S9.3：实现 warpPerspective 与 getRectSubPix。
- [x] P1-S9.4：补手工 contract、隔离 OpenCV 差分和 benchmark。
- [x] P1-S9.5：同步覆盖矩阵并执行阶段 gate。

完成记录：7 组手工 contract 和 5 条隔离 upstream 代表路径通过。
Mode A/B 已加入 float/fixed remap、warpPerspective 和 getRectSubPix，
compare quick smoke 生成 46 行；新增路径明确标为
`public_header_scalar`，不宣称 fast path。默认 CTest `16/16`、OpenCV
差分 `19/19`、header-only contract `5/5`。

## 14. P1-S10：第一阶段整体收口

### 如何落地

1. 逐项核对第一阶段 79 个操作族，没有遗漏、重复或仅声明未定义的接口。
2. 更新：
   - `doc/opencv-core-imgproc-api-coverage.md`；
   - `doc/opencv-core-imgproc-three-phase-support-plan.md`；
   - 根 README Operator Status；
   - Core/Imgproc 模块 README；
   - benchmark 操作矩阵和最新性能报告。
3. 检查新 public header 的 include 独立性和 umbrella 编译成本。
4. 运行完整 header-only、测试和 benchmark gate。

### 验收标准

- [x] Core 新增 43 个、Imgproc 新增 36 个，共 79 个操作族。
- [x] 覆盖文档统计变为 Core 57 / 97、Imgproc 50 / 123，总计
      107 / 220。
- [x] 第一阶段不存在 Declared only API。
- [x] 每个 API 都能追溯到 public header、contract test 和支持矩阵。
- [x] 所有不支持组合均显式失败，不存在静默错误结果。
- [x] Mode A 包含第一阶段主要热点的稳定基线。
- [x] Mode B 报告能够展示新增热点与 OpenCV 的差距。
- [x] `cvh::headers` 与 `cvh::headers_fast` API 集合一致。
- [x] 没有新增产品 `.cpp`、native 依赖或 xsimd 依赖。
- [x] ARM NEON 与 x86 AVX/SSE smoke 保持通过；RVV 仍未启用。

最终命令：

```bash
cmake -S . -B build-api-p1 \
  -DCVH_BUILD_NATIVE_BACKEND=OFF \
  -DCVH_BUILD_TESTS=ON \
  -DCVH_BUILD_BENCHMARKS=ON
cmake --build build-api-p1 -j
ctest --test-dir build-api-p1 --output-on-failure
./scripts/check_header_only_contract.sh
./scripts/ci_headers_all.sh
./scripts/ci_benchmark_headers_quick.sh
git diff --check
```

只有上述验收全部通过，才可以把第一阶段标记为完成并进入第二阶段。

### 执行记录

- [x] P1-S10.0：审计 79 个新增操作族的 public header、定义、测试与支持矩阵。
- [x] P1-S10.1：同步根 README、模块 README、覆盖矩阵与阶段列表。
- [x] P1-S10.2：更新 Mode A/B 操作矩阵和 upstream 性能报告。
- [x] P1-S10.3：运行完整 header-only、测试、benchmark 与架构 smoke gate。

完成状态：第一阶段清单计数确认为 Core 43、Imgproc 36；
所有分组均有 public header、contract test 和
`opencv-core-imgproc-api-coverage.md` 支持矩阵记录，没有第一阶段
Declared-only 入口。

追溯审计：

| Step | 操作族数 | Public headers | Contract test |
|---|---:|---|---|
| P1-S1 | 8 | `core/basic_op.h` | `core/binary_op_contract_test.cpp` |
| P1-S2 | 9 | `core/math.h` | `core/math_ops_contract_test.cpp` |
| P1-S3 | 13 | `core/reduce.h` | `core/reduction_ops_contract_test.cpp` |
| P1-S4 | 13 | `core/array.h` | `core/array_ops_contract_test.cpp`、`core/layout_ops_contract_test.cpp` |
| P1-S5 | 10 | `imgproc/kernels.h`、`integral.h`、`derivatives.h`、`sqr_box_filter.h` | `imgproc/imgproc_phase1_kernels_contract_test.cpp` |
| P1-S6 | 7 | `imgproc/median_blur.h`、`bilateral_filter.h`、`stack_blur.h`、`adaptive_threshold.h`、`threshold.h`、`equalize_hist.h`、`colormap.h` | `imgproc/imgproc_phase1_intensity_contract_test.cpp` |
| P1-S7 | 10 | `imgproc/accumulate.h`、`blend_linear.h`、`pyramid.h`、`cvtcolor_two_plane.h`、`demosaicing.h` | `imgproc/imgproc_phase1_pyramid_color_contract_test.cpp` |
| P1-S8 | 5 | `imgproc/geometry_transform.h` | `imgproc/imgproc_phase1_geometry_matrix_contract_test.cpp` |
| P1-S9 | 4 | `imgproc/convert_maps.h`、`remap.h`、`warp_perspective.h`、`rect_sub_pix.h` | `imgproc/imgproc_phase1_geometric_sampling_contract_test.cpp` |
| **合计** | **79** | 均由 module umbrella 和 `<cvh/cvh.h>` 公开 | 均进入默认 CTest；代表路径进入隔离 OpenCV 差分 |

Mode B 初始 stable 报告（2026-07-24）包含 188 个有效 case；随后补齐
第一阶段 79/79 个操作族并运行 full profile，当前报告包含 321 个 case，
其中 320 个有效、1 个既有 `cvtColor` 组合明确记录为 unsupported。P1 新增
操作族共有 92 个性能 case。scalar 几何采样路径仍作为后续共享 sampler
SIMD/分块优化的基线，不宣称当前已具备 fast path。

最终 gate（2026-07-24）：

- 配置和完整构建通过，`CVH_BUILD_NATIVE_BACKEND=OFF`。
- 默认 CTest `16/16` 通过；OpenCV 隔离差分 `19/19` 通过。
- `ci_headers_all.sh` 通过：Core `162` 个 case 通过、`2` 个已记录的
  `OutputArray` 非目标 case 跳过；Imgproc `173/173` 通过。
- header-only contract `5/5` 通过，public header dependency 检查通过。
- Mode A quick benchmark 构建并运行完成；Mode B full 报告覆盖 P1
  `79/79` 个操作族，共 `321` 个 case、`320` 个有效结果和 `1` 个明确
  unsupported 结果。
- Apple ARM64 主机上的 Universal Intrinsics/NEON smoke 实际运行通过；
  x86_64 AVX2 smoke 使用 Apple Clang `-target x86_64-apple-macos13
  -mavx2 -fsyntax-only` 交叉编译通过。x86 运行时性能仍由 x86 CI/机器
  验证。
- `git diff --check` 通过；产品代码未增加 `.cpp`、native、xsimd 或 RVV
  依赖。
