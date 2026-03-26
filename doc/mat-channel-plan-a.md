# Mat 通道语义方案 A 落地文档

- 文档版本：v1.0
- 日期：2026-03-26
- 适用范围：`cvh::Mat`、`core` 兼容层、后续 `imgproc`/`imgcodecs` 的输入输出约定
- 对齐目标：OpenCV 接口兼容优先

## 1. 背景与问题

当前项目已经具备多通道基础能力（`type` 编码、`channels()`、`elemSize()`、`step` 计算）。
但在设计讨论层面仍存在一个歧义：

- 一种理解：`channel` 是 `Mat` 的显式维度（如 `[C,H,W]` 或 `[W,H,C]`）。
- 另一种理解：`channel` 是元素类型的一部分（OpenCV 风格 `CV_8UC3`）。

如果这个问题不先冻结，后续 `merge/split/compare/reinterpret`、ROI 步长、上游测试迁移都会持续产生语义冲突。

## 2. 核心决策（方案 A）

方案 A：**`Mat` 严格采用 OpenCV 语义；张量布局由独立适配层承载。**

### 2.1 语义冻结

1. `channels` 只来自 `type`（`CV_MAKETYPE/CV_MAT_CN`），不是 `dims` 的某一轴。
2. `dims/shape` 只表达几何维度，不表达通道数。
3. 连续内存下，`Mat` 默认是交织通道（interleaved）布局。
4. `Mat({C,H,W}, CV_32F)` 的语义是“3D 单通道张量”，不是“CHW 图像”。
5. 任意“CHW/HWC/NCHW/NHWC”语义，必须通过显式适配 API 或转换函数表达。

### 2.2 为什么选 A

- 与 OpenCV 接口心智一致，迁移成本最低。
- 避免把“图像容器语义”和“张量布局语义”混在同一类型里。
- 为后续上游兼容用例（`merge/split/compare/reinterpret`）提供稳定前提。
- 与现有实现和 `mat-contract-v1` 已对齐，落地成本低。

## 3. 数据模型与行为合同

### 3.1 Mat 基础模型

- `type = depth + channels`。
- `total()` 统计的是几何元素数，不乘 `channels`。
- 标量元素总数为 `total() * channels()`。
- `step(dim)` 与 `step1(dim)` 只基于几何维度推导。

### 3.2 关键 API 合同

- `create/clone/copyTo/setTo/convertTo`：多通道与单通道统一规则。
- `convertTo(dst, depth_only)`：保持源 `channels` 不变。
- `convertTo(dst, full_type)`：若 `channels` 不一致，返回 `StsBadArg`。
- `reshape`：只改几何形状，不改通道语义。
- `rowRange/colRange/operator()(Range, Range)`：维持浅视图与步长语义。

### 3.3 明确禁止（防止语义漂移）

- 禁止把 `dims` 最后一维自动解释为 channel。
- 禁止根据 `shape` 猜测 channel。
- 禁止在 API 内隐式进行 HWC/CHW 重排。

## 4. 张量布局适配层（与 Mat 解耦）

`Mat` 不承载框架张量布局语义。新增一个轻量适配层（可放在 `core/detail` 或独立 `tensor` 子目录）：

- `enum class LayoutTag { HWC, CHW, NHWC, NCHW }`
- `struct TensorView { void* data; std::vector<int> shape; std::vector<size_t> stride; int depth; LayoutTag layout; }`

设计规则：

1. 适配层显式声明布局，不允许隐式推断。
2. HWC 与 `Mat(CV_*C*)` 可在满足条件时零拷贝映射。
3. CHW/NCHW 默认走显式重排函数（可选 in-place 优化，但不是语义前提）。
4. 任何跨布局转换函数必须在命名上体现方向，如：
   - `matInterleavedToCHW(...)`
   - `chwToMatInterleaved(...)`

## 5. 迁移与实现分阶段

### Phase A1：语义冻结与文档收口（当前阶段）

- 在设计文档与 `mat-contract-v1` 明确“channel 非维度”。
- 增加“歧义用法禁止项”。
- 在 `README` 的兼容声明中补充一段 Mat 语义说明。

### Phase A2：适配层最小可用

- 提供 `LayoutTag` 与 `TensorView` 最小结构。
- 提供 2 个显式转换入口：
  - `Mat(interleaved) -> TensorView(CHW/HWC)`
  - `TensorView(CHW/HWC) -> Mat(interleaved)`
- 出错统一走 `cvh::error/CV_Error`。

### Phase A3：上游兼容能力推进

- 按 `channel_manifest` 解锁：
  - `merge/split`
  - `compare`
  - `Mat::reinterpret`
  - MatExpr 相关通道语义
- 每解锁一项，将对应 case 从 `PENDING_CHANNEL` 升级为 `PASS_NOW`。

## 6. 测试与验收

### 6.1 合同测试（必须）

1. `CV_MAT_CN/CV_MAKETYPE` 编解码正确。
2. `create/setTo/copyTo/convertTo` 覆盖多通道元素。
3. 非连续 ROI + 多通道在 `copyTo/convertTo/setTo/clone` 上行为稳定。
4. `convertTo(depth_only)` 保持通道数；`convertTo(type_with_diff_cn)` 报错。
5. `reshape` 仅影响几何维度，不产生通道歧义。

### 6.2 适配层测试（新增）

1. HWC 零拷贝映射条件验证（可映射/不可映射分支）。
2. CHW<->interleaved 重排正确性（值、shape、stride）。
3. 大尺寸与非连续输入的错误路径验证。
4. 空输入、非法 layout、通道不匹配的错误码验证。

### 6.3 上游对齐测试（持续）

- 以 `test/upstream/opencv/core/channel_manifest.json` 为台账。
- 不允许 silent skip，pending 必须附带 `reason/unblock_by`。

## 7. 风险与缓解

1. 风险：历史代码继续把 `[C,H,W]` 当 Mat 通道语义。
   - 缓解：加 lint/评审规则，新增相关注释与测试用例。
2. 风险：适配层命名不清导致误用。
   - 缓解：函数名必须包含布局方向，禁止通用 `convertLayout` 模糊接口。
3. 风险：后续算子偷偷引入隐式布局转换。
   - 缓解：在 API 评审模板中加入“是否隐式重排”检查项。

## 8. 非目标（本阶段不做）

- 不把 `Mat` 改造成通用张量容器。
- 不在 v1 追求 CHW 零拷贝全覆盖。
- 不承诺与 OpenCV ABI/对象内存布局兼容。

## 9. 完成定义（DoD）

满足以下条件即认为方案 A 落地完成：

1. 文档层：`design + mat-contract + 本文档` 三处语义一致。
2. 代码层：`Mat` 不再存在 channel 维度歧义入口。
3. 测试层：合同测试和适配层测试通过，且无 silent pending 漏洞。
4. 兼容层：`channel_manifest` 可追踪每个未完成项与解锁条件。

## 10. 一句话原则

`Mat` 负责 OpenCV 兼容的“图像容器语义”；`TensorView/Layout` 负责框架侧“张量布局语义”。两者协作，但不混用。
