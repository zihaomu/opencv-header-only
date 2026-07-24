# Mat Contract v1（冻结稿）

- 文档版本：v1.2
- 冻结日期：2026-03-10
- 适用范围：`cvh::Mat` 及其直接相关 API（`clone/copyTo/convertTo/reshape` 等）
- 生效目标：作为当前 header-only `Mat` 行为与测试基线；后续行为变更必须先更新本合同

## 1. 设计目标

- 给 `Mat` 提供可预测、可测试、可迁移的行为合同。
- 保持与 OpenCV 核心语义方向一致，但允许在 v1 保留明确约束。
- 为 header-only 迁移提供稳定边界，避免“代码先改、语义后补”。

## 2. 术语定义

- `depth`：基础数据深度（如 `CV_8U`、`CV_32F`）。
- `channels`：通道数。
- `type`：`depth + channels` 的组合类型标识。
- `dims`：维度数量。
- `shape`：每个维度长度（`size[i]`）。
- `continuous`：内存连续布局。

## 3. v1 语义合同（必须满足）

### 3.1 类型与通道

- `Mat::type()` 返回当前元素类型标识。
- `Mat` 采用 OpenCV 风格 `type = depth + channels` 编码（`CV_MAKETYPE/CV_MAT_DEPTH/CV_MAT_CN`）。
- v1.1 的强制支持范围：连续内存语义下 `channels >= 1` 的 `create/clone/copyTo/convertTo/setTo`。
- 当前稳定存储深度为 `CV_8U/CV_8S/CV_16U/CV_16S/CV_32S/CV_32U/CV_16F/CV_32F/CV_64F`；
  其他已预留编码的实验深度必须被明确拒绝，不能依赖 depth 编号连续性误判为已支持。
- `convertTo` 在 `rtype` 仅给 depth 时，必须保持源通道数不变；形状不变。
- `convertTo` 若显式给出目标 type 且通道数与源不一致，必须返回明确错误（`StsBadArg`）。

### 3.2 维度与形状

- `dims > 0` 的 `Mat` 必须有有效 `shape`。
- `total()` 等于所有维度乘积。
- `reshape` 不复制数据，仅改变视图形状。
- `reshape` 前后 `total()` 必须相等，否则报错。
- `reshape` 当前仅支持连续内存输入；对非连续 `Mat` 必须返回明确错误。

### 3.3 内存与所有权

- 默认构造得到空 `Mat`（`data == nullptr`）。
- 复制构造和赋值为浅拷贝（引用计数 +1）。
- `clone()` 必须深拷贝。
- `copyTo(dst)` 必须把数据复制到 `dst`，并在类型不匹配时返回明确错误。
- 由外部指针构造的 `Mat` 视为用户内存，不由 `Mat` 释放。

### 3.4 连续性与步长（v1.1）

- v1 默认只承诺连续内存语义。
- 对连续内存 `Mat`，`isContinuous()` 返回 `true`；空 `Mat` 返回 `false`。
- `step(dim)` 返回对应维度的字节步长，`step1(dim)` 返回按标量元素计的步长。
- 已支持首批 2D view：`rowRange/colRange/operator()(Range, Range)`，返回共享存储的浅视图。
- 对非连续 view，`clone/copyTo/convertTo/setTo` 提供稳定语义保证（按步长处理，不允许 silent wrong result）。
- 任何依赖更通用 ND submat/高级 stride 变换的行为，需在后续版本扩展并补充合同。

### 3.5 错误语义

- 所有前置条件失败必须走统一错误模型（`cvh::error` / `CV_Error`）。
- 不允许返回未定义状态（例如对象半初始化、silent wrong result）。

## 4. v1 非目标（明确不支持）

- 多通道下高阶算子（表达式系统、广播算子等）的完整语义对齐。
- 通用 ND ROI/view 的完整步长模型与子矩阵引用语义。
- 非连续 `Mat` 的 `reshape` 语义。
- 与 OpenCV 完全二进制兼容。
- 与 `cv::Mat` 完全一致的对象内存布局兼容（字段布局、引用计数对象、allocator 内部组织）。

## 5. API 行为约束表

| API | 输入前置条件 | v1 结果保证 | 失败行为 |
|---|---|---|---|
| `Mat::create` | `dims>0` 且 shape 合法 | 分配连续内存并初始化元信息 | 抛错/报错 |
| `Mat::clone` | 任意合法 Mat | 深拷贝，类型和形状一致 | 抛错/报错 |
| `Mat::copyTo` | 源合法 | 数据复制到目标，形状一致 | 类型不匹配时报错 |
| `Mat::convertTo` | 源合法，目标类型有效 | 类型转换 + 形状保持 | 不支持类型时报错 |
| `Mat::reshape` | 新形状总元素数一致 | 不复制数据，仅改形状 | 元素总数不一致时报错 |
| `Mat::setTo` | 类型受支持 | 全量填充指定值 | 不支持类型时报错 |

## 6. 实现约束

- 任何改动 `Mat` 内部字段含义（`dims/data/u/size/matType`）都要同步更新本合同。
- 先保证语义正确，再做性能优化；优化不得改变语义。
- 若发现现有实现与合同冲突，以合同为准修复实现。

## 7. 测试验收清单

- `clone` 深拷贝验证（改副本不影响原对象）。
- `copyTo` 类型匹配/不匹配路径验证。
- `reshape` 成功/失败路径验证。
- `convertTo` 常见类型转换正确性验证（含边界值）。
- `CV_32F <-> CV_64F` 与 `CV_8U <-> CV_64F` 双向转换验证。
- 空 `Mat` 与非法输入的错误路径验证。
- 连续多通道（例如 `CV_8UC3`）的 `create/setTo/copyTo/convertTo` 成功路径验证。
- 2D submat（`rowRange/colRange`）共享视图与非连续步长路径验证。

## 8. 变更流程

- 本合同冻结后，任何行为变更需：
  - 先提合同修订（`doc/mat-contract-v1.md`）。
  - 再提交代码与测试。
  - 最后更新目录文档与计划文档引用。
