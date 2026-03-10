# OpenCV Header-Only 项目总规划（v1）

- 更新时间：2026-03-10
- 适用仓库：`opencv-header-only`
- 当前阶段：迁移清理 + 基线收口（不是功能扩张期）

## 1. 项目定位

目标是做一个可直接 `#include` 的 OpenCV 风格子集库，优先覆盖高频能力，而不是完整复制 OpenCV 全模块。

首版目标（MVP）：

- `core`：`Mat`、基础类型、逐元素算子、基础变换与拷贝。
- `imgproc`：颜色转换、resize、基础滤波、基础几何变换。
- `imgcodecs`：有限格式读写（基于 `stb_image` / `stb_image_write`）。
- 工程能力：可编译、可测试、可基准、可安装。

## 2. 明确边界（防止范围失控）

MVP 暂不进入以下范围：

- `dnn`、`videoio`、`calib3d`、`features2d` 等大模块。
- 完整 OpenCV ABI/二进制兼容承诺。
- 与推理框架强绑定的算子族（如 `softmax/rmsnorm/rope/silu`）进入主线 API。

处理方式：

- 非 MVP 算子留在“过渡层/实验层”，不阻塞主线里程碑。
- 等 `core`/`imgproc` 稳定后再评估是否保留为扩展模块。

## 3. 当前基线判断（2026-03-10）

仓库已经具备了目录骨架和部分实现，但仍有明显迁移债务：

- `src/core/*.cpp` 仍承载大量核心实现，header-only 尚未闭环。
- 命名和错误处理中仍有 `minfer` 残留。
- `test/core` 部分用例仍依赖旧 backend 头文件。
- `imgproc` / `imgcodecs` 基本处于占位状态。

因此短期策略是“先收敛，再扩张”。

## 4. 技术路线

### 4.1 目录分层

- `include/cvh/*`：最终对外 API 与 header-only 实现。
- `src/core/*`：过渡实现区，仅做迁移，不新增长期功能。
- `test/*`：按模块维护回归与差分测试。
- `benchmark/*`：性能回归与版本对比。

### 4.2 Header-Only 实施原则

- 对外可见能力必须在 `include/` 闭环（不依赖 `src/` 才能使用）。
- 复杂实现拆到 `*.inl.h` / `detail/*.h`，但仍只通过头文件交付。
- 需要可选加速时，使用编译期宏开关（如 `OpenMP`、`xsimd`）。

### 4.3 API 对齐原则

- 命名空间统一 `cvh`，并保持“从 `cv::` 迁移成本低”。
- 先对齐语义，再对齐细节行为；先保证可预测，再优化性能。
- 对不兼容行为在文档中显式声明，不做隐式“看起来兼容”。

## 5. 里程碑与门禁

### Phase 0：基线收口（必须先完成）

目标：

- 构建、测试、安装主干可稳定运行。
- 公共头不依赖测试目录与旧工程路径。
- 清理核心命名空间与错误处理残留。

门禁（全部满足才进入下一阶段）：

- `cvh/cvh.h` 可在仅 `-Iinclude` 下被最小程序编译。
- `ctest` 至少包含 smoke + core 基础测试并稳定通过。
- `src/core` 中每个文件都标记迁移归属与删除条件。

### Phase 1：Core MVP

目标：

- 冻结 `Mat` 首版语义（type/channel/shape/stride/ROI/clone/copyTo/convertTo）。
- 完成常用逐元素运算与基础 shape 操作。
- 对齐最小错误模型与断言模型。

门禁：

- `test/core` 覆盖核心行为，含边界和异常路径。
- 新增 API 必须同时有示例与测试。

### Phase 2：Imgproc MVP

目标：

- 颜色转换：`BGR<->RGB`、`BGR<->GRAY`。
- 几何：`resize`（nearest/bilinear）+ 基础 `warpAffine`。
- 滤波：`blur`、`GaussianBlur`（先标量后 SIMD）。

门禁：

- `test/imgproc` 用固定输入数据集做差分校验。
- 每个算子至少有 1 组 correctness case + 1 组回归 case。

### Phase 3：Imgcodecs + 示例 + 基准

目标：

- 打通 `imread` / `imwrite` 的最小可用链路。
- 给出端到端示例（读图 -> 处理 -> 写图）。
- 建立 `benchmark` 基线（与 OpenCV 对比或历史版本对比）。

门禁：

- 示例可在 CI 编译并运行。
- benchmark 输出固定格式，可用于回归对比。

### Phase 4：性能与发布

目标：

- 引入 `OpenMP` / `xsimd` 可选优化并验证收益。
- 补齐安装与导出配置，准备 `v0.x` 发布。

门禁：

- 有性能收益数据，不允许“优化后变慢但默认启用”。
- 公开 API 文档与迁移文档齐全。

## 6. 风险矩阵与应对

| 风险 | 等级 | 触发信号 | 应对策略 |
|---|---|---|---|
| Header-only 迟迟不闭环 | 高 | 新功能继续写进 `src/core` | 规定 `include/` 为唯一主线，`src/` 只迁移不扩展 |
| 旧项目语义污染 | 高 | `minfer`、旧 backend include 持续新增 | 设立 CI 检查，新增代码禁止旧命名 |
| `Mat` 语义反复变更 | 高 | 上层模块频繁返工 | Phase 1 冻结 `Mat` 合同，变更走 RFC |
| 测试无法反映真实进度 | 中高 | 测试通过但对外 API 仍不可用 | 测试分层：smoke / API / module / benchmark |
| 性能优化破坏可维护性 | 中 | 内核散落且无回归基线 | 统一 `kernel` 边界 + benchmark 门禁 |

## 7. 执行规则（建议写入开发规范）

- 每个目录必须维护自己的 `readme.md`，内容至少包含：职责、阶段目标、完成定义、风险。
- 新增 API 必须同步更新：对应目录文档 + 测试 + 示例（至少其一可运行）。
- 每个阶段结束要做一次“删减评审”：移除过渡代码和失效 TODO。
- 非主线能力（实验算子）不得阻塞主线里程碑。

## 8. 目录规划文档索引

- `doc/phase0-execution-plan.md`
- `doc/src-core-migration-tracker.md`
- `include/cvh/readme.md`
- `include/cvh/core/readme.md`
- `include/cvh/imgproc/readme.md`
- `include/cvh/imgcodecs/readme.md`
- `src/readme.md`
- `src/core/readme.md`
- `src/core/kernel/readme.md`
- `test/readme.md`
- `test/core/readme.md`
- `test/imgproc/readme.md`
- `test/imgcodecs/readme.md`
- `test/utils/readme.md`
- `test/smoke/readme.md`
- `benchmark/readme.md`
- `example/readme.md`
