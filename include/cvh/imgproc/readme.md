# `include/cvh/imgproc` 目录规划

## 目录职责

承载图像处理 API（与 OpenCV `imgproc` 风格对齐），包括颜色转换、几何变换、滤波和基础特征提取。

## 当前文件组织（已落地）

- `imgproc.h`：唯一聚合入口（对外建议只包含这个头）。
- `detail/common.h`：枚举、公共 helper、后端注册初始化。
- `resize.h`：`resize` 算子与 dispatch/fallback。
- `cvtcolor.h`：`cvtColor` 算子与 dispatch/fallback。
- `threshold.h`：`threshold` 算子与 dispatch/fallback。
- `box_filter.h`：`boxFilter` 算子与 dispatch/fallback。
- `blur.h`：`blur`（对 `boxFilter` 的语义封装）。
- `gaussian_blur.h`：`GaussianBlur` 算子与 dispatch/fallback。

## 当前能力快照（2026-04）

- `resize`：`CV_8U/CV_32F`，`INTER_NEAREST/INTER_NEAREST_EXACT/INTER_LINEAR`
- `cvtColor`：`CV_8U`，`BGR2GRAY/GRAY2BGR`
- `threshold`：`CV_8U` 全基础阈值；`CV_32F` 固定阈值（`OTSU/TRIANGLE` 仅 `CV_8UC1`）
- `boxFilter/blur`：`CV_8U/CV_32F`（`BORDER_CONSTANT/REPLICATE/REFLECT/REFLECT_101`）
- `GaussianBlur`：`CV_8U/CV_32F`（odd `ksize` + `sigma` 基础路径）

## 支持矩阵（当前实现）

| 算子 | 已支持类型/通道 | 当前优化重点 | 说明 |
|---|---|---|---|
| `resize` | `CV_8U/CV_32F`, `C1/C3/C4` | `INTER_NEAREST/INTER_NEAREST_EXACT/INTER_LINEAR` | fast-path + fallback，`cvh_test_imgproc` 已覆盖 ROI/边界尺寸 |
| `cvtColor` | `CV_8U`, `GRAY(C1) <-> BGR(C3)` | `BGR2GRAY`、`GRAY2BGR` | 仅覆盖当前 benchmark/contract test 使用组合 |
| `threshold` | `CV_8U`, `CV_32F` | `THRESH_BINARY/...` 固定阈值 | `THRESH_OTSU/THRESH_TRIANGLE` 仅 `CV_8UC1` |
| `boxFilter/blur` | `CV_8U/CV_32F`, `C1/C3/C4` | `3x3` 热路径 | `blur` 为 `boxFilter` 语义封装 |
| `GaussianBlur` | `CV_8U/CV_32F`, `C1/C3/C4` | odd `ksize` 可分离卷积 | 当前 benchmark 主打 `5x5` |

## 当前限制

- `cvtColor` 暂未扩展到 `CV_32F` 或更广的颜色空间转换组合。
- `threshold` 的自动阈值（`OTSU/TRIANGLE`）仍严格限定 `CV_8UC1`，对 `CV_32F` 会显式报错。
- benchmark baseline 是“固定机器 / 固定 runner class”相关资产，跨硬件直接复用不保证通过。

## 开发前提

- 依赖 `core` 的 `Mat` 语义先稳定（type/channel/stride/ROI）。
- 先保证 correctness，再做 SIMD/OpenMP 优化。

## 阶段计划

### P1：基础颜色与像素级操作

- `cvtColor`：`BGR <-> RGB`、`BGR <-> GRAY`。
- 基础像素级变换：阈值、归一化、简单 LUT（可选）。
- 优先支持 `CV_8U`、`CV_32F`。

### P2：几何变换

- `resize`：nearest、bilinear。
- `warpAffine`：先覆盖最常见场景。
- 明确边界处理模式（`BORDER_CONSTANT`/`BORDER_REPLICATE` 起步）。

### P3：滤波与邻域操作

- `blur`、`GaussianBlur`、`Sobel`。
- 卷积公共实现抽象成可复用内部接口。
- 在 correctness 稳定后引入 SIMD 路径。

### P4：进阶算子（可选）

- `Canny`、形态学（`erode/dilate`）等。
- 进入该阶段前，必须先完成 P1-P3 的测试闭环。

## 风险控制

- 不在 `imgproc` 里绕过 `Mat` 语义做“临时特判”。
- 每新增一个算子都要定义输入约束与边界行为，避免“接口有了但行为不稳定”。

## 完成定义（DoD）

- 每个公开 API 至少有 1 个固定数据集回归测试。
- `test/imgproc` 覆盖正常路径和边界路径。
- 示例目录有对应最小演示代码。
