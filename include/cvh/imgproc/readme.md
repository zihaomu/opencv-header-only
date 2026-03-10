# `include/cvh/imgproc` 目录规划

## 目录职责

承载图像处理 API（与 OpenCV `imgproc` 风格对齐），包括颜色转换、几何变换、滤波和基础特征提取。

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
