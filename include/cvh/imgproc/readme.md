# `include/cvh/imgproc` 目录规划

## 目录职责

承载图像处理 API（与 OpenCV `imgproc` 风格对齐），包括颜色转换、几何变换、滤波和基础特征提取。

## 当前文件组织（已落地）

- `imgproc.h`：唯一聚合入口（对外建议只包含这个头）。
- `detail/common.h`：枚举、公共 helper、后端注册初始化。
- `resize.h`：`resize` 算子与 dispatch/fallback。
- `cvtcolor.h`：`cvtColor` 算子与 dispatch/fallback。
- `threshold.h`：`threshold` 算子与 dispatch/fallback。
- `lut.h`：`LUT` 算子与 dispatch/fallback。
- `copy_make_border.h`：`copyMakeBorder` 算子与 dispatch/fallback。
- `filter2d.h`：`filter2D` 算子与 dispatch/fallback。
- `sep_filter2d.h`：`sepFilter2D` 算子与 dispatch/fallback。
- `box_filter.h`：`boxFilter` 算子与 dispatch/fallback。
- `blur.h`：`blur`（对 `boxFilter` 的语义封装）。
- `gaussian_blur.h`：`GaussianBlur` 算子与 dispatch/fallback。
- `canny.h`：`Canny`（图像输入与导数输入两种重载）与 dispatch/fallback。
- `morphology.h`：`erode/dilate/morphologyEx`（最小形态学）与 dispatch/fallback。
- `warp_affine.h`：`warpAffine` 算子与 dispatch/fallback。

## 当前能力快照（2026-04）

- `resize`：`CV_8U/CV_32F`，`INTER_NEAREST/INTER_NEAREST_EXACT/INTER_LINEAR`
- `cvtColor`：`CV_8U/CV_32F`，`GRAY2BGR/BGR2GRAY/GRAY2BGRA/BGRA2GRAY/GRAY2RGBA/RGBA2GRAY/BGR2RGB/RGB2BGR/BGR2BGRA/BGRA2BGR/RGB2RGBA/RGBA2RGB/BGR2RGBA/RGBA2BGR/RGB2BGRA/BGRA2RGB/BGRA2RGBA/RGBA2BGRA/BGR2YUV/YUV2BGR/RGB2YUV/YUV2RGB/BGR2YUV_NV12/RGB2YUV_NV12/BGR2YUV_NV21/RGB2YUV_NV21/BGR2YUV_I420/RGB2YUV_I420/BGR2YUV_YV12/RGB2YUV_YV12/BGR2YUV_NV16/RGB2YUV_NV16/BGR2YUV_NV61/RGB2YUV_NV61/BGR2YUV_YUY2/RGB2YUV_YUY2/BGR2YUV_UYVY/RGB2YUV_UYVY/BGR2YUV_NV24/RGB2YUV_NV24/BGR2YUV_NV42/RGB2YUV_NV42/BGR2YUV_I444/RGB2YUV_I444/BGR2YUV_YV24/RGB2YUV_YV24/YUV2BGR_NV12/YUV2RGB_NV12/YUV2BGR_NV21/YUV2RGB_NV21/YUV2BGR_I420/YUV2RGB_I420/YUV2BGR_YV12/YUV2RGB_YV12/YUV2BGR_I444/YUV2RGB_I444/YUV2BGR_YV24/YUV2RGB_YV24/YUV2BGR_NV16/YUV2RGB_NV16/YUV2BGR_NV61/YUV2RGB_NV61/YUV2BGR_NV24/YUV2RGB_NV24/YUV2BGR_NV42/YUV2RGB_NV42/YUV2BGR_YUY2/YUV2RGB_YUY2/YUV2BGR_UYVY/YUV2RGB_UYVY`（`NV12/NV21/I420/YV12/NV16/NV61/YUY2/UYVY/NV24/NV42/I444/YV24` 均已支持 `CV_8U` encode/decode）
- `threshold`：`CV_8U` 全基础阈值；`CV_32F` 固定阈值（`OTSU/TRIANGLE` 仅 `CV_8UC1`）
- `LUT`：`src=CV_8U`，`lut.total()==256`，`lut.channels()==1` 或 `src.channels()`
- `copyMakeBorder`：`CV_8U/CV_32F`（`BORDER_CONSTANT/REPLICATE/REFLECT/REFLECT_101/WRAP`）
- `filter2D`：`src=CV_8U/CV_32F`，`kernel=CV_32F(C1)`，`ddepth=-1/CV_8U/CV_32F`，`BORDER_CONSTANT/REPLICATE/REFLECT/REFLECT_101`
- `sepFilter2D`：`src=CV_8U/CV_32F`，`kernelX/kernelY=CV_32F(C1, 1xN/Nx1)`，`ddepth=-1/CV_8U/CV_32F`，`BORDER_CONSTANT/REPLICATE/REFLECT/REFLECT_101`
- `warpAffine`：`CV_8U/CV_32F`（`INTER_NEAREST/INTER_LINEAR` + `WARP_INVERSE_MAP`，`BORDER_CONSTANT/REPLICATE/REFLECT/REFLECT_101`）
- `boxFilter/blur`：`CV_8U/CV_32F`（`BORDER_CONSTANT/REPLICATE/REFLECT/REFLECT_101`）
- `GaussianBlur`：`CV_8U/CV_32F`（odd `ksize` + `sigma` 基础路径）
- `Sobel`：`CV_8U/CV_16S/CV_32F -> CV_16S/CV_32F`（当前支持 `ksize=3/5`，`dx/dy` 一阶）
- `Canny`：`CV_8UC1`（`apertureSize=3/5`，`L1/L2` 梯度）+ `CV_16SC1` 导数输入重载
- `erode/dilate`：`CV_8U`（支持自定义 kernel + 迭代 + 基础 border）
- `morphologyEx`：当前支持 `MORPH_ERODE/MORPH_DILATE/MORPH_OPEN/MORPH_CLOSE/MORPH_GRADIENT/MORPH_TOPHAT/MORPH_BLACKHAT/MORPH_HITMISS`

## 支持矩阵（当前实现）

| 算子 | 已支持类型/通道 | 当前优化重点 | 说明 |
|---|---|---|---|
| `resize` | `CV_8U/CV_32F`, `C1/C3/C4` | `INTER_NEAREST/INTER_NEAREST_EXACT/INTER_LINEAR` | fast-path + fallback，`cvh_test_imgproc` 已覆盖 ROI/边界尺寸 |
| `cvtColor` | `CV_8U/CV_32F`, `GRAY(C1) <-> BGR(C3)/BGRA(C4)/RGBA(C4)`, `BGR(C3) <-> RGB(C3)`, `BGR/RGB(C3) <-> BGRA/RGBA(C4)`, `BGRA(C4) <-> RGBA(C4)`, `BGR/RGB(C3) <-> YUV(C3)`, `BGR/RGB(C3) <-> NV12/NV21(C1, H*3/2 x W)`, `BGR/RGB(C3) <-> I420/YV12(C1, H*3/2 x W)`, `BGR/RGB(C3) <-> I444/YV24(C1, H*3 x W)`, `BGR/RGB(C3) <-> NV16/NV61(C1, H*2 x W)`, `BGR/RGB(C3) <-> NV24/NV42(C1, H*3 x W)`, `BGR/RGB(C3) <-> YUY2/UYVY(C2, H x W)` | `GRAY family`、`BGR2GRAY`、`GRAY2BGR`、`RGB/BGR/RGBA/BGRA family`、`YUV family`、`YUV420/YUV422/YUV444 decode/encode` | `NV12/NV21/I420/YV12/NV16/NV61/YUY2/UYVY/NV24/NV42/I444/YV24` 已覆盖 `CV_8U` encode/decode；已覆盖连续/ROI/单行单列与 benchmark 主组合 |
| `threshold` | `CV_8U`, `CV_32F` | `THRESH_BINARY/...` 固定阈值 | `THRESH_OTSU/THRESH_TRIANGLE` 仅 `CV_8UC1` |
| `LUT` | `src=CV_8U`, `lut.total()==256`, `lut C1/Csrc` | 查表映射（逐像素） | 输出深度由 `lut.depth()` 决定，支持 ROI/non-contiguous |
| `copyMakeBorder` | `CV_8U/CV_32F`, `C1/C3/C4` | 常用 border 路径 | 支持 `BORDER_CONSTANT/REPLICATE/REFLECT/REFLECT_101/WRAP` |
| `filter2D` | `src=CV_8U/CV_32F`, `kernel=CV_32FC1`, `C1/C3/C4` | 通用卷积主路径 | 支持 `ddepth=-1/CV_8U/CV_32F`、`delta`、自定义 `anchor`，覆盖 ROI/non-contiguous 与 in-place |
| `sepFilter2D` | `src=CV_8U/CV_32F`, `kernelX/kernelY=CV_32FC1 vector`, `C1/C3/C4` | 可分离卷积主路径 | 支持 `ddepth=-1/CV_8U/CV_32F`、`delta`、自定义 `anchor`，并与 `filter2D` 外积卷积交叉校验 |
| `warpAffine` | `CV_8U/CV_32F`, `C1/C3/C4` | `INTER_NEAREST/INTER_LINEAR` | 支持 `WARP_INVERSE_MAP`，覆盖 ROI/non-contiguous 与 in-place |
| `boxFilter/blur` | `CV_8U/CV_32F`, `C1/C3/C4` | `3x3` 热路径 | `blur` 为 `boxFilter` 语义封装 |
| `GaussianBlur` | `CV_8U/CV_32F`, `C1/C3/C4` | odd `ksize` 可分离卷积 | 当前 benchmark 主打 `5x5` |
| `Sobel` | `CV_8U/CV_16S/CV_32F -> CV_16S/CV_32F`, `C1/C3/C4` | `ksize=3/5` 一阶梯度 | 当前聚焦 `(dx,dy)=(1,0)/(0,1)` |
| `Canny` | 图像重载：`CV_8UC1`；导数重载：`dx/dy=CV_16SC1` | `apertureSize=3/5`，`L1/L2` 梯度 | NMS + 双阈值滞后连接（fallback） |
| `erode/dilate` | `CV_8U`, `C1/C3/C4` | kernel + iterations correctness | fallback 路径 |
| `morphologyEx` | `CV_8U`, `C1/C3/C4`（`HITMISS` 限 `C1`） | `MORPH_ERODE/MORPH_DILATE/MORPH_OPEN/MORPH_CLOSE/MORPH_GRADIENT/MORPH_TOPHAT/MORPH_BLACKHAT/MORPH_HITMISS` | 其它 `MORPH_*` 暂未支持 |

## 当前限制

- `cvtColor` 目前已覆盖 `GRAY(C1) <-> BGR(C3)/BGRA(C4)/RGBA(C4)`、`BGR(C3) <-> RGB(C3)`、`BGR/RGB(C3) <-> BGRA/RGBA(C4)`、`BGRA(C4) <-> RGBA(C4)`、`BGR/RGB(C3) <-> YUV(C3)`、`BGR/RGB(C3) <-> NV12/NV21(C1, H*3/2 x W)`、`BGR/RGB(C3) <-> I420/YV12(C1, H*3/2 x W)`、`BGR/RGB(C3) <-> I444/YV24(C1, H*3 x W)`、`BGR/RGB(C3) <-> NV16/NV61(C1, H*2 x W)`、`BGR/RGB(C3) <-> NV24/NV42(C1, H*3 x W)`、`BGR/RGB(C3) <-> YUY2/UYVY(C2, H x W)`；本轮计划内 YUV family 已收口。
- `NV12/NV21` 输入/输出约定为单 `Mat` 的 `CV_8UC1(H*3/2 x W)`：上 `H` 行为 `Y`，下 `H/2` 行为连续 `UV` / `VU` 字节流，每 `2x2` 像素块共享一组 `U/V`。
- `I420/YV12` 输入/输出约定为单 `Mat` 的 `CV_8UC1(H*3/2 x W)`：上 `H` 行为 `Y`，下 `H/2` 行按平面排列；`I420` 为 `U` 后 `V`，`YV12` 为 `V` 后 `U`，每 `2x2` 像素块共享一组 `U/V`。
- `NV16/NV61` 输入/输出约定为单 `Mat` 的 `CV_8UC1(H*2 x W)`：上 `H` 行为 `Y`，下 `H` 行为连续 `UV` / `VU` 字节流，每 2 个水平像素共享一组 `U/V`。
- `YUY2/UYVY` 输入/输出约定为单 `Mat` 的 `CV_8UC2(H x W)`：`YUY2` 按 `[Y0 U][Y1 V]` 写出，`UYVY` 按 `[U Y0][V Y1]` 写出，每 2 个水平像素共享一组 `U/V`。
- `NV24/NV42` 输入/输出约定为单 `Mat` 的 `CV_8UC1(H*3 x W)`：上 `H` 行为 `Y`，下 `2H` 行分别按连续 `UV` / `VU` 字节流解释。
- `I444/YV24` 输入/输出约定为单 `Mat` 的 `CV_8UC1(H*3 x W)`：上 `H` 行为 `Y`，中间 `H` 行为 `U/V` 平面，最后 `H` 行为 `V/U` 平面。
- `threshold` 的自动阈值（`OTSU/TRIANGLE`）仍严格限定 `CV_8UC1`，对 `CV_32F` 会显式报错。
- `LUT` 当前限定 `src.depth()==CV_8U`；`lut.total()` 必须为 `256`，且 `lut.channels()` 必须是 `1` 或 `src.channels()`。
- `filter2D` 当前限定 `kernel.depth()==CV_32F && kernel.channels()==1`，暂不支持 `CV_64F` kernel。
- `sepFilter2D` 当前限定 `kernelX/kernelY` 为 `CV_32F` 单通道向量（`1xN` 或 `Nx1`）。
- `warpAffine` 当前支持 `INTER_NEAREST/INTER_LINEAR` 与 `WARP_INVERSE_MAP`，不支持 `INTER_NEAREST_EXACT` / `WARP_FILL_OUTLIERS` 等扩展 flag。
- `Sobel` 当前支持 `ksize=3/5` 与一阶导数组合；输出 `ddepth` 当前支持 `CV_16S/CV_32F`。
- `Canny` 当前图像重载仅支持 `CV_8UC1`，导数重载仅支持 `CV_16SC1` 的 `dx/dy`，`apertureSize` 支持 `3/5`。
- `erode/dilate` 当前仅支持 `CV_8U`。
- `morphologyEx` 当前支持 `MORPH_ERODE/MORPH_DILATE/MORPH_OPEN/MORPH_CLOSE/MORPH_GRADIENT/MORPH_TOPHAT/MORPH_BLACKHAT/MORPH_HITMISS`；其余 op 会显式报错。
- `MORPH_HITMISS` 当前限定 `CV_8UC1` 输入，kernel 支持 `CV_8UC1/CV_8SC1`。
  - `CV_8SC1` 语义：`1`=前景命中、`-1`=背景命中、`0`=忽略。
- benchmark baseline 是“固定机器 / 固定 runner class”相关资产，跨硬件直接复用不保证通过。

## 开发前提

- 依赖 `core` 的 `Mat` 语义先稳定（type/channel/stride/ROI）。
- 先保证 correctness，再做 SIMD/OpenMP 优化。

## 阶段计划

### P1：基础颜色与像素级操作

- `cvtColor`：`BGR <-> RGB`、`BGR <-> GRAY`。
- 基础像素级变换：阈值、归一化、简单 LUT（已落地）。
- 优先支持 `CV_8U`、`CV_32F`。

### P2：几何变换

- `resize`：nearest、bilinear。
- `warpAffine`：先覆盖最常见场景。
- 明确边界处理模式（`BORDER_CONSTANT`/`BORDER_REPLICATE` 起步）。

### P3：滤波与邻域操作（已完成首轮）

- `blur`、`GaussianBlur`、`Sobel`（M1 最小可用版本已落地）。
- 卷积公共实现抽象成可复用内部接口。
- 在 correctness 稳定后引入 SIMD 路径。

### P4：进阶算子（进行中）

- `Canny`、形态学扩展（`erode/dilate` kernel/border 扩展）等。
- 进入该阶段前，必须先完成 P1-P3 的测试闭环。

## 风险控制

- 不在 `imgproc` 里绕过 `Mat` 语义做“临时特判”。
- 每新增一个算子都要定义输入约束与边界行为，避免“接口有了但行为不稳定”。

## 完成定义（DoD）

- 每个公开 API 至少有 1 个固定数据集回归测试。
- `test/imgproc` 覆盖正常路径和边界路径。
- 示例目录有对应最小演示代码。
