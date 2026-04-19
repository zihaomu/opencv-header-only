# `test/imgproc` 说明

## 当前覆盖范围（v23）

基于当前 `include/cvh/imgproc/imgproc.h` 能力，单测覆盖：

- `resize`：`CV_8U/CV_32F` + `INTER_NEAREST` / `INTER_NEAREST_EXACT` / `INTER_LINEAR`
- `cvtColor`：`COLOR_BGR2GRAY` / `COLOR_GRAY2BGR` / `COLOR_GRAY2BGRA` / `COLOR_BGRA2GRAY` / `COLOR_GRAY2RGBA` / `COLOR_RGBA2GRAY` / `COLOR_BGR2RGB` / `COLOR_RGB2BGR` / `COLOR_BGR2BGRA` / `COLOR_BGRA2BGR` / `COLOR_RGB2RGBA` / `COLOR_RGBA2RGB` / `COLOR_BGR2RGBA` / `COLOR_RGBA2BGR` / `COLOR_RGB2BGRA` / `COLOR_BGRA2RGB` / `COLOR_BGRA2RGBA` / `COLOR_RGBA2BGRA` / `COLOR_BGR2YUV` / `COLOR_YUV2BGR` / `COLOR_RGB2YUV` / `COLOR_YUV2RGB` / `COLOR_BGR2YUV_NV12` / `COLOR_RGB2YUV_NV12` / `COLOR_BGR2YUV_NV21` / `COLOR_RGB2YUV_NV21` / `COLOR_BGR2YUV_I420` / `COLOR_RGB2YUV_I420` / `COLOR_BGR2YUV_YV12` / `COLOR_RGB2YUV_YV12` / `COLOR_BGR2YUV_NV16` / `COLOR_RGB2YUV_NV16` / `COLOR_BGR2YUV_NV61` / `COLOR_RGB2YUV_NV61` / `COLOR_BGR2YUV_YUY2` / `COLOR_RGB2YUV_YUY2` / `COLOR_BGR2YUV_UYVY` / `COLOR_RGB2YUV_UYVY` / `COLOR_BGR2YUV_NV24` / `COLOR_RGB2YUV_NV24` / `COLOR_BGR2YUV_NV42` / `COLOR_RGB2YUV_NV42` / `COLOR_BGR2YUV_I444` / `COLOR_RGB2YUV_I444` / `COLOR_BGR2YUV_YV24` / `COLOR_RGB2YUV_YV24` / `COLOR_YUV2BGR_NV12` / `COLOR_YUV2RGB_NV12` / `COLOR_YUV2BGR_NV21` / `COLOR_YUV2RGB_NV21` / `COLOR_YUV2BGR_I420` / `COLOR_YUV2RGB_I420` / `COLOR_YUV2BGR_YV12` / `COLOR_YUV2RGB_YV12` / `COLOR_YUV2BGR_I444` / `COLOR_YUV2RGB_I444` / `COLOR_YUV2BGR_YV24` / `COLOR_YUV2RGB_YV24` / `COLOR_YUV2BGR_NV16` / `COLOR_YUV2RGB_NV16` / `COLOR_YUV2BGR_NV61` / `COLOR_YUV2RGB_NV61` / `COLOR_YUV2BGR_NV24` / `COLOR_YUV2RGB_NV24` / `COLOR_YUV2BGR_NV42` / `COLOR_YUV2RGB_NV42` / `COLOR_YUV2BGR_YUY2` / `COLOR_YUV2RGB_YUY2` / `COLOR_YUV2BGR_UYVY` / `COLOR_YUV2RGB_UYVY`
- `threshold`：`CV_8U` 全基础阈值 + `CV_32F` 固定阈值（自动阈值仍限定 `CV_8UC1`）
- `LUT`：`src=CV_8U`，`lut.total()==256`，`lut C1/Csrc`，覆盖输出类型跟随 `lut.depth`、ROI、in-place 与异常输入
- `boxFilter/blur`：`CV_8U/CV_32F`、`BORDER_REPLICATE/BORDER_CONSTANT/BORDER_REFLECT_101`
- `GaussianBlur`：`CV_8U/CV_32F` odd `ksize` + `sigma` 基础路径，含 `BORDER_ISOLATED` 位兼容
- `copyMakeBorder`：`CV_8U/CV_32F`，`BORDER_CONSTANT/REPLICATE/REFLECT/REFLECT_101/WRAP`（含 ROI 与 in-place）
- `filter2D`：`src=CV_8U/CV_32F`，`kernel=CV_32FC1`，`ddepth=-1/CV_8U/CV_32F`，`BORDER_CONSTANT/REPLICATE/REFLECT/REFLECT_101`（含 ROI 与 in-place）
- `sepFilter2D`：`src=CV_8U/CV_32F`，`kernelX/kernelY=CV_32FC1 vector`，`ddepth=-1/CV_8U/CV_32F`，`BORDER_CONSTANT/REPLICATE/REFLECT/REFLECT_101`（含 ROI 与 in-place，且与 `filter2D` 外积卷积交叉校验）
- `warpAffine`：`CV_8U/CV_32F`，`INTER_NEAREST/INTER_LINEAR`，`WARP_INVERSE_MAP`，`BORDER_CONSTANT/REPLICATE/REFLECT/REFLECT_101`（含 ROI 与 in-place）
- `Sobel`：`CV_8U/CV_16S/CV_32F -> CV_16S/CV_32F`，当前覆盖 `ksize=3/5`、`dx/dy` 一阶组合（含 ROI）
- `Canny`：图像重载（`CV_8UC1`，`aperture=3/5`，`L1/L2`）+ 导数重载（`dx/dy=CV_16SC1`），覆盖 ROI 与异常输入
- `erode/dilate`：`CV_8U`，当前覆盖默认 `3x3` 核、迭代等价性（含 ROI）
- `morphologyEx`：当前覆盖 `MORPH_OPEN/MORPH_CLOSE/MORPH_GRADIENT/MORPH_TOPHAT/MORPH_BLACKHAT/MORPH_HITMISS` 合同路径 + `MORPH_DILATE` 小输入回归（含 `HITMISS` 的 `CV_8SC1` kernel 语义）
- 真实图像 pipeline 回归：`imread -> resize -> cvtColor -> threshold`

## 数据来源

样本从本地 `opencv_extra-4.x/testdata` 同步到：

- `test/imgproc/data/opencv_extra`
- `test/imgproc/data/manifest.json`

同步命令：

```bash
python3 scripts/sync_opencv_imgproc_fixtures.py
```

可选同步大图（如 `lena.png`）：

```bash
python3 scripts/sync_opencv_imgproc_fixtures.py --with-large
```

## upstream 测试片段追踪

从 `opencv/modules/imgproc/test` 截取对应 TEST 块快照到：

- `test/upstream/opencv/imgproc/<opencv-commit>/`
- `test/upstream/opencv/imgproc/case_manifest.json`

同步命令：

```bash
python3 scripts/sync_opencv_imgproc_cases.py
```

## 当前对齐状态（M1 + M2）

`case_manifest` 中当前选定的 21 个 imgproc upstream case 均为 `PASS_NOW`，其中包含：

- `Imgproc_Morphology.iterated`
- `Imgproc.filter_empty_src_16857`（已实现算子的空输入覆盖）
- `Imgproc.morphologyEx_small_input_22893`
- `Imgproc_MorphEx.hitmiss_regression_8957`
- `Imgproc_MorphEx.hitmiss_zero_kernel`
- `Imgproc_Sobel.borderTypes`（含 `BORDER_ISOLATED` ROI 语义）
- `Imgproc_Sobel.s16_regression_13506`（`CV_16S` + `ksize=5`）
- `Imgproc_GaussianBlur.regression_11303`（`CV_32F` 常量图 + `sigma` 自动核大小路径）
- `Imgproc_Filter2D.dftFilter2d_regression_13179`（`filter2D` 回归子集）
- `Imgproc_sepFilter2D.identity`（`sepFilter2D` identity kernel 语义）
- `Imgproc_WarpAffine.accuracy`（`warpAffine` 固定参数子集：几何/逆映射/边界/异常参数）
- `Imgproc_Warp.regression_19566`（`warpAffine` 常量边界多通道回归子集）
- `Imgproc_FindContours.border`（`copyMakeBorder` upstream 前置语义子集）
- `Canny_Modes.accuracy`（固定参数子集，图像重载 + 导数重载对齐）
