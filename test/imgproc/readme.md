# `test/imgproc` 说明

## 当前覆盖范围（v17）

基于当前 `include/cvh/imgproc/imgproc.h` 能力，单测覆盖：

- `resize`：`CV_8U/CV_32F` + `INTER_NEAREST` / `INTER_NEAREST_EXACT` / `INTER_LINEAR`
- `cvtColor`：`COLOR_BGR2GRAY` / `COLOR_GRAY2BGR` / `COLOR_GRAY2BGRA` / `COLOR_BGRA2GRAY` / `COLOR_GRAY2RGBA` / `COLOR_RGBA2GRAY` / `COLOR_BGR2RGB` / `COLOR_RGB2BGR` / `COLOR_BGR2BGRA` / `COLOR_BGRA2BGR` / `COLOR_RGB2RGBA` / `COLOR_RGBA2RGB` / `COLOR_BGR2RGBA` / `COLOR_RGBA2BGR` / `COLOR_RGB2BGRA` / `COLOR_BGRA2RGB` / `COLOR_BGRA2RGBA` / `COLOR_RGBA2BGRA` / `COLOR_BGR2YUV` / `COLOR_YUV2BGR` / `COLOR_RGB2YUV` / `COLOR_YUV2RGB` / `COLOR_BGR2YUV_NV12` / `COLOR_RGB2YUV_NV12` / `COLOR_BGR2YUV_NV21` / `COLOR_RGB2YUV_NV21` / `COLOR_BGR2YUV_I420` / `COLOR_RGB2YUV_I420` / `COLOR_BGR2YUV_YV12` / `COLOR_RGB2YUV_YV12` / `COLOR_BGR2YUV_NV16` / `COLOR_RGB2YUV_NV16` / `COLOR_BGR2YUV_NV61` / `COLOR_RGB2YUV_NV61` / `COLOR_BGR2YUV_YUY2` / `COLOR_RGB2YUV_YUY2` / `COLOR_BGR2YUV_UYVY` / `COLOR_RGB2YUV_UYVY` / `COLOR_BGR2YUV_NV24` / `COLOR_RGB2YUV_NV24` / `COLOR_BGR2YUV_NV42` / `COLOR_RGB2YUV_NV42` / `COLOR_BGR2YUV_I444` / `COLOR_RGB2YUV_I444` / `COLOR_BGR2YUV_YV24` / `COLOR_RGB2YUV_YV24` / `COLOR_YUV2BGR_NV12` / `COLOR_YUV2RGB_NV12` / `COLOR_YUV2BGR_NV21` / `COLOR_YUV2RGB_NV21` / `COLOR_YUV2BGR_I420` / `COLOR_YUV2RGB_I420` / `COLOR_YUV2BGR_YV12` / `COLOR_YUV2RGB_YV12` / `COLOR_YUV2BGR_I444` / `COLOR_YUV2RGB_I444` / `COLOR_YUV2BGR_YV24` / `COLOR_YUV2RGB_YV24` / `COLOR_YUV2BGR_NV16` / `COLOR_YUV2RGB_NV16` / `COLOR_YUV2BGR_NV61` / `COLOR_YUV2RGB_NV61` / `COLOR_YUV2BGR_NV24` / `COLOR_YUV2RGB_NV24` / `COLOR_YUV2BGR_NV42` / `COLOR_YUV2RGB_NV42` / `COLOR_YUV2BGR_YUY2` / `COLOR_YUV2RGB_YUY2` / `COLOR_YUV2BGR_UYVY` / `COLOR_YUV2RGB_UYVY`
- `threshold`：`CV_8U` 全基础阈值 + `CV_32F` 固定阈值（自动阈值仍限定 `CV_8UC1`）
- `boxFilter/blur`：`CV_8U/CV_32F`、`BORDER_REPLICATE/BORDER_CONSTANT/BORDER_REFLECT_101`
- `GaussianBlur`：`CV_8U/CV_32F` odd `ksize` + `sigma` 基础路径，含 `BORDER_ISOLATED` 位兼容
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

## 当前仍待对齐的 upstream case

当前 `case_manifest` 中已无 pending 项（当前同步集 7/7 为 `PASS_NOW`）。
