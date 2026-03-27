# `test/imgproc` 说明

## 当前覆盖范围（v1）

基于当前 `include/cvh/imgproc/imgproc.h` 能力，单测覆盖：

- `resize`：`INTER_NEAREST` / `INTER_NEAREST_EXACT` / `INTER_LINEAR`
- `cvtColor`：`COLOR_BGR2GRAY` / `COLOR_GRAY2BGR`
- `threshold`：`THRESH_BINARY` / `THRESH_BINARY_INV`
- `boxFilter/blur`：`CV_8U`、`BORDER_REPLICATE/BORDER_CONSTANT/BORDER_REFLECT_101`
- `GaussianBlur`：odd `ksize` + `sigma` 基础路径，含 `BORDER_ISOLATED` 位兼容
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
