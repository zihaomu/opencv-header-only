# `test/imgcodecs` 说明

## 当前覆盖范围

基于 `include/cvh/3rdparty/std/stb_image.h` 与当前 `imgcodecs.h` 实现，单测覆盖：

- 读取：`png/jpg/bmp/gif/ppm`
- 写入：`png/jpg/bmp`
- 可选读取：`hdr`

## 数据来源

样本从本地 `opencv_extra-4.x/testdata` 同步到本仓库：

- 目标目录：`test/imgcodecs/data/opencv_extra`
- 同步脚本：`scripts/sync_opencv_imgcodecs_cases.py`

## 使用方式

```bash
python3 scripts/sync_opencv_imgcodecs_cases.py
```

如需额外同步可选 HDR 样本：

```bash
python3 scripts/sync_opencv_imgcodecs_cases.py --with-hdr
```

脚本会生成 `test/imgcodecs/data/manifest.json` 记录来源、大小和哈希。
