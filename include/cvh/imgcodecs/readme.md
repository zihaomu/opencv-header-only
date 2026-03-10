# `include/cvh/imgcodecs` 目录规划

## 目录职责

提供最小可用的图像读写接口，让 `Mat` 和文件系统可打通（`imread`/`imwrite` 主链路）。

## 技术路线

- 基础实现基于：
  - `include/cvh/3rdparty/std/stb_image.h`
  - `include/cvh/3rdparty/std/stb_image_write.h`
- API 风格尽量贴近 OpenCV，但先做能力子集，不追求一次性全格式覆盖。

## 阶段计划

### P1：最小链路

- `imread`：支持 `png/jpg` 读取到 `Mat`（`CV_8U` 主路径）。
- `imwrite`：支持 `png/jpg` 写出。
- 颜色布局明确（默认 BGR 或 RGB，需要文档固定）。

### P2：行为对齐

- 对齐读取失败、格式不支持、路径错误等异常语义。
- 增加通道转换与 bit depth 约束说明。

### P3：稳定化

- 补齐 `test/imgcodecs` 数据驱动测试。
- 增加跨平台路径与编码稳定性验证（Linux/macOS/Windows）。

## 边界与约束

- 先不做视频编解码、动画格式、超大格式矩阵优化。
- 不在该目录引入与 `Mat` 无关的额外 I/O 框架依赖。

## 完成定义（DoD）

- 能完成“读图 -> core/imgproc 处理 -> 写图”的端到端流程。
- 失败场景有可预测错误码/异常信息。
- `example/` 至少包含一个完整 I/O 示例。
