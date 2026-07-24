# `test/core` 目录规划

## 目录职责

验证 `core` 模块行为，包括 `Mat` 语义、基础算子、错误路径和数值稳定性。

## 核心覆盖目标

- `Mat` 生命周期：创建、拷贝、释放、浅拷贝/深拷贝。
- `Mat` 结构语义：shape、type、channel、stride、reshape、transpose。
- 基础算子：`add/sub/mul/div/compare/convertTo`。
- 异常语义：非法输入、维度不匹配、类型不支持。

## 阶段计划

### P0：现有用例治理

- 清理对旧 backend 头文件的依赖。
- 将测试按 API 分组，提升定位效率。

### P1：Mat 合同测试

- 增加通道语义和边界行为测试。
- 建立与 OpenCV 行为对照的关键 case（选取高频路径）。
- 已接线 core 测试二进制：
  - `cvh_test_core_lite`（Lite/header-only 套件）
  - `cvh_test_core`（Full 套件，包含 binary/gemm/transpose 等增强路径）
- 运行示例：
  - `cmake -S . -B build-core -DCVH_BUILD_TESTS=ON`
  - `cmake --build build-core -j --target cvh_test_core_lite cvh_test_core`
  - `./build-core/cvh_test_core '--gtest_filter=MatContract_TEST.*'`

## 文件职责划分（当前）

- `core_ops_test.cpp`：常用功能路径与结果正确性（例如 `convertTo/copyTo` 正常路径）。
- `array_ops_contract_test.cpp`：`absdiff`、bitwise、`inRange`、`min/max`
  的 Mat/Scalar、mask、浮点位模式、ROI 和边界语义。
- `math_ops_contract_test.cpp`：缩放转换、FP16 bits、F32/F64 数学函数、
  `checkRange/patchNaNs` 的精度、特殊值、ROI 和原地语义。
- `mat_channel_contract_test.cpp`：多通道合同首批覆盖（OpenCV type 宏、`CV_*C(n)`、连续多通道 `create/setTo/copyTo/convertTo`）。
- `mat_opencv_compat_test.cpp`：高频 OpenCV 风格接口对齐（`depth/channels/elemSize/isContinuous/step/step1`）与生命周期安全回归。
- `mat_submat_test.cpp`：2D submat/view 与非连续步长路径（`rowRange/colRange`、非连续 `setTo/copyTo/convertTo/clone`）回归。
- `mat_upstream_channel_port_test.cpp`：上游 channel 兼容移植入口（可运行项直接验证；未实现 API 用 `GTEST_SKIP` 明确挂起原因）。
- `mat_contract_test.cpp`：`Mat` 合同与安全基线（错误路径、所有权、边界与历史回归）。
- `geometry_types_contract_test.cpp`：`Point_`、`Size_` 及整数/浮点别名的构造、比较和转换合同。
- `test/opencv_contract`：可选 OpenCV 隔离差分 smoke；仅在
  `CVH_ENABLE_OPENCV_COMPARE=ON` 时构建，cvh 测试端不包含 OpenCV 头。
- 规则：同一行为只在一个文件中维护，避免重复 case 导致维护分叉。

### P2：算子回归

- 每个公开算子至少包含：
  - 一条正常路径 case
  - 一条边界或异常路径 case

## 完成定义（DoD）

- `core` 公开 API 均有可追溯测试。
- 任何 `Mat` 语义变更都会触发对应回归测试。
