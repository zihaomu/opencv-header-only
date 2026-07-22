# `test` 目录规划

## 目录职责

维护项目测试体系，保证 API 正确性、行为稳定性和迁移可控性。

## 测试分层

- `smoke`：最小编译/链接与入口可用性检查。
- `core`：`Mat` 与基础算子行为测试。
- `upstream`：OpenCV 上游用例快照与兼容迁移台账（用于接口完整性跟踪）。
- `imgproc`：图像处理算法正确性测试。
- `imgcodecs`：读写链路与异常路径测试。
- `utils`：测试工具与数据加载辅助。

## 阶段计划

### P0：测试主干打通

- `ctest` 默认运行 smoke + core 基础集。
- 清理旧工程路径依赖，确保测试仅依赖当前仓库。
- 当前已接线 core 测试二进制：
  - `cvh_test_core_lite`（header-only, Lite CI）
  - `cvh_test_core_native`（native backend, Native CI）
  （通过 `CVH_BUILD_TESTS=ON` 启用，支持 `--gtest_filter` 过滤）。

### P1：模块化覆盖

- `core/imgproc/imgcodecs` 各自建立稳定 case 集。
- 引入固定数据集和误差阈值策略。

### P2：回归与扩展

- 增加随机测试、边界测试和性能回归挂钩。
- 建立测试数据生成脚本的可复现流程。

## 完成定义（DoD）

- 关键 API 在对应目录均有测试覆盖。
- 测试失败可快速定位到模块，不依赖人工猜测。

## 测试失败台账

- `failing-tests.md`：记录当前未通过/未运行/挂起测试列表、原因和解锁条件。
