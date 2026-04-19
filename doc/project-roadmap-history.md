# Project Roadmap History (Merged)

更新时间：2026-04-19

本文件合并了以下重复/早期规划文档：

- `doc/header-only-opencv-plan.md`
- `doc/dual-mode-gap-tracker.md`

## 1. 早期共识（已落地）

1. 项目定位为 OpenCV-like 高频子集，不追求全模块复制。
2. 双模式路线：`Lite`（header-first）+ `Full`（backend增强）。
3. 对外优先承诺 API/行为兼容，不承诺 ABI/对象内存布局兼容。

## 2. 早期里程碑结果

已完成的关键基础：

- Lite/Full 模式宏与构建分流。
- Lite 最小链路可运行（读图-处理-写图）。
- Full 模式后端注册链路打通。
- smoke 与模块测试骨架建立。

## 3. 现行文档归一

项目当前应以以下文档为主：

- 总体设计：`doc/design.md`
- 主 README（国际化）：`README.md` / `README.zh-CN.md`
- 性能对比：`opencv_compare/README.md` 与自动生成报告

本文件仅保留历史语境，不再作为执行计划入口。
