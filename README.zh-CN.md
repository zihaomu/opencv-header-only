# opencv-header-only

中文 | [English](README.md)

`opencv-header-only` 是一个 OpenCV 风格的轻量子集实现，目标是在尽量保持接口习惯的前提下，提供两种可选运行形态：

- `Lite`：header-only 优先，开箱即用、依赖轻。
- `Full`：链接 `src/` 后端实现，覆盖更完整、性能更高。

项目面向需要 OpenCV 常见 API 体验、但希望更可裁剪、更易集成的场景。

## 使用介绍

### 1) 基础构建

```bash
cmake -S . -B build
cmake --build build -j
```

### 2) 运行测试

```bash
# Lite 全量（core-lite + imgproc）
./scripts/ci_lite_all.sh

# Full 全量（core-full + imgproc）
./scripts/ci_full_all.sh
```

### 3) cvh / OpenCV 对比测试

```bash
# 结果输出到 CI/终端日志（Markdown），中间产物写临时目录
./scripts/ci_compare_log_only.sh
```

PR 中可由管理员通过评论触发 compare 模块：

```text
/cvh-compare on
/cvh-compare off
```

其中 `/cvh-compare on` 会在加上 compare label 的同时，立即触发
独立的 `CI Compare On Demand` 工作流。

## 性能对比

对比专区入口：

- [OpenCV Compare README](opencv_compare/README.md)

项目已提供可直接查看的对比报告（Markdown）：

- Quick: [opencv_compare/opencv_compare_quick.md](opencv_compare/opencv_compare_quick.md)
- Stable: [opencv_compare/opencv_compare_stable.md](opencv_compare/opencv_compare_stable.md)
- Baseline Stable: [opencv_compare/opencv_compare_baseline_stable.md](opencv_compare/opencv_compare_baseline_stable.md)

对比脚本与配置位于：

- 脚本：`opencv_compare/run_compare.sh`
- CI 日志模式脚本：`scripts/ci_compare_log_only.sh`

## License

本项目采用 [LICENSE](LICENSE) 中声明的开源协议。
