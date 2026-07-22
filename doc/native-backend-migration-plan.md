# 历史归档：旧 compiled extension 改造计划

## 状态

此文档是历史归档，不再代表当前产品路线。

当前路线已在 [opencv-universal-intrinsics-adapter-plan.md](opencv-universal-intrinsics-adapter-plan.md) 的 P5 中改为：

- `cvh::headers`：默认纯 header-only baseline。
- `cvh::headers_fast`：纯 header-only + 已验证 SIMD fast-path。
- 历史 `.cpp` 实现只作为 legacy/experimental 代码存在。
- 安装包和公开文档只推荐 `cvh::headers` / `cvh::headers_fast`。
- xsimd 进入 P5.2 quarantine 和 P5.3 removal，不再作为公开 fast profile 路线。

## 为什么归档

旧计划试图解释项目名 `opencv-header-only` 与仓库内 `.cpp` 实现共存的问题，并提出一个可选编译层来缓解认知不对称。

这个方向现在被否定：项目应保持更纯粹的 header-only 产品定位，而不是把 header-only 和 C++ 编译扩展共同作为公开结构。

## 当前规则

- 公开入口只有 `cvh::headers` 和 `cvh::headers_fast`。
- `.cpp` 路径不能成为公开 API 可用性的前提。
- `.cpp` 路径不进入安装导出的公开 target 集合。
- 用户文档不应引导用户通过可选编译层获得常规能力。
- 任何保留的历史宏、target 或源码路径都必须被视为 internal/development-only。

## 后续处理

本文件只用于解释历史背景。后续实施以 P5 文档为准：

- P5.0：公开面收口。
- P5.1：Header-only contract gate。
- P5.2：xsimd quarantine。
- P5.3：xsimd removal。
- P5.4：OpenCV Universal Intrinsics 常态维护。
