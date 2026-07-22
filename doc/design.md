opencv-header-only 设计文档
1. 项目概述

opencv-header-only（以下简称 cvh）是一个 OpenCV-like 的轻量高频子集库，目标是以极低接入成本提供常见图像处理能力，并保持与 OpenCV 常用 API 风格尽可能一致。

本项目的核心理念不是“完整替代 OpenCV”，而是：

提供一个真实可运行的 header-only 最小版本
覆盖高频基础图像处理需求
对外保持低依赖、低接入门槛、熟悉接口
对内允许通过 src backend 提供更强实现、更完整 dispatch 与更高性能

项目强调的是：

Lite 模式真的可运行
Native backend 只是增强，不是补洞
API 一致
兼容边界清晰
性能目标现实可控

支持的模块包括： core, imgpro

2. 项目目标
2.1 核心目标

cvh 的核心目标如下：

提供接近 OpenCV 的常用基础数据结构
包括但不限于 Mat、Point、Size、Rect、Scalar、Range。
提供高频图像处理 API
优先覆盖 resize、GaussianBlur、boxFilter、threshold、少量 cvtColor、有限 imgcodecs。
提供真实可运行的 header-only 子集
用户仅包含头文件时，必须能完成基本图像处理链路。
提供可选增强 backend
当用户链接 src 时，应自动获得更强 dispatch、更多类型支持、更优性能与可选 SIMD/并行能力。
保持 OpenCV-like 的使用体验
用户尽量只需从 cv:: 迁移到 cvh::，即可完成常见代码迁移。
2.2 非目标

本项目明确不追求以下目标：

不追求完整复现 OpenCV 的全部模块
不追求完整复现 OpenCV 的所有数据结构细节
不追求与 OpenCV bit-exact 一致
不追求所有平台、所有类型、所有边界条件下达到 OpenCV 同等级性能
不将 Native backend 作为公开 API 的唯一实现来源
不承诺“仅改命名空间即可无缝替换所有 OpenCV 代码”
3. 产品定位

cvh 是一个：

面向高频图像处理场景的轻量 OpenCV-like 子集库

它适合：

小型工具和脚本化集成
SDK runtime
插件系统
依赖敏感的工程环境
不需要完整 OpenCV 的场景
需要基础图像处理但希望降低引入成本的项目

它不适合作为：

OpenCV 的全量替代品
深度依赖 OpenCV 全模块能力的项目
对 bit-exact 数值一致性有强要求的系统
依赖大量高级视觉模块的研究或生产系统
4. 架构原则

本项目遵循以下架构原则：

4.1 Lite 必须独立可运行

Header-only 模式不是“只有声明”，也不是“只能跑 toy demo”。
用户只包含头文件时，必须能完成一条真实图像处理链路。

4.2 Native backend 只是增强，不是修复

链接 src 获得的是性能、覆盖度、dispatch、SIMD、多线程等增强能力，而不是“原本不能用的 API 才终于能用”。

4.3 API 统一

无论是 Lite 还是 Native backend，公开 API 的命名空间、函数签名、数据结构语义应尽可能一致。

4.4 fallback 优先定义，backend 负责增强

所有公开 API 都必须先有可运行的 fallback 实现。
src backend 通过 dispatch 覆盖或增强 fallback，而不是提供唯一实现。

4.5 正确性优先于优化

所有优化路径（SIMD、多线程、平台特化）都必须建立在正确的 scalar/fallback 实现之上。

4.6 基础语义集中统一

Mat、类型系统、边界模式、参数推导等基础行为必须由头文件层统一定义，避免 Lite / Native backend 语义分裂。

5. 编译模式设计

项目正式支持两种模式，并允许能力开关按需叠加。

5.1 CVH_LITE
定义

CVH_LITE 表示纯头文件模式，不依赖 src backend。

特点
只需包含头文件
不需要链接额外库
所有公开 API 均有 fallback 实现
功能覆盖有限，但真实可运行
默认不依赖 xsimd
默认以 scalar 实现为主
使用场景
demo
原型验证
小工具
教学
对依赖、构建、体积敏感的环境
5.2 CVH_NATIVE
定义

CVH_NATIVE 表示在相同头文件 API 基础上，显式链接 src 编译出的 native backend，实现自动增强。

特点
使用与 Lite 相同的头文件
通过 dispatch 自动切换到更强实现
提供更多类型支持
提供更优性能
可集成 SIMD、多线程、平台特化
可增强 imgcodecs 能力
使用场景
正式工程
SDK
对性能有要求的项目
需要更广类型覆盖和更完整边界处理的环境
5.3 模式关系

必须满足以下关系：

Lite 是一个完整、闭环、可运行的最小子集
Native backend 建立在 Lite 之上
Native backend 不能破坏 Lite 已定义的核心语义
当 Native backend 不可用时，必须自动退回 Lite/fallback 路径
6. 能力开关设计

除模式外，项目还支持独立能力开关：

CVH_ENABLE_XSIMD
CVH_ENABLE_THREADS
CVH_ENABLE_FAST_MATH
CVH_ENABLE_OPENCV_INTRIN
CVH_ENABLE_PLATFORM_INTRINSICS

这些宏不决定项目模式，而只控制特定增强能力。

6.1 原则
CVH_LITE 不自动开启任何增强能力
CVH_NATIVE 可以按需开启能力开关
即使 header 中存在 SIMD 实现，Lite 也不应默认依赖它
任一能力开关缺失，都不得导致公开 API 不可运行
OpenCV Universal Intrinsics 只能作为 header-only SIMD adapter，不引入 OpenCV 完整 HAL
平台 intrinsic 只能作为 header 内可选优化，不进入 native backend 语义
7. 总体目录结构

推荐目录如下：

include/cvh/
  cvh.hpp

  core/
    mat.hpp
    types.hpp
    point.hpp
    size.hpp
    rect.hpp
    scalar.hpp
    range.hpp
    error.hpp
    log.hpp

  imgproc/
    resize.hpp
    filter.hpp
    color.hpp
    threshold.hpp

  imgcodecs/
    imgcodecs.hpp

  simd/
    pack.hpp
    load_store.hpp
    math.hpp
    traits.hpp

  detail/
    config.hpp
    dispatch.hpp
    registry.hpp

    fallback/
      resize_fallback.hpp
      gaussian_fallback.hpp
      box_filter_fallback.hpp
      color_fallback.hpp
      threshold_fallback.hpp
      imgcodecs_fallback.hpp
src/
  register_all.cpp

  imgproc/
    resize_backend.cpp
    gaussian_backend.cpp
    box_filter_backend.cpp
    color_backend.cpp
    threshold_backend.cpp

  imgcodecs/
    imread_backend.cpp
    imwrite_backend.cpp

  backend/
    scalar/
    simd/
    parallel/
8. 模块划分

项目分为三个主要模块：

8.1 core

职责：

基础数据结构
类型系统
内存与生命周期管理
日志与错误处理
常用宏与基础辅助能力

包括：

Mat
Point
Size
Rect
Scalar
Range
基础 type 定义
错误与断言机制
8.2 imgproc

职责：

高频图像处理 API
参数检查
输出分配
dispatch 调度
fallback 与 backend 统一对接

优先覆盖：

resize
GaussianBlur
boxFilter
threshold
少量 cvtColor
8.3 imgcodecs

职责：

提供轻量级图像读写能力
与 Mat 对齐数据结构
在 Lite 模式下提供有限格式支持
在 Native backend 下扩展更多 codec 能力

注意：

imgcodecs 在本项目中属于配套模块，不作为主要差异化战场。

9. Mat 设计

Mat 是整个项目的第一优先级地基。
本项目不要求完整复刻 OpenCV 的 cv::Mat，但必须保证常见核心语义稳定。

9.1 设计目标

Mat 应承担三项职责：

描述数据
管理数据生命周期
提供轻量视图与访问能力

Mat 不应承担复杂算法逻辑。

9.2 最小能力要求

Mat 必须支持：

rows / cols / type / channels
elemSize / elemSize1 / step
data 指针访问
isContinuous()
浅拷贝
深拷贝 clone()
ROI / submatrix view
基本外部内存包装能力（若接口开放）

9.3 语义约定
浅拷贝

拷贝构造与赋值默认为浅拷贝，只复制 header / view，底层数据共享。

深拷贝

clone() 与 copyTo() 采用深拷贝语义。

ROI

ROI 通过共享底层存储 + 调整偏移与尺寸实现，不复制底层数据。

连续性

isContinuous() 必须可判断连续内存路径，用于后续优化和快速分支。

9.5 当前非目标

当前 Mat 明确不追求：

expression template
lazy evaluation
copy-on-write
N 维通用张量全覆盖
GPU/CPU 统一存储抽象
完整复刻 OpenCV 内部内存模型
10. 类型系统设计

项目保留 OpenCV-like 的类型直觉，但不必完整照搬其所有宏细节。

推荐定义：

CVH_8UC1
CVH_8UC2
CVH_8UC3
CVH_8UC4
CVH_32FC1
CVH_32FC3
...

并提供：

depth()
channels()
elemSize()
elemSize1()

类型系统必须由 core 统一定义，Lite / Native backend 共用。

11. API / fallback / backend 三层结构

所有公开算子统一采用三层结构：

11.1 API 层

位于 include/cvh/imgproc/ 或 include/cvh/imgcodecs/。

职责：

参数接收
输出尺寸 / 类型推导
基础检查
调用 dispatch

禁止在 API 层直接堆算法主体。

11.2 fallback 层

位于 include/cvh/detail/fallback/。

职责：

提供 Lite 模式下的默认可运行实现
提供 Native backend 的安全兜底实现

要求：

独立、可运行
简洁、可维护
优先正确性与可读性
性能目标为“可接受”，非极致优化
11.3 backend 层

位于 src/。

职责：

提供更完整 dispatch
提供 SIMD、多线程、平台特化
扩展更广类型与边界支持
替换 fallback 的热路径实现
12. Dispatch 注册机制

项目统一采用函数表 dispatch，避免 API 直接依赖 src 唯一符号。

12.1 基本模式

每个算子定义：

fallback 实现
dispatch 函数指针
backend 注册入口
统一对外 API

示例：

namespace cvh::detail {

using ResizeFn = void(*)(const Mat&, Mat&, Size, double, double, int);

inline void resize_fallback(const Mat& src, Mat& dst, Size dsize,
                            double fx, double fy, int interp);

inline ResizeFn& resize_dispatch() {
    static ResizeFn fn = &resize_fallback;
    return fn;
}

inline void register_resize_backend(ResizeFn fn) {
    if (fn) resize_dispatch() = fn;
}

} // namespace cvh::detail

namespace cvh {

inline void resize(const Mat& src, Mat& dst, Size dsize,
                   double fx = 0, double fy = 0,
                   int interp = INTER_LINEAR) {
    detail::resize_dispatch()(src, dst, dsize, fx, fy, interp);
}

}
12.2 backend 注册方式

在 src 中通过静态注册器或统一注册入口安装 backend：

namespace cvh::detail {

static void resize_backend_impl(const Mat& src, Mat& dst, Size dsize,
                                double fx, double fy, int interp) {
    // 更强实现
}

struct ResizeBackendRegister {
    ResizeBackendRegister() {
        register_resize_backend(&resize_backend_impl);
    }
};

static ResizeBackendRegister g_resize_backend_register;

}
12.3 注册粒度

注册按“算子级”进行，而不是“模块级全量切换”。

例如：

resize 可单独增强
GaussianBlur 可单独增强
imread 可单独增强

这样便于：

渐进演进
单算子优化
不完整 Native backend 也能安全回退
12.4 dispatch 规则

必须保证：

backend 存在则优先走 backend
backend 不存在则自动回退 fallback
不允许某个公开 API 仅在 Native backend 下可运行
不允许因注册失败导致 API 崩溃或链接失败
13. SIMD 设计

SIMD 是可选优化能力，不是 Lite 模式成立的前提。

13.1 架构角色

SIMD 可以以两种形式存在：

Native backend 中的增强实现
可选的 header 内联优化路径

项目不要求 SIMD 只能存在于 src 中；但项目要求 Lite 默认不依赖 SIMD。

13.2 Lite / Native backend 关系

必须满足：

Lite 默认不以 SIMD 为成立前提
Lite 即使完全禁用 SIMD，也必须完整可运行
Native backend 可以自由使用 SIMD 做增强
header 内若存在 SIMD 优化实现，也必须是可选启用，而不是 Lite 默认必需
13.3 回退规则

无论 SIMD 代码位于 header 还是 backend，都必须满足：

无 SIMD 支持时，自动回退 scalar fallback 或 scalar backend
不得因平台、编译器、依赖差异导致 API 不可运行
所有 SIMD 路径都必须有可验证的非 SIMD 对应实现
13.4 xsimd 使用策略

推荐方式：

simd/ 目录中封装 xsimd 的薄包装
业务层不要直接依赖 xsimd 细节
通过 CVH_ENABLE_XSIMD 显式启用 SIMD 路径
不定义该宏时，自动退回 scalar 实现
13.5 OpenCV Universal Intrinsics adapter 策略

OpenCV Universal Intrinsics 是 header-only CPU 加速层的可选 adapter。

基本约束：

不复用 OpenCV 完整 HAL 函数接口层
不引入 OpenCV binary 依赖
不要求用户安装 OpenCV
不依赖 Native backend
不把 cv::v_* 作为 cvh 公开 API

推荐方式：

通过 CVH_ENABLE_OPENCV_INTRIN 显式启用
通过 cvh::detail::simd facade 间接使用
业务 kernel 不直接包含 vendored OpenCV intrin 头文件
OpenCV upstream 版本必须固定，并记录导入文件、license、local patches

13.6 平台 intrinsic 策略

平台 intrinsic 是 header-only 层的可选最高性能路径。

基本约束：

通过 CVH_ENABLE_PLATFORM_INTRINSICS 显式启用
由项目内部根据编译器宏判断 NEON / AVX / AVX2 等能力
用户不应直接定义 CVH_ARCH_* 平台宏
任一平台 intrinsic 不可用时，必须回退到 OpenCV intrin、xsimd 或 scalar

13.7 优先优化路径

SIMD 优先覆盖：

8UC1
8UC3
32FC1

优先场景：

连续内存
高频主路径
常见核尺寸与插值方式
14. 多线程设计

多线程属于增强能力，不属于 Lite 成立前提。

建议引入统一的轻量抽象：

template<typename F>
void parallel_for(int begin, int end, F&& f);

要求：

Lite 模式下可退化为单线程实现
Native backend 下可切换为更优线程 backend
业务层只依赖 parallel_for，不依赖具体线程库
15. Header-only 最小能力边界

Lite 模式必须形成一个真实可运行的最小闭环。

15.1 Lite 必须包含的基础能力
Mat
Point
Size
Rect
Scalar
Range
基础类型系统
基础错误处理
15.2 Lite 必须包含的最小 API

第一阶段最小边界建议为：

imread
imwrite
resize
boxFilter
GaussianBlur
threshold
cvtColor（至少 BGR2GRAY、GRAY2BGR）
15.3 Lite 最小闭环要求

Lite 模式下，用户必须可以完成：

读图
resize
blur
灰度化
threshold 或写图

若不能完成该闭环，则不得宣称“header-only 可运行”。

15.4 Lite 可接受限制

Lite 模式允许限制包括：

仅支持有限格式
仅支持少量 border type
仅支持少量 interpolation
优先支持连续内存主路径
仅覆盖主流类型
与 OpenCV 数值结果可能有差异
性能目标是“可用”，不是“最优”

这些限制必须明确写入 README。

16. imgproc 设计

imgproc 是项目最核心模块，优先级高于 imgcodecs。

推荐设计模式：

16.1 每个算子的统一结构

以 GaussianBlur 为例：

一个公开 API
一个参数规格化结构
一个 dispatch 入口
一个 fallback 实现
若干 backend kernel

示意：

struct GaussianParams {
    int kx;
    int ky;
    double sigmaX;
    double sigmaY;
    int borderType;
};
void gaussianBlur(const Mat& src, Mat& dst, Size ksize,
                  double sigmaX, double sigmaY = 0,
                  int borderType = BORDER_DEFAULT);
16.2 border 逻辑收口

边界处理必须集中管理，不允许散落在各个 kernel 中随意复制。

建议单独提供：

int border_interpolate(int p, int len, int borderType);

第一阶段优先支持：

BORDER_CONSTANT
BORDER_REPLICATE
BORDER_REFLECT_101
16.3 separable filter 优先

对于 boxFilter、GaussianBlur、后续 Sobel，优先使用可分离滤波架构：

横向 pass
纵向 pass

优点：

实现更稳
易于复用
更容易接入 SIMD
更利于后续优化
17. imgcodecs 设计

imgcodecs 只承担轻量图像 I/O 配套职责。

17.1 Lite 模式
提供有限格式支持
可基于轻量库实现
仅保证基础读写闭环
17.2 Native backend
扩展更多格式支持
增强参数检查
优化内部实现
17.3 约束

项目不承诺 imgcodecs 与 OpenCV 完全一致，必须明确说明：

支持哪些格式
哪些 flag/参数未实现
哪些行为可能与 OpenCV 存在差异
18. 一致性规则

为避免 Lite / Native backend 长期分裂，项目必须遵守以下规则。

18.1 必须共享的内容
类型系统
Mat 语义
API 签名
border 枚举
参数检查基本规则
错误与断言机制
18.2 允许分叉的内容
kernel 细节
dispatch 粒度
SIMD 与并行实现
性能路径
扩展类型与格式支持
18.3 明确禁止

禁止：

Lite / Native backend 使用两套不同公开头文件
Lite / Native backend 对同一 API 使用不同参数语义
Native backend 成为公开 API 的唯一来源
因优化路径引入不可回退行为
19. 对外文档表述规范

推荐 README 表述：

opencv-header-only 提供一个真实可运行的 header-only 最小子集，用于轻量集成与快速使用；同时可选编译 native backend，以获得更完整的 dispatch、更高性能和更广的类型支持。

禁止表述：

“OpenCV 完全替代品”
“100% 兼容 OpenCV”
“只改命名空间即可无缝替换”
“与 OpenCV 完全一致”
20. 演进路线
Phase 1
Mat 与基础类型稳定
Lite 最小闭环打通
实现 resize / threshold / GaussianBlur / boxFilter / BGR2GRAY
Phase 2
引入算子级 dispatch 注册
增强 imgproc backend
增加初步 SIMD 支持
Phase 3
引入并行 backend
扩展更多类型支持
扩展更多 border/interpolation/path
Phase 4
建立 benchmark 体系
建立 Lite / Native backend 一致性测试
完善兼容边界说明
21. 最终结论

cvh 的最终架构定义如下：

CVH_LITE：一个真实可运行、闭环的 header-only 最小子集
CVH_NATIVE：在相同 API 基础上的 native backend 增强模式
include/：定义公开 API、基础类型与 fallback 实现
src/：通过 dispatch 注册更强实现，而不是作为 API 唯一来源
SIMD 是可选优化能力，不是 Lite 成立前提
所有增强能力都必须遵守“可回退、可验证、不破坏 API”的原则

本项目的价值不在于形式上的“纯 header-only”，而在于：

提供一个诚实的、真实可运行的 header-only Lite 版本，并在需要时无缝升级到更强的 native backend。
