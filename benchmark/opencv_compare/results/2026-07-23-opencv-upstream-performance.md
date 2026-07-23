# opencv-header-only vs OpenCV Upstream 性能对比（2026-07-23）

生成时间（UTC）：`2026-07-23 09:55:21Z`

## 当前项目状态

- `opencv-header-only` 当前公共定位是纯 header-only，不依赖项目内 `.cpp` 扩展层。
- Mode B 只比较当前 `cvh::headers_fast` 与同机编译的 upstream OpenCV；`cvh::headers_fast` 表示最快 header-only 构建配置。
- `cvh::headers_fast` 完整继承 `cvh::headers`。算子没有专用 fast-path 时继续执行继承的 header 实现并参与 benchmark，不因缺少 SIMD 特化而跳过。
- Core 的 `add/subtract/multiply/divide/transpose/GEMM` 已迁入 ODR-safe headers；本报告通过公共 API 测量，不链接 legacy core 对象。
- OpenCV Universal Intrinsics 是默认 SIMD 方言，kernel 直接使用 OpenCV UI；项目已移除 xsimd 性能路径。
- ARM 当前关注 NEON，本次实测平台为 Apple ARM；x86 目标是 SSE/AVX 系列，RVV 因 scalable vector 设计问题暂缓。
- 当前已验证的 UI 专用路径主要包括 `BGR/RGB2GRAY` 和 `CV_8UC1` 精确 2x `INTER_LINEAR` 下采样；其余本报告算子多数仍走继承的 header baseline。

## 高层优化结构

| 层次 | 当前实现 | 本报告中的含义 |
| --- | --- | --- |
| 公共 API | OpenCV-compatible header API | 所有 case 均从 `cvh::headers_fast` 公共入口调用 |
| SIMD 方言 | OpenCV Universal Intrinsics | 在 Apple ARM 上映射到 NEON |
| 专用 kernel | `cvtColor`、特定 `resize` UI kernel | 记录为 `dispatch_path=opencv_ui`；core 计算当前仍为 baseline |
| 通用实现 | `cvh::headers` 中的 header baseline | 无专用 fast-path 时自动继承，记录为 `headers_baseline` |
| 对照实现 | upstream OpenCV `core` / `imgproc` | 相同输入、尺寸、border 和线程配置 |

## 运行配置

- Profile：`stable`
- CVH 实现：`cvh_headers_fast`
- 采样：`warmup=2, iters=20, repeats=5`
- 线程数：`1`
- OpenMP：`dynamic=false, proc_bind=close`
- 主机：`Darwin arm64`
- CPU：`Apple M5`
- 编译器：`Apple clang version 21.0.0 (clang-2100.0.123.102)`
- 构建类型：`Release`
- CVH commit：`d6ab4e928fddd7f9e89aa2a83a8f7d9d4d28ab3d` + dirty
- OpenCV：`4.14.0`，commit `d48bf69f65444a13f8a34b8982b083c1b78fa0e8` + dirty
- 原始数据：`2026-07-23-opencv-upstream-performance.csv`；元数据：`2026-07-23-opencv-upstream-performance.meta.json`

## 汇总

- 总 case：`126`；有效：`126`；不支持：`0`。
- `OpenCV/CVH` 几何平均：`0.1155`；中位数：`0.1968`。
- CVH 更快：`11` 个；OpenCV 更快或相当：`115` 个。

| Suite | Cases | 几何平均 OpenCV/CVH | 中位数 | CVH 更快 | OpenCV 更快/相当 |
| --- | --- | --- | --- | --- | --- |
| core_mat | 84 | 0.2194 | 0.3252 | 10 | 74 |
| imgproc | 42 | 0.0320 | 0.0229 | 1 | 41 |

## 算子级概览

### `core_mat`

| Op | CVH dispatch | Cases | 几何平均 OpenCV/CVH | 领先方 |
| --- | --- | --- | --- | --- |
| ADD | headers_baseline | 12 | 0.1845 | OpenCV `5.42x` |
| DIVIDE | headers_baseline | 12 | 0.3396 | OpenCV `2.94x` |
| GEMM | headers_baseline | 6 | 0.0129 | OpenCV `77.65x` |
| MAT_CLONE | headers_baseline | 3 | 0.9982 | OpenCV `1.00x` |
| MAT_CONVERTTO | headers_baseline | 3 | 0.9845 | OpenCV `1.02x` |
| MAT_COPYTO | headers_baseline | 3 | 0.9960 | OpenCV `1.00x` |
| MAT_CREATE | headers_baseline | 3 | 0.1133 | OpenCV `8.82x` |
| MAT_RESHAPE | headers_baseline | 3 | 0.3962 | OpenCV `2.52x` |
| MAT_SETTO | headers_baseline | 3 | 0.0129 | OpenCV `77.53x` |
| MULTIPLY | headers_baseline | 12 | 0.1857 | OpenCV `5.39x` |
| SUBTRACT | headers_baseline | 12 | 0.1907 | OpenCV `5.24x` |
| TRANSPOSE | headers_baseline | 12 | 0.6292 | OpenCV `1.59x` |

### `imgproc`

| Op | CVH dispatch | Cases | 几何平均 OpenCV/CVH | 领先方 |
| --- | --- | --- | --- | --- |
| BOX_FILTER | headers_baseline | 3 | 0.0088 | OpenCV `113.71x` |
| CANNY | headers_baseline | 3 | 0.2510 | OpenCV `3.98x` |
| COPY_MAKE_BORDER | headers_baseline | 3 | 0.0276 | OpenCV `36.26x` |
| CVTCOLOR | opencv_ui | 3 | 1.0011 | CVH `1.00x` |
| DILATE | headers_baseline | 3 | 0.0037 | OpenCV `271.09x` |
| ERODE | headers_baseline | 3 | 0.0042 | OpenCV `240.41x` |
| FILTER2D | headers_baseline | 3 | 0.0031 | OpenCV `325.97x` |
| GAUSSIAN | headers_baseline | 3 | 0.0078 | OpenCV `129.02x` |
| LUT | headers_baseline | 3 | 0.0201 | OpenCV `49.83x` |
| RESIZE | opencv_ui | 3 | 0.6533 | OpenCV `1.53x` |
| SEP_FILTER2D | headers_baseline | 3 | 0.0224 | OpenCV `44.56x` |
| SOBEL | headers_baseline | 3 | 0.0235 | OpenCV `42.63x` |
| THRESHOLD | headers_baseline | 3 | 0.0341 | OpenCV `29.35x` |
| WARP_AFFINE | headers_baseline | 3 | 0.2307 | OpenCV `4.33x` |

## 详细结果

### `core_mat`

| Op | Variant | CVH dispatch | Depth | Ch | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADD | mat_mat_continuous | headers_baseline | CV_32F | 1 | 1080x1920 | 0.661119 | 0.214535 | 0.3245 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_32F | 1 | 480x640 | 0.097952 | 0.031879 | 0.3255 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_32F | 1 | 720x1280 | 0.287463 | 0.096148 | 0.3345 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_32F | 3 | 1080x1920 | 1.199756 | 0.615465 | 0.5130 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_32F | 3 | 480x640 | 0.180798 | 0.096979 | 0.5364 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_32F | 3 | 720x1280 | 0.534727 | 0.286921 | 0.5366 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_8U | 1 | 1080x1920 | 0.793737 | 0.049794 | 0.0627 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_8U | 1 | 480x640 | 0.119394 | 0.007046 | 0.0590 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_8U | 1 | 720x1280 | 0.356387 | 0.022373 | 0.0628 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_8U | 3 | 1080x1920 | 1.466194 | 0.158200 | 0.1079 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_8U | 3 | 480x640 | 0.209635 | 0.022877 | 0.1091 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_8U | 3 | 720x1280 | 0.652121 | 0.070935 | 0.1088 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_32F | 1 | 1080x1920 | 0.623706 | 0.200742 | 0.3219 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_32F | 1 | 480x640 | 0.093881 | 0.030244 | 0.3221 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_32F | 1 | 720x1280 | 0.279883 | 0.092560 | 0.3307 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_32F | 3 | 1080x1920 | 1.496950 | 0.662817 | 0.4428 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_32F | 3 | 480x640 | 0.222521 | 0.090700 | 0.4076 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_32F | 3 | 720x1280 | 0.663123 | 0.273300 | 0.4121 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_8U | 1 | 1080x1920 | 1.771271 | 0.452829 | 0.2557 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_8U | 1 | 480x640 | 0.261704 | 0.067112 | 0.2564 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_8U | 1 | 720x1280 | 0.788119 | 0.200481 | 0.2544 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_8U | 3 | 1080x1920 | 3.572377 | 1.358869 | 0.3804 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_8U | 3 | 480x640 | 0.527002 | 0.201046 | 0.3815 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_8U | 3 | 720x1280 | 1.588242 | 0.606019 | 0.3816 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| GEMM | fp32_nn_end_to_end | headers_baseline | CV_32F | 1 | 128x128x128 | 0.211969 | 0.003703 | 0.0175 | correctness=upstream_pass;iters=8 |
| GEMM | fp32_nn_end_to_end | headers_baseline | CV_32F | 1 | 256x256x256 | 2.003625 | 0.025084 | 0.0125 | correctness=upstream_pass;iters=1 |
| GEMM | fp32_nn_end_to_end | headers_baseline | CV_32F | 1 | 512x512x512 | 17.904792 | 0.193916 | 0.0108 | correctness=upstream_pass;iters=1 |
| GEMM | fp32_nn_pack_once | headers_baseline | CV_32F | 1 | 128x128x128 | 0.223885 | 0.003599 | 0.0161 | correctness=upstream_pass;opencv_reuses_B_without_public_pack_handle;iters=8 |
| GEMM | fp32_nn_pack_once | headers_baseline | CV_32F | 1 | 256x256x256 | 2.094834 | 0.021625 | 0.0103 | correctness=upstream_pass;opencv_reuses_B_without_public_pack_handle;iters=1 |
| GEMM | fp32_nn_pack_once | headers_baseline | CV_32F | 1 | 512x512x512 | 19.035500 | 0.221000 | 0.0116 | correctness=upstream_pass;opencv_reuses_B_without_public_pack_handle;iters=1 |
| MAT_CLONE | full_copy | headers_baseline | CV_8U | 1 | 1080x1920 | 0.025048 | 0.025092 | 1.0017 |  |
| MAT_CLONE | full_copy | headers_baseline | CV_8U | 1 | 480x640 | 0.003737 | 0.003656 | 0.9783 |  |
| MAT_CLONE | full_copy | headers_baseline | CV_8U | 1 | 720x1280 | 0.010571 | 0.010729 | 1.0150 |  |
| MAT_CONVERTTO | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | 1080x1920 | 0.078512 | 0.078158 | 0.9955 |  |
| MAT_CONVERTTO | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | 480x640 | 0.013021 | 0.012031 | 0.9240 |  |
| MAT_CONVERTTO | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | 720x1280 | 0.035325 | 0.036644 | 1.0373 |  |
| MAT_COPYTO | continuous_reuse | headers_baseline | CV_8U | 1 | 1080x1920 | 0.024800 | 0.024983 | 1.0074 |  |
| MAT_COPYTO | continuous_reuse | headers_baseline | CV_8U | 1 | 480x640 | 0.003544 | 0.003427 | 0.9671 |  |
| MAT_COPYTO | continuous_reuse | headers_baseline | CV_8U | 1 | 720x1280 | 0.010202 | 0.010348 | 1.0143 |  |
| MAT_CREATE | reuse_same_shape | headers_baseline | CV_8U | 1 | 1080x1920 | 0.000010 | 0.000001 | 0.1196 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | reuse_same_shape | headers_baseline | CV_8U | 1 | 480x640 | 0.000010 | 0.000001 | 0.1136 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | reuse_same_shape | headers_baseline | CV_8U | 1 | 720x1280 | 0.000011 | 0.000001 | 0.1071 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_RESHAPE | to_column_view | headers_baseline | CV_8U | 1 | 1080x1920 | 0.000039 | 0.000015 | 0.3944 | micro_iters_x1000 |
| MAT_RESHAPE | to_column_view | headers_baseline | CV_8U | 1 | 480x640 | 0.000036 | 0.000015 | 0.4236 | micro_iters_x1000 |
| MAT_RESHAPE | to_column_view | headers_baseline | CV_8U | 1 | 720x1280 | 0.000041 | 0.000015 | 0.3721 | micro_iters_x1000 |
| MAT_SETTO | scalar_all | headers_baseline | CV_8U | 1 | 1080x1920 | 1.996469 | 0.024075 | 0.0121 |  |
| MAT_SETTO | scalar_all | headers_baseline | CV_8U | 1 | 480x640 | 0.291087 | 0.003962 | 0.0136 |  |
| MAT_SETTO | scalar_all | headers_baseline | CV_8U | 1 | 720x1280 | 0.863325 | 0.011283 | 0.0131 |  |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_32F | 1 | 1080x1920 | 0.619033 | 0.200888 | 0.3245 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_32F | 1 | 480x640 | 0.092923 | 0.030200 | 0.3250 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_32F | 1 | 720x1280 | 0.293923 | 0.095923 | 0.3264 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_32F | 3 | 1080x1920 | 1.200869 | 0.638442 | 0.5316 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_32F | 3 | 480x640 | 0.171519 | 0.092258 | 0.5379 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_32F | 3 | 720x1280 | 0.500685 | 0.273017 | 0.5453 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_8U | 1 | 1080x1920 | 0.798383 | 0.049710 | 0.0623 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_8U | 1 | 480x640 | 0.117708 | 0.007008 | 0.0595 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_8U | 1 | 720x1280 | 0.354619 | 0.022048 | 0.0622 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_8U | 3 | 1080x1920 | 1.376013 | 0.159969 | 0.1163 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_8U | 3 | 480x640 | 0.205577 | 0.022027 | 0.1071 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_8U | 3 | 720x1280 | 0.613729 | 0.066896 | 0.1090 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_32F | 1 | 1080x1920 | 0.632567 | 0.202458 | 0.3201 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_32F | 1 | 480x640 | 0.093981 | 0.030829 | 0.3280 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_32F | 1 | 720x1280 | 0.294177 | 0.096521 | 0.3281 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_32F | 3 | 1080x1920 | 1.160750 | 0.611956 | 0.5272 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_32F | 3 | 480x640 | 0.180510 | 0.092765 | 0.5139 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_32F | 3 | 720x1280 | 0.506498 | 0.272540 | 0.5381 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_8U | 1 | 1080x1920 | 0.752983 | 0.049656 | 0.0659 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_8U | 1 | 480x640 | 0.111354 | 0.007010 | 0.0630 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_8U | 1 | 720x1280 | 0.335235 | 0.021931 | 0.0654 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_8U | 3 | 1080x1920 | 1.330213 | 0.152031 | 0.1143 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_8U | 3 | 480x640 | 0.186348 | 0.022367 | 0.1200 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_8U | 3 | 720x1280 | 0.575642 | 0.071042 | 0.1234 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_32F | 1 | 1080x1920 | 0.793465 | 0.602904 | 0.7598 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_32F | 1 | 480x640 | 0.160488 | 0.075290 | 0.4691 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_32F | 1 | 720x1280 | 0.378240 | 0.306225 | 0.8096 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_32F | 3 | 1080x1920 | 0.730654 | 1.420619 | 1.9443 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_32F | 3 | 480x640 | 0.118321 | 0.155806 | 1.3168 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_32F | 3 | 720x1280 | 0.292979 | 0.561375 | 1.9161 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_8U | 1 | 1080x1920 | 0.592610 | 0.130581 | 0.2203 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_8U | 1 | 480x640 | 0.140527 | 0.006892 | 0.0490 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_8U | 1 | 720x1280 | 0.291469 | 0.028602 | 0.0981 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_8U | 3 | 1080x1920 | 0.417619 | 0.748317 | 1.7919 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_8U | 3 | 480x640 | 0.113079 | 0.089627 | 0.7926 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_8U | 3 | 720x1280 | 0.217123 | 0.392194 | 1.8063 | correctness=upstream_pass |

### `imgproc`

| Op | Variant | CVH dispatch | Depth | Ch | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BOX_FILTER | 3x3_replicate | headers_baseline | CV_8U | 1 | 1080x1920 | 29.503427 | 0.254992 | 0.0086 |  |
| BOX_FILTER | 3x3_replicate | headers_baseline | CV_8U | 1 | 480x640 | 4.475544 | 0.040056 | 0.0089 |  |
| BOX_FILTER | 3x3_replicate | headers_baseline | CV_8U | 1 | 720x1280 | 13.099240 | 0.115173 | 0.0088 |  |
| CANNY | aperture3_l1 | headers_baseline | CV_8U | 1 | 1080x1920 | 110.895704 | 27.852394 | 0.2512 |  |
| CANNY | aperture3_l1 | headers_baseline | CV_8U | 1 | 480x640 | 16.135579 | 3.992065 | 0.2474 |  |
| CANNY | aperture3_l1 | headers_baseline | CV_8U | 1 | 720x1280 | 48.030223 | 12.229798 | 0.2546 |  |
| COPY_MAKE_BORDER | 2px_replicate | headers_baseline | CV_8U | 1 | 1080x1920 | 1.576060 | 0.041598 | 0.0264 |  |
| COPY_MAKE_BORDER | 2px_replicate | headers_baseline | CV_8U | 1 | 480x640 | 0.227767 | 0.006235 | 0.0274 |  |
| COPY_MAKE_BORDER | 2px_replicate | headers_baseline | CV_8U | 1 | 720x1280 | 0.669131 | 0.019417 | 0.0290 |  |
| CVTCOLOR | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | 1080x1920 | 0.204288 | 0.205375 | 1.0053 |  |
| CVTCOLOR | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | 480x640 | 0.039598 | 0.039590 | 0.9998 |  |
| CVTCOLOR | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | 720x1280 | 0.097596 | 0.097410 | 0.9981 |  |
| DILATE | 3x3_replicate | headers_baseline | CV_8U | 1 | 1080x1920 | 39.371325 | 0.132513 | 0.0034 |  |
| DILATE | 3x3_replicate | headers_baseline | CV_8U | 1 | 480x640 | 5.623844 | 0.021163 | 0.0038 |  |
| DILATE | 3x3_replicate | headers_baseline | CV_8U | 1 | 720x1280 | 16.698931 | 0.066173 | 0.0040 |  |
| ERODE | 3x3_replicate | headers_baseline | CV_8U | 1 | 1080x1920 | 35.502950 | 0.141894 | 0.0040 |  |
| ERODE | 3x3_replicate | headers_baseline | CV_8U | 1 | 480x640 | 4.980773 | 0.020927 | 0.0042 |  |
| ERODE | 3x3_replicate | headers_baseline | CV_8U | 1 | 720x1280 | 15.336125 | 0.065715 | 0.0043 |  |
| FILTER2D | 3x3_replicate | headers_baseline | CV_8U | 1 | 1080x1920 | 197.430175 | 0.584600 | 0.0030 |  |
| FILTER2D | 3x3_replicate | headers_baseline | CV_8U | 1 | 480x640 | 27.790931 | 0.084165 | 0.0030 |  |
| FILTER2D | 3x3_replicate | headers_baseline | CV_8U | 1 | 720x1280 | 85.482817 | 0.275248 | 0.0032 |  |
| GAUSSIAN | 5x5_replicate | headers_baseline | CV_8U | 1 | 1080x1920 | 24.425944 | 0.194731 | 0.0080 |  |
| GAUSSIAN | 5x5_replicate | headers_baseline | CV_8U | 1 | 480x640 | 3.669371 | 0.027517 | 0.0075 |  |
| GAUSSIAN | 5x5_replicate | headers_baseline | CV_8U | 1 | 720x1280 | 10.825290 | 0.084308 | 0.0078 |  |
| LUT | invert_u8 | headers_baseline | CV_8U | 1 | 1080x1920 | 9.329473 | 0.185890 | 0.0199 |  |
| LUT | invert_u8 | headers_baseline | CV_8U | 1 | 480x640 | 1.292712 | 0.025573 | 0.0198 |  |
| LUT | invert_u8 | headers_baseline | CV_8U | 1 | 720x1280 | 3.894881 | 0.079858 | 0.0205 |  |
| RESIZE | linear_half_u8c1 | opencv_ui | CV_8U | 1 | 1080x1920 | 0.121550 | 0.080960 | 0.6661 |  |
| RESIZE | linear_half_u8c1 | opencv_ui | CV_8U | 1 | 480x640 | 0.025669 | 0.015737 | 0.6131 |  |
| RESIZE | linear_half_u8c1 | opencv_ui | CV_8U | 1 | 720x1280 | 0.058335 | 0.039825 | 0.6827 |  |
| SEP_FILTER2D | 3x3_replicate | headers_baseline | CV_8U | 1 | 1080x1920 | 27.383225 | 0.610262 | 0.0223 |  |
| SEP_FILTER2D | 3x3_replicate | headers_baseline | CV_8U | 1 | 480x640 | 3.765662 | 0.082667 | 0.0220 |  |
| SEP_FILTER2D | 3x3_replicate | headers_baseline | CV_8U | 1 | 720x1280 | 11.296533 | 0.260923 | 0.0231 |  |
| SOBEL | dx1_ksize3_replicate | headers_baseline | CV_8U | 1 | 1080x1920 | 31.578829 | 0.779340 | 0.0247 |  |
| SOBEL | dx1_ksize3_replicate | headers_baseline | CV_8U | 1 | 480x640 | 4.658667 | 0.106952 | 0.0230 |  |
| SOBEL | dx1_ksize3_replicate | headers_baseline | CV_8U | 1 | 720x1280 | 13.610654 | 0.310148 | 0.0228 |  |
| THRESHOLD | binary_u8 | headers_baseline | CV_8U | 1 | 1080x1920 | 0.963300 | 0.033244 | 0.0345 |  |
| THRESHOLD | binary_u8 | headers_baseline | CV_8U | 1 | 480x640 | 0.177069 | 0.005975 | 0.0337 |  |
| THRESHOLD | binary_u8 | headers_baseline | CV_8U | 1 | 720x1280 | 0.435477 | 0.014796 | 0.0340 |  |
| WARP_AFFINE | linear_inverse_replicate | headers_baseline | CV_8U | 1 | 1080x1920 | 11.200396 | 1.940852 | 0.1733 |  |
| WARP_AFFINE | linear_inverse_replicate | headers_baseline | CV_8U | 1 | 480x640 | 1.100106 | 0.308183 | 0.2801 |  |
| WARP_AFFINE | linear_inverse_replicate | headers_baseline | CV_8U | 1 | 720x1280 | 3.467054 | 0.877206 | 0.2530 |  |

## 说明

- 比值统一为 `OpenCV耗时 / CVH耗时`：大于 `1` 表示 CVH 更快，小于 `1` 表示 OpenCV 更快。
- 表内耗时取各 repeat 的最小单次耗时，用于降低系统抖动影响；本报告不是跨机器排名。
- Mat case 对比相同的分配/复用语义；imgproc case 对齐输入尺寸、类型、kernel、border 和主要参数。
- `headers_baseline` 不等于跳过优化，它表示 `cvh::headers_fast` 当前继承了 `cvh::headers` 的通用实现。
- 原始 CSV 和 metadata 是可再生成的运行产物，日期命名 Markdown 是阶段性快照。
