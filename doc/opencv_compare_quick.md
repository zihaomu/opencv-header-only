# cvh vs OpenCV Benchmark Report (quick)

Generated at (UTC): `2026-04-18 06:58:00Z`

Source CSV: `/Volumes/SSD1T/work_ssd/my_project/opencv-header-only/opencv_compare/results/current_compare_quick.csv`

## Summary

- Total rows: `58`
- Supported rows (`status=OK`): `49`
- Unsupported rows: `9`
- Mean speedup (`OpenCV/CVH`): `0.1441`
- Median speedup (`OpenCV/CVH`): `0.1388`
- Cases where CVH is faster (`OpenCV/CVH > 1`): `0`
- Cases where OpenCV is faster or equal (`OpenCV/CVH <= 1`): `49`

## Supported Cases

| Op | Depth | Ch | Shape | CVH (ms) | OpenCV (ms) | OpenCV/CVH |
| --- | --- | --- | --- | --- | --- | --- |
| ADD | CV_32F | 1 | 480x640 | 0.280658 | 0.039192 | 0.139642 |
| ADD | CV_32F | 3 | 480x640 | 1.063300 | 0.213992 | 0.201252 |
| ADD | CV_32F | 4 | 480x640 | 0.888508 | 0.241492 | 0.271795 |
| ADD | CV_8U | 1 | 480x640 | 0.178092 | 0.010483 | 0.058865 |
| ADD | CV_8U | 3 | 480x640 | 0.239350 | 0.033217 | 0.138778 |
| ADD | CV_8U | 4 | 480x640 | 0.285058 | 0.040900 | 0.143479 |
| BOX_11X11 | CV_32F | 1 | 480x640 | 68.334858 | 0.400158 | 0.005856 |
| BOX_11X11 | CV_32F | 3 | 480x640 | 221.180650 | 0.669258 | 0.003026 |
| BOX_11X11 | CV_32F | 4 | 480x640 | 313.395208 | 1.054367 | 0.003364 |
| BOX_11X11 | CV_8U | 1 | 480x640 | 0.441692 | 0.229550 | 0.519707 |
| BOX_11X11 | CV_8U | 3 | 480x640 | 0.936825 | 0.402175 | 0.429296 |
| BOX_11X11 | CV_8U | 4 | 480x640 | 1.190183 | 0.509033 | 0.427693 |
| BOX_3X3 | CV_32F | 1 | 480x640 | 7.150183 | 0.132408 | 0.018518 |
| BOX_3X3 | CV_32F | 3 | 480x640 | 23.321458 | 0.435067 | 0.018655 |
| BOX_3X3 | CV_32F | 4 | 480x640 | 29.427483 | 0.521775 | 0.017731 |
| BOX_3X3 | CV_8U | 1 | 480x640 | 0.384958 | 0.064475 | 0.167486 |
| BOX_3X3 | CV_8U | 3 | 480x640 | 1.047917 | 0.186142 | 0.177630 |
| BOX_3X3 | CV_8U | 4 | 480x640 | 1.181425 | 0.210525 | 0.178196 |
| BOX_5X5 | CV_32F | 1 | 480x640 | 17.503592 | 0.177717 | 0.010153 |
| BOX_5X5 | CV_32F | 3 | 480x640 | 55.780692 | 0.606525 | 0.010873 |
| BOX_5X5 | CV_32F | 4 | 480x640 | 72.592725 | 0.686842 | 0.009462 |
| BOX_5X5 | CV_8U | 1 | 480x640 | 0.464608 | 0.066433 | 0.142988 |
| BOX_5X5 | CV_8U | 3 | 480x640 | 0.987542 | 0.173158 | 0.175343 |
| BOX_5X5 | CV_8U | 4 | 480x640 | 1.183342 | 0.228467 | 0.193069 |
| GAUSSIAN_11X11 | CV_32F | 1 | 480x640 | 12.725000 | 0.299708 | 0.023553 |
| GAUSSIAN_11X11 | CV_32F | 3 | 480x640 | 37.678925 | 0.937292 | 0.024876 |
| GAUSSIAN_11X11 | CV_32F | 4 | 480x640 | 48.358217 | 1.239675 | 0.025635 |
| GAUSSIAN_11X11 | CV_8U | 1 | 480x640 | 0.931775 | 0.226292 | 0.242861 |
| GAUSSIAN_11X11 | CV_8U | 3 | 480x640 | 1.175208 | 0.434883 | 0.370048 |
| GAUSSIAN_11X11 | CV_8U | 4 | 480x640 | 0.922633 | 0.611000 | 0.662235 |
| GAUSSIAN_3X3 | CV_32F | 1 | 480x640 | 3.928658 | 0.108758 | 0.027683 |
| GAUSSIAN_3X3 | CV_32F | 3 | 480x640 | 12.393267 | 0.391125 | 0.031559 |
| GAUSSIAN_3X3 | CV_32F | 4 | 480x640 | 15.093933 | 0.516700 | 0.034232 |
| GAUSSIAN_3X3 | CV_8U | 1 | 480x640 | 0.396300 | 0.049908 | 0.125936 |
| GAUSSIAN_3X3 | CV_8U | 3 | 480x640 | 0.675958 | 0.068800 | 0.101781 |
| GAUSSIAN_3X3 | CV_8U | 4 | 480x640 | 0.445333 | 0.075100 | 0.168638 |
| GAUSSIAN_5X5 | CV_32F | 1 | 480x640 | 5.850217 | 0.155242 | 0.026536 |
| GAUSSIAN_5X5 | CV_32F | 3 | 480x640 | 17.998158 | 0.563900 | 0.031331 |
| GAUSSIAN_5X5 | CV_32F | 4 | 480x640 | 24.105817 | 0.723800 | 0.030026 |
| GAUSSIAN_5X5 | CV_8U | 1 | 480x640 | 0.540083 | 0.071458 | 0.132310 |
| GAUSSIAN_5X5 | CV_8U | 3 | 480x640 | 0.811758 | 0.134758 | 0.166008 |
| GAUSSIAN_5X5 | CV_8U | 4 | 480x640 | 0.562458 | 0.118433 | 0.210563 |
| GEMM | CV_32F | 1 | 256x256x256 | 0.763225 | 0.038608 | 0.050586 |
| SUB | CV_32F | 1 | 480x640 | 0.276592 | 0.041008 | 0.148263 |
| SUB | CV_32F | 3 | 480x640 | 0.559117 | 0.185242 | 0.331311 |
| SUB | CV_32F | 4 | 480x640 | 0.665725 | 0.187983 | 0.282374 |
| SUB | CV_8U | 1 | 480x640 | 0.178850 | 0.010383 | 0.058056 |
| SUB | CV_8U | 3 | 480x640 | 0.240192 | 0.034242 | 0.142560 |
| SUB | CV_8U | 4 | 480x640 | 0.272033 | 0.041217 | 0.151513 |

## Unsupported Cases

| Op | Depth | Ch | Shape | OpenCV (ms) | Status | Note |
| --- | --- | --- | --- | --- | --- | --- |
| DILATE | CV_8U | 1 | 480x640 | 0.034817 | UNSUPPORTED_CVH | cvh_api_not_available_yet |
| DILATE | CV_8U | 3 | 480x640 | 0.077725 | UNSUPPORTED_CVH | cvh_api_not_available_yet |
| DILATE | CV_8U | 4 | 480x640 | 0.121550 | UNSUPPORTED_CVH | cvh_api_not_available_yet |
| ERODE | CV_8U | 1 | 480x640 | 0.033550 | UNSUPPORTED_CVH | cvh_api_not_available_yet |
| ERODE | CV_8U | 3 | 480x640 | 0.077625 | UNSUPPORTED_CVH | cvh_api_not_available_yet |
| ERODE | CV_8U | 4 | 480x640 | 0.132883 | UNSUPPORTED_CVH | cvh_api_not_available_yet |
| SOBEL | CV_8U | 1 | 480x640 | 0.166275 | UNSUPPORTED_CVH | cvh_api_not_available_yet |
| SOBEL | CV_8U | 3 | 480x640 | 0.581958 | UNSUPPORTED_CVH | cvh_api_not_available_yet |
| SOBEL | CV_8U | 4 | 480x640 | 0.884275 | UNSUPPORTED_CVH | cvh_api_not_available_yet |

## Notes

- Speedup column is `OpenCV/CVH`; values `< 1` mean OpenCV is faster for that case.
- This report is generated automatically from the compare CSV.
