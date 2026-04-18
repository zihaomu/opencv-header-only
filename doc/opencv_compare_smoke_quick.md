# cvh vs OpenCV Benchmark Report (smoke quick)

Generated at (UTC): `2026-04-18 05:14:53Z`

Source CSV: `opencv_compare/results/smoke_compare_quick.csv`

## Summary

- Total rows: `19`
- Supported rows (`status=OK`): `13`
- Unsupported rows: `6`
- Mean speedup (`OpenCV/CVH`): `0.0619`
- Median speedup (`OpenCV/CVH`): `0.0168`
- Cases where CVH is faster (`OpenCV/CVH > 1`): `0`
- Cases where OpenCV is faster or equal (`OpenCV/CVH <= 1`): `13`

## Supported Cases

| Op | Depth | Ch | Shape | CVH (ms) | OpenCV (ms) | OpenCV/CVH |
| --- | --- | --- | --- | --- | --- | --- |
| ADD | CV_32F | 1 | 480x640 | 8.693083 | 0.135458 | 0.015582 |
| ADD | CV_32F | 3 | 480x640 | 25.979625 | 0.435416 | 0.016760 |
| ADD | CV_8U | 1 | 480x640 | 2.288833 | 0.422333 | 0.184519 |
| ADD | CV_8U | 3 | 480x640 | 6.205500 | 0.109833 | 0.017699 |
| GAUSSIAN_5X5 | CV_32F | 1 | 480x640 | 20.826292 | 1.056334 | 0.050721 |
| GAUSSIAN_5X5 | CV_32F | 3 | 480x640 | 63.097584 | 0.533417 | 0.008454 |
| GAUSSIAN_5X5 | CV_8U | 1 | 480x640 | 5.830916 | 2.500208 | 0.428785 |
| GAUSSIAN_5X5 | CV_8U | 3 | 480x640 | 13.224292 | 0.246666 | 0.018652 |
| GEMM | CV_32F | 1 | 256x256x256 | 175.123084 | 4.499708 | 0.025695 |
| SUB | CV_32F | 1 | 480x640 | 8.670000 | 0.047959 | 0.005532 |
| SUB | CV_32F | 3 | 480x640 | 25.730750 | 0.301375 | 0.011713 |
| SUB | CV_8U | 1 | 480x640 | 2.200042 | 0.028834 | 0.013106 |
| SUB | CV_8U | 3 | 480x640 | 6.229042 | 0.043292 | 0.006950 |

## Unsupported Cases

| Op | Depth | Ch | Shape | OpenCV (ms) | Status | Note |
| --- | --- | --- | --- | --- | --- | --- |
| DILATE | CV_8U | 1 | 480x640 | 0.135375 | UNSUPPORTED_CVH | cvh_api_not_available_yet |
| DILATE | CV_8U | 3 | 480x640 | 0.087500 | UNSUPPORTED_CVH | cvh_api_not_available_yet |
| ERODE | CV_8U | 1 | 480x640 | 0.329125 | UNSUPPORTED_CVH | cvh_api_not_available_yet |
| ERODE | CV_8U | 3 | 480x640 | 0.086500 | UNSUPPORTED_CVH | cvh_api_not_available_yet |
| SOBEL | CV_8U | 1 | 480x640 | 0.541625 | UNSUPPORTED_CVH | cvh_api_not_available_yet |
| SOBEL | CV_8U | 3 | 480x640 | 0.467291 | UNSUPPORTED_CVH | cvh_api_not_available_yet |

## Notes

- Speedup column is `OpenCV/CVH`; values `< 1` mean OpenCV is faster for that case.
- This report is generated automatically from the compare CSV.
