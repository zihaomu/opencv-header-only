#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${CVH_BENCH_BUILD_DIR:-${ROOT_DIR}/build-bench-headers-quick}"
RESULT_DIR="${CVH_BENCH_RESULT_DIR:-${ROOT_DIR}/benchmark/results/internal/ci_headers_quick}"
BASELINE_REF="${CVH_BENCH_BASELINE_REF:-}"
MAX_SLOWDOWN="${CVH_BENCH_MAX_SLOWDOWN:-0.08}"

if [[ -n "${BASELINE_REF}" ]]; then
  "${ROOT_DIR}/benchmark/internal/run_header_regression.sh" \
    --baseline-ref "${BASELINE_REF}" \
    --suite all \
    --profile quick \
    --target headers_fast \
    --output-dir "${ROOT_DIR}/benchmark/results/internal" \
    --max-slowdown "${MAX_SLOWDOWN}"
  exit 0
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCVH_BUILD_NATIVE_BACKEND=OFF \
  -DCVH_BUILD_TESTS=OFF \
  -DCVH_BUILD_BENCHMARKS=ON

cmake --build "${BUILD_DIR}" -j --target \
  cvh_benchmark_core_mat_header \
  cvh_benchmark_imgproc_header \
  cvh_benchmark_cvtcolor_bgr2gray_header \
  cvh_benchmark_resize_bilinear_header

mkdir -p "${RESULT_DIR}"

"${BUILD_DIR}/cvh_benchmark_core_mat_header" \
  --profile quick --warmup 0 --iters 1 --repeats 1 \
  --output "${RESULT_DIR}/core_mat_current.csv" >/dev/null

"${BUILD_DIR}/cvh_benchmark_imgproc_header" \
  --profile quick --warmup 0 --iters 1 --repeats 1 \
  --output "${RESULT_DIR}/imgproc_current.csv" >/dev/null

"${BUILD_DIR}/cvh_benchmark_cvtcolor_bgr2gray_header" \
  --profile quick --warmup 0 --iters 1 --repeats 1 \
  --output "${RESULT_DIR}/cvtcolor_current.csv" >/dev/null

"${BUILD_DIR}/cvh_benchmark_resize_bilinear_header" \
  --profile quick --warmup 0 --iters 1 --repeats 1 \
  --output "${RESULT_DIR}/resize_current.csv" >/dev/null

echo "ci_benchmark_headers_quick_done: result_dir=${RESULT_DIR}"
