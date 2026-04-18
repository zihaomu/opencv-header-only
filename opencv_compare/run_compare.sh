#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPARE_DIR="${ROOT_DIR}/opencv_compare"
SETUP_SCRIPT="${COMPARE_DIR}/setup_opencv_bench_slim.sh"

PROFILE="${CVH_COMPARE_PROFILE:-quick}"
WARMUP="${CVH_COMPARE_WARMUP:-1}"
ITERS="${CVH_COMPARE_ITERS:-5}"
REPEATS="${CVH_COMPARE_REPEATS:-1}"
THREADS="${CVH_COMPARE_THREADS:-}"
BUILD_OPENCV="${CVH_COMPARE_BUILD_OPENCV:-auto}"
RENDER_MD="${CVH_COMPARE_RENDER_MD:-1}"
BUILD_TYPE="${CVH_COMPARE_BUILD_TYPE:-Release}"

OPENCV_DIR="${CVH_OPENCV_DIR:-${COMPARE_DIR}/opencv-bench-slim}"
BUILD_DIR="${CVH_COMPARE_BUILD_DIR:-${ROOT_DIR}/build-opencv-compare}"
OUTPUT_CSV="${CVH_COMPARE_OUTPUT:-${COMPARE_DIR}/results/current_compare_${PROFILE}.csv}"
OUTPUT_MD="${CVH_COMPARE_OUTPUT_MD:-${ROOT_DIR}/doc/opencv_compare_${PROFILE}.md}"
REPORT_SCRIPT="${COMPARE_DIR}/csv_to_markdown.py"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--profile quick|full] [--warmup N] [--iters N] [--repeats N] [--output path]

Environment:
  CVH_COMPARE_PROFILE   (default: ${PROFILE})
  CVH_COMPARE_WARMUP    (default: ${WARMUP})
  CVH_COMPARE_ITERS     (default: ${ITERS})
  CVH_COMPARE_REPEATS   (default: ${REPEATS})
  CVH_COMPARE_THREADS   (optional, exports OMP_NUM_THREADS)
  CVH_COMPARE_BUILD_OPENCV (default: ${BUILD_OPENCV}, values: auto|0|1)
  CVH_COMPARE_RENDER_MD (default: ${RENDER_MD}, set 0 to skip Markdown generation)
  CVH_COMPARE_BUILD_TYPE (default: ${BUILD_TYPE}, e.g. Release|RelWithDebInfo|Debug)
  CVH_COMPARE_BUILD_DIR (default: ${BUILD_DIR})
  CVH_COMPARE_OUTPUT    (default: ${OUTPUT_CSV})
  CVH_COMPARE_OUTPUT_MD (default: ${OUTPUT_MD})
  CVH_OPENCV_DIR        (default: ${OPENCV_DIR})
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --iters)
      ITERS="$2"
      shift 2
      ;;
    --repeats)
      REPEATS="$2"
      shift 2
      ;;
    --output)
      OUTPUT_CSV="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${PROFILE}" != "quick" && "${PROFILE}" != "full" ]]; then
  echo "Unsupported profile: ${PROFILE} (expected quick|full)" >&2
  exit 2
fi

if ! [[ "${WARMUP}" =~ ^[0-9]+$ && "${ITERS}" =~ ^[1-9][0-9]*$ && "${REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid warmup/iters/repeats config" >&2
  exit 2
fi

find_opencv_dir() {
  local src_dir="$1"
  local -a candidates=(
    "${src_dir}/build-slim"
    "${src_dir}/build-slim/lib/cmake/opencv4"
    "${src_dir}/build-slim/install/lib/cmake/opencv4"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}/OpenCVConfig.cmake" ]]; then
      echo "${candidate}"
      return 0
    fi
  done

  return 1
}

if [[ "${BUILD_OPENCV}" == "auto" ]]; then
  if find_opencv_dir "${OPENCV_DIR}" >/dev/null 2>&1; then
    BUILD_OPENCV="0"
  else
    BUILD_OPENCV="1"
  fi
fi

if [[ "${BUILD_OPENCV}" != "0" && "${BUILD_OPENCV}" != "1" ]]; then
  echo "Invalid CVH_COMPARE_BUILD_OPENCV=${BUILD_OPENCV} (expected auto|0|1)" >&2
  exit 2
fi

if [[ "${BUILD_OPENCV}" == "0" ]]; then
  echo "opencv_compare: setup mode=clone_or_update_only"
  "${SETUP_SCRIPT}"
else
  echo "opencv_compare: setup mode=clone_or_update_and_build"
  "${SETUP_SCRIPT}" --build
fi

if ! OPENCV_CONFIG_DIR="$(find_opencv_dir "${OPENCV_DIR}")"; then
  echo "Cannot find OpenCVConfig.cmake under ${OPENCV_DIR}/build-slim" >&2
  exit 2
fi

mkdir -p "${BUILD_DIR}" "$(dirname "${OUTPUT_CSV}")"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCVH_BUILD_FULL_BACKEND=ON \
  -DCVH_BUILD_LEGACY_CORE=ON \
  -DCVH_BUILD_BACKEND_KERNEL_SOURCES=ON \
  -DCVH_BUILD_TESTS=OFF \
  -DCVH_BUILD_BENCHMARKS=ON \
  -DCVH_ENABLE_OPENCV_COMPARE=ON \
  -DCVH_OPENCV_BENCH_DIR="${OPENCV_DIR}" \
  -DOpenCV_DIR="${OPENCV_CONFIG_DIR}"

cmake --build "${BUILD_DIR}" --target cvh_benchmark_compare -j

BENCH_BIN="${BUILD_DIR}/cvh_benchmark_compare"
if [[ ! -x "${BENCH_BIN}" ]]; then
  echo "Missing benchmark binary: ${BENCH_BIN}" >&2
  exit 2
fi

if [[ -n "${THREADS}" ]]; then
  echo "opencv_compare: running benchmark (profile=${PROFILE}, warmup=${WARMUP}, iters=${ITERS}, repeats=${REPEATS}, threads=${THREADS})"
  echo "opencv_compare: benchmark stage can take several minutes for quick profile with large kernels."
  OMP_NUM_THREADS="${THREADS}" "${BENCH_BIN}" \
    --profile "${PROFILE}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --repeats "${REPEATS}" \
    --output "${OUTPUT_CSV}"
else
  echo "opencv_compare: running benchmark (profile=${PROFILE}, warmup=${WARMUP}, iters=${ITERS}, repeats=${REPEATS})"
  echo "opencv_compare: benchmark stage can take several minutes for quick profile with large kernels."
  "${BENCH_BIN}" \
    --profile "${PROFILE}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --repeats "${REPEATS}" \
    --output "${OUTPUT_CSV}"
fi

if [[ "${RENDER_MD}" != "0" ]]; then
  if [[ ! -f "${REPORT_SCRIPT}" ]]; then
    echo "Missing report script: ${REPORT_SCRIPT}" >&2
    exit 2
  fi

  python3 "${REPORT_SCRIPT}" \
    --input "${OUTPUT_CSV}" \
    --output "${OUTPUT_MD}" \
    --title "cvh vs OpenCV Benchmark Report (${PROFILE})"
fi

echo "opencv_compare_done: output_csv=${OUTPUT_CSV}, output_md=${OUTPUT_MD}, profile=${PROFILE}, warmup=${WARMUP}, iters=${ITERS}, repeats=${REPEATS}, build_type=${BUILD_TYPE}, opencv_dir=${OPENCV_DIR}"
