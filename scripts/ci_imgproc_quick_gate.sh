#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-imgproc-benchmark-gate"

PROFILE="${CVH_IMGPROC_BENCH_PROFILE:-quick}"
WARMUP="${CVH_IMGPROC_BENCH_WARMUP:-2}"
ITERS="${CVH_IMGPROC_BENCH_ITERS:-20}"
REPEATS="${CVH_IMGPROC_BENCH_REPEATS:-5}"
MAX_SLOWDOWN="${CVH_IMGPROC_BENCH_MAX_SLOWDOWN:-0.08}"
THREADS="${CVH_IMGPROC_BENCH_THREADS:-1}"
OMP_DYNAMIC_MODE="${CVH_IMGPROC_BENCH_OMP_DYNAMIC:-false}"
OMP_PROC_BIND_MODE="${CVH_IMGPROC_BENCH_OMP_PROC_BIND:-close}"
CPU_LIST="${CVH_IMGPROC_BENCH_CPU_LIST:-auto}"

if [[ "${PROFILE}" != "quick" && "${PROFILE}" != "full" ]]; then
  echo "Unsupported CVH_IMGPROC_BENCH_PROFILE=${PROFILE}, expected quick|full" >&2
  exit 2
fi

DEFAULT_BASELINE="${ROOT_DIR}/benchmark/baseline_imgproc_${PROFILE}.csv"
DEFAULT_CURRENT="${BUILD_DIR}/current_imgproc_${PROFILE}.csv"
BASELINE_CSV="${CVH_IMGPROC_BENCH_BASELINE:-${DEFAULT_BASELINE}}"
CURRENT_CSV="${CVH_IMGPROC_BENCH_CURRENT:-${DEFAULT_CURRENT}}"

if [[ "${BASELINE_CSV}" != /* ]]; then
  BASELINE_CSV="${ROOT_DIR}/${BASELINE_CSV}"
fi
if [[ "${CURRENT_CSV}" != /* ]]; then
  CURRENT_CSV="${ROOT_DIR}/${CURRENT_CSV}"
fi

mkdir -p "${BUILD_DIR}" "$(dirname "${CURRENT_CSV}")"

if [[ ! -f "${BASELINE_CSV}" ]]; then
  echo "imgproc benchmark baseline not found: ${BASELINE_CSV}" >&2
  echo "Create baseline first, for example:" >&2
  echo "  ./build-full-test/cvh_benchmark_imgproc_ops --profile ${PROFILE} --warmup ${WARMUP} --iters ${ITERS} --repeats ${REPEATS} --output benchmark/baseline_imgproc_${PROFILE}.csv" >&2
  exit 2
fi

echo "imgproc_bench_gate_config: profile=${PROFILE}, warmup=${WARMUP}, iters=${ITERS}, repeats=${REPEATS}, max_slowdown=${MAX_SLOWDOWN}"
TASKSET_CPU_LIST=""
if command -v taskset >/dev/null 2>&1; then
  if [[ "${CPU_LIST}" == "auto" ]]; then
    ALLOWED_CPU_LIST="$(awk '/Cpus_allowed_list:/ {print $2}' /proc/self/status 2>/dev/null || true)"
    TASKSET_CPU_LIST="${ALLOWED_CPU_LIST%%,*}"
    TASKSET_CPU_LIST="${TASKSET_CPU_LIST%%-*}"
  elif [[ "${CPU_LIST}" != "off" ]]; then
    TASKSET_CPU_LIST="${CPU_LIST}"
  fi
fi
echo "imgproc_bench_runtime: threads=${THREADS}, omp_dynamic=${OMP_DYNAMIC_MODE}, omp_proc_bind=${OMP_PROC_BIND_MODE}, cpu_list=${TASKSET_CPU_LIST:-none}"
echo "imgproc_bench_gate_files: baseline=${BASELINE_CSV}, current=${CURRENT_CSV}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCVH_BUILD_LEGACY_CORE=ON \
  -DCVH_BUILD_BACKEND_KERNEL_SOURCES=ON \
  -DCVH_BUILD_TESTS=OFF \
  -DCVH_BUILD_BENCHMARKS=ON \
  -DCVH_USE_OPENMP=ON

cmake --build "${BUILD_DIR}" -j

if [[ -n "${TASKSET_CPU_LIST}" ]]; then
  taskset -c "${TASKSET_CPU_LIST}" env \
    OMP_NUM_THREADS="${THREADS}" \
    OMP_DYNAMIC="${OMP_DYNAMIC_MODE}" \
    OMP_PROC_BIND="${OMP_PROC_BIND_MODE}" \
    "${BUILD_DIR}/cvh_benchmark_imgproc_ops" \
      --profile "${PROFILE}" \
      --warmup "${WARMUP}" \
      --iters "${ITERS}" \
      --repeats "${REPEATS}" \
      --output "${CURRENT_CSV}" \
      >/dev/null
else
  OMP_NUM_THREADS="${THREADS}" \
  OMP_DYNAMIC="${OMP_DYNAMIC_MODE}" \
  OMP_PROC_BIND="${OMP_PROC_BIND_MODE}" \
  "${BUILD_DIR}/cvh_benchmark_imgproc_ops" \
    --profile "${PROFILE}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --repeats "${REPEATS}" \
    --output "${CURRENT_CSV}" \
    >/dev/null
fi

python3 "${ROOT_DIR}/scripts/check_imgproc_benchmark_regression.py" \
  --baseline "${BASELINE_CSV}" \
  --current "${CURRENT_CSV}" \
  --max-slowdown "${MAX_SLOWDOWN}"
