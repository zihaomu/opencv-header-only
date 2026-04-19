#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPARE_DIR="${ROOT_DIR}/opencv_compare"
SETUP_SCRIPT="${COMPARE_DIR}/setup_opencv_bench_slim.sh"

PROFILE="${CVH_COMPARE_PROFILE:-quick}"
IMPLS="${CVH_COMPARE_IMPLS:-full,lite}"
WARMUP=""
ITERS=""
REPEATS=""
THREADS="${CVH_COMPARE_THREADS:-1}"
BUILD_OPENCV="${CVH_COMPARE_BUILD_OPENCV:-auto}"
RENDER_MD="${CVH_COMPARE_RENDER_MD:-1}"
BUILD_TYPE="${CVH_COMPARE_BUILD_TYPE:-Release}"
OMP_DYNAMIC_MODE="${CVH_COMPARE_OMP_DYNAMIC:-false}"
OMP_PROC_BIND_MODE="${CVH_COMPARE_OMP_PROC_BIND:-close}"
MARK_AS_BASELINE=0

OPENCV_DIR="${CVH_OPENCV_DIR:-${COMPARE_DIR}/opencv-bench-slim}"
BUILD_DIR="${CVH_COMPARE_BUILD_DIR:-${ROOT_DIR}/build-opencv-compare}"
OUTPUT_CSV=""
OUTPUT_MD=""
OUTPUT_META=""
REPORT_SCRIPT="${COMPARE_DIR}/csv_to_markdown.py"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--profile quick|stable|full] [--impls full|lite|full,lite] [--warmup N] [--iters N] [--repeats N] [--output path] [--baseline]

Environment:
  CVH_COMPARE_PROFILE   (default: ${PROFILE})
  CVH_COMPARE_IMPLS     (default: ${IMPLS}, values: full|lite|full,lite)
  CVH_COMPARE_WARMUP    (profile default, quick=1 stable=2 full=1)
  CVH_COMPARE_ITERS     (profile default, quick=5 stable=20 full=10)
  CVH_COMPARE_REPEATS   (profile default, quick=1 stable=5 full=3)
  CVH_COMPARE_THREADS   (default: ${THREADS}, exports OMP_NUM_THREADS)
  CVH_COMPARE_OMP_DYNAMIC (default: ${OMP_DYNAMIC_MODE})
  CVH_COMPARE_OMP_PROC_BIND (default: ${OMP_PROC_BIND_MODE})
  CVH_COMPARE_BUILD_OPENCV (default: ${BUILD_OPENCV}, values: auto|0|1)
  CVH_COMPARE_RENDER_MD (default: ${RENDER_MD}, set 0 to skip Markdown generation)
  CVH_COMPARE_BUILD_TYPE (default: ${BUILD_TYPE}, e.g. Release|RelWithDebInfo|Debug)
  CVH_COMPARE_BUILD_DIR (default: ${BUILD_DIR})
  CVH_COMPARE_OUTPUT    (default: opencv_compare/results/current_compare_<profile>.csv, or baseline_* with --baseline)
  CVH_COMPARE_OUTPUT_MD (default: opencv_compare/opencv_compare_<profile>.md, or baseline_* with --baseline)
  CVH_COMPARE_OUTPUT_META (default: <output_csv>.meta.json)
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
    --impls)
      IMPLS="$2"
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
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --output)
      OUTPUT_CSV="$2"
      shift 2
      ;;
    --output-md)
      OUTPUT_MD="$2"
      shift 2
      ;;
    --output-meta)
      OUTPUT_META="$2"
      shift 2
      ;;
    --baseline)
      MARK_AS_BASELINE=1
      shift
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

if [[ "${PROFILE}" != "quick" && "${PROFILE}" != "stable" && "${PROFILE}" != "full" ]]; then
  echo "Unsupported profile: ${PROFILE} (expected quick|stable|full)" >&2
  exit 2
fi

IFS=',' read -r -a RAW_IMPLS <<< "${IMPLS}"
REQUESTED_IMPLS=()
for raw_impl in "${RAW_IMPLS[@]}"; do
  impl="${raw_impl//[[:space:]]/}"
  if [[ -z "${impl}" ]]; then
    continue
  fi

  if [[ "${impl}" != "full" && "${impl}" != "lite" ]]; then
    echo "Unsupported impl: ${impl} (expected full|lite)" >&2
    exit 2
  fi

  impl_exists=0
  for existing_impl in "${REQUESTED_IMPLS[@]-}"; do
    if [[ "${existing_impl}" == "${impl}" ]]; then
      impl_exists=1
      break
    fi
  done
  if [[ "${impl_exists}" == "0" ]]; then
    REQUESTED_IMPLS+=("${impl}")
  fi
done

if [[ "${#REQUESTED_IMPLS[@]}" -eq 0 ]]; then
  echo "No valid impl selected. Use --impls full|lite|full,lite" >&2
  exit 2
fi

IMPLS_NORMALIZED="$(IFS=,; echo "${REQUESTED_IMPLS[*]-}")"

case "${PROFILE}" in
  quick)
    DEFAULT_WARMUP=1
    DEFAULT_ITERS=5
    DEFAULT_REPEATS=1
    ;;
  stable)
    DEFAULT_WARMUP=2
    DEFAULT_ITERS=20
    DEFAULT_REPEATS=5
    ;;
  full)
    DEFAULT_WARMUP=1
    DEFAULT_ITERS=10
    DEFAULT_REPEATS=3
    ;;
esac

WARMUP="${WARMUP:-${CVH_COMPARE_WARMUP:-${DEFAULT_WARMUP}}}"
ITERS="${ITERS:-${CVH_COMPARE_ITERS:-${DEFAULT_ITERS}}}"
REPEATS="${REPEATS:-${CVH_COMPARE_REPEATS:-${DEFAULT_REPEATS}}}"

if [[ "${MARK_AS_BASELINE}" == "1" ]]; then
  DEFAULT_OUTPUT_CSV="${COMPARE_DIR}/results/baseline_compare_${PROFILE}.csv"
  DEFAULT_OUTPUT_MD="${COMPARE_DIR}/opencv_compare_baseline_${PROFILE}.md"
else
  DEFAULT_OUTPUT_CSV="${COMPARE_DIR}/results/current_compare_${PROFILE}.csv"
  DEFAULT_OUTPUT_MD="${COMPARE_DIR}/opencv_compare_${PROFILE}.md"
fi

OUTPUT_CSV="${OUTPUT_CSV:-${CVH_COMPARE_OUTPUT:-${DEFAULT_OUTPUT_CSV}}}"
OUTPUT_MD="${OUTPUT_MD:-${CVH_COMPARE_OUTPUT_MD:-${DEFAULT_OUTPUT_MD}}}"
OUTPUT_META="${OUTPUT_META:-${CVH_COMPARE_OUTPUT_META:-${OUTPUT_CSV}.meta.json}}"

if ! [[ "${WARMUP}" =~ ^[0-9]+$ && "${ITERS}" =~ ^[1-9][0-9]*$ && "${REPEATS}" =~ ^[1-9][0-9]*$ && "${THREADS}" =~ ^[1-9][0-9]*$ ]]; then
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

detect_cpu_model() {
  if [[ "$(uname -s)" == "Darwin" ]]; then
    sysctl -n machdep.cpu.brand_string 2>/dev/null || printf '%s' "unknown"
    return
  fi

  if [[ -f /proc/cpuinfo ]]; then
    awk -F ':' '/model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo
    return
  fi

  printf '%s' "unknown"
}

write_compare_metadata() {
  local output_meta="$1"
  local output_csv="$2"
  local output_md="$3"
  local opencv_config_dir="$4"
  local generated_at
  generated_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  local repo_git_commit
  repo_git_commit="$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || printf '%s' 'unknown')"
  local opencv_git_commit
  opencv_git_commit="$(git -C "${OPENCV_DIR}" rev-parse HEAD 2>/dev/null || printf '%s' 'unknown')"
  local cpu_model
  cpu_model="$(detect_cpu_model)"

  mkdir -p "$(dirname "${output_meta}")"
  env \
    META_OUTPUT="${output_meta}" \
    META_PROFILE="${PROFILE}" \
    META_IMPLS="${IMPLS_NORMALIZED}" \
    META_WARMUP="${WARMUP}" \
    META_ITERS="${ITERS}" \
    META_REPEATS="${REPEATS}" \
    META_SAMPLE_COUNT="$((ITERS * REPEATS))" \
    META_THREADS="${THREADS}" \
    META_OMP_DYNAMIC="${OMP_DYNAMIC_MODE}" \
    META_OMP_PROC_BIND="${OMP_PROC_BIND_MODE}" \
    META_BUILD_TYPE="${BUILD_TYPE}" \
    META_COMPARE_MODE="$([[ "${MARK_AS_BASELINE}" == "1" ]] && printf '%s' baseline || printf '%s' current)" \
    META_OUTPUT_CSV="${output_csv}" \
    META_OUTPUT_MD="${output_md}" \
    META_OPENCV_DIR="${OPENCV_DIR}" \
    META_OPENCV_CONFIG_DIR="${opencv_config_dir}" \
    META_OPENCV_GIT_COMMIT="${opencv_git_commit}" \
    META_REPO_GIT_COMMIT="${repo_git_commit}" \
    META_SYSTEM="$(uname -s)" \
    META_KERNEL="$(uname -r)" \
    META_ARCH="$(uname -m)" \
    META_HOSTNAME="$(hostname)" \
    META_CPU_MODEL="${cpu_model}" \
    META_GENERATED_AT="${generated_at}" \
    python3 - <<'PY'
import json
import os
import pathlib

data = {
    "profile": os.environ["META_PROFILE"],
    "impls": [s for s in os.environ["META_IMPLS"].split(",") if s],
    "warmup": int(os.environ["META_WARMUP"]),
    "iters": int(os.environ["META_ITERS"]),
    "repeats": int(os.environ["META_REPEATS"]),
    "sample_count": int(os.environ["META_SAMPLE_COUNT"]),
    "threads": int(os.environ["META_THREADS"]),
    "omp_dynamic": os.environ["META_OMP_DYNAMIC"],
    "omp_proc_bind": os.environ["META_OMP_PROC_BIND"],
    "build_type": os.environ["META_BUILD_TYPE"],
    "compare_mode": os.environ["META_COMPARE_MODE"],
    "output_csv": os.environ["META_OUTPUT_CSV"],
    "output_md": os.environ["META_OUTPUT_MD"],
    "opencv_dir": os.environ["META_OPENCV_DIR"],
    "opencv_config_dir": os.environ["META_OPENCV_CONFIG_DIR"],
    "opencv_git_commit": os.environ["META_OPENCV_GIT_COMMIT"],
    "repo_git_commit": os.environ["META_REPO_GIT_COMMIT"],
    "system": os.environ["META_SYSTEM"],
    "kernel": os.environ["META_KERNEL"],
    "arch": os.environ["META_ARCH"],
    "hostname": os.environ["META_HOSTNAME"],
    "cpu_model": os.environ["META_CPU_MODEL"],
    "generated_at_utc": os.environ["META_GENERATED_AT"],
}
data["fingerprint"] = {
    "profile": data["profile"],
    "impls": data["impls"],
    "warmup": data["warmup"],
    "iters": data["iters"],
    "repeats": data["repeats"],
    "sample_count": data["sample_count"],
    "threads": data["threads"],
    "omp_dynamic": data["omp_dynamic"],
    "omp_proc_bind": data["omp_proc_bind"],
    "build_type": data["build_type"],
    "opencv_config_dir": data["opencv_config_dir"],
    "system": data["system"],
    "arch": data["arch"],
    "cpu_model": data["cpu_model"],
}

meta_path = pathlib.Path(os.environ["META_OUTPUT"])
meta_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
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

mkdir -p "${BUILD_DIR}" "$(dirname "${OUTPUT_CSV}")" "$(dirname "${OUTPUT_META}")" "$(dirname "${OUTPUT_MD}")"

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

BENCH_TARGETS=()
for impl in "${REQUESTED_IMPLS[@]-}"; do
  if [[ "${impl}" == "full" ]]; then
    BENCH_TARGETS+=("cvh_benchmark_compare")
  else
    BENCH_TARGETS+=("cvh_benchmark_compare_lite")
  fi
done

cmake --build "${BUILD_DIR}" --target "${BENCH_TARGETS[@]}" -j

TMP_OUTPUT_CSVS=()
for impl in "${REQUESTED_IMPLS[@]-}"; do
  if [[ "${impl}" == "full" ]]; then
    BENCH_BIN="${BUILD_DIR}/cvh_benchmark_compare"
  else
    BENCH_BIN="${BUILD_DIR}/cvh_benchmark_compare_lite"
  fi

  if [[ ! -x "${BENCH_BIN}" ]]; then
    echo "Missing benchmark binary for impl=${impl}: ${BENCH_BIN}" >&2
    exit 2
  fi

  IMPL_OUTPUT_CSV="${OUTPUT_CSV}.${impl}.tmp.csv"
  TMP_OUTPUT_CSVS+=("${IMPL_OUTPUT_CSV}")

  echo "opencv_compare: running benchmark (impl=${impl}, profile=${PROFILE}, warmup=${WARMUP}, iters=${ITERS}, repeats=${REPEATS}, threads=${THREADS}, omp_dynamic=${OMP_DYNAMIC_MODE}, omp_proc_bind=${OMP_PROC_BIND_MODE})"
  echo "opencv_compare: benchmark stage can take several minutes for quick profile with large kernels."
  OMP_NUM_THREADS="${THREADS}" \
  OMP_DYNAMIC="${OMP_DYNAMIC_MODE}" \
  OMP_PROC_BIND="${OMP_PROC_BIND_MODE}" \
  "${BENCH_BIN}" \
    --profile "${PROFILE}" \
    --impl "${impl}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --repeats "${REPEATS}" \
    --output "${IMPL_OUTPUT_CSV}"
done

python3 - "${OUTPUT_CSV}" "${TMP_OUTPUT_CSVS[@]}" <<'PY'
import csv
import pathlib
import sys

if len(sys.argv) < 3:
    raise SystemExit("merge requires output path + at least one input csv")

out_path = pathlib.Path(sys.argv[1])
in_paths = [pathlib.Path(p) for p in sys.argv[2:]]

header = None
rows = []
for in_path in in_paths:
    with in_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            local_header = next(reader)
        except StopIteration:
            continue
        if header is None:
            header = local_header
        elif local_header != header:
            raise SystemExit(f"CSV header mismatch in {in_path}")
        rows.extend(reader)

if header is None:
    raise SystemExit("no rows to merge")

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)
PY

for impl_csv in "${TMP_OUTPUT_CSVS[@]}"; do
  rm -f "${impl_csv}"
done

write_compare_metadata "${OUTPUT_META}" "${OUTPUT_CSV}" "${OUTPUT_MD}" "${OPENCV_CONFIG_DIR}"

if [[ "${RENDER_MD}" != "0" ]]; then
  if [[ ! -f "${REPORT_SCRIPT}" ]]; then
    echo "Missing report script: ${REPORT_SCRIPT}" >&2
    exit 2
  fi

  python3 "${REPORT_SCRIPT}" \
    --input "${OUTPUT_CSV}" \
    --output "${OUTPUT_MD}" \
    --meta "${OUTPUT_META}" \
    --title "cvh vs OpenCV Benchmark Report (${PROFILE})"
fi

echo "opencv_compare_done: output_csv=${OUTPUT_CSV}, output_md=${OUTPUT_MD}, output_meta=${OUTPUT_META}, mode=$([[ "${MARK_AS_BASELINE}" == "1" ]] && printf '%s' baseline || printf '%s' current), profile=${PROFILE}, impls=${IMPLS_NORMALIZED}, warmup=${WARMUP}, iters=${ITERS}, repeats=${REPEATS}, threads=${THREADS}, build_type=${BUILD_TYPE}, opencv_dir=${OPENCV_DIR}"
