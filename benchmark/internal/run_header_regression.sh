#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPORT_SCRIPT="${ROOT_DIR}/benchmark/common/benchmark_report.py"

BASELINE_REF=""
SUITE="all"
PROFILE="quick"
BUILD_TYPE="Release"
OUTPUT_DIR="${ROOT_DIR}/benchmark/results/internal"
WARMUP=""
ITERS=""
REPEATS=""
MAX_SLOWDOWN="0.08"
KEEP_WORKTREE=0

usage() {
  cat <<USAGE
Usage: $(basename "$0") --baseline-ref <git-ref> [options]

Options:
  --suite core_mat|imgproc|all
  --profile quick|stable|full|micro
  --build-type Release|RelWithDebInfo|Debug
  --output-dir <path>
  --warmup N
  --iters N
  --repeats N
  --max-slowdown FLOAT
  --keep-worktree
  --help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline-ref)
      BASELINE_REF="$2"
      shift 2
      ;;
    --suite)
      SUITE="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --build-type)
      BUILD_TYPE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
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
    --max-slowdown)
      MAX_SLOWDOWN="$2"
      shift 2
      ;;
    --keep-worktree)
      KEEP_WORKTREE=1
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

if [[ -z "${BASELINE_REF}" ]]; then
  echo "--baseline-ref is required" >&2
  usage >&2
  exit 2
fi

if [[ "${SUITE}" != "core_mat" && "${SUITE}" != "imgproc" && "${SUITE}" != "all" ]]; then
  echo "Unsupported --suite=${SUITE}" >&2
  exit 2
fi

if [[ "${PROFILE}" == "micro" ]]; then
  BENCH_PROFILE="quick"
else
  BENCH_PROFILE="${PROFILE}"
fi

CURRENT_COMMIT="$(git -C "${ROOT_DIR}" rev-parse --short HEAD)"
CURRENT_DIRTY_JSON="false"
CURRENT_LABEL="${CURRENT_COMMIT}"
if [[ -n "$(git -C "${ROOT_DIR}" status --porcelain)" ]]; then
  CURRENT_DIRTY_JSON="true"
  CURRENT_LABEL="${CURRENT_COMMIT}-dirty"
fi
BASELINE_COMMIT="$(git -C "${ROOT_DIR}" rev-parse --short "${BASELINE_REF}")"
RUN_ID="${BASELINE_COMMIT}_to_${CURRENT_LABEL}_${BENCH_PROFILE}_headers_fast"
RUN_DIR="${OUTPUT_DIR}/${RUN_ID}"
BASELINE_WORKTREE="${RUN_DIR}/baseline-worktree"
BASELINE_BUILD="${RUN_DIR}/build-baseline"
CURRENT_BUILD="${RUN_DIR}/build-current"

mkdir -p "${RUN_DIR}"

cleanup() {
  if [[ "${KEEP_WORKTREE}" != "1" && -e "${BASELINE_WORKTREE}/.git" ]]; then
    git -C "${ROOT_DIR}" worktree remove --force "${BASELINE_WORKTREE}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ ! -e "${BASELINE_WORKTREE}/.git" ]]; then
  git -C "${ROOT_DIR}" worktree add --detach "${BASELINE_WORKTREE}" "${BASELINE_REF}" >/dev/null
fi

configure_build() {
  local src_dir="$1"
  local build_dir="$2"
  cmake -S "${src_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCVH_BUILD_NATIVE_BACKEND=OFF \
    -DCVH_BUILD_TESTS=OFF \
    -DCVH_BUILD_BENCHMARKS=ON >/dev/null
}

target_for_suite() {
  local suite="$1"
  case "${suite}" in
    core_mat) printf '%s\n' "cvh_benchmark_core_mat_header" ;;
    imgproc) printf '%s\n' "cvh_benchmark_imgproc_header" ;;
    *) return 2 ;;
  esac
}

run_suite() {
  local suite="$1"
  local target
  target="$(target_for_suite "${suite}")"
  local baseline_csv="${RUN_DIR}/${suite}_baseline.csv"
  local current_csv="${RUN_DIR}/${suite}_current.csv"
  local report_md="${RUN_DIR}/${suite}_report.md"
  local summary_json="${RUN_DIR}/${suite}_summary.json"

  cmake --build "${BASELINE_BUILD}" --target "${target}" -j >/dev/null
  cmake --build "${CURRENT_BUILD}" --target "${target}" -j >/dev/null

  local -a args=(--profile "${BENCH_PROFILE}")
  if [[ -n "${WARMUP}" ]]; then args+=(--warmup "${WARMUP}"); fi
  if [[ -n "${ITERS}" ]]; then args+=(--iters "${ITERS}"); fi
  if [[ -n "${REPEATS}" ]]; then args+=(--repeats "${REPEATS}"); fi
  "${BASELINE_BUILD}/${target}" "${args[@]}" --output "${baseline_csv}" >/dev/null
  "${CURRENT_BUILD}/${target}" "${args[@]}" --output "${current_csv}" >/dev/null

  python3 "${REPORT_SCRIPT}" internal \
    --suite "${suite}" \
    --baseline "${baseline_csv}" \
    --current "${current_csv}" \
    --output-md "${report_md}" \
    --output-json "${summary_json}" \
    --max-slowdown "${MAX_SLOWDOWN}"
}

echo "header_regression: baseline=${BASELINE_REF} (${BASELINE_COMMIT}) current=${CURRENT_LABEL} suite=${SUITE} profile=${BENCH_PROFILE}"
configure_build "${BASELINE_WORKTREE}" "${BASELINE_BUILD}"
configure_build "${ROOT_DIR}" "${CURRENT_BUILD}"

if [[ "${SUITE}" == "all" ]]; then
  run_suite core_mat
  run_suite imgproc
else
  run_suite "${SUITE}"
fi

cat > "${RUN_DIR}/meta.json" <<META
{
  "mode": "internal_regression",
  "baseline_ref": "${BASELINE_REF}",
  "baseline_commit": "${BASELINE_COMMIT}",
  "current_commit": "${CURRENT_COMMIT}",
  "current_dirty": ${CURRENT_DIRTY_JSON},
  "suite": "${SUITE}",
  "profile": "${BENCH_PROFILE}",
  "target_profile": "headers_fast",
  "benchmark_schema_version": 2,
  "build_type": "${BUILD_TYPE}",
  "max_slowdown": ${MAX_SLOWDOWN}
}
META

echo "header_regression_done: output_dir=${RUN_DIR}"
