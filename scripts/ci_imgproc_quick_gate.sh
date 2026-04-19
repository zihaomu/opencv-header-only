#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-imgproc-benchmark-gate"

PROFILE="${CVH_IMGPROC_BENCH_PROFILE:-quick}"

if [[ "${PROFILE}" != "quick" && "${PROFILE}" != "full" ]]; then
  echo "Unsupported CVH_IMGPROC_BENCH_PROFILE=${PROFILE}, expected quick|full" >&2
  exit 2
fi

POLICY_EXPORTS="$(python3 "${ROOT_DIR}/scripts/read_gate_policy.py" --repo-root "${ROOT_DIR}" --profile "${PROFILE}" --emit shell)"
eval "${POLICY_EXPORTS}"

WARMUP="${CVH_IMGPROC_BENCH_WARMUP:-${CVH_POLICY_IMGPROC_WARMUP}}"
ITERS="${CVH_IMGPROC_BENCH_ITERS:-${CVH_POLICY_IMGPROC_ITERS}}"
REPEATS="${CVH_IMGPROC_BENCH_REPEATS:-${CVH_POLICY_IMGPROC_REPEATS}}"
MAX_SLOWDOWN="${CVH_IMGPROC_BENCH_MAX_SLOWDOWN:-${CVH_POLICY_IMGPROC_MAX_SLOWDOWN}}"
MAX_SLOWDOWN_BY_OP_DEPTH="${CVH_IMGPROC_BENCH_MAX_SLOWDOWN_BY_OP_DEPTH:-${CVH_POLICY_IMGPROC_MAX_SLOWDOWN_BY_OP_DEPTH}}"
THREADS="${CVH_IMGPROC_BENCH_THREADS:-${CVH_POLICY_IMGPROC_THREADS}}"
MINIMUM_SAMPLES="${CVH_IMGPROC_BENCH_MINIMUM_SAMPLES:-${CVH_POLICY_IMGPROC_MINIMUM_SAMPLES}}"
ENFORCE_FINGERPRINT="${CVH_IMGPROC_BENCH_ENFORCE_FINGERPRINT:-${CVH_POLICY_IMGPROC_ENFORCE_FINGERPRINT}}"
BASE_REF="${CVH_IMGPROC_BENCH_BASE_REF:-}"

OMP_DYNAMIC_MODE="${CVH_IMGPROC_BENCH_OMP_DYNAMIC:-false}"
OMP_PROC_BIND_MODE="${CVH_IMGPROC_BENCH_OMP_PROC_BIND:-close}"
CPU_LIST="${CVH_IMGPROC_BENCH_CPU_LIST:-auto}"

if ! [[ "${WARMUP}" =~ ^[0-9]+$ && "${ITERS}" =~ ^[0-9]+$ && "${REPEATS}" =~ ^[0-9]+$ && "${THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid warmup/iters/repeats/threads config" >&2
  exit 2
fi
if ! [[ "${MINIMUM_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid CVH_IMGPROC_BENCH_MINIMUM_SAMPLES=${MINIMUM_SAMPLES}" >&2
  exit 2
fi

DEFAULT_BASELINE="${ROOT_DIR}/benchmark/baseline_imgproc_${PROFILE}.csv"
DEFAULT_CURRENT="${BUILD_DIR}/current_imgproc_${PROFILE}.csv"
BASELINE_CSV="${CVH_IMGPROC_BENCH_BASELINE:-${DEFAULT_BASELINE}}"
CURRENT_CSV="${CVH_IMGPROC_BENCH_CURRENT:-${DEFAULT_CURRENT}}"
BASELINE_META="${CVH_IMGPROC_BENCH_BASELINE_META:-${BASELINE_CSV}.meta.json}"
CURRENT_META="${CVH_IMGPROC_BENCH_CURRENT_META:-${CURRENT_CSV}.meta.json}"

if [[ "${BASELINE_CSV}" != /* ]]; then
  BASELINE_CSV="${ROOT_DIR}/${BASELINE_CSV}"
fi
if [[ "${CURRENT_CSV}" != /* ]]; then
  CURRENT_CSV="${ROOT_DIR}/${CURRENT_CSV}"
fi
if [[ "${BASELINE_META}" != /* ]]; then
  BASELINE_META="${ROOT_DIR}/${BASELINE_META}"
fi
if [[ "${CURRENT_META}" != /* ]]; then
  CURRENT_META="${ROOT_DIR}/${CURRENT_META}"
fi

mkdir -p "${BUILD_DIR}" "$(dirname "${CURRENT_CSV}")" "$(dirname "${CURRENT_META}")"

to_lower() {
  printf '%s' "${1}" | tr '[:upper:]' '[:lower:]'
}

is_true() {
  local value
  value="$(to_lower "${1:-}")"
  case "${value}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

REDUCED_CONFIDENCE_REASONS=()
mark_reduced_confidence() {
  REDUCED_CONFIDENCE_REASONS+=("${1}")
}

if (( ITERS * REPEATS < MINIMUM_SAMPLES )); then
  mark_reduced_confidence "sample_budget_low:iters*repeats=$((ITERS * REPEATS))<${MINIMUM_SAMPLES}"
fi

TASKSET_CPU_LIST=""
if command -v taskset >/dev/null 2>&1; then
  if [[ "${CPU_LIST}" == "auto" ]]; then
    ALLOWED_CPU_LIST="$(awk '/Cpus_allowed_list:/ {print $2}' /proc/self/status 2>/dev/null || true)"
    FIRST_SEGMENT="${ALLOWED_CPU_LIST%%,*}"
    if [[ "${FIRST_SEGMENT}" == *-* ]]; then
      RANGE_START="${FIRST_SEGMENT%%-*}"
      RANGE_END="${FIRST_SEGMENT##*-}"
      if [[ "${RANGE_START}" =~ ^[0-9]+$ && "${RANGE_END}" =~ ^[0-9]+$ && "${RANGE_END}" -ge "${RANGE_START}" ]]; then
        RANGE_LEN=$((RANGE_END - RANGE_START + 1))
        RANGE_OFFSET=$((RANGE_LEN / 4))
        TASKSET_CPU_LIST="$((RANGE_START + RANGE_OFFSET))"
      fi
    elif [[ "${FIRST_SEGMENT}" =~ ^[0-9]+$ ]]; then
      TASKSET_CPU_LIST="${FIRST_SEGMENT}"
    fi
  elif [[ "${CPU_LIST}" != "off" ]]; then
    TASKSET_CPU_LIST="${CPU_LIST}"
  fi
fi

if [[ -z "${TASKSET_CPU_LIST}" ]]; then
  mark_reduced_confidence "cpu_affinity_not_pinned"
fi

echo "imgproc_bench_gate_config: profile=${PROFILE}, warmup=${WARMUP}, iters=${ITERS}, repeats=${REPEATS}, max_slowdown=${MAX_SLOWDOWN}, minimum_samples=${MINIMUM_SAMPLES}"
echo "imgproc_bench_runtime: threads=${THREADS}, omp_dynamic=${OMP_DYNAMIC_MODE}, omp_proc_bind=${OMP_PROC_BIND_MODE}, cpu_list=${TASKSET_CPU_LIST:-none}"
echo "imgproc_bench_threshold_overrides: ${MAX_SLOWDOWN_BY_OP_DEPTH:-none}"
echo "imgproc_bench_gate_files: baseline=${BASELINE_CSV}, current=${CURRENT_CSV}, baseline_meta=${BASELINE_META}, current_meta=${CURRENT_META}"
echo "imgproc_bench_gate_mode: base_ref=${BASE_REF:-none}, enforce_fingerprint=${ENFORCE_FINGERPRINT}"

BASELINE_WORKTREE=""
cleanup() {
  if [[ -n "${BASELINE_WORKTREE}" ]]; then
    git -C "${ROOT_DIR}" worktree remove --force "${BASELINE_WORKTREE}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

write_bench_metadata() {
  local output_meta="$1"
  local output_csv="$2"
  local git_ref_label="$3"
  local git_commit="$4"
  local build_path="$5"
  local generated_at
  generated_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  mkdir -p "$(dirname "${output_meta}")"
  env \
    META_OUTPUT="${output_meta}" \
    META_PROFILE="${PROFILE}" \
    META_WARMUP="${WARMUP}" \
    META_ITERS="${ITERS}" \
    META_REPEATS="${REPEATS}" \
    META_THREADS="${THREADS}" \
    META_OMP_DYNAMIC="${OMP_DYNAMIC_MODE}" \
    META_OMP_PROC_BIND="${OMP_PROC_BIND_MODE}" \
    META_CPU_LIST_MODE="${CPU_LIST}" \
    META_CPU_LIST_SELECTED="${TASKSET_CPU_LIST}" \
    META_MAX_SLOWDOWN="${MAX_SLOWDOWN}" \
    META_MAX_SLOWDOWN_BY_OP_DEPTH="${MAX_SLOWDOWN_BY_OP_DEPTH}" \
    META_SAMPLE_COUNT="$((ITERS * REPEATS))" \
    META_GIT_REF="${git_ref_label}" \
    META_GIT_COMMIT="${git_commit}" \
    META_BUILD_DIR="${build_path}" \
    META_OUTPUT_CSV="${output_csv}" \
    META_GENERATED_AT="${generated_at}" \
    python3 - <<'PY'
import json
import os
import pathlib

meta_path = pathlib.Path(os.environ["META_OUTPUT"])
data = {
    "profile": os.environ["META_PROFILE"],
    "warmup": int(os.environ["META_WARMUP"]),
    "iters": int(os.environ["META_ITERS"]),
    "repeats": int(os.environ["META_REPEATS"]),
    "sample_count": int(os.environ["META_SAMPLE_COUNT"]),
    "threads": int(os.environ["META_THREADS"]),
    "omp_dynamic": os.environ["META_OMP_DYNAMIC"],
    "omp_proc_bind": os.environ["META_OMP_PROC_BIND"],
    "cpu_list_mode": os.environ["META_CPU_LIST_MODE"],
    "cpu_list_selected": os.environ["META_CPU_LIST_SELECTED"],
    "max_slowdown": float(os.environ["META_MAX_SLOWDOWN"]),
    "max_slowdown_by_op_depth": os.environ["META_MAX_SLOWDOWN_BY_OP_DEPTH"],
    "git_ref": os.environ["META_GIT_REF"],
    "git_commit": os.environ["META_GIT_COMMIT"],
    "build_dir": os.environ["META_BUILD_DIR"],
    "output_csv": os.environ["META_OUTPUT_CSV"],
    "generated_at_utc": os.environ["META_GENERATED_AT"],
}
meta_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_imgproc_benchmark() {
  local repo_dir="$1"
  local out_csv="$2"
  local out_meta="$3"
  local build_dir="$4"
  local git_ref_label="$5"

  mkdir -p "${build_dir}" "$(dirname "${out_csv}")" "$(dirname "${out_meta}")"

  cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCVH_BUILD_FULL_BACKEND=ON \
    -DCVH_BUILD_BACKEND_KERNEL_SOURCES=ON \
    -DCVH_BUILD_TESTS=OFF \
    -DCVH_BUILD_BENCHMARKS=ON \
    -DCVH_USE_OPENMP=ON

  cmake --build "${build_dir}" -j

  if [[ -n "${TASKSET_CPU_LIST}" ]]; then
    taskset -c "${TASKSET_CPU_LIST}" env \
      OMP_NUM_THREADS="${THREADS}" \
      OMP_DYNAMIC="${OMP_DYNAMIC_MODE}" \
      OMP_PROC_BIND="${OMP_PROC_BIND_MODE}" \
      "${build_dir}/cvh_benchmark_imgproc_ops" \
        --profile "${PROFILE}" \
        --warmup "${WARMUP}" \
        --iters "${ITERS}" \
        --repeats "${REPEATS}" \
        --output "${out_csv}" \
        >/dev/null
  else
    OMP_NUM_THREADS="${THREADS}" \
    OMP_DYNAMIC="${OMP_DYNAMIC_MODE}" \
    OMP_PROC_BIND="${OMP_PROC_BIND_MODE}" \
    "${build_dir}/cvh_benchmark_imgproc_ops" \
      --profile "${PROFILE}" \
      --warmup "${WARMUP}" \
      --iters "${ITERS}" \
      --repeats "${REPEATS}" \
      --output "${out_csv}" \
      >/dev/null
  fi

  local git_commit
  git_commit="$(git -C "${repo_dir}" rev-parse HEAD 2>/dev/null || printf '%s' 'unknown')"
  write_bench_metadata "${out_meta}" "${out_csv}" "${git_ref_label}" "${git_commit}" "${build_dir}"
}

metadata_fingerprint() {
  local meta_path="$1"
  python3 - "${meta_path}" <<'PY'
import json
import pathlib
import sys

meta = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
keys = [
    "profile",
    "warmup",
    "iters",
    "repeats",
    "sample_count",
    "threads",
    "omp_dynamic",
    "omp_proc_bind",
    "cpu_list_mode",
    "cpu_list_selected",
    "max_slowdown",
    "max_slowdown_by_op_depth",
]
fingerprint = {key: meta.get(key) for key in keys}
print(json.dumps(fingerprint, sort_keys=True, separators=(",", ":")))
PY
}

if [[ -n "${BASE_REF}" ]]; then
  if ! git -C "${ROOT_DIR}" rev-parse --verify "${BASE_REF}^{commit}" >/dev/null 2>&1; then
    echo "base-ref commit not found: ${BASE_REF}" >&2
    exit 2
  fi
  BASELINE_COMMIT="$(git -C "${ROOT_DIR}" rev-parse --verify "${BASE_REF}^{commit}")"
  BASELINE_WORKTREE="${BUILD_DIR}/baseline-worktree"
  rm -rf "${BASELINE_WORKTREE}"
  git -C "${ROOT_DIR}" worktree add --detach "${BASELINE_WORKTREE}" "${BASELINE_COMMIT}" >/dev/null
  BASELINE_CSV="${BUILD_DIR}/baseline_imgproc_${PROFILE}_from_base_ref.csv"
  BASELINE_META="${BASELINE_CSV}.meta.json"
  run_imgproc_benchmark "${BASELINE_WORKTREE}" "${BASELINE_CSV}" "${BASELINE_META}" "${BASELINE_WORKTREE}/build-imgproc-benchmark-gate" "${BASE_REF}"
else
  if [[ ! -f "${BASELINE_CSV}" ]]; then
    echo "imgproc benchmark baseline not found: ${BASELINE_CSV}" >&2
    echo "Create baseline first, for example:" >&2
    echo "  ./build-full-test/cvh_benchmark_imgproc_ops --profile ${PROFILE} --warmup ${WARMUP} --iters ${ITERS} --repeats ${REPEATS} --output benchmark/baseline_imgproc_${PROFILE}.csv" >&2
    exit 2
  fi
  if [[ ! -f "${BASELINE_META}" ]]; then
    mark_reduced_confidence "baseline_metadata_missing:${BASELINE_META}"
  fi
fi

run_imgproc_benchmark "${ROOT_DIR}" "${CURRENT_CSV}" "${CURRENT_META}" "${BUILD_DIR}" "HEAD"

if [[ -f "${BASELINE_META}" ]]; then
  BASELINE_FP="$(metadata_fingerprint "${BASELINE_META}")"
  CURRENT_FP="$(metadata_fingerprint "${CURRENT_META}")"
  if [[ "${BASELINE_FP}" != "${CURRENT_FP}" ]]; then
    if is_true "${ENFORCE_FINGERPRINT}"; then
      echo "benchmark runtime fingerprint mismatch between baseline and current" >&2
      echo "baseline_meta=${BASELINE_META}" >&2
      echo "current_meta=${CURRENT_META}" >&2
      exit 2
    fi
    mark_reduced_confidence "fingerprint_mismatch"
  fi
fi

REGRESSION_ARGS=(
  --baseline "${BASELINE_CSV}"
  --current "${CURRENT_CSV}"
  --max-slowdown "${MAX_SLOWDOWN}"
)

if [[ -n "${MAX_SLOWDOWN_BY_OP_DEPTH}" ]]; then
  IFS=',' read -r -a OVERRIDE_RULES <<< "${MAX_SLOWDOWN_BY_OP_DEPTH}"
  for rule in "${OVERRIDE_RULES[@]}"; do
    trimmed_rule="$(echo "${rule}" | xargs)"
    if [[ -n "${trimmed_rule}" ]]; then
      REGRESSION_ARGS+=(--max-slowdown-by-op-depth "${trimmed_rule}")
    fi
  done
fi

python3 "${ROOT_DIR}/scripts/check_imgproc_benchmark_regression.py" "${REGRESSION_ARGS[@]}"

if ((${#REDUCED_CONFIDENCE_REASONS[@]} > 0)); then
  CONFIDENCE_REPORT="${BUILD_DIR}/imgproc_gate_confidence.json"
  printf 'imgproc gate running with reduced confidence:\n'
  for reason in "${REDUCED_CONFIDENCE_REASONS[@]}"; do
    printf '  - %s\n' "${reason}"
  done
  python3 - "${CONFIDENCE_REPORT}" "${BASELINE_CSV}" "${CURRENT_CSV}" "${BASELINE_META}" "${CURRENT_META}" "${REDUCED_CONFIDENCE_REASONS[@]}" <<'PY'
import json
import pathlib
import sys

report = pathlib.Path(sys.argv[1])
data = {
    "confidence": "reduced",
    "baseline_csv": sys.argv[2],
    "current_csv": sys.argv[3],
    "baseline_meta": sys.argv[4],
    "current_meta": sys.argv[5],
    "reasons": sys.argv[6:],
}
report.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  echo "confidence_report=${CONFIDENCE_REPORT}"
  if is_true "${CVH_IMGPROC_BENCH_FAIL_ON_REDUCED_CONFIDENCE:-false}"; then
    echo "failing due to CVH_IMGPROC_BENCH_FAIL_ON_REDUCED_CONFIDENCE=true" >&2
    exit 3
  fi
else
  echo "imgproc gate confidence: high"
fi
