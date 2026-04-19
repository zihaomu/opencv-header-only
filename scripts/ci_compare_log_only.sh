#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPARE_RUN_SCRIPT="${ROOT_DIR}/benchmark/opencv_compare/run_compare.sh"

PROFILE="${CVH_COMPARE_PROFILE:-quick}"
IMPLS="${CVH_COMPARE_IMPLS:-full,lite}"

TMP_BASE="${RUNNER_TEMP:-$(mktemp -d)}"
WORK_DIR="$(mktemp -d "${TMP_BASE%/}/cvh-compare.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT

export CVH_OPENCV_DIR="${WORK_DIR}/opencv-bench-slim"
export CVH_OPENCV_BUILD_DIR="${CVH_OPENCV_DIR}/build-slim"
export CVH_COMPARE_BUILD_DIR="${WORK_DIR}/build-opencv-compare"
export CVH_COMPARE_OUTPUT="${WORK_DIR}/current_compare_${PROFILE}.csv"
export CVH_COMPARE_OUTPUT_MD="${WORK_DIR}/opencv_compare_${PROFILE}.md"
export CVH_COMPARE_OUTPUT_META="${WORK_DIR}/opencv_compare_${PROFILE}.meta.json"
export CVH_COMPARE_RENDER_MD=1

"${COMPARE_RUN_SCRIPT}" --profile "${PROFILE}" --impls "${IMPLS}" "$@"

echo "opencv_compare_ci_report_begin"
if [[ -f "${CVH_COMPARE_OUTPUT_MD}" ]]; then
  cat "${CVH_COMPARE_OUTPUT_MD}"
elif [[ -f "${CVH_COMPARE_OUTPUT}" ]]; then
  cat "${CVH_COMPARE_OUTPUT}"
else
  echo "opencv compare output not found" >&2
  exit 2
fi
echo "opencv_compare_ci_report_end"
