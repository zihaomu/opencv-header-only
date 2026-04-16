#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-core-basic"
BUILD_LOG="${BUILD_DIR}/build.log"

"${ROOT_DIR}/scripts/check_public_headers.sh"
python3 "${ROOT_DIR}/scripts/verify_opencv_core_channel_cases.py" --repo-root "${ROOT_DIR}"
POLICY_EXPORTS="$(python3 "${ROOT_DIR}/scripts/read_gate_policy.py" --repo-root "${ROOT_DIR}" --profile quick --emit shell)"
eval "${POLICY_EXPORTS}"

CORE_CONTRACT_PATH="${CVH_CORE_CONTRACT_PATH:-${CVH_POLICY_CORE_CONTRACT_PATH}}"
if [[ "${CORE_CONTRACT_PATH}" != /* ]]; then
  CORE_CONTRACT_PATH="${ROOT_DIR}/${CORE_CONTRACT_PATH}"
fi

CORE_CONTRACT_GTEST_FILTER="$(python3 "${ROOT_DIR}/scripts/verify_core_contract.py" \
  --repo-root "${ROOT_DIR}" \
  --manifest test/upstream/opencv/core/channel_manifest.json \
  --contract "${CORE_CONTRACT_PATH}" \
  --emit-gtest-filter)"

python3 -m unittest discover -s "${ROOT_DIR}/test/scripts" -p 'test_*.py'

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCVH_BUILD_FULL_BACKEND=ON \
  -DCVH_BUILD_LEGACY_CORE=ON \
  -DCVH_BUILD_BACKEND_KERNEL_SOURCES=ON \
  -DCVH_BUILD_TESTS=ON

cmake --build "${BUILD_DIR}" -j 2>&1 | tee "${BUILD_LOG}"

warning_count="$(grep -E -c 'warning:' "${BUILD_LOG}" || true)"
warning_budget="${CVH_WARNING_BUDGET:-0}"
echo "core_basic_warning_count=${warning_count} (budget=${warning_budget})"

if (( warning_count > warning_budget )); then
  echo "warning count exceeds budget" >&2
  exit 1
fi

if command -v ctest >/dev/null 2>&1; then
  ctest --test-dir "${BUILD_DIR}" --output-on-failure
else
  cmake --build "${BUILD_DIR}" --target test
fi

if [[ -x "${BUILD_DIR}/cvh_test_core" ]]; then
  "${BUILD_DIR}/cvh_test_core" "--gtest_filter=${CORE_CONTRACT_GTEST_FILTER}"
fi
