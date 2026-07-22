#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-ci-native-all"

print_env_fingerprint() {
  echo "ci_native_env_begin"
  echo "uname: $(uname -a)"
  echo "compiler: $(c++ --version | head -n 1)"
  echo "cmake: $(cmake --version | head -n 1)"
  echo "python: $(python3 --version)"
  echo "ci_native_env_end"
}

run_gtest_with_case_log() {
  local bin_path="$1"
  local tag="$2"

  if [[ ! -x "${bin_path}" ]]; then
    echo "Missing test binary: ${bin_path}" >&2
    exit 2
  fi

  echo "${tag}_all_cases_begin"
  "${bin_path}" --gtest_list_tests
  echo "${tag}_all_cases_end"

  "${bin_path}" --gtest_brief=1
}

print_env_fingerprint
"${ROOT_DIR}/scripts/check_public_headers.sh"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -U CVH_BUILD_FULL_BACKEND \
  -DCVH_BUILD_NATIVE_BACKEND=ON \
  -DCVH_BUILD_BACKEND_KERNEL_SOURCES=ON \
  -DCVH_BUILD_TESTS=ON \
  -DCVH_BUILD_BENCHMARKS=OFF

echo "ci_native_cmake_cache_begin"
grep -E '^(CVH_BUILD_NATIVE_BACKEND|CVH_BUILD_FULL_BACKEND|CVH_BUILD_BACKEND_KERNEL_SOURCES|CVH_BUILD_TESTS|CVH_BUILD_BENCHMARKS|CMAKE_BUILD_TYPE):' "${BUILD_DIR}/CMakeCache.txt" || true
echo "ci_native_cmake_cache_end"

cmake --build "${BUILD_DIR}" --target cvh_test_core_native cvh_test_imgproc -j

run_gtest_with_case_log "${BUILD_DIR}/cvh_test_core_native" "cvh_test_core_native"
run_gtest_with_case_log "${BUILD_DIR}/cvh_test_imgproc" "cvh_test_imgproc_lite_under_native_config"
