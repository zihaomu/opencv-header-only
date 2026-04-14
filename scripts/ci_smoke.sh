#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-smoke"

"${ROOT_DIR}/scripts/check_public_headers.sh"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCVH_BUILD_FULL_BACKEND=OFF \
  -DCVH_BUILD_LEGACY_CORE=OFF \
  -DCVH_BUILD_BACKEND_KERNEL_SOURCES=OFF \
  -DCVH_BUILD_TESTS=ON

cmake --build "${BUILD_DIR}" -j

if command -v ctest >/dev/null 2>&1; then
  ctest --test-dir "${BUILD_DIR}" --output-on-failure -R 'smoke'
else
  "${BUILD_DIR}/cvh_header_compile_smoke"
  "${BUILD_DIR}/cvh_include_only_smoke"
  if [[ -x "${BUILD_DIR}/cvh_legacy_core_smoke" ]]; then
    "${BUILD_DIR}/cvh_legacy_core_smoke"
  fi
fi
