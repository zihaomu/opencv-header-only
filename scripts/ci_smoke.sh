#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-smoke"

"${ROOT_DIR}/scripts/check_public_headers.sh"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCVH_BUILD_LEGACY_CORE=OFF \
  -DCVH_BUILD_BACKEND_KERNEL_SOURCES=OFF \
  -DCVH_BUILD_LEGACY_TESTS=OFF \
  -DCVH_BUILD_SMOKE_TESTS=ON

cmake --build "${BUILD_DIR}" -j

if command -v ctest >/dev/null 2>&1; then
  ctest --test-dir "${BUILD_DIR}" --output-on-failure
else
  cmake --build "${BUILD_DIR}" --target test
fi
