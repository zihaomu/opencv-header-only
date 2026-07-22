#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/cvh_header_only_contract.XXXXXX")"

cleanup() {
  if [[ "${CVH_KEEP_CONTRACT_TMP:-0}" != "1" ]]; then
    rm -rf "${TMP_ROOT}"
  else
    echo "Keeping contract check temp dir: ${TMP_ROOT}" >&2
  fi
}
trap cleanup EXIT

BUILD_DIR="${TMP_ROOT}/build"
INSTALL_DIR="${TMP_ROOT}/install"
HEADERS_CONSUMER_DIR="${TMP_ROOT}/consumer-headers"
FAST_CONSUMER_DIR="${TMP_ROOT}/consumer-headers-fast"
LEGACY_XSIMD_DENY_DIR="${TMP_ROOT}/consumer-legacy-xsimd-deny"
LEGACY_ON_BUILD_DIR="${TMP_ROOT}/build-legacy-on"
LEGACY_ON_INSTALL_DIR="${TMP_ROOT}/install-legacy-on"

require_no_legacy_export() {
  local cmake_dir="$1"

  if ! grep -R "cvh::headers" "${cmake_dir}" >/dev/null; then
    echo "Installed package does not export cvh::headers." >&2
    return 1
  fi

  if ! grep -R "cvh::headers_fast" "${cmake_dir}" >/dev/null; then
    echo "Installed package does not export cvh::headers_fast." >&2
    return 1
  fi

  if grep -R -E "cvh::native|cvh::full|full_backend|cvh_native_backend" "${cmake_dir}" >/dev/null; then
    echo "Installed package exports legacy .cpp targets." >&2
    grep -R -n -E "cvh::native|cvh::full|full_backend|cvh_native_backend" "${cmake_dir}" >&2
    return 1
  fi
}

"${ROOT_DIR}/scripts/check_public_headers.sh"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCVH_BUILD_NATIVE_BACKEND=OFF \
  -DCVH_BUILD_BACKEND_KERNEL_SOURCES=OFF \
  -DCVH_BUILD_TESTS=ON \
  -DCVH_BUILD_BENCHMARKS=OFF \
  >/dev/null

cmake --build "${BUILD_DIR}" -j --target \
  cvh_header_compile_smoke \
  cvh_include_only_smoke \
  cvh_headers_fast_smoke \
  >/dev/null

ctest --test-dir "${BUILD_DIR}" --output-on-failure \
  -R 'cvh_header_compile_smoke|cvh_include_only_smoke|cvh_headers_fast_smoke'

cmake --install "${BUILD_DIR}" --prefix "${INSTALL_DIR}" >/dev/null
require_no_legacy_export "${INSTALL_DIR}/lib/cmake/opencv_header_only"

mkdir -p "${HEADERS_CONSUMER_DIR}"
cat > "${HEADERS_CONSUMER_DIR}/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.16)
project(cvh_headers_consumer LANGUAGES CXX)

find_package(opencv_header_only CONFIG REQUIRED)

if(NOT TARGET cvh::headers)
    message(FATAL_ERROR "Missing cvh::headers target")
endif()
if(NOT TARGET cvh::headers_fast)
    message(FATAL_ERROR "Missing cvh::headers_fast target")
endif()
if(TARGET cvh::native OR TARGET cvh::full OR TARGET cvh::full_backend)
    message(FATAL_ERROR "Installed package must not expose legacy .cpp targets")
endif()

add_executable(headers_consumer main.cpp)
target_link_libraries(headers_consumer PRIVATE cvh::headers)
target_compile_features(headers_consumer PRIVATE cxx_std_17)
EOF

cat > "${HEADERS_CONSUMER_DIR}/main.cpp" <<'EOF'
#include <cvh/cvh.h>

#ifndef CVH_LITE
#error "cvh::headers must keep the header-only compatibility macro"
#endif

#ifdef CVH_NATIVE
#error "cvh::headers must not enable legacy .cpp mode"
#endif

#if CVH_ENABLE_OPENCV_INTRIN
#error "cvh::headers must not enable OpenCV Universal Intrinsics by default"
#endif

#if CVH_ENABLE_PLATFORM_INTRINSICS
#error "cvh::headers must not enable platform intrinsics by default"
#endif

#if CVH_ENABLE_XSIMD
#error "cvh::headers must not enable xsimd"
#endif

#if CVH_ENABLE_LEGACY_XSIMD
#error "cvh::headers must not enable legacy xsimd"
#endif

int main()
{
    cvh::Mat src({2, 2}, CV_8UC1);
    src = 7;

    cvh::Mat dst;
    cvh::resize(src, dst, cvh::Size(1, 1), 0.0, 0.0, cvh::INTER_LINEAR);

    return (dst.dims == 2 && dst.type() == CV_8UC1 && dst.size[0] == 1 && dst.size[1] == 1) ? 0 : 1;
}
EOF

cmake -S "${HEADERS_CONSUMER_DIR}" -B "${HEADERS_CONSUMER_DIR}/build" \
  -DCMAKE_PREFIX_PATH="${INSTALL_DIR}" \
  >/dev/null
cmake --build "${HEADERS_CONSUMER_DIR}/build" -j >/dev/null
"${HEADERS_CONSUMER_DIR}/build/headers_consumer"

mkdir -p "${FAST_CONSUMER_DIR}"
cat > "${FAST_CONSUMER_DIR}/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.16)
project(cvh_headers_fast_consumer LANGUAGES CXX)

find_package(opencv_header_only CONFIG REQUIRED)

if(NOT TARGET cvh::headers)
    message(FATAL_ERROR "Missing cvh::headers target")
endif()
if(NOT TARGET cvh::headers_fast)
    message(FATAL_ERROR "Missing cvh::headers_fast target")
endif()
if(TARGET cvh::native OR TARGET cvh::full OR TARGET cvh::full_backend)
    message(FATAL_ERROR "Installed package must not expose legacy .cpp targets")
endif()

add_executable(headers_fast_consumer main.cpp)
target_link_libraries(headers_fast_consumer PRIVATE cvh::headers_fast)
target_compile_features(headers_fast_consumer PRIVATE cxx_std_17)
EOF

cat > "${FAST_CONSUMER_DIR}/main.cpp" <<'EOF'
#include <cvh/cvh.h>
#include <cvh/core/simd/simd.h>

#include <cstring>

#ifndef CVH_LITE
#error "cvh::headers_fast must keep the header-only compatibility macro"
#endif

#ifdef CVH_NATIVE
#error "cvh::headers_fast must not enable legacy .cpp mode"
#endif

#if !CVH_ENABLE_OPENCV_INTRIN
#error "cvh::headers_fast must enable OpenCV Universal Intrinsics"
#endif

#if !CVH_ENABLE_PLATFORM_INTRINSICS
#error "cvh::headers_fast must enable platform intrinsics"
#endif

#if CVH_ENABLE_XSIMD
#error "cvh::headers_fast must not enable xsimd"
#endif

#if CVH_ENABLE_LEGACY_XSIMD
#error "cvh::headers_fast must not enable legacy xsimd"
#endif

int main()
{
    if (std::strcmp(cvh::detail::simd::backend_name(), "opencv_intrin") != 0)
    {
        return 1;
    }

    cvh::Mat src({2, 2}, CV_8UC1);
    src.at<uchar>(0, 0, 0) = 4;
    src.at<uchar>(0, 1, 0) = 8;
    src.at<uchar>(1, 0, 0) = 12;
    src.at<uchar>(1, 1, 0) = 16;

    cvh::Mat dst;
    cvh::resize(src, dst, cvh::Size(1, 1), 0.0, 0.0, cvh::INTER_LINEAR);

    return dst.at<uchar>(0, 0, 0) == 10 ? 0 : 2;
}
EOF

cmake -S "${FAST_CONSUMER_DIR}" -B "${FAST_CONSUMER_DIR}/build" \
  -DCMAKE_PREFIX_PATH="${INSTALL_DIR}" \
  >/dev/null
cmake --build "${FAST_CONSUMER_DIR}/build" -j >/dev/null
"${FAST_CONSUMER_DIR}/build/headers_fast_consumer"

mkdir -p "${LEGACY_XSIMD_DENY_DIR}"
cat > "${LEGACY_XSIMD_DENY_DIR}/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.16)
project(cvh_legacy_xsimd_denied_consumer LANGUAGES CXX)

find_package(opencv_header_only CONFIG REQUIRED)

add_executable(legacy_xsimd_denied main.cpp)
target_link_libraries(legacy_xsimd_denied PRIVATE cvh::headers)
target_compile_features(legacy_xsimd_denied PRIVATE cxx_std_17)
target_compile_definitions(legacy_xsimd_denied PRIVATE CVH_ENABLE_XSIMD=1)
EOF

cat > "${LEGACY_XSIMD_DENY_DIR}/main.cpp" <<'EOF'
#include <cvh/core/simd/simd.h>

int main()
{
    return cvh::detail::simd::backend_name()[0] == '\0';
}
EOF

cmake -S "${LEGACY_XSIMD_DENY_DIR}" -B "${LEGACY_XSIMD_DENY_DIR}/build" \
  -DCMAKE_PREFIX_PATH="${INSTALL_DIR}" \
  >/dev/null
if cmake --build "${LEGACY_XSIMD_DENY_DIR}/build" -j >"${LEGACY_XSIMD_DENY_DIR}/build.log" 2>&1; then
  echo "CVH_ENABLE_XSIMD=1 unexpectedly compiled without legacy opt-in." >&2
  exit 1
fi
if ! grep -F "CVH_ENABLE_XSIMD is legacy/experimental" "${LEGACY_XSIMD_DENY_DIR}/build.log" >/dev/null; then
  echo "CVH_ENABLE_XSIMD=1 failed for an unexpected reason." >&2
  cat "${LEGACY_XSIMD_DENY_DIR}/build.log" >&2
  exit 1
fi

cmake -S "${ROOT_DIR}" -B "${LEGACY_ON_BUILD_DIR}" \
  -DCVH_BUILD_NATIVE_BACKEND=ON \
  -DCVH_BUILD_TESTS=OFF \
  -DCVH_BUILD_BENCHMARKS=OFF \
  >/dev/null
cmake --install "${LEGACY_ON_BUILD_DIR}" --prefix "${LEGACY_ON_INSTALL_DIR}" >/dev/null
require_no_legacy_export "${LEGACY_ON_INSTALL_DIR}/lib/cmake/opencv_header_only"

echo "Header-only contract check passed."
