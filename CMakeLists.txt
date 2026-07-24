cmake_minimum_required(VERSION 3.16)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION.txt" CVH_PROJECT_VERSION_RAW)
string(STRIP "${CVH_PROJECT_VERSION_RAW}" CVH_PROJECT_VERSION)
if(NOT CVH_PROJECT_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR "VERSION.txt must be semantic version (MAJOR.MINOR.PATCH), got: '${CVH_PROJECT_VERSION}'")
endif()

project(opencv_header_only VERSION ${CVH_PROJECT_VERSION} LANGUAGES CXX)

# default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(CVH_BUILD_NATIVE_BACKEND "Internal/development-only: build legacy .cpp experiments; not installed/exported" OFF)
set(CVH_BUILD_FULL_BACKEND "" CACHE STRING "Deprecated internal alias for CVH_BUILD_NATIVE_BACKEND")
if(NOT "${CVH_BUILD_FULL_BACKEND}" STREQUAL "")
    message(DEPRECATION "CVH_BUILD_FULL_BACKEND is deprecated and only forwards to the internal CVH_BUILD_NATIVE_BACKEND switch")
    set(CVH_BUILD_NATIVE_BACKEND "${CVH_BUILD_FULL_BACKEND}" CACHE BOOL "Internal/development-only: build legacy .cpp experiments; not installed/exported" FORCE)
endif()
option(CVH_BUILD_TESTS "Build all test targets (smoke + core)" ON)
option(CVH_BUILD_BENCHMARKS "Build benchmark targets" ON)
option(CVH_USE_OPENMP "Internal/development-only: enable OpenMP in legacy .cpp experiments when available" ON)
option(CVH_ENABLE_OPENCV_COMPARE "Build cvh vs OpenCV compare benchmark target" OFF)
set(CVH_OPENCV_BENCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/benchmark/opencv_compare/opencv-bench-slim" CACHE PATH "Path to opencv-bench-slim source repository")

if(CVH_ENABLE_OPENCV_COMPARE)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc)
endif()

set(CVH_ENABLE_NATIVE_BACKEND OFF)
if(CVH_BUILD_NATIVE_BACKEND)
    set(CVH_ENABLE_NATIVE_BACKEND ON)
endif()

if(APPLE AND CVH_ENABLE_NATIVE_BACKEND)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cvh_headers INTERFACE)
add_library(cvh::headers ALIAS cvh_headers)
set_target_properties(cvh_headers PROPERTIES EXPORT_NAME headers)
target_include_directories(cvh_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(cvh_headers INTERFACE cxx_std_17)

set(CVH_OPENCV_INTRIN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/cvh/3rdparty/opencv_intrin")
target_include_directories(cvh_headers INTERFACE
    $<BUILD_INTERFACE:${CVH_OPENCV_INTRIN_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cvh/3rdparty/opencv_intrin>
)

add_library(cvh_headers_fast INTERFACE)
add_library(cvh::headers_fast ALIAS cvh_headers_fast)
set_target_properties(cvh_headers_fast PROPERTIES EXPORT_NAME headers_fast)
target_link_libraries(cvh_headers_fast INTERFACE cvh_headers)
target_compile_definitions(cvh_headers_fast INTERFACE
    CVH_ENABLE_PLATFORM_INTRINSICS=1
)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64|AARCH64)$")
    target_compile_definitions(cvh_headers_fast INTERFACE CV_NEON=1)
endif()

set(CVH_NATIVE_BACKEND_SOURCES
    src/highgui/highgui.cpp
)

set(CVH_NATIVE_BACKEND_PRIVATE_LIBS)
set(CVH_NATIVE_BACKEND_PRIVATE_INCLUDE_DIRS)
set(CVH_NATIVE_BACKEND_PRIVATE_COMPILE_DEFINITIONS)
set(CVH_NATIVE_BACKEND_LINK_DEPS)

if(CVH_ENABLE_NATIVE_BACKEND)
    # Legacy .cpp experiments remain build-tree only. The installed package
    # public surface is cvh::headers and cvh::headers_fast.
    if(WIN32)
        list(APPEND CVH_NATIVE_BACKEND_SOURCES src/highgui/display_win32.cpp)
    elseif(APPLE)
        find_library(CVH_COCOA_FRAMEWORK Cocoa REQUIRED)
        list(APPEND CVH_NATIVE_BACKEND_SOURCES src/highgui/display_cocoa.mm)
        list(APPEND CVH_NATIVE_BACKEND_PRIVATE_LIBS ${CVH_COCOA_FRAMEWORK})
        list(APPEND CVH_NATIVE_BACKEND_PRIVATE_COMPILE_DEFINITIONS CVH_HAVE_COCOA=1)
    elseif(UNIX AND NOT APPLE AND NOT ANDROID)
        list(APPEND CVH_NATIVE_BACKEND_SOURCES src/highgui/display_fb.cpp)

        find_package(X11 QUIET)
        if(X11_FOUND)
            list(APPEND CVH_NATIVE_BACKEND_SOURCES src/highgui/display_x11.cpp)
            list(APPEND CVH_NATIVE_BACKEND_PRIVATE_LIBS ${X11_LIBRARIES})
            list(APPEND CVH_NATIVE_BACKEND_PRIVATE_INCLUDE_DIRS ${X11_INCLUDE_DIR})
            list(APPEND CVH_NATIVE_BACKEND_PRIVATE_COMPILE_DEFINITIONS CVH_HAVE_X11=1)
        endif()
    endif()

    add_library(cvh_native_backend STATIC ${CVH_NATIVE_BACKEND_SOURCES})
    add_library(cvh::native ALIAS cvh_native_backend)
    add_library(cvh::full_backend ALIAS cvh_native_backend)
    add_library(cvh::full ALIAS cvh_native_backend)
    add_library(cvh_full_backend ALIAS cvh_native_backend)
    set_target_properties(cvh_native_backend PROPERTIES EXPORT_NAME native)
    target_compile_features(cvh_native_backend PUBLIC cxx_std_17)
    target_compile_definitions(cvh_native_backend PUBLIC CVH_NATIVE=1 CVH_FULL=1)
    target_compile_definitions(cvh_native_backend PRIVATE ${CVH_NATIVE_BACKEND_PRIVATE_COMPILE_DEFINITIONS})
    target_link_libraries(cvh_native_backend PUBLIC cvh::headers PRIVATE ${CVH_NATIVE_BACKEND_PRIVATE_LIBS})
    target_include_directories(cvh_native_backend
        PRIVATE
            ${CVH_NATIVE_BACKEND_PRIVATE_INCLUDE_DIRS}
    )

    if(CVH_USE_OPENMP)
        find_package(OpenMP QUIET)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(cvh_native_backend PRIVATE OpenMP::OpenMP_CXX)
            list(APPEND CVH_NATIVE_BACKEND_LINK_DEPS OpenMP::OpenMP_CXX)
            target_compile_definitions(cvh_native_backend PRIVATE CVH_WITH_OPENMP=1)
            message(STATUS "cvh_native_backend: OpenMP enabled")
        else()
            message(STATUS "cvh_native_backend: OpenMP requested but not found; building without OpenMP")
        endif()
    endif()

    add_executable(cvh_example_highgui_imshow
        example/highgui_imshow.cpp
    )
    target_link_libraries(cvh_example_highgui_imshow PRIVATE cvh::native)
    target_compile_features(cvh_example_highgui_imshow PRIVATE cxx_std_17)
    target_compile_definitions(cvh_example_highgui_imshow PRIVATE CVH_EXAMPLE_ROOT_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
    target_include_directories(cvh_example_highgui_imshow PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
endif()

if(CVH_BUILD_BENCHMARKS)
    add_executable(cvh_benchmark_imgproc_coverage
        benchmark/imgproc_coverage_benchmark.cpp
    )
    target_link_libraries(cvh_benchmark_imgproc_coverage PRIVATE cvh::headers_fast)
    target_compile_features(cvh_benchmark_imgproc_coverage PRIVATE cxx_std_17)
    target_include_directories(cvh_benchmark_imgproc_coverage PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    add_executable(cvh_benchmark_imgproc_filter
        benchmark/imgproc_filter_benchmark.cpp
    )
    target_link_libraries(cvh_benchmark_imgproc_filter PRIVATE cvh::headers_fast)
    target_compile_features(cvh_benchmark_imgproc_filter PRIVATE cxx_std_17)
    target_include_directories(cvh_benchmark_imgproc_filter PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    add_executable(cvh_benchmark_core_mat_header
        benchmark/core_mat_header_benchmark.cpp
    )
    target_link_libraries(cvh_benchmark_core_mat_header PRIVATE cvh::headers_fast)
    target_compile_features(cvh_benchmark_core_mat_header PRIVATE cxx_std_17)

    add_executable(cvh_benchmark_cvtcolor_bgr2gray_header
        benchmark/cvtcolor_bgr2gray_header_benchmark.cpp
    )
    target_link_libraries(cvh_benchmark_cvtcolor_bgr2gray_header PRIVATE cvh::headers_fast)
    target_compile_features(cvh_benchmark_cvtcolor_bgr2gray_header PRIVATE cxx_std_17)

    add_executable(cvh_benchmark_resize_bilinear_header
        benchmark/resize_bilinear_header_benchmark.cpp
    )
    target_link_libraries(cvh_benchmark_resize_bilinear_header PRIVATE cvh::headers_fast)
    target_compile_features(cvh_benchmark_resize_bilinear_header PRIVATE cxx_std_17)

    add_executable(cvh_benchmark_imgproc_header
        benchmark/imgproc_header_benchmark.cpp
    )
    target_link_libraries(cvh_benchmark_imgproc_header PRIVATE cvh::headers_fast)
    target_compile_features(cvh_benchmark_imgproc_header PRIVATE cxx_std_17)

    if(CVH_ENABLE_OPENCV_COMPARE)
        add_library(cvh_benchmark_opencv_compare_backend_obj OBJECT
            benchmark/opencv_compare_opencv_backend.cpp
            benchmark/opencv_compare_phase1_opencv_backend.cpp
        )
        target_compile_features(cvh_benchmark_opencv_compare_backend_obj PRIVATE cxx_std_17)
        target_include_directories(cvh_benchmark_opencv_compare_backend_obj PRIVATE
            ${OpenCV_INCLUDE_DIRS}
        )

        add_executable(cvh_benchmark_opencv_compare_headers_fast
            benchmark/opencv_compare_header_benchmark.cpp
            benchmark/opencv_compare_phase1_header_benchmark.cpp
            $<TARGET_OBJECTS:cvh_benchmark_opencv_compare_backend_obj>
        )
        target_link_libraries(cvh_benchmark_opencv_compare_headers_fast PRIVATE cvh::headers_fast ${OpenCV_LIBS})
        target_compile_features(cvh_benchmark_opencv_compare_headers_fast PRIVATE cxx_std_17)
        target_compile_definitions(cvh_benchmark_opencv_compare_headers_fast PRIVATE CVH_COMPARE_IMPL_NAME="cvh_headers_fast")
        target_include_directories(cvh_benchmark_opencv_compare_headers_fast BEFORE PRIVATE
            ${OpenCV_INCLUDE_DIRS}
        )
        target_include_directories(cvh_benchmark_opencv_compare_headers_fast PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
    endif()
endif()

if(CVH_BUILD_TESTS)
    add_subdirectory(test/3rdparty/googletest EXCLUDE_FROM_ALL)

    add_executable(cvh_header_compile_smoke test/smoke/cvh_header_compile_smoke.cpp)
    target_link_libraries(cvh_header_compile_smoke PRIVATE cvh::headers)

    add_executable(cvh_core_header_odr_smoke
        test/smoke/cvh_core_header_odr_main.cpp
        test/smoke/cvh_core_header_odr_peer.cpp
    )
    target_link_libraries(cvh_core_header_odr_smoke PRIVATE cvh::headers)

    add_executable(cvh_imgproc_header_odr_smoke
        test/smoke/cvh_imgproc_header_odr_main.cpp
        test/smoke/cvh_imgproc_header_odr_peer.cpp
    )
    target_link_libraries(cvh_imgproc_header_odr_smoke PRIVATE cvh::headers_fast)

    add_executable(cvh_include_only_smoke test/smoke/cvh_include_only_smoke.cpp)
    target_include_directories(cvh_include_only_smoke PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CVH_OPENCV_INTRIN_INCLUDE_DIR}
    )
    target_compile_features(cvh_include_only_smoke PRIVATE cxx_std_17)

    add_executable(cvh_mode_lite_smoke test/smoke/cvh_mode_smoke.cpp)
    target_link_libraries(cvh_mode_lite_smoke PRIVATE cvh::headers)

    add_executable(cvh_headers_fast_smoke test/smoke/cvh_headers_fast_smoke.cpp)
    target_link_libraries(cvh_headers_fast_smoke PRIVATE cvh::headers_fast)

    add_executable(cvh_lite_pipeline_smoke test/smoke/cvh_lite_pipeline_smoke.cpp)
    target_link_libraries(cvh_lite_pipeline_smoke PRIVATE cvh::headers)

    add_executable(cvh_resize_dispatch_lite_smoke test/smoke/cvh_resize_dispatch_smoke.cpp)
    target_link_libraries(cvh_resize_dispatch_lite_smoke PRIVATE cvh::headers)

    add_executable(cvh_opencv_intrin_smoke test/smoke/cvh_opencv_intrin_smoke.cpp)
    target_link_libraries(cvh_opencv_intrin_smoke PRIVATE cvh::headers)
    target_compile_definitions(cvh_opencv_intrin_smoke PRIVATE CV_FORCE_SIMD128_CPP=1)

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|i386|i686|x86|X86)$")
        add_executable(cvh_opencv_intrin_x86_smoke test/smoke/cvh_opencv_intrin_x86_smoke.cpp)
        target_link_libraries(cvh_opencv_intrin_x86_smoke PRIVATE cvh::headers_fast)
        if(MSVC)
            target_compile_options(cvh_opencv_intrin_x86_smoke PRIVATE /arch:AVX2)
        else()
            target_compile_options(cvh_opencv_intrin_x86_smoke PRIVATE -mavx2)
        endif()
    endif()

    add_executable(cvh_cvtcolor_opencv_intrin_smoke test/smoke/cvh_cvtcolor_opencv_intrin_smoke.cpp)
    target_link_libraries(cvh_cvtcolor_opencv_intrin_smoke PRIVATE cvh::headers_fast)

    add_executable(cvh_resize_opencv_intrin_smoke test/smoke/cvh_resize_opencv_intrin_smoke.cpp)
    target_link_libraries(cvh_resize_opencv_intrin_smoke PRIVATE cvh::headers_fast)

    add_executable(cvh_test_imgcodecs
        test/imgcodecs/imgcodecs_format_test.cpp
    )
    target_link_libraries(cvh_test_imgcodecs PRIVATE cvh::headers gtest_main)
    target_compile_features(cvh_test_imgcodecs PRIVATE cxx_std_17)
    target_compile_definitions(cvh_test_imgcodecs PRIVATE M_ROOT_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
    target_include_directories(cvh_test_imgcodecs PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/test
    )

    add_executable(cvh_test_imgproc
        test/imgproc/imgproc_header_layout_test.cpp
        test/imgproc/imgproc_resize_contract_test.cpp
        test/imgproc/imgproc_cvtcolor_contract_test.cpp
        test/imgproc/imgproc_threshold_contract_test.cpp
        test/imgproc/imgproc_pipeline_regression_test.cpp
        test/imgproc/imgproc_lut_contract_test.cpp
        test/imgproc/imgproc_filter_contract_test.cpp
        test/imgproc/imgproc_phase1_kernels_contract_test.cpp
        test/imgproc/imgproc_phase1_intensity_contract_test.cpp
        test/imgproc/imgproc_phase1_pyramid_color_contract_test.cpp
        test/imgproc/imgproc_phase1_geometry_matrix_contract_test.cpp
        test/imgproc/imgproc_phase1_geometric_sampling_contract_test.cpp
        test/imgproc/imgproc_filter2d_contract_test.cpp
        test/imgproc/imgproc_sep_filter2d_contract_test.cpp
        test/imgproc/imgproc_copy_make_border_contract_test.cpp
        test/imgproc/imgproc_morph_gradient_contract_test.cpp
        test/imgproc/imgproc_canny_contract_test.cpp
        test/imgproc/imgproc_warp_affine_contract_test.cpp
    )
    target_link_libraries(cvh_test_imgproc PRIVATE cvh::headers gtest_main)
    target_compile_features(cvh_test_imgproc PRIVATE cxx_std_17)
    target_compile_definitions(cvh_test_imgproc PRIVATE M_ROOT_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
    target_include_directories(cvh_test_imgproc PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/test
    )

    add_executable(cvh_test_highgui
        test/highgui/highgui_contract_test.cpp
    )
    target_link_libraries(cvh_test_highgui PRIVATE cvh::headers gtest_main)
    target_compile_features(cvh_test_highgui PRIVATE cxx_std_17)
    target_compile_definitions(cvh_test_highgui PRIVATE M_ROOT_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
    target_include_directories(cvh_test_highgui PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/test
    )

    set(CVH_CORE_TEST_LITE_SOURCES
        test/core/array_ops_contract_test.cpp
        test/core/core_ops_test.cpp
        test/core/binary_op_contract_test.cpp
        test/core/gemm_pack_contract_test.cpp
        test/core/geometry_types_contract_test.cpp
        test/core/layout_ops_contract_test.cpp
        test/core/mat_channel_contract_test.cpp
        test/core/mat_layout_semantics_test.cpp
        test/core/mat_contract_test.cpp
        test/core/math_ops_contract_test.cpp
        test/core/mat_expr_scalar_compare_test.cpp
        test/core/mat_shape_display_test.cpp
        test/core/mat_opencv_compat_test.cpp
        test/core/mat_scalar_ops_test.cpp
        test/core/mat_submat_test.cpp
        test/core/mat_upstream_channel_port_test.cpp
        test/core/parallel_for_runtime_test.cpp
        test/core/reduction_ops_contract_test.cpp
        test/core/scalar_contract_test.cpp
    )

    add_executable(cvh_test_core_lite
        ${CVH_CORE_TEST_LITE_SOURCES}
    )
    add_executable(cvh_test_core ALIAS cvh_test_core_lite)
    target_link_libraries(cvh_test_core_lite PRIVATE cvh::headers gtest_main)
    target_compile_features(cvh_test_core_lite PRIVATE cxx_std_17)
    target_compile_definitions(cvh_test_core_lite PRIVATE M_ROOT_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
    target_include_directories(cvh_test_core_lite PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/test
    )

    if(CVH_ENABLE_OPENCV_COMPARE)
        add_library(cvh_test_opencv_contract_backend_obj OBJECT
            test/opencv_contract/opencv_contract_backend.cpp
        )
        target_compile_features(cvh_test_opencv_contract_backend_obj PRIVATE cxx_std_17)
        target_include_directories(cvh_test_opencv_contract_backend_obj PRIVATE
            ${OpenCV_INCLUDE_DIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/opencv_contract
        )

        add_executable(cvh_test_opencv_contract_smoke
            test/opencv_contract/opencv_contract_smoke_test.cpp
            $<TARGET_OBJECTS:cvh_test_opencv_contract_backend_obj>
        )
        target_link_libraries(cvh_test_opencv_contract_smoke PRIVATE
            cvh::headers
            gtest_main
            ${OpenCV_LIBS}
        )
        target_compile_features(cvh_test_opencv_contract_smoke PRIVATE cxx_std_17)
        target_include_directories(cvh_test_opencv_contract_smoke PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/test/opencv_contract
        )
    endif()

    if(TARGET cvh_native_backend)
        add_executable(cvh_test_highgui_native
            test/highgui/highgui_native_backend_test.cpp
        )
        target_link_libraries(cvh_test_highgui_native PRIVATE cvh::native gtest_main)
        target_compile_features(cvh_test_highgui_native PRIVATE cxx_std_17)
        target_compile_definitions(cvh_test_highgui_native PRIVATE M_ROOT_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
        target_include_directories(cvh_test_highgui_native PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/test
        )

        add_executable(cvh_native_backend_smoke test/smoke/cvh_native_backend_smoke.cpp)
        target_link_libraries(cvh_native_backend_smoke PRIVATE cvh::native)

        add_executable(cvh_mode_native_smoke test/smoke/cvh_mode_smoke.cpp)
        target_compile_definitions(cvh_mode_native_smoke PRIVATE CVH_EXPECT_NATIVE=1)
        target_link_libraries(cvh_mode_native_smoke PRIVATE cvh::native)

    endif()

    if(BUILD_TESTING)
        add_test(NAME cvh_header_compile_smoke COMMAND cvh_header_compile_smoke)
        add_test(NAME cvh_core_header_odr_smoke COMMAND cvh_core_header_odr_smoke)
        add_test(NAME cvh_imgproc_header_odr_smoke COMMAND cvh_imgproc_header_odr_smoke)
        add_test(NAME cvh_include_only_smoke COMMAND cvh_include_only_smoke)
        add_test(NAME cvh_mode_lite_smoke COMMAND cvh_mode_lite_smoke)
        add_test(NAME cvh_headers_fast_smoke COMMAND cvh_headers_fast_smoke)
        add_test(NAME cvh_lite_pipeline_smoke COMMAND cvh_lite_pipeline_smoke)
        add_test(NAME cvh_resize_dispatch_lite_smoke COMMAND cvh_resize_dispatch_lite_smoke)
        add_test(NAME cvh_opencv_intrin_smoke COMMAND cvh_opencv_intrin_smoke)
        if(TARGET cvh_opencv_intrin_x86_smoke)
            add_test(NAME cvh_opencv_intrin_x86_smoke COMMAND cvh_opencv_intrin_x86_smoke)
        endif()
        add_test(NAME cvh_cvtcolor_opencv_intrin_smoke COMMAND cvh_cvtcolor_opencv_intrin_smoke)
        add_test(NAME cvh_resize_opencv_intrin_smoke COMMAND cvh_resize_opencv_intrin_smoke)
        add_test(NAME cvh_test_imgcodecs COMMAND cvh_test_imgcodecs)
        add_test(NAME cvh_test_imgproc COMMAND cvh_test_imgproc)
        add_test(NAME cvh_test_highgui COMMAND cvh_test_highgui)
        add_test(NAME cvh_test_core_lite COMMAND cvh_test_core_lite)
        if(TARGET cvh_test_opencv_contract_smoke)
            add_test(NAME cvh_test_opencv_contract_smoke COMMAND cvh_test_opencv_contract_smoke)
        endif()
        add_test(NAME cvh_test_core COMMAND cvh_test_core_lite)
        if(TARGET cvh_native_backend)
            add_test(NAME cvh_test_highgui_native COMMAND cvh_test_highgui_native)
            add_test(NAME cvh_native_backend_smoke COMMAND cvh_native_backend_smoke)
            add_test(NAME cvh_mode_native_smoke COMMAND cvh_mode_native_smoke)
        endif()
    endif()

endif()

set(CVH_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/opencv_header_only" CACHE STRING "Install path for CMake package config files")

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS cvh_headers cvh_headers_fast
    EXPORT opencv_header_onlyTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT opencv_header_onlyTargets
    NAMESPACE cvh::
    FILE opencv_header_onlyTargets.cmake
    DESTINATION ${CVH_INSTALL_CMAKEDIR}
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/opencv_header_onlyConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/opencv_header_onlyConfig.cmake"
    INSTALL_DESTINATION ${CVH_INSTALL_CMAKEDIR}
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/opencv_header_onlyConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/opencv_header_onlyConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/opencv_header_onlyConfigVersion.cmake"
    DESTINATION ${CVH_INSTALL_CMAKEDIR}
)
