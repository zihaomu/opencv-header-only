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

option(CVH_BUILD_FULL_BACKEND "Build src/ backend library (CVH_FULL mode)" ON)
option(CVH_BUILD_BACKEND_KERNEL_SOURCES "Build sources requiring full backend kernels" ON)
option(CVH_BUILD_TESTS "Build all test targets (smoke + core)" ON)
option(CVH_BUILD_BENCHMARKS "Build benchmark targets" ON)
option(CVH_USE_OPENMP "Enable OpenMP acceleration in backend kernels when available" ON)
option(CVH_ENABLE_OPENCV_COMPARE "Build cvh vs OpenCV compare benchmark target" OFF)
set(CVH_OPENCV_BENCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv_compare/opencv-bench-slim" CACHE PATH "Path to opencv-bench-slim source repository")

if(APPLE)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
endif()

set(CVH_ENABLE_FULL_BACKEND OFF)
if(CVH_BUILD_FULL_BACKEND)
    set(CVH_ENABLE_FULL_BACKEND ON)
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

set(CVH_FULL_BACKEND_SOURCES
    src/core/basic_op_scalar.cpp
    src/core/kernel/binary_kernel_xsimd.cpp
    src/core/kernel/gemm_kernel_xsimd.cpp
    src/core/kernel/transpose_kernel.cpp
    src/core/mat.cpp
    src/core/mat_gemm.cpp
    src/core/mat_expr.cpp
    src/core/mat_convert.cpp
    src/core/system.cpp
    src/core/utils.cpp
    src/imgproc/resize.cpp
    src/imgproc/cvtcolor.cpp
    src/imgproc/cvtcolor_rgb_gray.cpp
    src/imgproc/cvtcolor_yuv420.cpp
    src/imgproc/cvtcolor_yuv422.cpp
    src/imgproc/cvtcolor_yuv444.cpp
    src/imgproc/threshold.cpp
    src/imgproc/lut.cpp
    src/imgproc/box_filter.cpp
    src/imgproc/gaussian_blur.cpp
    src/imgproc/sobel.cpp
    src/imgproc/canny_image.cpp
    src/imgproc/canny_deriv.cpp
    src/imgproc/copy_make_border.cpp
    src/imgproc/filter2d.cpp
    src/imgproc/sep_filter2d.cpp
    src/imgproc/warp_affine.cpp
    src/imgproc/erode.cpp
    src/imgproc/dilate.cpp
    src/imgproc/backend_registry.cpp
    src/highgui/highgui.cpp
)

set(CVH_FULL_BACKEND_PRIVATE_LIBS)
set(CVH_FULL_BACKEND_PRIVATE_INCLUDE_DIRS)
set(CVH_FULL_BACKEND_PRIVATE_COMPILE_DEFINITIONS)

if(WIN32)
    list(APPEND CVH_FULL_BACKEND_SOURCES src/highgui/display_win32.cpp)
elseif(APPLE)
    find_library(CVH_COCOA_FRAMEWORK Cocoa REQUIRED)
    list(APPEND CVH_FULL_BACKEND_SOURCES src/highgui/display_cocoa.mm)
    list(APPEND CVH_FULL_BACKEND_PRIVATE_LIBS ${CVH_COCOA_FRAMEWORK})
    list(APPEND CVH_FULL_BACKEND_PRIVATE_COMPILE_DEFINITIONS CVH_HAVE_COCOA=1)
elseif(UNIX AND NOT APPLE AND NOT ANDROID)
    list(APPEND CVH_FULL_BACKEND_SOURCES src/highgui/display_fb.cpp)

    find_package(X11 QUIET)
    if(X11_FOUND)
        list(APPEND CVH_FULL_BACKEND_SOURCES src/highgui/display_x11.cpp)
        list(APPEND CVH_FULL_BACKEND_PRIVATE_LIBS ${X11_LIBRARIES})
        list(APPEND CVH_FULL_BACKEND_PRIVATE_INCLUDE_DIRS ${X11_INCLUDE_DIR})
        list(APPEND CVH_FULL_BACKEND_PRIVATE_COMPILE_DEFINITIONS CVH_HAVE_X11=1)
    endif()
endif()

if(CVH_ENABLE_FULL_BACKEND)
    add_library(cvh_full_backend STATIC ${CVH_FULL_BACKEND_SOURCES})
    add_library(cvh::full_backend ALIAS cvh_full_backend)
    add_library(cvh::full ALIAS cvh_full_backend)
    set_target_properties(cvh_full_backend PROPERTIES EXPORT_NAME full_backend)
    target_compile_features(cvh_full_backend PUBLIC cxx_std_17)
    target_compile_definitions(cvh_full_backend PUBLIC CVH_FULL=1)
    target_compile_definitions(cvh_full_backend PRIVATE ${CVH_FULL_BACKEND_PRIVATE_COMPILE_DEFINITIONS})
    target_link_libraries(cvh_full_backend PUBLIC cvh::headers PRIVATE ${CVH_FULL_BACKEND_PRIVATE_LIBS})
    target_include_directories(cvh_full_backend
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernel
            ${CMAKE_CURRENT_SOURCE_DIR}/include/cvh/3rdparty/xsimd/include
            ${CVH_FULL_BACKEND_PRIVATE_INCLUDE_DIRS}
    )

    if(CVH_USE_OPENMP)
        find_package(OpenMP QUIET)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(cvh_full_backend PRIVATE OpenMP::OpenMP_CXX)
            target_compile_definitions(cvh_full_backend PRIVATE CVH_WITH_OPENMP=1)
            message(STATUS "cvh_full_backend: OpenMP enabled")
        else()
            message(STATUS "cvh_full_backend: OpenMP requested but not found; building without OpenMP")
        endif()
    endif()

    if(CVH_BUILD_BENCHMARKS)
        add_executable(cvh_benchmark_core_ops
            benchmark/core_ops_benchmark.cpp
        )
        target_link_libraries(cvh_benchmark_core_ops PRIVATE cvh::full_backend)
        target_compile_features(cvh_benchmark_core_ops PRIVATE cxx_std_17)
        target_include_directories(cvh_benchmark_core_ops PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )

        add_executable(cvh_benchmark_imgproc_ops
            benchmark/imgproc_ops_benchmark.cpp
        )
        target_link_libraries(cvh_benchmark_imgproc_ops PRIVATE cvh::full_backend)
        target_compile_features(cvh_benchmark_imgproc_ops PRIVATE cxx_std_17)
        target_include_directories(cvh_benchmark_imgproc_ops PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )

        add_executable(cvh_benchmark_imgproc_filter
            benchmark/imgproc_filter_benchmark.cpp
        )
        target_link_libraries(cvh_benchmark_imgproc_filter PRIVATE cvh::full_backend)
        target_compile_features(cvh_benchmark_imgproc_filter PRIVATE cxx_std_17)
        target_include_directories(cvh_benchmark_imgproc_filter PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )

        if(CVH_ENABLE_OPENCV_COMPARE)
            if(NOT EXISTS "${CVH_OPENCV_BENCH_DIR}")
                message(FATAL_ERROR "CVH_ENABLE_OPENCV_COMPARE=ON but CVH_OPENCV_BENCH_DIR does not exist: ${CVH_OPENCV_BENCH_DIR}")
            endif()

            if(NOT OpenCV_DIR)
                if(EXISTS "${CVH_OPENCV_BENCH_DIR}/build-slim/OpenCVConfig.cmake")
                    set(OpenCV_DIR "${CVH_OPENCV_BENCH_DIR}/build-slim" CACHE PATH "Path to OpenCVConfig.cmake for compare benchmark" FORCE)
                endif()
            endif()

            find_package(OpenCV REQUIRED COMPONENTS core imgproc)

            add_executable(cvh_benchmark_compare
                benchmark/opencv_compare_benchmark.cpp
                benchmark/opencv_compare_opencv_backend.cpp
            )
            target_link_libraries(cvh_benchmark_compare PRIVATE cvh::full_backend ${OpenCV_LIBS})
            target_compile_features(cvh_benchmark_compare PRIVATE cxx_std_17)
            target_include_directories(cvh_benchmark_compare PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${OpenCV_INCLUDE_DIRS}
            )

            add_executable(cvh_benchmark_compare_lite
                benchmark/opencv_compare_benchmark.cpp
                benchmark/opencv_compare_opencv_backend.cpp
            )
            target_link_libraries(cvh_benchmark_compare_lite PRIVATE
                cvh::headers
                ${OpenCV_LIBS}
                $<TARGET_FILE:cvh_full_backend>
            )
            add_dependencies(cvh_benchmark_compare_lite cvh_full_backend)
            target_compile_features(cvh_benchmark_compare_lite PRIVATE cxx_std_17)
            target_compile_definitions(cvh_benchmark_compare_lite PRIVATE CVH_LITE=1)
            target_include_directories(cvh_benchmark_compare_lite PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${OpenCV_INCLUDE_DIRS}
            )
        endif()
    endif()

    add_executable(cvh_example_highgui_imshow
        example/highgui_imshow.cpp
    )
    target_link_libraries(cvh_example_highgui_imshow PRIVATE cvh::full)
    target_compile_features(cvh_example_highgui_imshow PRIVATE cxx_std_17)
    target_compile_definitions(cvh_example_highgui_imshow PRIVATE CVH_EXAMPLE_ROOT_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
    target_include_directories(cvh_example_highgui_imshow PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
endif()

if(CVH_BUILD_TESTS)
    add_subdirectory(test/3rdparty/googletest EXCLUDE_FROM_ALL)

    add_executable(cvh_header_compile_smoke test/smoke/cvh_header_compile_smoke.cpp)
    target_link_libraries(cvh_header_compile_smoke PRIVATE cvh::headers)

    add_executable(cvh_include_only_smoke test/smoke/cvh_include_only_smoke.cpp)
    target_include_directories(cvh_include_only_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(cvh_include_only_smoke PRIVATE cxx_std_17)

    add_executable(cvh_mode_lite_smoke test/smoke/cvh_mode_smoke.cpp)
    target_link_libraries(cvh_mode_lite_smoke PRIVATE cvh::headers)

    add_executable(cvh_lite_pipeline_smoke test/smoke/cvh_lite_pipeline_smoke.cpp)
    target_link_libraries(cvh_lite_pipeline_smoke PRIVATE cvh::headers)

    add_executable(cvh_resize_dispatch_lite_smoke test/smoke/cvh_resize_dispatch_smoke.cpp)
    target_link_libraries(cvh_resize_dispatch_lite_smoke PRIVATE cvh::headers)

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

    if(TARGET cvh_full_backend)
        add_executable(cvh_test_highgui_full
            test/highgui/highgui_full_backend_test.cpp
        )
        target_link_libraries(cvh_test_highgui_full PRIVATE cvh::full_backend gtest_main)
        target_compile_features(cvh_test_highgui_full PRIVATE cxx_std_17)
        target_compile_definitions(cvh_test_highgui_full PRIVATE M_ROOT_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
        target_include_directories(cvh_test_highgui_full PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/test
        )

        add_executable(cvh_full_backend_smoke test/smoke/cvh_full_backend_smoke.cpp)
        target_link_libraries(cvh_full_backend_smoke PRIVATE cvh::full_backend)

        add_executable(cvh_mode_full_smoke test/smoke/cvh_mode_smoke.cpp)
        target_compile_definitions(cvh_mode_full_smoke PRIVATE CVH_EXPECT_FULL=1)
        target_link_libraries(cvh_mode_full_smoke PRIVATE cvh::full)

        add_executable(cvh_resize_dispatch_full_smoke test/smoke/cvh_resize_dispatch_smoke.cpp)
        target_compile_definitions(cvh_resize_dispatch_full_smoke PRIVATE CVH_EXPECT_FULL=1)
        target_link_libraries(cvh_resize_dispatch_full_smoke PRIVATE cvh::full)
    endif()

    if(BUILD_TESTING)
        add_test(NAME cvh_header_compile_smoke COMMAND cvh_header_compile_smoke)
        add_test(NAME cvh_include_only_smoke COMMAND cvh_include_only_smoke)
        add_test(NAME cvh_mode_lite_smoke COMMAND cvh_mode_lite_smoke)
        add_test(NAME cvh_lite_pipeline_smoke COMMAND cvh_lite_pipeline_smoke)
        add_test(NAME cvh_resize_dispatch_lite_smoke COMMAND cvh_resize_dispatch_lite_smoke)
        add_test(NAME cvh_test_imgcodecs COMMAND cvh_test_imgcodecs)
        add_test(NAME cvh_test_imgproc COMMAND cvh_test_imgproc)
        add_test(NAME cvh_test_highgui COMMAND cvh_test_highgui)
        if(TARGET cvh_full_backend)
            add_test(NAME cvh_test_highgui_full COMMAND cvh_test_highgui_full)
            add_test(NAME cvh_full_backend_smoke COMMAND cvh_full_backend_smoke)
            add_test(NAME cvh_mode_full_smoke COMMAND cvh_mode_full_smoke)
            add_test(NAME cvh_resize_dispatch_full_smoke COMMAND cvh_resize_dispatch_full_smoke)
        endif()
    endif()

    if(TARGET cvh_full_backend)
        add_executable(cvh_test_core
            test/core/core_ops_test.cpp
            test/core/binary_op_contract_test.cpp
            test/core/mat_channel_contract_test.cpp
            test/core/mat_layout_semantics_test.cpp
            test/core/mat_shape_display_test.cpp
            test/core/mat_expr_scalar_compare_test.cpp
            test/core/mat_opencv_compat_test.cpp
            test/core/parallel_for_runtime_test.cpp
            test/core/mat_scalar_ops_test.cpp
            test/core/mat_submat_test.cpp
            test/core/mat_upstream_channel_port_test.cpp
            test/core/gemm_pack_contract_test.cpp
            test/core/mat_contract_test.cpp
            test/core/scalar_contract_test.cpp
        )
        target_link_libraries(cvh_test_core PRIVATE cvh::full_backend gtest_main)
        target_compile_features(cvh_test_core PRIVATE cxx_std_17)
        target_compile_definitions(cvh_test_core PRIVATE M_ROOT_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
        target_include_directories(cvh_test_core PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/test
        )

        if(BUILD_TESTING)
            add_test(NAME cvh_test_core COMMAND cvh_test_core)
        endif()
    endif()
endif()

set(CVH_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/opencv_header_only" CACHE STRING "Install path for CMake package config files")

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS cvh_headers
    EXPORT opencv_header_onlyTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(TARGET cvh_full_backend)
    install(TARGETS cvh_full_backend
        EXPORT opencv_header_onlyTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

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
