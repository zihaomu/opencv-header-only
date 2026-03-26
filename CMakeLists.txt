cmake_minimum_required(VERSION 3.16)

project(opencv_header_only VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

option(CVH_BUILD_FULL_BACKEND "Build src/ backend library (CVH_FULL mode)" ON)
option(CVH_BUILD_LEGACY_CORE "Deprecated alias of CVH_BUILD_FULL_BACKEND" ON)
option(CVH_BUILD_BACKEND_KERNEL_SOURCES "Build sources requiring legacy backend kernels" ON)
option(CVH_BUILD_TESTS "Build all test targets (smoke + core)" ON)

set(CVH_ENABLE_FULL_BACKEND OFF)
if(CVH_BUILD_FULL_BACKEND OR CVH_BUILD_LEGACY_CORE)
    set(CVH_ENABLE_FULL_BACKEND ON)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cvh_headers INTERFACE)
add_library(cvh::headers ALIAS cvh_headers)
target_include_directories(cvh_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cvh_headers INTERFACE cxx_std_17)

set(CVH_LEGACY_CORE_SOURCES
    src/core/mat.cpp
    src/core/mat_convert.cpp
    src/core/system.cpp
    src/core/utils.cpp
)

if(CVH_ENABLE_FULL_BACKEND)
    add_library(cvh_legacy_core STATIC ${CVH_LEGACY_CORE_SOURCES})
    add_library(cvh::legacy_core ALIAS cvh_legacy_core)
    add_library(cvh::full ALIAS cvh_legacy_core)
    target_compile_features(cvh_legacy_core PUBLIC cxx_std_17)
    target_compile_definitions(cvh_legacy_core PUBLIC CVH_FULL=1)
    target_link_libraries(cvh_legacy_core PUBLIC cvh::headers)
    target_include_directories(cvh_legacy_core
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernel
            ${CMAKE_CURRENT_SOURCE_DIR}/include/cvh/3rdparty/xsimd/include
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

    if(TARGET cvh_legacy_core)
        add_executable(cvh_legacy_core_smoke test/smoke/cvh_legacy_core_smoke.cpp)
        target_link_libraries(cvh_legacy_core_smoke PRIVATE cvh::legacy_core)

        add_executable(cvh_lite_pipeline_smoke test/smoke/cvh_lite_pipeline_smoke.cpp)
        target_link_libraries(cvh_lite_pipeline_smoke PRIVATE cvh::legacy_core)

        add_executable(cvh_mode_full_smoke test/smoke/cvh_mode_smoke.cpp)
        target_compile_definitions(cvh_mode_full_smoke PRIVATE CVH_EXPECT_FULL=1)
        target_link_libraries(cvh_mode_full_smoke PRIVATE cvh::full)
    endif()

    if(BUILD_TESTING)
        add_test(NAME cvh_header_compile_smoke COMMAND cvh_header_compile_smoke)
        add_test(NAME cvh_include_only_smoke COMMAND cvh_include_only_smoke)
        add_test(NAME cvh_mode_lite_smoke COMMAND cvh_mode_lite_smoke)
        if(TARGET cvh_legacy_core)
            add_test(NAME cvh_legacy_core_smoke COMMAND cvh_legacy_core_smoke)
            add_test(NAME cvh_lite_pipeline_smoke COMMAND cvh_lite_pipeline_smoke)
            add_test(NAME cvh_mode_full_smoke COMMAND cvh_mode_full_smoke)
        endif()
    endif()

    if(TARGET cvh_legacy_core)
        add_executable(cvh_test_core
            test/core/core_ops_test.cpp
            test/core/mat_channel_contract_test.cpp
            test/core/mat_layout_semantics_test.cpp
            test/core/mat_opencv_compat_test.cpp
            test/core/mat_submat_test.cpp
            test/core/mat_upstream_channel_port_test.cpp
            test/core/mat_contract_test.cpp
        )
        target_link_libraries(cvh_test_core PRIVATE cvh::legacy_core gtest_main)
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

install(DIRECTORY include/ DESTINATION include)
