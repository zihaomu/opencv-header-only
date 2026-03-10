cmake_minimum_required(VERSION 3.16)

project(opencv_header_only VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

option(CVH_BUILD_LEGACY_CORE "Build temporary src/core implementation library" ON)
option(CVH_BUILD_BACKEND_KERNEL_SOURCES "Build sources requiring legacy backend kernels" OFF)
option(CVH_BUILD_SMOKE_TESTS "Build smoke test targets" ON)
option(CVH_BUILD_LEGACY_TESTS "Build old migrated tests (not wired yet)" OFF)

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


if(CVH_BUILD_SMOKE_TESTS)
    add_executable(cvh_header_compile_smoke test/smoke/cvh_header_compile_smoke.cpp)
    target_link_libraries(cvh_header_compile_smoke PRIVATE cvh::headers)

    if(CVH_BUILD_LEGACY_CORE)
        add_executable(cvh_legacy_core_smoke test/smoke/cvh_legacy_core_smoke.cpp)
        target_link_libraries(cvh_legacy_core_smoke PRIVATE cvh::legacy_core)
    endif()

    if(BUILD_TESTING)
        add_test(NAME cvh_header_compile_smoke COMMAND cvh_header_compile_smoke)
        if(CVH_BUILD_LEGACY_CORE)
            add_test(NAME cvh_legacy_core_smoke COMMAND cvh_legacy_core_smoke)
        endif()
    endif()
endif()

if(CVH_BUILD_LEGACY_TESTS)
    message(WARNING "CVH_BUILD_LEGACY_TESTS is ON, but legacy tests are not wired yet.")
endif()

install(DIRECTORY include/ DESTINATION include)
