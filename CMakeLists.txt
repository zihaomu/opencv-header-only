cmake_minimum_required(VERSION 3.16)

project(opencv_header_only VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

option(CVH_BUILD_LEGACY_CORE "Build temporary src/core implementation library" OFF)
option(CVH_BUILD_BACKEND_KERNEL_SOURCES "Build sources requiring legacy backend kernels" OFF)
option(CVH_BUILD_SMOKE_TESTS "Build smoke test targets" ON)
option(CVH_BUILD_LEGACY_TESTS "Build migrated core test subset" OFF)

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

if(CVH_BUILD_LEGACY_CORE OR CVH_BUILD_LEGACY_TESTS)
    add_library(cvh_legacy_core STATIC ${CVH_LEGACY_CORE_SOURCES})
    add_library(cvh::legacy_core ALIAS cvh_legacy_core)
    target_compile_features(cvh_legacy_core PUBLIC cxx_std_17)
    target_include_directories(cvh_legacy_core
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernel
            ${CMAKE_CURRENT_SOURCE_DIR}/include/cvh/3rdparty/xsimd/include
    )
endif()

if(CVH_BUILD_SMOKE_TESTS)
    add_executable(cvh_header_compile_smoke test/smoke/cvh_header_compile_smoke.cpp)
    target_link_libraries(cvh_header_compile_smoke PRIVATE cvh::headers)

    add_executable(cvh_include_only_smoke test/smoke/cvh_include_only_smoke.cpp)
    target_include_directories(cvh_include_only_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(cvh_include_only_smoke PRIVATE cxx_std_17)

    if(TARGET cvh_legacy_core)
        add_executable(cvh_legacy_core_smoke test/smoke/cvh_legacy_core_smoke.cpp)
        target_link_libraries(cvh_legacy_core_smoke PRIVATE cvh::legacy_core)
    endif()

    if(BUILD_TESTING)
        add_test(NAME cvh_header_compile_smoke COMMAND cvh_header_compile_smoke)
        add_test(NAME cvh_include_only_smoke COMMAND cvh_include_only_smoke)
        if(TARGET cvh_legacy_core)
            add_test(NAME cvh_legacy_core_smoke COMMAND cvh_legacy_core_smoke)
        endif()
    endif()
endif()

if(CVH_BUILD_LEGACY_TESTS)
    add_subdirectory(test/3rdparty/googletest EXCLUDE_FROM_ALL)

    add_executable(cvh_core_basic_tests
        test/core/core_basic_test.cpp
    )
    target_link_libraries(cvh_core_basic_tests PRIVATE cvh::legacy_core gtest_main)
    target_compile_features(cvh_core_basic_tests PRIVATE cxx_std_17)
    target_compile_definitions(cvh_core_basic_tests PRIVATE M_ROOT_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
    target_include_directories(cvh_core_basic_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/test
    )

    if(BUILD_TESTING)
        add_test(NAME cvh_core_basic_tests COMMAND cvh_core_basic_tests)
    endif()
endif()

install(DIRECTORY include/ DESTINATION include)
