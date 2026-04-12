cmake_minimum_required(VERSION 3.31)
project(marian-rknn VERSION 1.0 LANGUAGES CXX)

set(COPYRIGHT "Copyright (c) 2026 Tristan Penman")
set(IDENTIFIER "com.tristanpenman.marian-rknn")

# Force out-of-source builds
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# C++17 is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# FetchContent: Easier than Git submodules
include(FetchContent)

# Implements FetchContent steps directly instead of through a sub-build
set(CMAKE_POLICY_DEFAULT_CMP0168 NEW)

# Default to Debug for single-config generators if user didn't specify anything
get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT isMultiConfig AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Add -O0 to remove optimizations in debug builds
IF(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
ENDIF(CMAKE_COMPILER_IS_GNUCC)

option(MARIAN_RKNN_BUILD_BENCHMARK "Build benchmark executable" ON)

if(ANDROID)
    set(MARIAN_RKNN_TARGET ${CMAKE_ANDROID_ARCH_ABI})
    set(DEFAULT_RKNN_RUNTIME_LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/rknpu2/lib-android/${MARIAN_RKNN_TARGET}/librknnrt.so)
else()
    set(MARIAN_RKNN_TARGET aarch64)
    set(DEFAULT_RKNN_RUNTIME_LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/rknpu2/lib/${MARIAN_RKNN_TARGET}/librknnrt.so)
endif()

set(RKNN_RUNTIME_LIB ${DEFAULT_RKNN_RUNTIME_LIB} CACHE FILEPATH
    "Path to librknnrt.so for the selected target")
set(RKNN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/rknpu2/include CACHE PATH
    "Path containing rknn_api.h")

if(NOT EXISTS ${RKNN_RUNTIME_LIB})
    message(FATAL_ERROR
        "RKNN runtime library not found: ${RKNN_RUNTIME_LIB}. "
        "Set -DRKNN_RUNTIME_LIB=<path-to-librknnrt.so> (for Android use arm64-v8a runtime).")
endif()

if(NOT EXISTS ${RKNN_INCLUDE_DIR}/rknn_api.h)
    message(FATAL_ERROR
        "RKNN headers not found in ${RKNN_INCLUDE_DIR}. "
        "Set -DRKNN_INCLUDE_DIR=<path-containing-rknn_api.h>.")
endif()

# Avoid building shared libraries
set(BUILD_SHARED_LIBS OFF)

# -----------------------------------------------------------------------------
# Eigen
# -----------------------------------------------------------------------------

FetchContent_Declare(eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 5.0.1
)
FetchContent_MakeAvailable(eigen)

# -----------------------------------------------------------------------------
# JSON for Modern C++
# -----------------------------------------------------------------------------

FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
)
FetchContent_MakeAvailable(nlohmann_json)

# -----------------------------------------------------------------------------
# SentencePiece
# -----------------------------------------------------------------------------

set(SPM_ENABLE_SHARED OFF CACHE BOOL INTERNAL)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL INTERNAL)
FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece
    GIT_TAG 11051e3b73b3a6222a52acd720e39805dc7545ab
    EXCLUDE_FROM_ALL
)

# Aliased to sentencepiece-static
FetchContent_MakeAvailable(sentencepiece)
set(LIBSENTENCEPIECE_INCLUDES ${sentencepiece_SOURCE_DIR}/src)

# Suppress warning in sentencepiece build
target_compile_options(sentencepiece-static PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wno-stringop-overflow>
)

set(MARIAN_COMMON_SOURCES
    src/file_utils.cc
    src/logger.cc
    src/marian_rknn.cc
    src/rknn_utils.cc
)

set(MARIAN_COMMON_LIBS
    ${RKNN_RUNTIME_LIB}
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    sentencepiece
)

if(ANDROID)
    list(APPEND MARIAN_COMMON_LIBS log)
else()
    list(APPEND MARIAN_COMMON_LIBS dl)
endif()

set(MARIAN_COMMON_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${RKNN_INCLUDE_DIR}
    ${LIBSENTENCEPIECE_INCLUDES}
)

# -----------------------------------------------------------------------------
# Main executable
# -----------------------------------------------------------------------------

add_executable(${PROJECT_NAME}
    src/main.cc
    ${MARIAN_COMMON_SOURCES}
)

target_link_libraries(${PROJECT_NAME} PRIVATE ${MARIAN_COMMON_LIBS})

# Strip debug symbols from release builds
target_link_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Release>:-Wl,--strip-all>
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${MARIAN_COMMON_INCLUDES}
)

# -----------------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------------

if(MARIAN_RKNN_BUILD_BENCHMARK)
    add_executable(${PROJECT_NAME}-benchmark
        src/benchmark.cc
        ${MARIAN_COMMON_SOURCES}
    )

    target_link_libraries(${PROJECT_NAME}-benchmark PRIVATE ${MARIAN_COMMON_LIBS})

    target_include_directories(${PROJECT_NAME}-benchmark PRIVATE
        ${MARIAN_COMMON_INCLUDES}
    )
endif()

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

include(CTest)
if(BUILD_TESTING AND NOT ANDROID)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
    )
    FetchContent_MakeAvailable(googletest)

    add_executable(${PROJECT_NAME}-tests
        tests/file_utils_test.cc
        tests/marian_rknn_test.cc
        ${MARIAN_COMMON_SOURCES}
    )

    target_link_libraries(${PROJECT_NAME}-tests PRIVATE
        ${MARIAN_COMMON_LIBS}
        GTest::gtest_main
    )

    target_include_directories(${PROJECT_NAME}-tests PRIVATE
        ${MARIAN_COMMON_INCLUDES}
    )
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Generator: ${CMAKE_GENERATOR} (multi-config: ${isMultiConfig})")
message(STATUS "Target: ${MARIAN_RKNN_TARGET}")
message(STATUS "RKNN runtime: ${RKNN_RUNTIME_LIB}")
