cmake_minimum_required(VERSION 3.31)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# FetchContent: Easier than Git submodules
include(FetchContent)

# Implements FetchContent steps directly instead of through a sub-build
set(CMAKE_POLICY_DEFAULT_CMP0168 NEW)

project(marian-rknn)

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

set(TARGET_LIB_ARCH aarch64)
set(BUILD_SHARED_LIBS OFF)

# rknn
set(RKNN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/rknpu2)
set(LIBRKNNRT ${RKNN_PATH}/lib/${TARGET_LIB_ARCH}/librknnrt.so)
set(LIBRKNNRT_INCLUDES ${RKNN_PATH}/include)

# timer
set(TIMER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/timer)
set(LIBTIMER_INCLUDES ${TIMER_PATH})

# Eigen
FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)

FetchContent_MakeAvailable(eigen)

# sentencepiece options: statically linked, no tcmalloc
set(SPM_ENABLE_SHARED OFF CACHE BOOL INTERNAL)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL INTERNAL)

# sentencepiece
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

add_executable(${PROJECT_NAME}
    file_utils.cc
    main.cc
    marian_rknn.cc
    rknn_utils.cc
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${LIBRKNNRT}
    dl
    Eigen3::Eigen
    sentencepiece
)

# Strip debug symbols from release builds
target_link_options(${PROJECT_NAME} PRIVATE
  $<$<CONFIG:Release>:-Wl,--strip-all>
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${LIBSENTENCEPIECE_INCLUDES}
    ${LIBTIMER_INCLUDES}
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Generator: ${CMAKE_GENERATOR} (multi-config: ${isMultiConfig})")
