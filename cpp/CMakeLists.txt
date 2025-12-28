cmake_minimum_required(VERSION 3.31)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# FetchContent: Easier than Git submodules
include(FetchContent)

# Implements FetchContent steps directly instead of through a sub-build
set(CMAKE_POLICY_DEFAULT_CMP0168 NEW)

project(marian-rknn)

set(TARGET_LIB_ARCH aarch64)
set(BUILD_SHARED_LIBS OFF)

# rknn
set(RKNN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/rknpu2)
set(LIBRKNNRT ${RKNN_PATH}/lib/${TARGET_LIB_ARCH}/librknnrt.so)
set(LIBRKNNRT_INCLUDES ${RKNN_PATH}/include)

# timer
set(TIMER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/timer)
set(LIBTIMER_INCLUDES ${TIMER_PATH})

# sentencepiece options: statically linked, no tcmalloc
set(SPM_ENABLE_SHARED "OFF")
set(SPM_ENABLE_TCMALLOC "OFF")

# sentencepiece
FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece
    GIT_TAG 11051e3b73b3a6222a52acd720e39805dc7545ab
    EXCLUDE_FROM_ALL
)

# Aliased to sentencepiece-static
FetchContent_MakeAvailable(sentencepiece)

# Suppress warning in sentencepiece build
target_compile_options(sentencepiece-static PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wno-stringop-overflow>
)

add_executable(${PROJECT_NAME}
    bpe_tools.cc
    file_utils.cc
    main.cc
    marian_rknn.cc
    rknn_utils.cc
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${LIBRKNNRT}
    dl
    pthread
    sentencepiece
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${LIBSENTENCEPIECE_INCLUDES}
    ${LIBTIMER_INCLUDES}
)
