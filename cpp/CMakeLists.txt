cmake_minimum_required(VERSION 3.10)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project(marian-rknn)

set(TARGET_LIB_ARCH aarch64)

set(RKNN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/rknpu2)
set(LIBRKNNRT ${RKNN_PATH}/lib/${TARGET_LIB_ARCH}/librknnrt.so)
set(LIBRKNNRT_INCLUDES ${RKNN_PATH}/include)

set(TIMER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/timer)
set(LIBTIMER_INCLUDES ${TIMER_PATH})

add_executable(${PROJECT_NAME}
    bpe_tools.cc
    file_utils.cc
    main.cc
    marian_rknn.cc
    rknn_utils.cc
)

target_link_libraries(${PROJECT_NAME}
    ${LIBRKNNRT}
    dl
)

target_link_libraries(${PROJECT_NAME}
    pthread
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${LIBTIMER_INCLUDES}
)
