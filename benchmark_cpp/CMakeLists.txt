cmake_minimum_required(VERSION 3.14)
project(jetson_serial_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# 寻找 CUDA
find_package(CUDA REQUIRED)

# 寻找 OpenCV
find_package(OpenCV REQUIRED)

# 自动定位 Jetson Orin 上的 TensorRT 路径
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS
    /usr/include/aarch64-linux-gnu
    /usr/local/cuda/include
    /usr/include
    ${CUDA_INCLUDE_DIRS}
)

find_library(TENSORRT_LIBRARY nvinfer
    PATHS
    /usr/lib/aarch64-linux-gnu
    /usr/local/cuda/lib64
    /usr/lib
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

if (NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "未在系统中找到 TensorRT (NvInfer.h / libnvinfer.so)！请确认已安装 JetPack 或 TensorRT。")
endif()

message(STATUS "找到 TensorRT 头文件: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "找到 TensorRT 动态库: ${TENSORRT_LIBRARY}")
message(STATUS "找到 OpenCV 版本    : ${OpenCV_VERSION}")
message(STATUS "找到 CUDA 路径      : ${CUDA_TOOLKIT_ROOT_DIR}")

include_directories(
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(test_serial_trt main.cpp)

target_link_libraries(test_serial_trt
    ${TENSORRT_LIBRARY}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    pthread
    dl
)
