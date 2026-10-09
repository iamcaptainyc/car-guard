#!/bin/bash
set -e

# =============================================================================
# 🚀 Jetson Orin 原生 C++ 串行基准测试一键编译与运行脚本
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "===================================================================="
echo " 🚀 正在编译 Jetson Orin 原生 C++ TensorRT 串行测速程序 (test_serial_trt)"
echo "===================================================================="

mkdir -p "${SCRIPT_DIR}/build"
cd "${SCRIPT_DIR}/build"

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

echo ""
echo "===================================================================="
echo " ✅ 编译成功！可执行文件位于: ${SCRIPT_DIR}/build/test_serial_trt"
echo "===================================================================="
echo ""
echo "常用运行命令示例:"
echo "1. 默认参数运行 (后台无头模式):"
echo "   cd ${ROOT_DIR} && ./benchmark_cpp/build/test_serial_trt --video your_video.mp4 --headless"
echo ""
echo "2. 自定义指定模型与输出路径:"
echo "   cd ${ROOT_DIR} && ./benchmark_cpp/build/test_serial_trt \\"
echo "       --video sample_fisheye_left.mp4 \\"
echo "       --det models/yolo/yolo11n.engine \\"
echo "       --output tmp/serial_cpp_output.mp4 \\"
echo "       --headless"
echo ""
