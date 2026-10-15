#!/bin/bash
set -e

# 默认 TensorRT-Edge-LLM 源码路径 (可作为参数传入)
EDGELLM_DIR="${1:-/home/nvidia/qwen35-orin/TensorRT-Edge-LLM}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================================="
echo "[*] 开始部署 libvlm_engine.so 原生 C-API 推理内核"
echo "    - 目标源码目录: ${EDGELLM_DIR}"
echo "    - 源文件目录  : ${SCRIPT_DIR}"
echo "=================================================================="

if [ ! -d "${EDGELLM_DIR}/examples" ]; then
    echo "[!] 错误: 未在 ${EDGELLM_DIR} 找到 examples 目录，请传入正确的 TensorRT-Edge-LLM 路径。"
    exit 1
fi

DEST_DIR="${EDGELLM_DIR}/examples/vlm_engine"
mkdir -p "${DEST_DIR}"

echo "[*] 1. 复制 C/C++ 共享库源码与 CMakeLists.txt..."
cp -f "${SCRIPT_DIR}/CMakeLists.txt" "${DEST_DIR}/"
cp -f "${SCRIPT_DIR}/vlm_engine.h" "${DEST_DIR}/"
cp -f "${SCRIPT_DIR}/vlm_engine.cpp" "${DEST_DIR}/"

echo "[*] 2. 检查 examples/CMakeLists.txt 子目录注册..."
if ! grep -q "add_subdirectory(vlm_engine)" "${EDGELLM_DIR}/examples/CMakeLists.txt"; then
    echo "add_subdirectory(vlm_engine)" >> "${EDGELLM_DIR}/examples/CMakeLists.txt"
    echo "[+] 已将 add_subdirectory(vlm_engine) 追加至 examples/CMakeLists.txt"
else
    echo "[i] examples/CMakeLists.txt 已包含 vlm_engine"
fi

echo "[*] 3. 重新生成构建规则与优化编译 (cmake -DCMAKE_BUILD_TYPE=Release .. && make vlm_engine)..."
BUILD_DIR="${EDGELLM_DIR}/build"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake -DCMAKE_BUILD_TYPE=Release ..

make -j$(nproc) vlm_engine

SO_PATH="${BUILD_DIR}/examples/vlm_engine/libvlm_engine.so"
if [ -f "${SO_PATH}" ]; then
    echo "=================================================================="
    echo "[+] 编译成功！"
    echo "    - 共享库路径: ${SO_PATH}"
    echo "    - 大小: $(ls -lh "${SO_PATH}" | awk '{print $5}')"
    echo "    - 导出符号验证:"
    nm -D --defined-only "${SO_PATH}" | grep vlm_ || true
    echo "=================================================================="
else
    echo "[!] 警告: 未在预期位置找到 ${SO_PATH}"
fi
