#!/usr/bin/env bash
# 一键批量转换并生成 Markdown 性能评测报告
# 用法:
#   bash batch_convert_trt.sh [模型目录] [选项]
# 示例:
#   bash batch_convert_trt.sh /home/nvidia/qwen35-orin/models
#   bash batch_convert_trt.sh /home/nvidia/qwen35-orin/models --skip-existing

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
MODELS_DIR="${1:-$SCRIPT_DIR/models}"

# 如果第一个参数是目录，则 shift
if [[ $# -gt 0 && -d "$1" ]]; then
    MODELS_DIR="$1"
    shift
fi

echo "=========================================================="
echo "🚀 启动 Jetson Orin 批量模型 TensorRT 编译流水线"
echo "📂 目标目录: $MODELS_DIR"
echo "=========================================================="

# 优先解锁 Orin 硬件全性能锁频（如果具备 sudo 权限）
if command -v jetson_clocks >/dev/null 2>&1; then
    echo "[优化] 检测到 Jetson 平台，建议确保处于最高功耗模式："
    echo "       可手动执行: sudo nvpmodel -m 0 && sudo jetson_clocks"
fi

# 调用 Python 执行扫描、转换、指标解析与 Markdown 报告生成
python3 "$SCRIPT_DIR/batch_convert_trt.py" "$MODELS_DIR" "$@"
