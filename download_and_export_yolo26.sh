#!/usr/bin/env bash
# ==============================================================================
# Jetson Orin - YOLO26 & YOLO11 系列模型一键下载与部署准备脚本
# 包含：
#   - 基础目标检测: yolo26n.onnx, yolo26s.onnx
#   - 人体姿态估计: yolo26n-pose.onnx
#   - 人脸快速检测: yolo26n-face.onnx, yolo11n-face.onnx
# ==============================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
MODELS_DIR="${1:-$SCRIPT_DIR/models}"

echo "======================================================================"
echo "🚀 正在准备下载与导出 YOLO26 系列模型"
echo "📂 目标模型库: $MODELS_DIR"
echo "======================================================================"

# 1. 确保 Python 依赖就绪 (ultralytics, onnx, onnxslim)
echo "📦 检查并更新 Ultralytics 与 ONNX 工具链..."
pip install -U "ultralytics>=8.4.0" onnx onnxslim huggingface_hub requests --quiet

# 2. 执行自动化下载与导出脚本
python3 "$SCRIPT_DIR/download_and_export_yolo26.py" --models-dir "$MODELS_DIR"

echo ""
echo "======================================================================"
echo "🎯 完成！现在可以执行增量 TensorRT 编译和评测："
echo "   python batch_convert_trt.py \"$MODELS_DIR\" --skip-existing"
echo "======================================================================"
