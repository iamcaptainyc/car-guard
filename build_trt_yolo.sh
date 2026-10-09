#!/usr/bin/env bash
# Convert a single ONNX model directly to a TensorRT Engine on Jetson Orin.
# Input : /path/to/model.onnx (单文件输入)
# Output: /path/to/model.engine (单文件输出)

set -Eeuo pipefail

DEFAULT_BUILD_DIR="${EDGELLM_BUILD_DIR:-/home/nvidia/qwen35-orin/TensorRT-Edge-LLM/build}"
INPUT_FILE=""
OUTPUT_FILE=""
BUILD_DIR="$DEFAULT_BUILD_DIR"
INPUT_NAME="images"
IMG_HEIGHT=640
IMG_WIDTH=640
BATCH_SIZE=1
MAX_BATCH_SIZE=1
PRECISION="fp16"
WORKSPACE_MB=2048
DYNAMIC_BATCH=false

usage() {
    cat <<EOF
用法：
  $(basename "$0") <输入.onnx文件> [选项]
  $(basename "$0") --input <输入.onnx文件> [--output <输出.engine文件>] [选项]

基础用法（输入单文件，默认在同目录输出同名 .engine）：
  bash $(basename "$0") /path/to/yolov8n.onnx

默认 build 目录：
  $DEFAULT_BUILD_DIR

参数说明：
  -i, --input <文件>             必须。输入的单文件（.onnx 格式，也支持 .pt 自动导出）
  -o, --output <文件>            可选。输出的 .engine 文件路径（默认：<输入名>.engine）
  --build-dir <目录>             TensorRT-Edge-LLM build目录（默认 $DEFAULT_BUILD_DIR）
  --precision <fp16|fp32|int8>   编译精度（默认 fp16，Orin 性能最优）
  --img-size <H> [W]             输入图像尺寸（默认 640 640）
  --batch-size N                 推理 Batch 批大小（默认 1）
  --dynamic-batch                启用动态 Batch 范围（1 ~ max-batch-size）
  --max-batch-size N             动态 Batch 最大值（默认等于 batch-size）
  --input-name <节点名>          ONNX 输入节点名称（YOLO 默认: images，部分为: input）
  --workspace N                  构建显存工作区大小（MiB，默认 2048）

示例：
  # 1. 最简转换：单文件输入，同目录下输出 yolov8n.engine
  bash $(basename "$0") /home/nvidia/models/yolov8n.onnx

  # 2. 指定输出路径与精度
  bash $(basename "$0") -i ./yolo11n.onnx -o ./engines/yolo11n_fp16.engine --precision fp16

  # 3. 姿态模型指定输入尺寸与节点名
  bash $(basename "$0") ./rtmpose.onnx --img-size 256 192 --input-name input
EOF
}

log() { printf '\n[TRT 单文件转换] %s\n' "$*"; }
warn() { printf '\n[TRT 单文件转换][警告] %s\n' "$*" >&2; }
die() { printf '\n[TRT 单文件转换][失败] %s\n' "$*" >&2; exit 1; }
run() {
    printf '\n+ '
    printf '%q ' "$@"
    printf '\n'
    "$@"
}
require_file() { [[ -f "$1" ]] || die "找不到文件：$1"; }
require_dir() { [[ -d "$1" ]] || die "找不到目录：$1"; }

on_error() {
    local exit_code=$?
    printf '\n[TRT 单文件转换][失败] 第 %s 行命令执行出错（exit=%s）：%s\n' \
        "$1" "$exit_code" "$2" >&2
    exit "$exit_code"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

# 支持第一个参数直接传文件路径（如：bash build.sh model.onnx）
if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
    INPUT_FILE="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input) INPUT_FILE="${2:-}"; shift 2 ;;
        -o|--output) OUTPUT_FILE="${2:-}"; shift 2 ;;
        --build-dir) BUILD_DIR="${2:-}"; shift 2 ;;
        --precision) PRECISION="${2:-}"; shift 2 ;;
        --input-name) INPUT_NAME="${2:-}"; shift 2 ;;
        --img-size)
            IMG_HEIGHT="${2:-}"
            if [[ $# -gt 2 && ! "${3:-}" =~ ^-- ]]; then
                IMG_WIDTH="$3"
                shift 3
            else
                IMG_WIDTH="$IMG_HEIGHT"
                shift 2
            fi
            ;;
        --batch-size) BATCH_SIZE="${2:-}"; shift 2 ;;
        --max-batch-size) MAX_BATCH_SIZE="${2:-}"; shift 2 ;;
        --dynamic-batch) DYNAMIC_BATCH=true; shift 1 ;;
        --workspace) WORKSPACE_MB="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "未知参数：$1（使用 --help 查看帮助）" ;;
    esac
done

[[ -n "$INPUT_FILE" ]] || { usage >&2; die "必须提供输入模型文件"; }
require_file "$INPUT_FILE"

# 自动处理输出文件路径（未指定时，默认同目录下生成 .engine）
if [[ -z "$OUTPUT_FILE" ]]; then
    OUTPUT_FILE="${INPUT_FILE%.*}.engine"
fi

OUTPUT_DIR="$(dirname "$OUTPUT_FILE")"
mkdir -p "$OUTPUT_DIR"

if [[ -n "$BUILD_DIR" && -d "$BUILD_DIR" ]]; then
    BUILD_DIR="$(cd "$BUILD_DIR" && pwd -P)"
    PLUGIN_PATH="$BUILD_DIR/libNvInfer_edgellm_plugin.so"
else
    PLUGIN_PATH=""
fi

# 自动定位 trtexec 可执行文件：优先使用 build 目录下的，其次系统路径
TRTEXEC_BIN=""
CANDIDATES=()
if [[ -n "$BUILD_DIR" && -d "$BUILD_DIR" ]]; then
    CANDIDATES+=("$BUILD_DIR/bin/trtexec" "$BUILD_DIR/trtexec")
fi
CANDIDATES+=("/usr/src/tensorrt/bin/trtexec")

for candidate in "${CANDIDATES[@]}"; do
    if [[ -x "$candidate" ]]; then
        TRTEXEC_BIN="$candidate"
        break
    fi
done
if [[ -z "$TRTEXEC_BIN" ]]; then
    if command -v trtexec >/dev/null 2>&1; then
        TRTEXEC_BIN="$(command -v trtexec)"
    else
        die "未找到可执行的 trtexec（系统路径 /usr/src/tensorrt/bin/trtexec 不存在或不可执行）"
    fi
fi

# 处理 .pt 格式自动先导为 .onnx
COMPILE_ONNX="$INPUT_FILE"
EXT="${INPUT_FILE##*.}"
if [[ "$EXT" == "pt" ]]; then
    COMPILE_ONNX="${INPUT_FILE%.*}.onnx"
    log "检测到 PyTorch 格式 (.pt)，调用 Python 导出 ONNX..."
    python3 - <<EOF
from ultralytics import YOLO
print("[导出] 加载模型: $INPUT_FILE")
model = YOLO("$INPUT_FILE")
model.export(
    format="onnx",
    imgsz=($IMG_HEIGHT, $IMG_WIDTH),
    dynamic=$DYNAMIC_BATCH,
    opset=12,
    simplify=True
)
print("[导出] 导出完成: $COMPILE_ONNX")
EOF
    require_file "$COMPILE_ONNX"
fi

log "开始编译 TensorRT Engine"
printf '输入模型文件 : %s\n' "$COMPILE_ONNX"
printf '输出 Engine  : %s\n' "$OUTPUT_FILE"
printf '构建 Build 目录: %s\n' "$BUILD_DIR"
printf '使用 trtexec : %s\n' "$TRTEXEC_BIN"
printf '编译精度模式 : %s\n' "$PRECISION"
printf '输入分辨率   : %sx%s (节点: %s)\n' "$IMG_HEIGHT" "$IMG_WIDTH" "$INPUT_NAME"

# 架构与环境检查（如果指定了 EdgeLLM BUILD_DIR）
if [[ -n "$BUILD_DIR" && -d "$BUILD_DIR" ]]; then
    if ! grep -q '87' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
        warn "未能从 CMakeCache.txt 确认 SM87；如果使用了 EdgeLLM Plugin，请确认这是为 Jetson Orin 构建的目录。"
    fi

    for flags_file in \
        "$BUILD_DIR/cpp/CMakeFiles/edgellmCore.dir/flags.make" \
        "$BUILD_DIR/cpp/CMakeFiles/edgellmKernels.dir/flags.make"; do
        if [[ -f "$flags_file" ]] && grep -q -- '-DEXCLUDE_SM_87' "$flags_file"; then
            die "当前 build 排除了 SM87：$flags_file"
        fi
    done
fi

# 内存与磁盘状态打印
if command -v df >/dev/null 2>&1; then
    df -h "$(dirname "$COMPILE_ONNX")" || true
fi
if command -v free >/dev/null 2>&1; then
    free -h || true
fi

# 注入动态链接库与插件路径
export EDGELLM_PLUGIN_PATH="$PLUGIN_PATH"
export LD_LIBRARY_PATH="$BUILD_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 组装编译命令
CMD=(
    "$TRTEXEC_BIN"
    "--onnx=$COMPILE_ONNX"
    "--saveEngine=$OUTPUT_FILE"
    "--memPoolSize=workspace:${WORKSPACE_MB}"
    "--warmUp=200"
    "--iterations=50"
    "--verbose=false"
)

# 挂载自定义 Plugin
if [[ -f "$PLUGIN_PATH" ]]; then
    CMD+=("--plugins=$PLUGIN_PATH")
fi

# 精度配置
case "$PRECISION" in
    fp16) CMD+=("--fp16") ;;
    int8) CMD+=("--int8") ;;
    fp32) ;; # 默认即为 fp32
    *) die "不支持的精度：$PRECISION（仅支持 fp16, fp32, int8）" ;;
esac

# 形状配置（兼容 TensorRT 10+ 静态/动态模型规则）
IS_DYNAMIC=$(python3 - <<EOF 2>/dev/null || echo "false"
try:
    import onnx
    model = onnx.load("$COMPILE_ONNX")
    is_dyn = False
    for inp in model.graph.input:
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_param or d.dim_value <= 0:
                is_dyn = True
                break
    print("true" if is_dyn else "false")
except Exception:
    print("false")
EOF
)

if [ "$DYNAMIC_BATCH" = true ] || [ "$IS_DYNAMIC" = "true" ]; then
    log "检测到模型输入包含动态维度，配置 TensorRT Optimization Profile："
    printf 'Profile: min=1x3x%sx%s, opt=%sx3x%sx%s, max=%sx3x%sx%s\n' \
        "$IMG_HEIGHT" "$IMG_WIDTH" "$BATCH_SIZE" "$IMG_HEIGHT" "$IMG_WIDTH" "$MAX_BATCH_SIZE" "$IMG_HEIGHT" "$IMG_WIDTH"
    CMD+=(
        "--minShapes=${INPUT_NAME}:1x3x${IMG_HEIGHT}x${IMG_WIDTH}"
        "--optShapes=${INPUT_NAME}:${BATCH_SIZE}x3x${IMG_HEIGHT}x${IMG_WIDTH}"
        "--maxShapes=${INPUT_NAME}:${MAX_BATCH_SIZE}x3x${IMG_HEIGHT}x${IMG_WIDTH}"
    )
else
    log "检测到模型自带固定静态输入维度 (Static ONNX: 1x3x${IMG_HEIGHT}x${IMG_WIDTH})。"
    log "TensorRT 10 会自动继承模型内部维度，跳过显式 --shapes 传参（避免 Static model 报错）。"
fi

log "正在执行构建并测试吞吐延时..."
run "${CMD[@]}"

require_file "$OUTPUT_FILE"

log "转换完成，Engine 产物信息："
ls -lh "$OUTPUT_FILE"

cat <<EOF

==============================================================
[✓] TensorRT Engine 单文件构建成功！
输出文件路径：$OUTPUT_FILE
==============================================================
EOF
