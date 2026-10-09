#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
🚀 YOLO26 (YOLOv26) 系列模型自动下载与 ONNX 导出工具
支持：
  1. 目标检测 (Detection): yolo26n.pt / yolo26s.pt -> models/yolo/
  2. 人体姿态 (Pose):      yolo26n-pose.pt        -> models/pose/
  3. 人脸检测 (Face):      yolo26n-face.pt        -> models/insightface/ (及 yolo/)
特性：
  - 针对 Jetson Orin 与 TensorRT 优化：固定输入尺寸 640x640，Static Batch=1
  - 原生端到端（NMS-Free & DFL-Free），无需冗余的 TensorRT NMS Plugin
  - 自动归类存入对应子目录，可无缝对接 batch_convert_trt.py 一键编译评测
=============================================================================
"""

import os
import sys
import shutil
import argparse
import urllib.request
from pathlib import Path

# YOLO26 模型配置字典
MODEL_CONFIGS = [
    {
        "name": "yolo26n",
        "task": "detect",
        "description": "YOLO26 Nano 目标检测 (COCO预训练, 2.4M参数)",
        "pt_name": "yolo26n.pt",
        "category": "yolo",
        "imgsz": 640,
        "urls": [
            "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
            "https://github.com/ultralytics/assets/releases/latest/download/yolo26n.pt"
        ]
    },
    {
        "name": "yolo26s",
        "task": "detect",
        "description": "YOLO26 Small 目标检测 (COCO预训练, 9.5M参数)",
        "pt_name": "yolo26s.pt",
        "category": "yolo",
        "imgsz": 640,
        "urls": [
            "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt",
            "https://github.com/ultralytics/assets/releases/latest/download/yolo26s.pt"
        ]
    },
    {
        "name": "yolo26n-pose",
        "task": "pose",
        "description": "YOLO26 Nano 人体姿态估计 (17关键点, 2.7M参数)",
        "pt_name": "yolo26n-pose.pt",
        "category": "pose",
        "imgsz": 640,
        "urls": [
            "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt",
            "https://github.com/ultralytics/assets/releases/latest/download/yolo26n-pose.pt"
        ]
    },
    {
        "name": "yolo26n-face",
        "task": "detect",
        "description": "YOLO26 Nano 人脸检测 (WIDER FACE微调, 2.4M参数)",
        "pt_name": "yolo26n-face.pt",
        "category": "insightface",  # 默认放入人脸组，也会同步放置副本到 yolo 组
        "imgsz": 640,
        "urls": [
            "https://huggingface.co/ammirosoh/yolo26n-face/resolve/main/best.pt",
            "https://huggingface.co/theminji/yolo26n-face-detection/resolve/main/weights/best.pt",
            "https://huggingface.co/Shubham12864/YOLO26n-face/resolve/main/yolo26_widerdataset.pt"
        ],
        "hf_repo": "ammirosoh/yolo26n-face",
        "hf_file": "best.pt"
    },
    {
        "name": "yolo11n-face",
        "task": "detect",
        "description": "YOLO11 Nano 人脸检测 (WIDER FACE微调, 2.6M参数)",
        "pt_name": "yolo11n-face.pt",
        "category": "insightface",  # 默认放入人脸组，也会同步放置副本到 yolo 组
        "imgsz": 640,
        "urls": [
            "https://huggingface.co/AdamCodd/YOLOv11n-face-detection/resolve/main/model.pt",
            "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11n-face.pt",
            "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.pt"
        ],
        "hf_repo": "AdamCodd/YOLOv11n-face-detection",
        "hf_file": "model.pt"
    }
]

def check_dependencies():
    """检查环境依赖"""
    try:
        import ultralytics
        from ultralytics import YOLO
        print(f"✅ 检测到 Ultralytics 库已安装 (版本: {ultralytics.__version__})")
    except ImportError:
        print("❌ 未检测到 ultralytics 库！正在尝试提示安装...")
        print("请在终端运行：pip install -U ultralytics onnx onnxslim huggingface_hub")
        sys.exit(1)

def download_file(url_list, target_path: Path, hf_repo=None, hf_file=None):
    """从给定的 URL 列表或 HuggingFace Hub 尝试下载文件"""
    if target_path.exists() and target_path.stat().st_size > 1024 * 1024:
        print(f"📦 已存在权重文件: {target_path.name} ({target_path.stat().st_size / 1024 / 1024:.2f} MB)，跳过网络下载")
        return True

    # 优先尝试 huggingface_hub（若指定了 repo）
    if hf_repo and hf_file:
        try:
            from huggingface_hub import hf_hub_download
            print(f"🤗 正在通过 huggingface_hub ({hf_repo}) 获取 {hf_file}...")
            cached_file = hf_hub_download(repo_id=hf_repo, filename=hf_file)
            if cached_file and Path(cached_file).exists():
                shutil.copy2(cached_file, str(target_path))
                print(f"✅ 成功从 Hugging Face 获取权重: {target_path.name}")
                return True
        except Exception as e:
            print(f"ℹ️ HuggingFace Hub 备选直链模式: {e}")

    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'}
    for url in url_list:
        try:
            print(f"🌐 正在从 {url} 下载...")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as response, open(target_path, 'wb') as out_file:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                chunk_size = 1024 * 1024
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    out_file.write(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r   进度: {downloaded / 1024 / 1024:.2f}MB / {total_size / 1024 / 1024:.2f}MB ({percent:.1f}%)", end="", flush=True)
                print()
            if target_path.stat().st_size > 100 * 1024:
                print(f"✅ 成功下载: {target_path.name}")
                return True
            else:
                print(f"⚠️ 下载文件大小异常 ({target_path.stat().st_size} 字节)，尝试下一个源...")
                target_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"⚠️ 从 {url} 下载失败: {e}")
            target_path.unlink(missing_ok=True)

    return False

def export_to_onnx(pt_path: Path, output_onnx_path: Path, imgsz=640):
    """
    使用 Ultralytics 将 PyTorch 权重导出为标准的 TensorRT 友好型 ONNX 模型
    参数配置：
      - imgsz: (640, 640)
      - dynamic: False (Jetson Orin 优先使用固定尺寸以获得最高性能)
      - simplify: True (调用 onnxslim / onnxsim 简化算子图)
      - opset: 17
    """
    from ultralytics import YOLO
    
    print(f"⚙️ 正在导出 ONNX: {pt_path.name} -> {output_onnx_path.name} (imgsz={imgsz})")
    
    # 临时工作目录
    temp_work_dir = pt_path.parent
    
    try:
        model = YOLO(str(pt_path))
        # 执行导出
        res_path_str = model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=False,
            simplify=True,
            opset=17,
            batch=1
        )
        
        generated_onnx = Path(res_path_str)
        if not generated_onnx.exists():
            # 兼容有些版本可能直接在同一目录下生成同名 .onnx
            fallback_onnx = pt_path.with_suffix(".onnx")
            if fallback_onnx.exists():
                generated_onnx = fallback_onnx

        if generated_onnx.exists():
            output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(generated_onnx), str(output_onnx_path))
            print(f"🎉 导出成功并存入: {output_onnx_path} ({output_onnx_path.stat().st_size / 1024 / 1024:.2f} MB)")
            return True
        else:
            print(f"❌ 导出后未找到输出文件: {generated_onnx}")
            return False
            
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        return False

def verify_onnx(onnx_path: Path):
    """检查 ONNX 模型输入输出节点信息"""
    try:
        import onnx
        model = onnx.load(str(onnx_path), load_external_data=False)
        inp = model.graph.input[0]
        dims = [d.dim_value if d.dim_value > 0 else 'dyn' for d in inp.type.tensor_type.shape.dim]
        out = model.graph.output[0]
        out_dims = [d.dim_value if d.dim_value > 0 else 'dyn' for d in out.type.tensor_type.shape.dim]
        print(f"   [验证] 输入节点 '{inp.name}': {dims} | 输出节点 '{out.name}': {out_dims}")
    except Exception as e:
        pass

def main():
    parser = argparse.ArgumentParser(description="YOLO26 系列模型自动化下载与 ONNX 导出工具")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="/home/nvidia/qwen35-orin/tiny-models/models",
        help="模型根目录 (包含 yolo/, pose/, insightface/ 等分类子目录)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./.cache_weights",
        help="临时下载权重缓存目录"
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="仅下载 .pt 权重，不执行 ONNX 导出"
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="指定仅处理某些模型，英文逗号分隔，如: yolo26n,yolo26n-pose"
    )
    args = parser.parse_args()

    check_dependencies()
    from ultralytics import YOLO

    models_root = Path(args.models_dir).resolve()
    cache_root = Path(args.cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 YOLO26 自动化下载与 ONNX 部署导出")
    print(f"📂 模型输出根目录: {models_root}")
    print(f"💾 权重缓存目录  : {cache_root}")
    print("=" * 70)

    filter_names = [x.strip() for x in args.only.split(",") if x.strip()]

    for cfg in MODEL_CONFIGS:
        name = cfg["name"]
        if filter_names and name not in filter_names:
            continue

        print(f"\n▶ 正在处理: {name} ({cfg['description']})")
        pt_path = cache_root / cfg["pt_name"]
        
        # 1. 下载或加载权重
        if not pt_path.exists():
            # 优先尝试通过 Ultralytics 内置自动机制下载
            try:
                print(f"⏳ 正在尝试通过 Ultralytics 内置接口加载 {cfg['pt_name']}...")
                _ = YOLO(cfg["pt_name"])
                # 检查是否下载到当前目录或 Ultralytics 缓存中
                downloaded_default = Path(cfg["pt_name"])
                if downloaded_default.exists():
                    shutil.move(str(downloaded_default), str(pt_path))
            except Exception as e:
                print(f"ℹ️ 内置自动拉取未命中或需要官方地址，改用备用直链: {e}")

        # 若仍不存在，使用预配置直链列表下载
        if not pt_path.exists():
            download_success = download_file(
                cfg["urls"], 
                pt_path, 
                hf_repo=cfg.get("hf_repo"), 
                hf_file=cfg.get("hf_file")
            )
            if not download_success:
                print(f"❌ 无法下载 {name} 权重，跳过此模型。")
                continue

        if args.skip_export:
            continue

        # 2. 导出到目标分类目录
        target_dir = models_root / cfg["category"]
        target_dir.mkdir(parents=True, exist_ok=True)
        target_onnx = target_dir / f"{name}.onnx"

        success = export_to_onnx(pt_path, target_onnx, imgsz=cfg["imgsz"])
        if success:
            verify_onnx(target_onnx)
            
            # 针对人脸检测模型，额外在 yolo/ 目录下建立副本或链接，便于不同视角的管理
            if name.endswith("-face"):
                yolo_dir = models_root / "yolo"
                yolo_dir.mkdir(parents=True, exist_ok=True)
                yolo_face_onnx = yolo_dir / f"{name}.onnx"
                try:
                    shutil.copy2(str(target_onnx), str(yolo_face_onnx))
                    print(f"📋 同步副本至 yolo 分组: {yolo_face_onnx}")
                except Exception:
                    pass

    print("\n" + "=" * 70)
    print("✨ 所有指定的 YOLO26 模型已准备就绪！")
    print(f"👉 接下来可在终端运行以下命令进行增量 TensorRT 转换与测速：")
    print(f"   python tiny-models/batch_convert_trt.py {models_root} --skip-existing")
    print("=" * 70)

if __name__ == "__main__":
    main()
