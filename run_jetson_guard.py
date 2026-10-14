#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
🚀 Jetson Orin 专属车载露营守卫运行控制台 (run_jetson_guard.py)
专为 NVIDIA Jetson Orin (Orin Nano / Orin NX / AGX Orin) 打造的端到端 TensorRT 部署入口

特性：
1. 硬件状态与功耗模式自检 (Jetson Model, JetPack, nvpmodel, jetson_clocks)
2. 纯 TensorRT .engine 模型全管线挂载 (YOLO 目标检测 + RTMPose 姿态 + OSNet ReID + SCRFD/ArcFace 人脸)
3. 缺失模型自动化一键编译引导 (对接 batch_convert_trt.py 与 trtexec)
4. 多源视频流输入：MP4 视频文件、CSI 车载鱼眼相机 (nvarguscamerasrc)、USB (v4l2) 或 RTSP
5. 帧框同源合一实时渲染、1.0x 无卡顿录制与全维度 JSON 性能诊断报告自动导出
=============================================================================
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import cv2
import numpy as np

from camping_guard.engine import CampingGuardEngine
from camping_guard.types import GuardState, ThreatLevel, VehicleZone
from camping_guard.config import load_vehicle_calibration
from camping_guard.perception.trt_engine import (
    is_jetson,
    get_jetson_info,
    TRTInferenceSession,
    load_inference_backend
)
from camping_guard.perception.detector import ONNXObjectDetector
from camping_guard.perception.pose_estimator import RTMPoseEstimator
from camping_guard.perception.distance import GroundDistanceEstimator
from camping_guard.profiler import PipelineProfiler
from test_video_pipeline import run_video_pipeline


def print_banner(info: dict):
    print("=" * 76)
    print(" 🚀 NVIDIA Jetson Orin 露营守卫系统 - TensorRT Engine 高性能控制台")
    print("=" * 76)
    print(f" 平台设备   : {info.get('model', 'NVIDIA Jetson')}")
    print(f" L4T/驱动   : {info.get('l4t_ver', 'Linux for Tegra')}")
    print(f" TensorRT   : v{info.get('tensorrt_ver', '8.x/10.x')}")
    print(f" CUDA状态   : {'可用 (' + info.get('device_name', '') + ')' if info.get('cuda_available') else '未就绪'}")
    print("-" * 76)


def check_and_recommend_power_mode():
    """检查 Jetson Orin 功耗模式并给出调优建议"""
    if not is_jetson():
        return

    try:
        res = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            print(f"[*] 当前系统功耗模式: {res.stdout.strip()}")
            if "MAXN" not in res.stdout.upper() and "MODE 0" not in res.stdout.upper():
                print("[!] 建议提示: 为获得最强哨兵响应与满血 TensorRT 推理算力，建议执行:")
                print("    sudo nvpmodel -m 0 && sudo jetson_clocks")
    except Exception:
        pass


def check_required_engines(
    models_dir: Path = Path("models"),
    vlm_model_dir: Optional[str] = None,
    vlm_engine: Optional[str] = None,
    custom_detector: Optional[str] = None,
    custom_megadetector: Optional[str] = None,
    custom_cls: Optional[str] = None,
    custom_pose: Optional[str] = None,
    custom_fire: Optional[str] = None,
    enable_fire: bool = True
) -> Dict[str, Optional[Any]]:
    """检查系统所需的各核心 TensorRT Engine / ONNX 模型文件"""
    engine_status = {
        "yolo": None,
        "megadetector": None,
        "cls": None,
        "fire_det": None,
        "pose": None,
        "reid": None,
        "face_det": None,
        "face_rec": None,
        "vlm": None
    }

    # 1. 主目标检测器 (YOLO 与 MegaDetector 统一自由替换)
    chosen_det_input = custom_detector or custom_megadetector
    if chosen_det_input and Path(chosen_det_input).is_file():
        engine_status["detector"] = Path(chosen_det_input)
    else:
        det_cands = [
            models_dir / "yolo" / "yolo11s.engine",
            models_dir / "yolo" / "yolo11n.engine",
            models_dir / "megadetector" / "MDV6-yolov10-c.engine",
            models_dir / "yolo" / "MDV6-yolov10-c.engine",
            models_dir / "yolo" / "yolo11n-fisheye.engine",
            models_dir / "yolo" / "yolo26n.engine",
            models_dir / "yolo" / "yolov8n.engine",
            models_dir / "megadetector" / "MDV6-yolov10-c.onnx",
            models_dir / "yolo" / "MDV6-yolov10-c.onnx",
            models_dir / "yolo" / "yolo11s.onnx",
            models_dir / "yolo" / "yolo11n.onnx"
        ]
        for c in det_cands:
            if c.is_file():
                engine_status["detector"] = c
                break

    engine_status["yolo"] = engine_status["detector"]
    engine_status["megadetector"] = engine_status["detector"]

    # 1.2 二级动物细分类器 (MobileNetV3)
    if custom_cls and Path(custom_cls).is_file():
        engine_status["cls"] = Path(custom_cls)
    else:
        cls_cands = [
            models_dir / "cls" / "mobilenetv3-large.engine",
            models_dir / "cls" / "mobilenetv3.engine",
            models_dir / "cls" / "mobilenetv3-large.onnx",
            models_dir / "cls" / "mobilenetv3.onnx",
        ]
        for c in cls_cands:
            if c.is_file():
                engine_status["cls"] = c
                break

    # 1.3 辅助火灾/烟雾目标检测器 (多流并发，默认开启)
    if enable_fire:
        if custom_fire and Path(custom_fire).is_file():
            engine_status["fire_det"] = Path(custom_fire)
        else:
            fire_cands = [
                models_dir / "fire" / "yolov8n_fire.engine",
                models_dir / "fire" / "yolo11n_fire.engine",
                models_dir / "yolo" / "yolo11n_fire.engine",
                models_dir / "fire" / "fire.engine",
                models_dir / "fire" / "yolov8n_fire.onnx",
                models_dir / "fire" / "yolo11n_fire.onnx",
                models_dir / "fire" / "fire.onnx",
            ]
            for c in fire_cands:
                if c.is_file():
                    engine_status["fire_det"] = c
                    break
            if not engine_status["fire_det"]:
                engine_status["fire_det"] = "builtin:VisualFlameDetector"
    else:
        engine_status["fire_det"] = None

    # 2. RTMPose 人体姿态
    if custom_pose and Path(custom_pose).is_file():
        engine_status["pose"] = Path(custom_pose)
    else:
        pose_cands = [
            models_dir / "pose" / "rtmpose-s.engine",
            models_dir / "pose" / "yolo26n-pose.engine",
        ]
        for c in pose_cands:
            if c.is_file():
                engine_status["pose"] = c
                break

    # 3. OSNet 人员重识别
    reid_cand = models_dir / "reid" / "osnet_x0_25.engine"
    if reid_cand.is_file():
        engine_status["reid"] = reid_cand

    # 4. 人脸检测
    face_det_cand = models_dir / "insightface" / "det_500m.engine"
    if face_det_cand.is_file():
        engine_status["face_det"] = face_det_cand

    # 5. 人脸识别
    face_rec_cand = models_dir / "insightface" / "w600k_mbf.engine"
    if face_rec_cand.is_file():
        engine_status["face_rec"] = face_rec_cand

    # 6. 端侧 Qwen-VLM 按需多模态复核大模型
    if vlm_engine:
        ve_p = Path(vlm_engine).resolve()
        if ve_p.is_file() and ve_p.suffix == ".engine":
            engine_status["vlm"] = ve_p
        elif ve_p.is_dir() and ((ve_p / "config.json").is_file() or (ve_p / "llm.engine").is_file() or any(ve_p.glob("*.engine"))):
            engine_status["vlm"] = ve_p
    elif vlm_model_dir:
        vm_p = Path(vlm_model_dir).resolve()
        for cand in [vm_p / "engines" / "llm", vm_p / "llm", vm_p]:
            if cand.is_dir() and ((cand / "config.json").is_file() or (cand / "llm.engine").is_file() or any(cand.glob("*.engine"))):
                engine_status["vlm"] = cand
                break
    else:
        vlm_default = models_dir / "vlm"
        if vlm_default.is_dir() and (vlm_default / "engines" / "llm").is_dir():
            engine_status["vlm"] = vlm_default / "engines" / "llm"

    return engine_status


def auto_compile_engines(models_dir: Path = Path("models")):
    """调用 batch_convert_trt.py 批量将未编译的 ONNX 编译为 TensorRT Engine"""
    print("\n[*] 正在启动 Jetson Orin 自动化 TensorRT 引擎编译工具 (batch_convert_trt.py)...")
    convert_script = Path("batch_convert_trt.py")
    if not convert_script.is_file():
        print(f"[!] 错误: 未在当前目录找到 {convert_script}，无法自动编译！")
        return False

    cmd = [sys.executable, str(convert_script), str(models_dir), "--precision", "fp16", "--skip-existing"]
    print(f"[*] 执行命令: {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    return ret.returncode == 0


def get_csi_gstreamer_pipeline(
    sensor_id: int = 0,
    capture_w: int = 1920,
    capture_h: int = 1080,
    display_w: int = 640,
    display_h: int = 480,
    framerate: int = 30,
    flip_method: int = 0
) -> str:
    """生成适用于 Jetson Orin 的 CSI 鱼眼摄像头硬件解码 GStreamer 管道字符串"""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_w}, height=(int){capture_h}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_w}, height=(int){display_h}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1"
    )


def main():
    parser = argparse.ArgumentParser(description="Jetson Orin 专属车载露营守卫 - TensorRT 运行控制台")
    parser.add_argument("--video", type=str, default=None,
                        help="测试用本地鱼眼 MP4 视频路径 (若未指定则自动选用内置样本或摄像头)")
    parser.add_argument("--camera-id", type=int, default=1,
                        help="相机机位通道: 0前, 1左, 2右, 3后 (默认 1 左侧后视镜鱼眼)")
    parser.add_argument("--csi", type=int, default=None,
                        help="使用 Jetson 硬件 CSI 鱼眼相机传感器通道编号 (例如: --csi 0 或 --csi 1)")
    parser.add_argument("--usb", type=int, default=None,
                        help="使用 USB 视频摄像头设备编号 (例如: --usb 0 代表 /dev/video0)")
    parser.add_argument("--compile", action="store_true",
                        help="强制调用 batch_convert_trt.py 检查并编译所有 ONNX 为 TensorRT .engine 文件")
    parser.add_argument("--config", type=str, default="vehicle_calib.json",
                        help="全车标定配置文件 (默认: vehicle_calib.json)")
    parser.add_argument("--detector", "--megadetector", type=str, default=None,
                        help="显式指定目标检测器模型路径 (支持 YOLO 或 MegaDetector，如 models/yolo/yolo11s.engine 或 models/megadetector/MDV6-yolov10-c.engine)")
    parser.add_argument("--cls", "--animal-cls", dest="cls_model", type=str, default=None,
                        help="显式指定二级动物细分类模型路径 (如 models/cls/mobilenetv3-large.engine，默认自动发现)")
    parser.add_argument("--pose", type=str, default=None,
                        help="显式指定 RTMPose .engine 模型路径 (如 models/pose/rtmpose-s.engine)")
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值 (默认 0.25)")
    parser.add_argument("--headless", action="store_true", help="无头模式运行 (适用于车载后台守护进程或 SSH 调试)")
    parser.add_argument("--sidebar", action="store_true", default=False, help="保留旧版 OpenCV 绘制的 CPU 侧边栏 (默认已禁用以提升性能)")
    parser.add_argument("--no-sidebar", action="store_true", help="关闭右侧看板，使用极简纯画面显示")
    parser.add_argument("--output", type=str, default=None, help="保存渲染好的视频文件至路径 (如 output.mp4)")
    parser.add_argument("--loop", dest="loop_video", action="store_true", default=None,
                        help="强制循环播放视频")
    parser.add_argument("--no-loop", dest="loop_video", action="store_false",
                        help="单次处理后自动退出，不循环回放 (当指定 --output 时默认开启此项)")
    parser.add_argument("--web", action="store_true", default=True,
                        help="启用内置 Web 实时流仪表盘 (默认开启，远程浏览器直接查看效果)")
    parser.add_argument("--no-web", action="store_true", help="关闭 Web 实时推流仪表盘")
    parser.add_argument("--web-port", type=int, default=8080, help="Web 实时流端口号 (默认: 8080)")
    parser.add_argument("--model-dir", "--vlm-model-dir", dest="vlm_model_dir", type=str, default=None,
                        help="显式指定端侧 Qwen-VLM 模型总目录 (内部须包含 engines/llm 与 engines/visual，如 /home/nvidia/models/Qwen3.5-0.8B)")
    parser.add_argument("--build-dir", "--edgellm-build-dir", "--edgellm-dir", dest="edgellm_dir", type=str, default=None,
                        help="显式指定 TensorRT-Edge-LLM build 目录 (包含 examples/llm/llm_inference，如 /home/nvidia/qwen35-orin/TensorRT-Edge-LLM/build)")
    parser.add_argument("--vlm-engine", type=str, default=None,
                        help="显式指定端侧 Qwen-VLM LLM 引擎目录或路径 (如 /home/nvidia/models/Qwen3.5-0.8B/engines/llm)")
    parser.add_argument("--vlm-visual-engine", type=str, default=None,
                        help="显式指定端侧 Qwen-VLM Visual 视觉编码器引擎目录或路径 (如 /home/nvidia/models/Qwen3.5-0.8B/engines/visual)")
    parser.add_argument("--vlm-timeout", type=float, default=35.0,
                        help="端侧 Qwen-VLM 推理超时时间 (秒，默认 35.0，为 Orin 引擎反序列化加载与生成留足裕量)")
    parser.add_argument("--vlm-mode", type=str, default="auto", choices=["auto", "server", "subprocess"],
                        help="端侧 VLM 推理模式: auto (优先使用常驻持久化服务 vlm_server 零冷启动，自动降级 llm_inference), server (强制常驻服务模式), subprocess (传统一次性 llm_inference 进程模式)")
    parser.add_argument("--vlm-server-port", type=int, default=8090,
                        help="vlm_server 监听的 HTTP 端口 (默认 8090)")
    parser.add_argument("--vlm-idle-timeout", type=int, default=60,
                        help="vlm_server 空闲自动退出卸载显存超时 (秒，默认 60)")
    parser.add_argument("--vlm-server-bin", type=str, default=None,
                        help="显式指定 vlm_server 可执行程序路径 (默认自动在 build 目录检索)")
    parser.add_argument("--realtime", action="store_true", default=True, help="启用真实时间 30 FPS 物理流速控制 (默认开启)")
    parser.add_argument("--fire-detector", "--fire", dest="fire_detector", type=str, default=None,
                        help="显式指定火灾/浓烟目标检测 .engine 模型路径 (默认已启用火焰检测并发)")
    parser.add_argument("--detector-size", type=int, default=None,
                        help="主目标检测器输入分辨率 (如 640 或 1280，默认自动探测模型输入维度)")
    parser.add_argument("--fire-size", type=int, default=None,
                        help="火焰检测器输入分辨率 (如 640 或 1280，默认自动探测模型输入维度)")
    parser.add_argument("--max-frames", type=int, default=None, help="最大测试评估帧数 (达到后自动结束)")
    parser.add_argument("--no-fire", action="store_true",
                        help="显式关闭火焰检测并发 (默认开启)")
    parser.add_argument("--stats-json", type=str, default=None,
                        help="额外指定全维度性能诊断 JSON 报告保存路径 (默认始终自动导出至 logs/latest_perf_report.json)")

    args = parser.parse_args()

    jetson_info = get_jetson_info()
    print_banner(jetson_info)
    check_and_recommend_power_mode()

    # 1. 编译或检查 TensorRT 引擎
    if args.compile:
        auto_compile_engines()

    engines = check_required_engines(
        vlm_model_dir=args.vlm_model_dir,
        vlm_engine=args.vlm_engine,
        custom_detector=args.detector,
        custom_cls=args.cls_model,
        custom_pose=args.pose,
        custom_fire=args.fire_detector,
        enable_fire=not args.no_fire
    )
    # 统一主检测器与二级动物细分类器 (自由指定 YOLO 或 MegaDetector)
    chosen_det = args.detector or (str(engines["detector"]) if engines["detector"] else None)
    chosen_cls = args.cls_model or (str(engines["cls"]) if engines["cls"] else None)

    is_md = ("mdv" in str(chosen_det).lower() or "megadetector" in str(chosen_det).lower()) if chosen_det else False
    det_mode_str = "MegaDetector (野生动物监测模式)" if is_md else "YOLO (人车/野生动物统一检测模式)"

    print("\n[*] 检查 TensorRT Engine 模型挂载状态:")
    for k, p in engines.items():
        if k in ["yolo", "megadetector"]:
            continue
        elif k == "detector":
            status = f"✅ 已激活 (主检测器: {det_mode_str}): {chosen_det}" if chosen_det else "⚠️ 未找到对应 .engine 文件"
        elif k == "cls":
            status = f"✅ 已激活 (二级动物物种细分类): {p}" if p else "⏸️ 待命休眠 (未找到细分类模型，使用先验回退策略)"
        elif k == "fire_det":
            if p == "builtin:VisualFlameDetector":
                status = "✅ 已激活 (自适应视觉火焰并发引擎, 默认开启)"
            elif p:
                status = f"✅ 已激活 (多流并发专用模型): {p}"
            else:
                status = "⏸️ 已禁用 (--no-fire 显式关闭)"
        elif k == "vlm":
            mode_desc = f"模式: {args.vlm_mode}"
            status = f"✅ 已就绪 ({mode_desc}): {p}" if p else f"⚠️ 未指定/未挂载 ({mode_desc}，可传入 --model-dir 指定，当前使用端侧离线保底)"
        else:
            status = f"✅ 已就绪: {p}" if p else "⚠️ 未找到对应 .engine 文件 (将降级使用 ONNXRuntime 或跳过)"
        print(f"    - {k:<12}: {status}")

    if not chosen_det:
        print("\n[!] 提示: 检测到尚未编译或未找到主目标检测器 Engine。")
        print("    可在启动命令后追加 --compile 自动一键编译，或单独运行:")
        print("    python batch_convert_trt.py models --precision fp16")

    # 2. 视频输入源判定
    video_source = args.video
    if args.csi is not None:
        video_source = get_csi_gstreamer_pipeline(sensor_id=args.csi)
        print(f"[*] 已配置 Jetson CSI 摄像头硬件加速管道 (Sensor {args.csi})")
    elif args.usb is not None:
        video_source = args.usb
        print(f"[*] 已配置 USB 摄像头设备 (/dev/video{args.usb})")

    if video_source is None:
        # 寻找默认样本视频
        for cand in [
            "C:\\Users\\chanceyu\\Downloads\\开门碰撞1.mp4",
            "sample_fisheye_left.mp4",
            "models/sample.mp4"
        ]:
            if os.path.exists(cand):
                video_source = cand
                print(f"[*] 自动选用默认测试视频源: {cand}")
                break

    if video_source is None:
        print("[!] 错误: 未指定视频输入源！请使用 --video <mp4文件> 或 --csi 0。")
        sys.exit(1)

    # 3. 运行主视频/摄像头端到端管线 (强制启用 --use-trt)
    loop_val = args.loop_video
    if loop_val is None:
        loop_val = False if args.output else True

    show_sb = args.sidebar and not args.no_sidebar

    print("\n[*] 正在启动车载露营守卫 (TensorRT 模式)...")
    run_video_pipeline(
        video_path=video_source,
        config_path=args.config,
        camera_id=args.camera_id,
        detector_onnx_path=chosen_det,
        pose_onnx_path=args.pose or (str(engines["pose"]) if engines["pose"] else None),
        conf_threshold=args.conf,
        show_sidebar=show_sb,
        headless=args.headless,
        max_frames=args.max_frames,
        realtime=args.realtime,
        use_trt=True,
        loop_video=loop_val,
        output_path=args.output,
        stats_json_path=args.stats_json,
        enable_web=args.web and not args.no_web,
        web_port=args.web_port,
        vlm_model_dir=args.vlm_model_dir,
        vlm_engine_path=args.vlm_engine,
        vlm_visual_engine_path=args.vlm_visual_engine,
        edgellm_dir=args.edgellm_dir,
        build_dir=args.edgellm_dir,
        vlm_timeout=args.vlm_timeout,
        vlm_max_tokens=args.vlm_max_tokens,
        vlm_mode=args.vlm_mode,
        vlm_server_port=args.vlm_server_port,
        vlm_idle_timeout=args.vlm_idle_timeout,
        vlm_server_bin=args.vlm_server_bin,
        cls_onnx_path=chosen_cls,
        fire_detector_path=args.fire_detector or (str(engines["fire_det"]) if engines["fire_det"] and str(engines["fire_det"]) != "builtin:VisualFlameDetector" else None),
        enable_fire=not args.no_fire,
        detector_size=args.detector_size,
        fire_size=args.fire_size
    )


if __name__ == "__main__":
    main()
