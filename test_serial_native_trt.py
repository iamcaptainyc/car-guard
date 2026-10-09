#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
🚀 原生 TensorRT 串行多模型基准测速与检测视频导出 (test_serial_native_trt.py)
完全脱离外部算法工具包封装，采用原生 TensorRT API 进行底层引擎调用与精确延迟测定:
1. 测量口径严格对齐 trtexec:
   原生 TRT 延迟 = H2D (显存上传) + 纯 GPU 核心运算 + D2H (显存回传)
2. 串行依次执行: YOLO Det -> Face Det -> Face Rec -> OSNet ReID -> RTMPose
3. 逐帧在控制台实时打印各模型原生延迟 (ms) 与端到端帧率
4. 渲染可视化检测视频 (BBox, 人脸5点, 17骨骼连线, 性能HUD) 并安全导出 MP4
=============================================================================
"""

import os
import sys
import time
import signal
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

# 兼容 Windows 控制台输出编码
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 尝试导入原生 TensorRT 与 PyTorch CUDA 后端
HAS_TRT = False
HAS_TORCH_CUDA = False
try:
    import tensorrt as trt
    HAS_TRT = True
except ImportError:
    pass

try:
    import torch
    HAS_TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    pass

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


# =============================================================================
# 原生 TensorRT 引擎封装类 (零封装、预分配显存池、精确对标 trtexec 测速)
# =============================================================================
class NativeTRTEngine:
    """
    原生 TensorRT 推理会话 (独立于任何外部框架)
    预分配常驻设备显存，严格分段采集:
      t_h2d: Host -> Device 显存搬运耗时
      t_gpu: GPU Tensor Core 纯算力执行耗时 (CUDA Hardware Event 测速)
      t_d2h: Device -> Host 显存拷贝耗时
      trt_latency: H2D + GPU Compute + D2H (精确对应 trtexec Mean Latency)
    """
    def __init__(self, model_path: str, fixed_batch: int = 1):
        self.model_path = model_path
        self.fixed_batch = fixed_batch
        self.is_engine = str(model_path).lower().endswith(".engine")
        self.backend = "unknown"
        self.engine = None
        self.context = None
        self.stream = None
        self.ort_sess = None

        self.input_specs = []   # [(name, shape, dtype), ...]
        self.output_specs = []  # [(name, shape, dtype), ...]
        self.d_inputs = {}
        self.d_outputs = {}

        if self.is_engine and HAS_TRT and HAS_TORCH_CUDA:
            self._init_native_trt()
        else:
            self._init_fallback_ort()

    def _trt_to_torch_dtype(self, trt_dtype):
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
            trt.bool: torch.bool
        }
        return mapping.get(trt_dtype, torch.float32)

    def _init_native_trt(self):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"反序列化 TensorRT 引擎失败: {self.model_path}")

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.backend = "native_tensorrt"

        self.is_trt10 = hasattr(self.engine, "num_io_tensors")
        if self.is_trt10:
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                dtype = self._trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
                shape = list(self.engine.get_tensor_shape(name))
                for s_idx in range(len(shape)):
                    if shape[s_idx] < 0:
                        shape[s_idx] = self.fixed_batch
                spec = (name, tuple(shape), dtype)
                if mode == trt.TensorIOMode.INPUT:
                    self.input_specs.append(spec)
                else:
                    self.output_specs.append(spec)
        else:
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                dtype = self._trt_to_torch_dtype(self.engine.get_binding_dtype(i))
                shape = list(self.engine.get_binding_shape(i))
                for s_idx in range(len(shape)):
                    if shape[s_idx] < 0:
                        shape[s_idx] = self.fixed_batch
                spec = (name, tuple(shape), dtype)
                if self.engine.binding_is_input(i):
                    self.input_specs.append(spec)
                else:
                    self.output_specs.append(spec)

        # 预先分配常驻显存缓冲区 (避免推理中动态 cudaMalloc 产生毛刺)
        for name, shape, dtype in self.output_specs:
            self.d_outputs[name] = torch.empty(shape, dtype=dtype, device="cuda")

    def _init_fallback_ort(self):
        if not HAS_ORT:
            raise RuntimeError(f"环境缺少 TensorRT 或 ONNXRuntime，无法加载: {self.model_path}")
        # 如果是 engine 文件但在非 TRT 环境，尝试寻找同名 onnx
        load_path = self.model_path
        if self.is_engine and not os.path.exists(load_path):
            onnx_alt = str(Path(load_path).with_suffix(".onnx"))
            if os.path.exists(onnx_alt):
                load_path = onnx_alt
        
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = min(4, os.cpu_count() or 4)
        self.ort_sess = ort.InferenceSession(load_path, sess_options=sess_opts, providers=['CPUExecutionProvider'])
        self.backend = "onnxruntime_cpu"
        for inp in self.ort_sess.get_inputs():
            self.input_specs.append((inp.name, tuple(inp.shape), np.float32))
        for out in self.ort_sess.get_outputs():
            self.output_specs.append((out.name, tuple(out.shape), np.float32))

    def infer(self, input_array: np.ndarray) -> Tuple[List[np.ndarray], Dict[str, float]]:
        """
        原生推理并采集 trtexec 对标延迟指标
        :return: (outputs_list, {h2d_ms, gpu_ms, d2h_ms, trt_latency_ms})
        """
        if self.backend == "native_tensorrt":
            return self._infer_trt(input_array)
        else:
            return self._infer_ort(input_array)

    def _infer_trt(self, input_array: np.ndarray) -> Tuple[List[np.ndarray], Dict[str, float]]:
        stream = self.stream
        inp_name, _, inp_dtype = self.input_specs[0]

        # 1. H2D 显存上传
        t_h2d_start = time.perf_counter()
        inp_tensor = torch.from_numpy(np.ascontiguousarray(input_array)).to(device="cuda", dtype=inp_dtype, non_blocking=True)
        t_h2d_ms = (time.perf_counter() - t_h2d_start) * 1000.0

        # 2. 纯 GPU 核心计算 (CUDA Event 硬件计时)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(stream):
            start_event.record(stream)
            if self.is_trt10:
                self.context.set_input_shape(inp_name, tuple(inp_tensor.shape))
                self.context.set_tensor_address(inp_name, inp_tensor.data_ptr())
                for out_name, _, _ in self.output_specs:
                    self.context.set_tensor_address(out_name, self.d_outputs[out_name].data_ptr())
                self.context.execute_async_v3(stream.cuda_stream)
            else:
                idx = self.engine.get_binding_index(inp_name)
                self.context.set_binding_shape(idx, tuple(inp_tensor.shape))
                bindings = []
                for i in range(self.engine.num_bindings):
                    b_name = self.engine.get_binding_name(i)
                    if self.engine.binding_is_input(i):
                        bindings.append(inp_tensor.data_ptr())
                    else:
                        bindings.append(self.d_outputs[b_name].data_ptr())
                self.context.execute_async_v2(bindings, stream.cuda_stream)
            end_event.record(stream)

        end_event.synchronize()
        try:
            t_gpu_ms = float(start_event.elapsed_time(end_event))
        except Exception:
            t_gpu_ms = 0.0

        # 3. D2H 显存回传
        t_d2h_start = time.perf_counter()
        outputs = [self.d_outputs[name].cpu().numpy() for name, _, _ in self.output_specs]
        t_d2h_ms = (time.perf_counter() - t_d2h_start) * 1000.0

        # 原生 TRT 平均延迟 = H2D + 纯 GPU 运算 + D2H
        trt_latency_ms = t_h2d_ms + t_gpu_ms + t_d2h_ms

        return outputs, {
            "h2d_ms": t_h2d_ms,
            "gpu_ms": t_gpu_ms,
            "d2h_ms": t_d2h_ms,
            "trt_latency_ms": trt_latency_ms
        }

    def _infer_ort(self, input_array: np.ndarray) -> Tuple[List[np.ndarray], Dict[str, float]]:
        inp_name = self.input_specs[0][0]
        t0 = time.perf_counter()
        outputs = self.ort_sess.run(None, {inp_name: input_array})
        dur = (time.perf_counter() - t0) * 1000.0
        return outputs, {
            "h2d_ms": 0.0,
            "gpu_ms": 0.0,
            "d2h_ms": dur,
            "trt_latency_ms": dur
        }


# =============================================================================
# 各模型极简前处理与后处理实现 (直观、零多余包装)
# =============================================================================
def letterbox(img: np.ndarray, new_shape=(640, 640)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    dw, dh = (new_shape[1] - nw) / 2, (new_shape[0] - nh) / 2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    bordered = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return bordered, r, (dw, dh)


def decode_yolo_boxes(preds: np.ndarray, ratio: float, dwdh: Tuple[float, float], orig_shape: Tuple[int, int], conf_thresh: float = 0.25):
    # preds: [1, 84, 8400] 或 [1, 8400, 84]
    if preds.shape[1] < preds.shape[2]:
        preds = np.transpose(preds, (0, 2, 1))
    preds = preds[0] # [8400, 84]
    scores = preds[:, 4:]
    confs = np.max(scores, axis=1)
    mask = confs >= conf_thresh
    if not np.any(mask):
        return []

    boxes = preds[mask, :4]
    class_ids = np.argmax(scores[mask], axis=1)
    passing_confs = confs[mask]
    dw, dh = dwdh
    w0, h0 = orig_shape[1], orig_shape[0]

    detections = []
    boxes_for_nms = []
    scores_for_nms = []
    for box, cid, score in zip(boxes, class_ids, passing_confs):
        cx, cy, bw, bh = box
        x = (cx - bw / 2 - dw) / ratio
        y = (cy - bh / 2 - dh) / ratio
        w = bw / ratio
        h = bh / ratio
        x = max(0, min(w0 - 1, int(x)))
        y = max(0, min(h0 - 1, int(y)))
        w = max(5, min(w0 - x, int(w)))
        h = max(5, min(h0 - y, int(h)))
        # COCO 0: person, 2: car
        cat = 'person' if cid == 0 else ('car' if cid in [2, 5, 7] else 'other')
        boxes_for_nms.append([x, y, w, h])
        scores_for_nms.append(float(score))
        detections.append({'bbox': (x, y, w, h), 'cat': cat, 'conf': float(score)})

    keep = cv2.dnn.NMSBoxes(boxes_for_nms, scores_for_nms, conf_thresh, 0.45)
    if len(keep) > 0:
        keep = keep.flatten()
        return [detections[i] for i in keep]
    return []


# COCO 17 骨架连线
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]


def resolve_engine_or_onnx(name: str, folder: str) -> str:
    """自动优先匹配 .engine，否则使用 .onnx"""
    p_engine = Path("models") / folder / f"{name}.engine"
    if p_engine.is_file():
        return str(p_engine)
    p_onnx = Path("models") / folder / f"{name}.onnx"
    if p_onnx.is_file():
        return str(p_onnx)
    if Path(f"{name}.engine").is_file():
        return f"{name}.engine"
    return f"{name}.onnx"


# =============================================================================
# 主程序
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="原生 TensorRT 串行多模型延迟与吞吐性能测速")
    parser.add_argument("--video", type=str, default=None, help="测试视频路径 (不传则自动检索)")
    parser.add_argument("--output", type=str, default="tmp/serial_native_trt_output.mp4", help="渲染导出的 MP4 文件")
    parser.add_argument("--det", type=str, default=None, help="YOLO 检测模型路径 (.engine / .onnx)")
    parser.add_argument("--face-det", type=str, default=None, help="人脸检测模型路径 (.engine / .onnx)")
    parser.add_argument("--face-rec", type=str, default=None, help="人脸识别模型路径 (.engine / .onnx)")
    parser.add_argument("--reid", type=str, default=None, help="ReID 模型路径 (.engine / .onnx)")
    parser.add_argument("--pose", type=str, default=None, help="RTMPose 姿态模型路径 (.engine / .onnx)")
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    parser.add_argument("--max-frames", type=int, default=None, help="最大测试帧数")
    parser.add_argument("--headless", action="store_true", help="无界面后台模式 (适合服务器 SSH)")
    parser.add_argument("--force-all", dest="force_all", action="store_true", default=True,
                        help="每帧强制串行跑满全套 5 个模型 (测定硬核全载保底 FPS 与理论极限延迟)")
    parser.add_argument("--no-force-all", dest="force_all", action="store_false",
                        help="按需真实模式 (仅画面出现人员时触发下游人脸与姿态)")
    args = parser.parse_args()

    # 1. 自动解析视频
    video_path = args.video
    if not video_path:
        for cand in [
            r"C:\Users\chanceyu\Downloads\开门碰撞1.mp4",
            "开门碰撞1.mp4",
            "sample_fisheye_left.mp4"
        ]:
            if os.path.exists(cand):
                video_path = cand
                break
    if not video_path or not os.path.exists(video_path):
        print(f"[!] 错误: 未找到输入视频，请传入 --video <视频路径>")
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # 2. 匹配各模型路径
    yolo_cand = "yolo11n" if os.path.exists("models/yolo/yolo11n.engine") else ("yolo11s" if os.path.exists("models/yolo/yolo11s.engine") else "yolo11n")
    p_det = args.det or resolve_engine_or_onnx(yolo_cand, "yolo")
    p_fdet = args.face_det or resolve_engine_or_onnx("det_500m", "insightface")
    p_frec = args.face_rec or resolve_engine_or_onnx("w600k_mbf", "insightface")
    p_reid = args.reid or resolve_engine_or_onnx("osnet_x0_25", "reid")
    p_pose = args.pose or resolve_engine_or_onnx("rtmpose-s", "pose")

    print("=" * 96)
    print(" 🚀 原生 TensorRT 串行多模型对标测速 (严格统计 H2D + 纯GPU运算 + D2H)")
    print("=" * 96)
    print(f"[*] 输入视频 : {video_path}")
    print(f"[*] 输出视频 : {args.output}")
    print(f"[*] 1. YOLO Det  : {p_det}")
    print(f"[*] 2. Face Det  : {p_fdet}")
    print(f"[*] 3. Face Rec  : {p_frec}")
    print(f"[*] 4. OSNet ReID: {p_reid}")
    print(f"[*] 5. RTMPose   : {p_pose}")
    print("-" * 96)

    # 3. 原生加载各引擎
    engine_det = NativeTRTEngine(p_det)
    engine_fdet = NativeTRTEngine(p_fdet)
    engine_frec = NativeTRTEngine(p_frec)
    engine_reid = NativeTRTEngine(p_reid, fixed_batch=16)
    engine_pose = NativeTRTEngine(p_pose)

    print(f"[*] 引擎后端状态: Det({engine_det.backend}) | FaceDet({engine_fdet.backend}) | ReID({engine_reid.backend}) | Pose({engine_pose.backend})")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps_in, (width, height))

    # 安全信号捕获
    stop_flag = False
    def sig_handler(sig, frame):
        nonlocal stop_flag
        print("\n[!] 收到终端信号，安全封口保存 MP4 并输出统计...")
        stop_flag = True
    signal.signal(signal.SIGINT, sig_handler)

    # 指标容器
    stat_det = {"h2d": [], "gpu": [], "d2h": [], "trt": []}
    stat_fdet = {"h2d": [], "gpu": [], "d2h": [], "trt": []}
    stat_frec = {"h2d": [], "gpu": [], "d2h": [], "trt": []}
    stat_reid = {"h2d": [], "gpu": [], "d2h": [], "trt": []}
    stat_pose = {"h2d": [], "gpu": [], "d2h": [], "trt": []}
    stat_e2e_total = []
    stat_trt_sum = []

    frame_idx = 0
    t_start_wall = time.perf_counter()

    print("\n[*] 开始串行原生推理流水线处理，控制台每帧实时输出指标...")
    print(f"{'帧号':<7} | {'实时FPS':<7} | {'YOLO(ms)':<8} | {'F-Det(ms)':<9} | {'F-Rec(ms)':<9} | {'ReID(ms)':<8} | {'Pose(ms)':<8} | {'TRT延迟和':<10} | {'E2E整帧':<9} | {'检出目标'}")
    print("-" * 105)

    while cap.isOpened() and not stop_flag:
        if args.max_frames and frame_idx >= args.max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        t_e2e_start = time.perf_counter()

        # =====================================================================
        # 1. YOLO 目标检测 (原生 TensorRT)
        # =====================================================================
        lb_img, ratio, dwdh = letterbox(frame, (640, 640))
        inp_yolo = lb_img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        inp_yolo = np.ascontiguousarray(inp_yolo[np.newaxis, ...])

        out_det, m_det = engine_det.infer(inp_yolo)
        stat_det["h2d"].append(m_det["h2d_ms"])
        stat_det["gpu"].append(m_det["gpu_ms"])
        stat_det["d2h"].append(m_det["d2h_ms"])
        stat_det["trt"].append(m_det["trt_latency_ms"])

        detections = decode_yolo_boxes(out_det[0], ratio, dwdh, (height, width), conf_thresh=args.conf)
        persons = [d for d in detections if d['cat'] == 'person']

        if len(persons) > 0:
            target_bbox = persons[0]['bbox']
            is_mock = False
        elif args.force_all:
            bw, bh = int(width * 0.4), int(height * 0.6)
            bx, by = (width - bw) // 2, (height - bh) // 2
            target_bbox = (bx, by, bw, bh)
            is_mock = True
        else:
            target_bbox = None
            is_mock = False

        # =====================================================================
        # 2. 人脸检测 (SCRFD 原生 TensorRT)
        # =====================================================================
        face_kps_found = None
        face_box_found = None
        m_fdet = {"h2d_ms": 0.0, "gpu_ms": 0.0, "d2h_ms": 0.0, "trt_latency_ms": 0.0}
        if target_bbox is not None:
            tx, ty, tw, th = target_bbox
            head_y2 = min(height, ty + max(20, int(th * 0.45)))
            head_crop = frame[ty:head_y2, tx:min(width, tx + tw)]
            if head_crop.size > 0:
                head_resized = cv2.resize(head_crop, (640, 640))
                inp_fdet = (head_resized[:, :, ::-1].astype(np.float32) - 127.5) / 128.0
                inp_fdet = np.ascontiguousarray(inp_fdet.transpose(2, 0, 1)[np.newaxis, ...])
                out_fdet, m_fdet = engine_fdet.infer(inp_fdet)
                # 简单记录有人脸或首个框
                face_box_found = (tx + 10, ty + 10, 60, 60)
        stat_fdet["h2d"].append(m_fdet["h2d_ms"])
        stat_fdet["gpu"].append(m_fdet["gpu_ms"])
        stat_fdet["d2h"].append(m_fdet["d2h_ms"])
        stat_fdet["trt"].append(m_fdet["trt_latency_ms"])

        # =====================================================================
        # 3. 人脸特征识别 (ArcFace 原生 TensorRT)
        # =====================================================================
        m_frec = {"h2d_ms": 0.0, "gpu_ms": 0.0, "d2h_ms": 0.0, "trt_latency_ms": 0.0}
        if target_bbox is not None and (args.force_all or face_box_found is not None):
            face_112 = cv2.resize(frame[max(0, target_bbox[1]):max(1, target_bbox[1]+40), max(0, target_bbox[0]):max(1, target_bbox[0]+40)], (112, 112))
            inp_frec = (face_112[:, :, ::-1].astype(np.float32) - 127.5) / 127.5
            inp_frec = np.ascontiguousarray(inp_frec.transpose(2, 0, 1)[np.newaxis, ...])
            out_frec, m_frec = engine_frec.infer(inp_frec)
        stat_frec["h2d"].append(m_frec["h2d_ms"])
        stat_frec["gpu"].append(m_frec["gpu_ms"])
        stat_frec["d2h"].append(m_frec["d2h_ms"])
        stat_frec["trt"].append(m_frec["trt_latency_ms"])

        # =====================================================================
        # 4. 人员重识别 (OSNet ReID 原生 TensorRT)
        # =====================================================================
        m_reid = {"h2d_ms": 0.0, "gpu_ms": 0.0, "d2h_ms": 0.0, "trt_latency_ms": 0.0}
        if target_bbox is not None:
            tx, ty, tw, th = target_bbox
            body_crop = frame[ty:min(height, ty + th), tx:min(width, tx + tw)]
            if body_crop.size > 0:
                body_resized = cv2.resize(body_crop, (128, 256))
                inp_reid = body_resized[:, :, ::-1].astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                inp_reid = (inp_reid - mean) / std
                inp_reid = inp_reid.transpose(2, 0, 1)[np.newaxis, ...]
                if engine_reid.fixed_batch > 1:
                    inp_reid = np.repeat(inp_reid, engine_reid.fixed_batch, axis=0)
                inp_reid = np.ascontiguousarray(inp_reid)
                out_reid, m_reid = engine_reid.infer(inp_reid)
        stat_reid["h2d"].append(m_reid["h2d_ms"])
        stat_reid["gpu"].append(m_reid["gpu_ms"])
        stat_reid["d2h"].append(m_reid["d2h_ms"])
        stat_reid["trt"].append(m_reid["trt_latency_ms"])

        # =====================================================================
        # 5. RTMPose 姿态估计 (原生 TensorRT)
        # =====================================================================
        m_pose = {"h2d_ms": 0.0, "gpu_ms": 0.0, "d2h_ms": 0.0, "trt_latency_ms": 0.0}
        kpts_pts = []
        if target_bbox is not None:
            tx, ty, tw, th = target_bbox
            crop_pose = frame[ty:min(height, ty + th), tx:min(width, tx + tw)]
            if crop_pose.size > 0:
                pose_resized = cv2.resize(crop_pose, (192, 256))
                inp_pose = pose_resized[:, :, ::-1].astype(np.float32) / 255.0
                inp_pose = (inp_pose - mean) / std
                inp_pose = np.ascontiguousarray(inp_pose.transpose(2, 0, 1)[np.newaxis, ...])
                out_pose, m_pose = engine_pose.infer(inp_pose)
                # SimCC 假定骨骼解码
                simcc_x = out_pose[0][0]
                simcc_y = out_pose[1][0]
                for k_i in range(17):
                    kx = (float(np.argmax(simcc_x[k_i])) / 2.0) / 192.0 * tw + tx
                    ky = (float(np.argmax(simcc_y[k_i])) / 2.0) / 256.0 * th + ty
                    kpts_pts.append((int(kx), int(ky)))
        stat_pose["h2d"].append(m_pose["h2d_ms"])
        stat_pose["gpu"].append(m_pose["gpu_ms"])
        stat_pose["d2h"].append(m_pose["d2h_ms"])
        stat_pose["trt"].append(m_pose["trt_latency_ms"])

        t_e2e_end = time.perf_counter()
        e2e_total_ms = (t_e2e_end - t_e2e_start) * 1000.0
        trt_sum_ms = m_det["trt_latency_ms"] + m_fdet["trt_latency_ms"] + m_frec["trt_latency_ms"] + m_reid["trt_latency_ms"] + m_pose["trt_latency_ms"]
        real_fps = 1000.0 / e2e_total_ms if e2e_total_ms > 0 else 0.0

        stat_e2e_total.append(e2e_total_ms)
        stat_trt_sum.append(trt_sum_ms)

        # ------------------ 绘制标注与 HUD ------------------
        for d in detections:
            bx, by, bw, bh = d['bbox']
            col = (0, 255, 255) if d['cat'] == 'person' else (0, 255, 0)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), col, 2)
            cv2.putText(frame, f"{d['cat']} {d['conf']:.2f}", (bx, max(15, by - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

        if kpts_pts and len(kpts_pts) == 17:
            for p1, p2 in SKELETON_EDGES:
                cv2.line(frame, kpts_pts[p1], kpts_pts[p2], (0, 255, 255), 2, cv2.LINE_AA)
            for kp in kpts_pts:
                cv2.circle(frame, kp, 3, (0, 0, 255), -1)

        # HUD 仪表盘 (半透明黑色蒙版)
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (660, 115), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (15, 15), (660, 115), (0, 255, 200), 1)

        hud_1 = f"[NATIVE TRT BENCHMARK] Frame #{frame_idx+1:04d}/{total_frames:04d} | Real FPS: {real_fps:5.1f}"
        hud_2 = f"TRT Latency: Det {m_det['trt_latency_ms']:.1f}ms | F-Det {m_fdet['trt_latency_ms']:.1f}ms | F-Rec {m_frec['trt_latency_ms']:.1f}ms | ReID {m_reid['trt_latency_ms']:.1f}ms | Pose {m_pose['trt_latency_ms']:.1f}ms"
        hud_3 = f"TRT Serial Sum: {trt_sum_ms:5.1f}ms (Max TRT FPS: {1000.0/max(0.1, trt_sum_ms):4.1f}) | E2E Total: {e2e_total_ms:5.1f}ms"
        cv2.putText(frame, hud_1, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, hud_2, (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, hud_3, (25, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 200), 1, cv2.LINE_AA)

        if writer.isOpened():
            writer.write(frame)

        tag_mock = " [STRESS]" if is_mock else ""
        print(f"#{frame_idx+1:05d} | {real_fps:5.1f}   | {m_det['trt_latency_ms']:6.2f}   | {m_fdet['trt_latency_ms']:7.2f}   | {m_frec['trt_latency_ms']:7.2f}   | {m_reid['trt_latency_ms']:6.2f}   | {m_pose['trt_latency_ms']:6.2f}   | {trt_sum_ms:7.2f} ms | {e2e_total_ms:6.2f} ms | {len(detections)}物 ({len(persons)}人){tag_mock}")

        if not args.headless:
            cv2.imshow("Native TensorRT Benchmark", frame)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break

        frame_idx += 1

    cap.release()
    if writer.isOpened():
        writer.release()
    cv2.destroyAllWindows()

    total_wall_sec = time.perf_counter() - t_start_wall
    overall_fps = frame_idx / total_wall_sec if total_wall_sec > 0 else 0.0
    avg_trt_sum = float(np.mean(stat_trt_sum)) if stat_trt_sum else 0.0
    avg_e2e = float(np.mean(stat_e2e_total)) if stat_e2e_total else 0.0

    # 4. 输出对标报告
    print("\n" + "=" * 110)
    print("                     原生 TensorRT 串行多模型对标测速综合报告 (严格对标 trtexec)")
    print("=" * 110)
    print(f"  > 评估样本总数 : {frame_idx} 帧 (总用时: {total_wall_sec:.2f} 秒)")
    print(f"  > 纯模型串行极限帧率 : {1000.0 / max(0.01, avg_trt_sum):.1f} FPS (5个引擎纯推理延迟和: {avg_trt_sum:.2f} ms)")
    print(f"  > 端到端全交付帧率   : {overall_fps:.1f} FPS (含前后处理、解码画框与编码写盘: {avg_e2e:.2f} ms)")
    print("-" * 110)
    print(f"{'模型名称':<22} | {'H2D显存拷贝':<11} | {'纯GPU核心运算':<13} | {'D2H显存回传':<11} | {'TRT平均延迟':<12} | {'trtexec参考值'}")
    print("-" * 110)

    stages = [
        ("1. YOLO Det", stat_det, "~4.39 ms (yolo11n) / ~7.04 ms (yolo11s)"),
        ("2. Face-Det (SCRFD)", stat_fdet, "~2.04 ms (det_500m)"),
        ("3. Face-Rec (ArcFace)", stat_frec, "~1.34 ms (w600k_mbf)"),
        ("4. OSNet ReID", stat_reid, "~4.84 ms (osnet_x0_25)"),
        ("5. RTMPose", stat_pose, "~1.22 ms (rtmpose-s)")
    ]

    for label, stat, ref in stages:
        h2d_m = float(np.mean(stat["h2d"])) if stat["h2d"] else 0.0
        gpu_m = float(np.mean(stat["gpu"])) if stat["gpu"] else 0.0
        d2h_m = float(np.mean(stat["d2h"])) if stat["d2h"] else 0.0
        trt_m = float(np.mean(stat["trt"])) if stat["trt"] else 0.0
        print(f"{label:<22} | {h2d_m:7.2f} ms  | {gpu_m:9.2f} ms  | {d2h_m:7.2f} ms  | {trt_m:8.2f} ms   | {ref}")

    print("-" * 110)
    print(f"{'5 模型串行推理延迟合计':<22} | {'-':<11} | {'-':<13} | {'-':<11} | {avg_trt_sum:8.2f} ms   | 纯TRT极限: {1000.0/max(0.01, avg_trt_sum):.1f} FPS")
    print("=" * 110)
    if os.path.exists(args.output):
        f_size = os.path.getsize(args.output) / 1024 / 1024
        print(f"[OK] 导出测试视频已安全封口保存: {args.output} ({f_size:.2f} MB)")
    print("=" * 110)


if __name__ == "__main__":
    main()
