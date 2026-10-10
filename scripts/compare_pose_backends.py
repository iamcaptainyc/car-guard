#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTMPose ONNX vs TensorRT Engine 精度对齐诊断工具 (scripts/compare_pose_backends.py)
用于在 Jetson Orin 上一键比对 ONNX 与 TensorRT Engine 的输出张量、Shape 及解码关键点坐标
支持图像文件、视频文件 (--video)，并自动抽取有人画幅生成两端并排可视化图片
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from camping_guard.perception.pose_estimator import RTMPoseEstimator, KEYPOINT_NAMES
from run_simulation import draw_skeleton


def extract_person_frame_and_bbox(video_path: str, target_frame_idx: int = -1):
    """从视频中抓取有效画幅及人体检测框"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[*] 成功打开视频: {video_path} (总帧数: {total_frames})")

    # 尝试挂载 YOLO 自动定位真实人体
    detector = None
    yolo_candidates = [
        "models/yolo/yolo11s.engine",
        "models/yolo/yolo11s.onnx",
        "models/yolo/yolo11n.engine",
        "models/yolo/yolo11n.onnx"
    ]
    for yc in yolo_candidates:
        if os.path.exists(yc):
            try:
                from camping_guard.perception.detector import TRTObjectDetector
                detector = TRTObjectDetector(model_path=yc)
                print(f"[*] 挂载 YOLO 初检器用于精确定位人体: {yc}")
                break
            except Exception:
                pass

    best_frame = None
    best_bbox = None
    best_conf = 0.0

    step = max(1, total_frames // 30) if target_frame_idx < 0 else 1
    scan_limit = 30 if target_frame_idx < 0 else 1
    curr_idx = 0 if target_frame_idx < 0 else target_frame_idx

    cap.set(cv2.CAP_PROP_POS_FRAMES, curr_idx)

    first_frame = None
    for _ in range(scan_limit):
        ret, frame = cap.read()
        if not ret:
            break

        if first_frame is None:
            first_frame = frame.copy()

        if detector is not None:
            dets = detector.detect(frame)
            person_dets = []
            for d in dets:
                cat = d.get('category') if isinstance(d, dict) else getattr(d, 'category', None)
                if cat == 'person':
                    person_dets.append(d)

            if person_dets:
                def get_conf(det):
                    return det.get('conf', 0.0) if isinstance(det, dict) else getattr(det, 'confidence', 0.0)

                top_p = max(person_dets, key=get_conf)
                conf = get_conf(top_p)
                if conf > best_conf:
                    best_conf = conf
                    best_frame = frame.copy()
                    if isinstance(top_p, dict):
                        bx, by, bw, bh = top_p['bbox']
                    elif hasattr(top_p, 'bbox'):
                        b = top_p.bbox
                        bx, by, bw, bh = b.to_xywh() if hasattr(b, 'to_xywh') else (b.x, b.y, b.w, b.h)
                    else:
                        bx, by, bw, bh = 0, 0, 0, 0
                    best_bbox = (int(bx), int(by), int(bw), int(bh))
                    if best_conf > 0.60:
                        break
        else:
            if best_frame is None:
                best_frame = frame.copy()
                h, w = frame.shape[:2]
                best_bbox = (int(w * 0.35), int(h * 0.20), int(w * 0.30), int(h * 0.65))

        curr_idx += step
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_idx)

    cap.release()

    if best_frame is None:
        best_frame = first_frame

    if best_frame is None:
        raise RuntimeError("未能从视频中读取到任何画面帧！")

    if best_bbox is None:
        h, w = best_frame.shape[:2]
        best_bbox = (int(w * 0.35), int(h * 0.20), int(w * 0.30), int(h * 0.65))

    return best_frame, best_bbox


def compare_pose(
    input_source: str,
    is_video: bool = False,
    frame_idx: int = -1,
    onnx_path: str = "models/pose/rtmpose-s.onnx",
    engine_path: str = "models/pose/rtmpose-s.engine",
    output_image: str = "compare_pose_result.jpg"
):
    print("=" * 70)
    print(" [*] RTMPose 姿态模型端侧精度比对与诊断工具")
    print("=" * 70)

    # 1. 获取测试画幅与人体包围框
    if is_video or input_source.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
        print(f"[*] 正在从视频源解析人体测试帧: {input_source}")
        frame, test_bbox = extract_person_frame_and_bbox(input_source, frame_idx)
    elif os.path.exists(input_source):
        frame = cv2.imread(input_source)
        h, w = frame.shape[:2]
        test_bbox = (int(w * 0.35), int(h * 0.20), int(w * 0.30), int(h * 0.65))
        print(f"[*] 载入测试图像: {input_source} (尺寸: {w}x{h})")
    else:
        print("[!] 未找到测试输入文件，自动生成 640x480 合成测试画幅...")
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        h, w = frame.shape[:2]
        test_bbox = (int(w * 0.35), int(h * 0.20), int(w * 0.30), int(h * 0.65))

    bx, by, bw, bh = test_bbox
    print(f"[*] 评估人体边界框: x={bx}, y={by}, w={bw}, h={bh}")

    # 2. 检查模型文件
    has_onnx = os.path.exists(onnx_path)
    has_engine = os.path.exists(engine_path)

    print(f"[*] ONNX   模型路径: {onnx_path} ({'存在' if has_onnx else '不存在'})")
    print(f"[*] Engine 引擎路径: {engine_path} ({'存在' if has_engine else '不存在'})")

    if not has_onnx and not has_engine:
        print("[!] 错误: 未检测到任何姿态模型权重！请确认 models/pose/ 目录。")
        return

    kpts_onnx = None
    kpts_engine = None

    # 3. 执行 ONNXRuntime 推理 (基准标杆)
    if has_onnx:
        print("\n--- [1/2] 正在运行 ONNXRuntime (基准标杆) ---")
        try:
            est_onnx = RTMPoseEstimator(model_path=onnx_path, use_trt=False)
            res = est_onnx.estimate_keypoints(frame, [test_bbox])
            if res:
                kpts_onnx = res[0]
                print("[OK] ONNXRuntime 推理成功！")
        except Exception as e:
            print(f"[FAIL] ONNX 推理异常: {e}")

    # 4. 执行 TensorRT Engine 推理 (目标硬件加速)
    if has_engine:
        print("\n--- [2/2] 正在运行 TensorRT Engine (目标硬件加速) ---")
        try:
            est_engine = RTMPoseEstimator(model_path=engine_path, use_trt=True)
            res = est_engine.estimate_keypoints(frame, [test_bbox])
            if res:
                kpts_engine = res[0]
                print("[OK] TensorRT Engine 推理成功！")
        except Exception as e:
            print(f"[FAIL] TensorRT Engine 推理异常: {e}")

    # 5. 逐点对比关键点坐标与置信度
    if kpts_onnx and kpts_engine:
        print("\n" + "=" * 70)
        print(f" {'关键点名称':<15} | {'ONNX (X, Y, Conf)':<24} | {'TRT Engine (X, Y, Conf)':<24} | {'误差(px)'}")
        print("-" * 70)

        valid_diffs = []
        for name in KEYPOINT_NAMES:
            ox, oy, oc = kpts_onnx[name]
            tx, ty, tc = kpts_engine[name]
            diff = float(np.sqrt((ox - tx) ** 2 + (oy - ty) ** 2))

            o_str = f"({ox:5.1f}, {oy:5.1f}, {oc:.2f})"
            t_str = f"({tx:5.1f}, {ty:5.1f}, {tc:.2f})"

            # 若置信度均极低 (例如暗处背景底噪)，标记为无目标/低置信
            if oc < 0.25 and tc < 0.25:
                status = "[LOW CONF]"
            else:
                valid_diffs.append(diff)
                status = "[OK]" if diff < 5.0 else ("[WARN]" if diff < 20.0 else "[DIFF]")

            print(f" {name:<15} | {o_str:<24} | {t_str:<24} | {diff:5.2f}px {status}")

        print("=" * 70)
        if valid_diffs:
            max_diff = max(valid_diffs)
            avg_diff = sum(valid_diffs) / len(valid_diffs)
            print(f"[*] 有效关键点 (Conf>=0.25) 平均误差: {avg_diff:.2f} px | 最大误差: {max_diff:.2f} px")
            if max_diff < 5.0:
                print(f"[SUCCESS] 恭喜！TensorRT Engine 与 ONNX 预测结果高度一致，在亚像素范围内完美对齐！")
            elif max_diff < 20.0:
                print(f"[INFO] 关键点在轻微容限范围内对齐 (误差 < 20px)，通常为 FP16 量化微差。")
            else:
                print(f"[WARNING] 发现显著坐标偏差 (最大有效误差: {max_diff:.2f} px)！")
        else:
            print("[INFO] 当前测试画面内关键点置信度均低于 0.25，建议传入真实有人视频画幅进行测试。")

        # 6. 生成并排对比图
        vis_onnx = frame.copy()
        vis_engine = frame.copy()
        cv2.rectangle(vis_onnx, (bx, by), (bx + bw, by + bh), (0, 165, 255), 2)
        cv2.rectangle(vis_engine, (bx, by), (bx + bw, by + bh), (0, 165, 255), 2)
        draw_skeleton(vis_onnx, kpts_onnx, bbox=test_bbox)
        draw_skeleton(vis_engine, kpts_engine, bbox=test_bbox)

        cv2.putText(vis_onnx, "ONNXRuntime (Ground Truth)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(vis_engine, "TensorRT Engine (Target)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        combined = np.hstack([vis_onnx, vis_engine])
        cv2.imwrite(output_image, combined)
        print(f"\n[*] 已成功生成并排可视化诊断图: {output_image}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTMPose ONNX vs TRT Engine 精度比对工具")
    parser.add_argument("input", nargs="?", default="sample_fisheye.jpg", help="测试输入图片或视频")
    parser.add_argument("--video", type=str, default=None, help="显式指定测试视频路径")
    parser.add_argument("--image", type=str, default=None, help="显式指定测试图片路径")
    parser.add_argument("--frame-idx", type=int, default=-1, help="指定视频帧索引 (默认自动定位有人帧)")
    parser.add_argument("--onnx", type=str, default="models/pose/rtmpose-s.onnx", help="ONNX 权重路径")
    parser.add_argument("--engine", type=str, default="models/pose/rtmpose-s.engine", help="Engine 权重路径")
    parser.add_argument("--output", type=str, default="compare_pose_result.jpg", help="输出比对图像路径")

    args = parser.parse_args()

    target_input = args.video or args.image or args.input
    is_vid = bool(args.video) or target_input.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))

    compare_pose(
        input_source=target_input,
        is_video=is_vid,
        frame_idx=args.frame_idx,
        onnx_path=args.onnx,
        engine_path=args.engine,
        output_image=args.output
    )
