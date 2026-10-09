"""
单图端到端算法测试与可视化诊断工具 (test_single_image.py)
对单张输入图像进行：
1. S1 目标检测 (yolo11n.onnx)
2. 单目接地 IPM 测距与同心防区划分 (Zone 1 / 2)
3. 车身固定 ROI (门把手/车窗/车身防碰撞区) 空间碰撞判定
4. S2 RTMPose 姿态估计 (rtmpose-s.onnx, 17骨骼点)
5. 贴窗窥探脊柱倾角与拉门/踢踹动作时序推断
6. 生成带 HUD 仪表盘与骨骼点连线的高清分析图并导出
"""

import argparse
import os
import sys
import time
import math
from pathlib import Path
import cv2
import numpy as np

from camping_guard.perception.detector import ONNXObjectDetector
from camping_guard.perception.pose_estimator import RTMPoseEstimator
from camping_guard.perception.distance import GroundDistanceEstimator, ZoneHysteresisFilter
from camping_guard.analysis.pose_action import PoseActionClassifier
from camping_guard.types import TrackedEntity, BoundingBox, VehicleZone, ThreatLevel, ActionType
from camping_guard.config import load_vehicle_calibration
from run_simulation import draw_skeleton, draw_vehicle_and_zones
from camping_guard.engine import CampingGuardEngine
from camping_guard.profiler import PipelineProfiler


def run_single_image_test(
    image_path: str,
    config_path: str = "vehicle_calib.json",
    detector_path: str = "yolo11n.onnx",
    pose_path: str = "rtmpose-s.onnx",
    reid_path: str = "models/reid/osnet_x0_25.onnx",
    face_det_path: str = "models/insightface/det_500m.onnx",
    face_rec_path: str = "models/insightface/w600k_mbf.onnx",
    camera_id: int = 1,
    camera_height: float = None,
    camera_pitch: float = None,
    conf_threshold: float = 0.35,
    manual_bbox: tuple = None,
    output_path: str = "tmp/annotated_result.jpg",
    headless: bool = False
):
    img_file = Path(image_path)
    if not img_file.exists():
        print(f"[!] 错误: 输入图像文件不存在: {image_path}")
        return

    frame = cv2.imread(str(img_file))
    if frame is None:
        print(f"[!] 错误: 无法使用 OpenCV 读取图像: {image_path}")
        return

    # 1. 载入统一标定并初始化主引擎与性能剖析器
    profiler = PipelineProfiler()
    calib = load_vehicle_calibration(config_path)
    cam_info = calib.get_camera(camera_id)
    actual_height = camera_height if camera_height is not None else (cam_info.camera_height_m if cam_info else 1.2)
    actual_pitch = camera_pitch if camera_pitch is not None else (cam_info.camera_pitch_deg if cam_info else -15.0)

    dist_estimator = GroundDistanceEstimator(
        camera_height=actual_height,
        camera_pitch=actual_pitch,
        cameras_calib=calib.cameras
    )
    engine = CampingGuardEngine(calib=calib, distance_estimator=dist_estimator, profiler=profiler)
    engine.activate()
    engine.vehicle_bus.uwb_key_pos = None # 默认无钥匙在场

    h, w = frame.shape[:2]
    print("=" * 70)
    print(" 智能汽车露营守卫系统 - 单图端到端算法诊断")
    print(f" 车辆型号: {calib.vehicle_info.model_name} (VIN: {calib.vehicle_info.vin})")
    print(f" 标定配置: {config_path}")
    print(f" 输入图像: {img_file.name} (分辨率: {w}x{h})")
    print(f" 相机通道: Camera {camera_id} ({cam_info.name if cam_info else 'unknown'})")
    print(f" 相机外参: 安装高度={actual_height:.2f}m, 俯仰角={actual_pitch:+.1f}°")
    print("=" * 70)


    # 2. 载入模型并注册性能剖析元信息
    detector = None
    if detector_path and not os.path.exists(detector_path):
        sub_det = os.path.join("models", "yolo", detector_path)
        if os.path.exists(sub_det):
            detector_path = sub_det
    if detector_path and os.path.exists(detector_path):
        print(f"[*] 载入 YOLO 检测模型: {detector_path}")
        detector = ONNXObjectDetector(model_path=detector_path)
        profiler.register_model_metadata("YOLO11n-Det", detector_path, "S1单图全图目标检测 (640x480)")
    else:
        print(f"[!] 未找到 YOLO 模型 '{detector_path}'，将跳过全图检测")

    pose_estimator = None
    if pose_path and not os.path.exists(pose_path):
        sub_pose = os.path.join("models", "pose", pose_path)
        if os.path.exists(sub_pose):
            pose_path = sub_pose
    if pose_path and os.path.exists(pose_path):
        print(f"[*] 载入 RTMPose 姿态模型: {pose_path}")
        pose_estimator = RTMPoseEstimator(model_path=pose_path)
        profiler.register_model_metadata("RTMPose-s", pose_path, "Zone2/3人员姿态与17骨骼点提取")
    else:
        print(f"[!] 未找到 RTMPose 模型 '{pose_path}'，将跳过骨骼关键点提取")

    profiler.register_model_metadata("OSNet-ReID", reid_path, "人员跨时空重识别512维特征建档")
    profiler.register_model_metadata("SCRFD-500M", face_det_path, "4m以内人员上半身人脸定位初筛")
    profiler.register_model_metadata("ArcFace-MobileFaceNet", face_rec_path, "FQA质检通过后白名单比对")

    zone_filter = ZoneHysteresisFilter()
    pose_classifier = PoseActionClassifier()

    now = time.time()
    profiler.start_frame(1)

    # 3. 执行目标检测
    raw_detections = []
    if detector is not None:
        with profiler.profile_stage("Object_Detection"):
            with profiler.profile_model("YOLO11n-Det", condition="S1单图全图初筛 (640x480)"):
                raw_detections = detector.detect(frame, conf_threshold=conf_threshold)
        det_time = profiler.model_stats["YOLO11n-Det"].last_ms if "YOLO11n-Det" in profiler.model_stats else 0.0
        print(f"[OK] YOLO 检测完成 (耗时: {det_time:.1f}ms): 检出 {len(raw_detections)} 个目标")
    else:
        print("[i] 提示: 无检测模型，请使用 --detector 挂载 yolo11n.onnx")

    if manual_bbox is not None:
        raw_detections.append({
            'bbox': manual_bbox,
            'category': 'person',
            'conf': 0.90
        })
        print(f"[*] 已手动注入测试目标框: {manual_bbox}")

    # 4. 目标测距、防区分级与姿态估计
    processed_entities = []
    handle_polys = [r.polygon for r in engine.zone3_daemon.rois if "handle" in r.roi_id]
    window_polys = [r.polygon for r in engine.zone3_daemon.rois if "window" in r.roi_id]

    print("\n" + "-" * 70)
    print("【目标详细检测与几何研判报表】")
    print(f"{'ID':<4} | {'类别':<10} | {'置信度':<8} | {'接地点 (X,Y)':<14} | {'车身距离':<10} | {'同心防区':<12} | {'行为/姿态':<12}")
    print("-" * 70)

    highest_threat = ThreatLevel.L0_SAFE
    highest_desc = "安全免扰"

    for idx, det in enumerate(raw_detections):
        tid = idx + 1
        cat = det.get('category', 'unknown')
        conf = det.get('conf', 0.0)
        bx, by, bw, bh = det['bbox']
        bbox_obj = BoundingBox(bx, by, bw, bh)

        # 单目接地测距 (IPM)
        ground_pt = bbox_obj.bottom_center
        dist = dist_estimator.estimate_ground_distance(ground_pt, (h, w))
        zone = zone_filter.update_zone(tid, dist)

        entity = TrackedEntity(
            track_id=tid,
            category=cat,
            confidence=conf,
            bbox=bbox_obj,
            ground_dist=dist,
            zone=zone
        )

        # 若为人且进入 Zone 2 (0~3.0m 直达车身)，运行 RTMPose-s 姿态估计
        posture_desc = "常规经过"
        if cat == 'person' and pose_estimator is not None and zone == VehicleZone.ZONE_2_INNER:
            with profiler.profile_model("RTMPose-s", condition=f"ID:{tid} Zone 2 姿态骨骼提取"):
                kpts_list = pose_estimator.estimate_keypoints(frame, [(bx, by, bw, bh)])
            if kpts_list:
                entity.keypoints = kpts_list[0]
                # 分析姿态动作
                act, act_threat, desc = pose_classifier.classify_action(
                    entity, now,
                    handle_roi_polygons=handle_polys,
                    window_roi_polygons=window_polys
                )
                entity.action_type = act
                posture_desc = act.value

                # 计算脊柱前倾夹角
                ls = entity.keypoints.get("left_shoulder")
                rs = entity.keypoints.get("right_shoulder")
                lh = entity.keypoints.get("left_hip")
                rh = entity.keypoints.get("right_hip")
                if ls and rs and lh and rh:
                    mid_shoulder_x = (ls[0] + rs[0]) / 2
                    mid_shoulder_y = (ls[1] + rs[1]) / 2
                    mid_hip_x = (lh[0] + rh[0]) / 2
                    mid_hip_y = (lh[1] + rh[1]) / 2
                    dx = mid_shoulder_x - mid_hip_x
                    dy = mid_hip_y - mid_shoulder_y
                    tilt_deg = math.degrees(math.atan2(abs(dx), max(dy, 1e-3)))
                    posture_desc += f" (脊柱倾角:{tilt_deg:.1f}°)"

                if act_threat.value > highest_threat.value:
                    highest_threat = act_threat
                    highest_desc = desc
        elif cat == 'person':
            posture_desc = "外围常规"
            if zone == VehicleZone.ZONE_1_OUTER and highest_threat.value < ThreatLevel.L1_NOTICE.value:
                highest_threat = ThreatLevel.L1_NOTICE
                highest_desc = "外围目标经过"

        # 若为人员，运行 OSNet ReID 提取
        if cat == 'person' and engine.reid_extractor is not None:
            with profiler.profile_model("OSNet-ReID", condition=f"ID:{tid} 人员ReID特征建档"):
                feat = engine.reid_extractor.extract_from_frame(frame, entity.bbox)
                entity.reid_feature = feat
                if engine.session_manager:
                    sess, _ = engine.session_manager.match_or_create_session(
                        track_uid=f"cam{camera_id}_trk{tid}",
                        feature=feat,
                        current_time=now,
                        quality_score=conf,
                        camera_id=camera_id
                    )
                    entity.person_session_id = sess.session_id

        # 若为人员且在 4 米内，运行 FaceID 机会主义识别状态机 (内含 SCRFD 与 ArcFace 耗时剖析)
        if cat == 'person' and engine.faceid_pipeline is not None and dist <= 4.0:
            face_state = engine.faceid_pipeline.process_target_face(frame, entity, now)
            entity.face_state = face_state
            if entity.face_id_name:
                posture_desc += f" [Face:{entity.face_id_name}]"

        print(f"{tid:<4} | {cat:<10} | {conf:<8.2f} | {f'{ground_pt[0]},{ground_pt[1]}':<14} | {dist:<10.2f}m | {zone.value:<12} | {posture_desc:<12}")
        processed_entities.append(entity)

    print("-" * 70)

    # 5. 图像渲染与可视化标注
    annotated = frame.copy()

    # 结束单帧性能剖析并输出统计报表
    frame_metrics = profiler.end_frame(
        threat_level=highest_threat.value,
        target_count=len(processed_entities)
    )
    print("\n" + profiler.format_single_frame_report(frame_metrics))

    # 绘制地面同心防区与车身敏感固定 ROI
    draw_vehicle_and_zones(annotated, engine)

    # 绘制检测框与骨骼点连线
    for ent in processed_entities:
        bx, by, bw, bh = ent.bbox.x, ent.bbox.y, ent.bbox.w, ent.bbox.h

        # 防区颜色映射
        if ent.zone == VehicleZone.ZONE_2_INNER:
            box_color = (0, 160, 255) # 橙色 (Zone 2 核心研判区 0~3.0m)
        elif ent.zone == VehicleZone.ZONE_1_OUTER:
            box_color = (255, 200, 0) # 黄色 (Zone 1 外围初筛区 3.0~6.0m)
        else:
            box_color = (180, 180, 180)

        # 绘制包围框
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), box_color, 2)

        # 绘制接地脚点标靶
        gc_x, gc_y = ent.bbox.bottom_center
        cv2.circle(annotated, (gc_x, gc_y), 5, (0, 255, 255), -1)
        cv2.line(annotated, (gc_x - 8, gc_y), (gc_x + 8, gc_y), (0, 255, 255), 1)

        # 绘制目标标签条
        label_text = f"ID:{ent.track_id} {ent.category} {ent.confidence:.2f} | {ent.ground_dist:.2f}m | {ent.zone.value[:6]}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (bx, max(0, by - 20)), (bx + tw + 6, max(20, by)), box_color, -1)
        cv2.putText(annotated, label_text, (bx + 3, max(14, by - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # 绘制 17 骨骼点连线
        if ent.keypoints:
            draw_skeleton(annotated, ent.keypoints)

    # 绘制顶部 HUD 综合诊断条与性能指标
    hud_h = 105
    hud_overlay = annotated.copy()
    cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (20, 20, 25), -1)
    cv2.addWeighted(hud_overlay, 0.85, annotated, 0.15, 0, annotated)
    cv2.line(annotated, (0, hud_h), (w, hud_h), (70, 70, 80), 1)

    threat_colors = {
        ThreatLevel.L0_SAFE: (0, 255, 120),
        ThreatLevel.L1_NOTICE: (255, 200, 0),
        ThreatLevel.L2_WARNING: (0, 140, 255),
        ThreatLevel.L3_CRITICAL: (0, 0, 255)
    }
    tc = threat_colors.get(highest_threat, (200, 200, 200))

    cv2.putText(annotated, "CAMPING GUARD | SINGLE IMAGE DIAGNOSTIC", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 2)
    cv2.putText(annotated, f"Detected: {len(processed_entities)} targets", (15, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 240, 255), 1)
    cv2.putText(annotated, f"Threat Level: {highest_threat.value}", (235, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, tc, 2)
    cv2.putText(annotated, f"Analysis: {highest_desc}", (420, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 220, 250), 1)

    # 绘制 E2E 与各模型调用耗时指标条
    perf_summary = f"E2E: {frame_metrics.e2e_ms:.1f}ms ({frame_metrics.fps:.1f} FPS)"
    model_snippets = []
    for m_name, m_info in frame_metrics.get_model_summary().items():
        s_name = m_name.replace("ArcFace-MobileFaceNet", "ArcFace").replace("YOLO11n-Det", "YOLO").replace("SCRFD-500M", "SCRFD").replace("RTMPose-s", "RTMPose").replace("OSNet-ReID", "ReID")
        model_snippets.append(f"{s_name}:{m_info['total_ms']:.1f}ms({m_info['count']}次)")
    if model_snippets:
        perf_summary += " | " + " | ".join(model_snippets)
    cv2.putText(annotated, perf_summary, (15, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 240, 255), 1)

    # 保存输出图像
    out_path_obj = Path(output_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path_obj), annotated)
    print(f"\n[OK] 分析图已成功导出至: {out_path_obj.resolve()}")

    # 弹窗预览
    if not headless:
        try:
            print("[*] 正在打开可视化窗口 (按任意键或 ESC 关闭窗口)...")
            cv2.imshow("Camping Guard - Single Image Diagnostic", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("[!] 无图形显示环境，已直接保存图片。")


def main():
    parser = argparse.ArgumentParser(description="智能汽车露营守卫 - 单图端到端算法测试工具")
    parser.add_argument("--image", type=str, default="sample_fisheye.jpg",
                        help="输入的单张鱼眼测试图片路径 (默认: sample_fisheye.jpg)")
    parser.add_argument("--config", type=str, default="vehicle_calib.json",
                        help="全车统一标定配置文件 (包含相机外参、几何尺寸、车身固定 ROI 与阈值，默认: vehicle_calib.json)")
    parser.add_argument("--detector", type=str, default="models/yolo/yolo11n-fisheye.onnx",
                        help="YOLO 目标检测模型 (默认: models/yolo/yolo11n-fisheye.onnx)")
    parser.add_argument("--pose", type=str, default="models/pose/rtmpose-s.onnx",
                        help="RTMPose 姿态估计模型 (默认: models/pose/rtmpose-s.onnx)")
    parser.add_argument("--reid", type=str, default="models/reid/osnet_x0_25.onnx",
                        help="OSNet 人员重识别模型 (默认: models/reid/osnet_x0_25.onnx)")
    parser.add_argument("--face-det", type=str, default="models/insightface/det_500m.onnx",
                        help="SCRFD 人脸检测模型 (默认: models/insightface/det_500m.onnx)")
    parser.add_argument("--face-rec", type=str, default="models/insightface/w600k_mbf.onnx",
                        help="ArcFace 人脸特征识别模型 (默认: models/insightface/w600k_mbf.onnx)")
    parser.add_argument("--camera-id", type=int, default=1, choices=[0, 1, 2, 3],
                        help="相机通道编号: 0前, 1左, 2右, 3后 (默认 1 左侧)")
    parser.add_argument("--camera-height", type=float, default=None,
                        help="手动覆盖相机离地安装高度 (米，默认从 --config 读取)")
    parser.add_argument("--camera-pitch", type=float, default=None,
                        help="手动覆盖相机俯仰角 (度，向下为负，默认从 --config 读取)")
    parser.add_argument("--conf", type=float, default=0.35, help="检测置信度阈值 (默认 0.35)")
    parser.add_argument("--bbox", type=str, default=None,
                        help="手动指定测试目标包围框 (x,y,w,h)，例如 '200,260,80,160'")
    parser.add_argument("--output", type=str, default="tmp/annotated_result.jpg",
                        help="标注后的分析结果图保存路径 (默认: tmp/annotated_result.jpg)")
    parser.add_argument("--headless", action="store_true", help="无头模式运行，不弹窗仅保存分析图")

    args = parser.parse_args()

    manual_box = None
    if args.bbox:
        parts = [int(v.strip()) for v in args.bbox.split(",")]
        if len(parts) == 4:
            manual_box = tuple(parts)

    # 若默认 sample_fisheye.jpg 不存在，自动尝试从 sample_fisheye_left.mp4 截取
    if not os.path.exists(args.image) and args.image == "sample_fisheye.jpg":
        if os.path.exists("sample_fisheye_left.mp4"):
            cap = cv2.VideoCapture("sample_fisheye_left.mp4")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite("sample_fisheye.jpg", frame)
                print("[*] 已从 sample_fisheye_left.mp4 截取默认测试帧: sample_fisheye.jpg")
            cap.release()

    run_single_image_test(
        image_path=args.image,
        config_path=args.config,
        detector_path=args.detector,
        pose_path=args.pose,
        reid_path=args.reid,
        face_det_path=args.face_det,
        face_rec_path=args.face_rec,
        camera_id=args.camera_id,
        camera_height=args.camera_height,
        camera_pitch=args.camera_pitch,
        conf_threshold=args.conf,
        manual_bbox=manual_box,
        output_path=args.output,
        headless=args.headless
    )



if __name__ == "__main__":
    main()
