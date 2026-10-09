"""
智能汽车露营守卫系统 - 交互式端到端全流程仿真控制台 (run_simulation.py)
基于 camping_guard_algorithm_flow_revised_1.md 方案
支持多防区动态渲染、车身固定 ROI 常驻监视、CAN/IMU 硬件信号注入与中继攻击模拟

交互快捷键说明:
  [1] - 生成外围 Zone 1 陌生人走过 (L1 提示)
  [2] - 模拟陌生人在核心 Zone 2 徘徊滞留 (>10s -> L2 预警)
  [3] - 模拟人员贴近车身尝试拉门撬锁 (纯视觉 -> L2 预警)
  [h] - 【硬件注入】向 CAN 注入门把手被机械拉动信号 (时序联合确认 -> L3 紧急高危)
  [k] - 【硬件注入】向 IMU 注入车身冲击震动峰值 (踢踹联合确认 -> L3 紧急高危)
  [l] - 【安全测试】注入车门解锁信号 (测试钥匙不在场时的【中继攻击】防御拦截！)
  [u] - 切换车主 UWB 钥匙在线/离线 (测试白名单与中继攻击鉴权)
  [f] - 注入营地篝火/烟雾 (测试稳定有人看管 vs 蔓延逼近 L3 撤离报警)
  [o] - 模拟单路/多路相机遮挡故障 (测试 5s 延时 L2 与双路遮挡 L3 降级)
  [m] - 切换工作模式 (车内模式 vs 车外模式)
  [c] - 清空当前场景所有目标
  [q] 或 ESC - 退出仿真
"""

import cv2
import numpy as np
import time
import sys

from camping_guard.engine import CampingGuardEngine
from camping_guard.types import GuardState, ThreatLevel, OperatingMode, VehicleZone, ActionType
from camping_guard.config import GuardConfig

def draw_simulation_hud(frame: np.ndarray, engine: CampingGuardEngine, tick_result: dict, fps: float, sim_status: str):
    h, w = frame.shape[:2]

    # 1. 绘制顶部信息仪表盘黑底背景
    hud_h = 155
    hud_overlay = frame.copy()
    cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (20, 20, 25), -1)
    cv2.addWeighted(hud_overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, hud_h), (w, hud_h), (70, 70, 80), 1)

    # 2. 状态与威胁等级颜色标识
    threat_str = tick_result.get("threat_level", "L0")
    if "L3" in threat_str:
        threat_color = (0, 0, 255) # 红色高亮
    elif "L2" in threat_str:
        threat_color = (0, 140, 255) # 橙黄色
    elif "L1" in threat_str:
        threat_color = (255, 200, 0) # 青黄色
    else:
        threat_color = (0, 255, 120) # 绿色安全

    # 3. 绘制文字信息
    state_str = tick_result.get("state", "")
    mode_str = tick_result.get("mode", "")
    desc_str = tick_result.get("description", "")
    vlm_summary = tick_result.get("vlm_summary") or "慢通路静默等待"

    cv2.putText(frame, f"CAMPING GUARD V2.0 | FPS: {fps:.1f} | Mode: {mode_str}", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

    cv2.putText(frame, f"State: {state_str}", (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f"Threat: {threat_str}", (260, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, threat_color, 2)

    uwb_online = (engine.vehicle_bus.uwb_key_pos is not None)
    uwb_text = "UWB Key: PRESENT (<=1.5m)" if uwb_online else "UWB Key: AWAY / NONE"
    uwb_color = (0, 255, 120) if uwb_online else (0, 0, 255)
    cv2.putText(frame, uwb_text, (480, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, uwb_color, 2)

    cv2.putText(frame, f"Analysis: {desc_str}", (15, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 240, 255), 1)

    cv2.putText(frame, f"VLM Report: {vlm_summary}", (15, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 255, 180), 1)

    latest_hash = engine.recorder.prev_hash[:12] + "..." if engine.recorder.prev_hash != engine.recorder.GENESIS_HASH else "GENESIS"
    clock_info = f"Clock: {engine.recorder.clock_source.value} | HashChain: {latest_hash} (Blocks: {len(engine.recorder.manifest)})"
    cv2.putText(frame, clock_info, (15, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 210, 240), 1)

    # 4. 底部执行动作横幅
    actions = tick_result.get("actions", [])
    if actions:
        bar_y = h - 35
        cv2.rectangle(frame, (0, bar_y), (w, h), (10, 10, 15), -1)
        action_summary = " | ".join(actions)
        act_color = (0, 0, 255) if "L3" in threat_str else (0, 220, 255)
        cv2.putText(frame, f"ACTION: {action_summary}", (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, act_color, 1)

    # 5. 右上角提示模拟状态
    if sim_status:
        cv2.putText(frame, f"[EVENT] {sim_status}", (15, hud_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 255), 2)

def draw_vehicle_and_zones(frame: np.ndarray, engine: CampingGuardEngine, camera_id: int = 1):
    """绘制地面同心防区与当前相机对应的车身敏感部件固定 ROI"""
    h, w = frame.shape[:2]

    # 1. 查找当前相机的专属标定配置
    cam_calib = engine.calib.get_camera(camera_id) if hasattr(engine, 'calib') and engine.calib else None
    cam_name = cam_calib.name if cam_calib else f"Camera_{camera_id}"
    body_side = getattr(cam_calib, "vehicle_body_side", "left" if camera_id == 1 else ("right" if camera_id == 2 else ("front" if camera_id == 0 else "rear")))

    # 2. 绘制专属防区多边形 (优先使用标定中的 zone1_polygon / zone2_polygon)
    calib_w, calib_h = (getattr(cam_calib, "resolution", (640, 480))) if cam_calib else (640, 480)
    scale_x = w / float(calib_w)
    scale_y = h / float(calib_h)

    if cam_calib and cam_calib.zone1_polygon and cam_calib.zone2_polygon:
        z1_poly = np.array([[int(p[0] * scale_x), int(p[1] * scale_y)] for p in cam_calib.zone1_polygon], np.int32)
        z2_poly = np.array([[int(p[0] * scale_x), int(p[1] * scale_y)] for p in cam_calib.zone2_polygon], np.int32)
    elif camera_id in (1, 2):
        # 左右侧视相机专属防区 (环绕侧面车身展开)
        z1_poly = np.array([[int(w*0.06), int(h*0.30)], [int(w*0.94), int(h*0.30)],
                            [int(w*0.97), int(h*0.75)], [int(w*0.03), int(h*0.75)]], np.int32)
        z2_poly = np.array([[int(w*0.12), int(h*0.42)], [int(w*0.88), int(h*0.42)],
                            [int(w*0.94), int(h*0.92)], [int(w*0.06), int(h*0.92)]], np.int32)
    else:
        # 前/后视相机专属防区 (沿纵深展开的梯形投影)
        z1_poly = np.array([[int(w*0.10), int(h*0.35)], [int(w*0.90), int(h*0.35)],
                            [int(w*0.98), int(h*0.95)], [int(w*0.02), int(h*0.95)]], np.int32)
        z2_poly = np.array([[int(w*0.22), int(h*0.50)], [int(w*0.78), int(h*0.50)],
                            [int(w*0.92), int(h*0.95)], [int(w*0.08), int(h*0.95)]], np.int32)

    cv2.polylines(frame, [z1_poly], True, (60, 60, 90), 1, cv2.LINE_AA)
    cv2.polylines(frame, [z2_poly], True, (90, 90, 120), 1, cv2.LINE_AA)

    side_label_map = {"front": "前向", "left": "左侧", "right": "右侧", "rear": "后向"}
    side_cn = side_label_map.get(body_side, "车周")
    cv2.putText(frame, f"[{side_cn} Cam{camera_id}] Zone 1 (3.0~6.0m)", (z1_poly[0][0] + 10, z1_poly[0][1] + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (110, 110, 140), 1)
    cv2.putText(frame, f"[{side_cn} Cam{camera_id}] Zone 2 (0~3.0m)", (z2_poly[0][0] + 10, z2_poly[0][1] + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (140, 140, 170), 1)

    # 3. 严格仅绘制属于当前相机 camera_id 的车身敏感部件与防撞区
    for roi in engine.zone3_daemon.rois:
        if roi.camera_id != camera_id:
            continue
        scaled_poly = [[int(pt[0] * scale_x), int(pt[1] * scale_y)] for pt in roi.polygon]
        pts = np.array(scaled_poly, np.int32)

        # 若该 ROI 当前被扰动触发，显示红色高亮闪烁
        is_disturbed = any(d["roi_id"] == roi.roi_id for d in engine.active_roi_disturbances)
        if is_disturbed:
            color = (0, 0, 255)
            thickness = 2
        elif getattr(roi, "roi_type", "part") == "body":
            color = (180, 140, 60) # 车身大面积防碰撞区低饱和度轮廓
            thickness = 1
        else:
            color = (0, 220, 255) # 敏感部件 (门把手/充电口/锁扣)
            thickness = 1
        cv2.polylines(frame, [pts], True, color, thickness)
        rx, ry = pts[0]
        label = roi.name if getattr(roi, "roi_type", "part") == "body" else roi.roi_id
        cv2.putText(frame, label, (rx, max(15, ry - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

def draw_skeleton(frame: np.ndarray, keypoints: dict):
    """绘制 COCO 17 点人体骨骼连线与关节点"""
    links = [
        ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"), ("right_eye", "right_ear"),
        ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle")
    ]
    # 绘制骨骼连线
    for p1, p2 in links:
        if p1 in keypoints and p2 in keypoints:
            pt1 = (int(keypoints[p1][0]), int(keypoints[p1][1]))
            pt2 = (int(keypoints[p2][0]), int(keypoints[p2][1]))
            conf1, conf2 = keypoints[p1][2], keypoints[p2][2]
            if conf1 > 0.4 and conf2 > 0.4:
                cv2.line(frame, pt1, pt2, (0, 255, 180), 2, cv2.LINE_AA)

    # 绘制关节点圆圈
    for name, (kx, ky, conf) in keypoints.items():
        if conf > 0.4:
            color = (0, 0, 255) if "wrist" in name else ((255, 0, 255) if "ankle" in name else (0, 220, 255))
            cv2.circle(frame, (int(kx), int(ky)), 3, color, -1)

def run_simulation_loop(config_path: str = "vehicle_calib.json"):
    print("=" * 70)
    print(" 智能汽车露营守卫算法端到端仿真系统已启动")
    print(f" 挂载标定文件: {config_path}")
    print(" 基于 camping_guard_algorithm_flow_revised_1.md 方案构建")
    print(" 快捷键操作指南:")
    print("   [1] 注入陌生人常规经过 Zone 1 (L1)")
    print("   [2] 注入陌生人 Zone 2 核心区徘徊 (>10s 升 L2)")
    print("   [3] 注入目标近车尝试拉门手部动作 (纯视觉 L2)")
    print("   [h] 硬件注入: CAN 门把手机械拉动 (时序联合确认秒升 L3!)")
    print("   [k] 硬件注入: IMU 车身冲击震动峰值 (踢踹联合确认秒升 L3!)")
    print("   [l] 安全对抗测试: 注入解锁信号 (测试钥匙不在场时的中继攻击防御)")
    print("   [u] 切换车主 UWB 钥匙在线/远离 (测试白名单豁免)")
    print("   [f] 注入露营营火 (测试稳定有人火源 vs 失控蔓延 L3)")
    print("   [o] 模拟相机遮挡故障降级 (单路 5s 报 L2, 双路报 L3)")
    print("   [8] 注入纯视觉假阳性 (测试 VLM 语义复核安全降级通道至 L1)")
    print("   [9] 审计防篡改哈希链完整性 (检验 SHA-256 签名与防篡改报告)")
    print("   [m] 切换模式 (车内模式 vs 车外模式)")
    print("   [c] 清空场景所有目标")
    print("   [q] 退出仿真")
    print("=" * 70)

    engine = CampingGuardEngine(config_path=config_path, use_bytetrack=True)
    success, msg = engine.activate()
    print(f"[*] {msg}")


    # 初始默认车主 UWB 钥匙在车内/离场
    engine.vehicle_bus.uwb_key_pos = None

    # 仿真目标池
    sim_targets = []
    sim_event_msg = "系统就绪，按下按键注入测试工况"
    sim_event_time = time.time()

    # 视频帧规格
    w, h = 640, 480
    base_frame = np.zeros((h, w, 3), dtype=np.uint8)

    # 模拟环境底图 (带简单透视地平面)
    cv2.rectangle(base_frame, (0, 0), (w, int(h * 0.35)), (30, 25, 20), -1) # 远景
    cv2.rectangle(base_frame, (0, int(h * 0.35)), (w, h), (40, 45, 40), -1) # 地面

    fps_count = 0
    fps_start = time.time()
    current_fps = 30.0

    occluded_cameras = []
    door_pulling_sim = False
    door_pull_start = 0.0

    while True:
        loop_now = time.time()

        # 统计 FPS
        fps_count += 1
        if loop_now - fps_start >= 1.0:
            current_fps = fps_count / (loop_now - fps_start)
            fps_count = 0
            fps_start = loop_now

        # 构造当帧图像
        frame = base_frame.copy()

        # 如果相机被遮挡，将画面全黑
        if 1 in occluded_cameras:
            frame[:, :] = 10
            cv2.putText(frame, "[CAMERA 1 OCCLUDED / BLIND]", (int(w*0.25), int(h*0.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        draw_vehicle_and_zones(frame, engine)

        # 更新拉门模拟持续时间
        if door_pulling_sim:
            engine.pose_classifier.action_states[101] = {
                "action": ActionType.DOOR_HANDLE_PULL,
                "start_time": door_pull_start,
                "frame_count": 30
            }

        # 绘制仿真目标并送入引擎
        dets_input = []
        for t in sim_targets:
            bx, by, bw, bh = t['bbox']
            # 绘制目标包围框
            cat = t['category']
            color = (255, 180, 0) if cat == 'person' else ((0, 160, 255) if cat == 'fire' else (0, 255, 255))
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
            cv2.putText(frame, f"ID:{t['track_id']} {cat}", (bx, by - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            dets_input.append({
                'track_id': t['track_id'],
                'category': cat,
                'conf': t.get('conf', 0.88),
                'bbox': (bx, by, bw, bh)
            })

        # 执行引擎 tick
        tick_res = engine.process_tick(
            camera_frames={1: frame},
            raw_detections=dets_input,
            occluded_cams=occluded_cameras
        )

        # 绘制活跃目标的骨骼关键点 (若已由 RTMPose 提取)
        for t in engine.active_targets:
            if t.keypoints:
                draw_skeleton(frame, t.keypoints)

        # 渲染 HUD 仪表盘
        if loop_now - sim_event_time > 4.0:
            sim_event_msg = ""
        draw_simulation_hud(frame, engine, tick_res, current_fps, sim_event_msg)

        # 显示窗口
        cv2.imshow("Smart Camping Guard System - Full Pipeline Simulation", frame)
        key = cv2.waitKey(30) & 0xFF

        # 按键交互
        if key in [ord('q'), 27]: # q or ESC
            print("[*] 正在退出仿真系统...")
            break

        elif key == ord('1'):
            # 注入陌生人在 Zone 1 常规经过
            sim_targets = [{
                'track_id': 101, 'category': 'person', 'conf': 0.85,
                'bbox': (180, 200, 50, 100) # 远端 Zone 1
            }]
            door_pulling_sim = False
            sim_event_msg = "已注入陌生人在 Zone 1 外围经过 (应触发 L1 提示)"
            sim_event_time = loop_now

        elif key == ord('2'):
            # 注入陌生人在 Zone 2 核心区停留
            sim_targets = [{
                'track_id': 101, 'category': 'person', 'conf': 0.88,
                'bbox': (200, 270, 70, 140) # 中间 Zone 2
            }]
            door_pulling_sim = False
            sim_event_msg = "已注入陌生人进入 Zone 2 核心研判区 (停留 >10s 将升 L2)"
            sim_event_time = loop_now

        elif key == ord('3'):
            # 注入目标近车并尝试拉门 (纯视觉)
            sim_targets = [{
                'track_id': 101, 'category': 'person', 'conf': 0.92,
                'bbox': (175, 345, 65, 125) # 贴车极近距离 (<0.5m) Zone 2
            }]
            door_pulling_sim = True
            door_pull_start = loop_now - 2.0 # 模拟已抓握 2 秒
            sim_event_msg = "已注入目标在门把手附近持续拉扯 (纯视觉未达 L3，触发 L2 预警)"
            sim_event_time = loop_now

        elif key == ord('h'):
            # 硬件注入: CAN 门把手机械拉动
            engine.vehicle_bus.inject_door_handle_pull("handle_fl")
            sim_event_msg = "【CAN 硬件注入】门把手被机械拉动！(联合确认秒升 L3 紧急高危)"
            sim_event_time = loop_now

        elif key == ord('k'):
            # 硬件注入: 车身 IMU 剧烈冲击震动
            engine.vehicle_bus.inject_vibration_peak(9.2)
            sim_event_msg = "【IMU 硬件注入】车身检测到剧烈冲击！(踢踹/碰撞联合确认升 L3)"
            sim_event_time = loop_now

        elif key == ord('l'):
            # 安全对抗测试: 注入解锁信号 (测试中继攻击拦截)
            uwb_d = 6.0 if engine.vehicle_bus.uwb_key_pos is None else 0.8
            can_exit, new_threat, desc = engine.state_machine.handle_unlock_request(
                unlock_source="SIMULATED_BLE_RELAY",
                uwb_distance_to_car=uwb_d,
                current_time=loop_now
            )
            if not can_exit:
                sim_event_msg = "【中继攻击防御触发】检测到解锁但现场无 UWB 钥匙！坚决不退出守卫并升 L3！"
            else:
                sim_event_msg = "【正常退出】UWB 钥匙鉴权通过，系统安全终止守卫"
            sim_event_time = loop_now

        elif key == ord('u'):
            # 切换 UWB 车钥匙在场/远离
            if engine.vehicle_bus.uwb_key_pos is None:
                engine.vehicle_bus.uwb_key_pos = (0.5, 0.8) # 在车旁
                sim_event_msg = "【UWB 钥匙】车主钥匙已在车身旁 0.9m 范围内 (白名单激活)"
            else:
                engine.vehicle_bus.uwb_key_pos = None
                sim_event_msg = "【UWB 钥匙】车主钥匙已离开现场 (无白名单保护)"
            sim_event_time = loop_now

        elif key == ord('f'):
            # 注入露营营火
            sim_targets = [
                {'track_id': 201, 'category': 'fire', 'conf': 0.93, 'bbox': (240, 360, 120, 80)}
            ]
            door_pulling_sim = False
            sim_event_msg = "已注入营地失控火源逼近 (<3m，触发 L3 紧急撤离且不驱逐)"
            sim_event_time = loop_now

        elif key == ord('o'):
            # 模拟相机遮挡
            if not occluded_cameras:
                occluded_cameras = [1]
                sim_event_msg = "【故障注入】相机 1 已被完全遮挡 (>5s 将报 L2)"
            elif len(occluded_cameras) == 1:
                occluded_cameras = [1, 2]
                sim_event_msg = "【故障注入】多路相机 (1, 2) 同时被遮挡！(立即升 L3)"
            else:
                occluded_cameras = []
                sim_event_msg = "【故障恢复】所有相机视野恢复正常"
            sim_event_time = loop_now

        elif key == ord('m'):
            # 切换工作模式
            if engine.state_machine.operating_mode == OperatingMode.IN_CAR:
                engine.state_machine.operating_mode = OperatingMode.OUT_CAR
                sim_event_msg = "已切换为【车外模式】(车内无人，侧重车外声光爆闪驱逐)"
            else:
                engine.state_machine.operating_mode = OperatingMode.IN_CAR
                sim_event_msg = "已切换为【车内模式】(车主就寝，侧重主驾座椅震动与静默推送)"
            sim_event_time = loop_now

        elif key == ord('8'):
            # 注入纯视觉环境假阳性 (测试 VLM 语义复核安全降级通道)
            if hasattr(engine.vlm_reviewer.backend, 'force_false_alarm'):
                engine.vlm_reviewer.backend.force_false_alarm = True
            sim_targets = [
                {'track_id': 301, 'category': 'person', 'conf': 0.70, 'bbox': (200, 300, 50, 100)}
            ]
            sim_event_msg = "【VLM 测试】注入纯视觉假阳性干扰 (树叶/光影) -> 经 VLM 语义复核安全降至 L1"
            sim_event_time = loop_now

        elif key == ord('9'):
            # 审计防篡改哈希链完整性
            is_valid, report_msg, bad_idx = engine.recorder.verify_hash_chain()
            if is_valid:
                sim_event_msg = f"【防篡改审计通过】哈希链无缝闭环 (已验 {len(engine.recorder.manifest)} 个区块)"
            else:
                sim_event_msg = f"【警报：哈希链被篡改】区块 {bad_idx} 签名不一致！"
            sim_event_time = loop_now

        elif key == ord('c'):
            sim_targets = []
            door_pulling_sim = False
            if hasattr(engine.vlm_reviewer.backend, 'force_false_alarm'):
                engine.vlm_reviewer.backend.force_false_alarm = False
            sim_event_msg = "已清空场景所有动态目标并重置假阳性标志"
            sim_event_time = loop_now

    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="智能汽车露营守卫算法交互仿真控制台")
    parser.add_argument("--config", type=str, default="vehicle_calib.json",
                        help="全车统一标定配置文件 (默认: vehicle_calib.json)")
    args = parser.parse_args()
    run_simulation_loop(config_path=args.config)

