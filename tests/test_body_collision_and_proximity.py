"""
车身防碰撞区域运动检测、极近距离防丢与 VLM/录像联动专项测试套件
(tests/test_body_collision_and_proximity.py)

验证内容:
1. 目标进入 0.5m 内未进入 ROI 时绝不降级为 L0 安全 (彻底修复 <0.5m 无威胁 Bug)
2. 车身区域 (侧围门板、保险杠、车顶) 运动扰动检测 (模拟邻车开门碰撞、倒木、外物砸击)
3. 运动检测触发自动录像 (lock_event_clip) 与 VLM 多模态语义复核
4. 车身运动扰动与整车 IMU 震动信号联合确认升 L3
"""

import time
import numpy as np
import pytest

from camping_guard.engine import CampingGuardEngine
from camping_guard.types import (
    TrackedEntity, BoundingBox, VehicleZone, ActionType,
    ThreatLevel, GuardState, ContactSensorEvent, Zone3ROI
)
from camping_guard.config import GuardConfig
from camping_guard.analysis.pose_action import PoseActionClassifier
from camping_guard.analysis.fusion import MultiSourceFusionEngine
from camping_guard.review.vlm_reviewer import AsyncVLMReviewer, MockVLMBackend


def test_near_proximity_never_safe():
    """测试当目标进入 0.5m 以内且未触发任何 ROI 时，系统绝不会判定为 L0 无威胁"""
    classifier = PoseActionClassifier()
    now = time.time()

    # 1. 模拟车辆目标在 0.35m 处 (例如邻车贴身停放或极近距离倒车)
    car_target = TrackedEntity(
        track_id=101,
        category='car',
        confidence=0.85,
        bbox=BoundingBox(100, 200, 200, 250),
        ground_dist=0.35, # < 0.5m
        zone=VehicleZone.ZONE_3_CONTACT
    )
    act_car, lvl_car, desc_car = classifier.classify_action(car_target, now)
    assert lvl_car != ThreatLevel.L0_SAFE, "0.35m 外部车辆绝不能判定为 L0 安全！"
    assert lvl_car in [ThreatLevel.L1_NOTICE, ThreatLevel.L2_WARNING]

    # 2. 模拟人员目标在 0.4m 处 (未触碰把手)
    person_target = TrackedEntity(
        track_id=102,
        category='person',
        confidence=0.88,
        bbox=BoundingBox(150, 180, 80, 220),
        ground_dist=0.40, # < 0.5m
        zone=VehicleZone.ZONE_3_CONTACT
    )
    act_person, lvl_person, desc_person = classifier.classify_action(person_target, now)
    assert lvl_person != ThreatLevel.L0_SAFE, "0.4m 贴身人员绝不能判定为 L0 安全！"
    assert lvl_person in [ThreatLevel.L1_NOTICE, ThreatLevel.L2_WARNING]

    # 3. 引擎端到端输入目标：验证整体 highest_threat 绝不为 L0
    engine = CampingGuardEngine()
    engine.activate()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    dets = [{
        'track_id': 103,
        'category': 'car',
        'conf': 0.85,
        'bbox': (120, 200, 180, 260)
    }]
    tick_res = engine.process_tick({1: dummy_frame}, raw_detections=dets)
    assert tick_res["threat_level"] != ThreatLevel.L0_SAFE.value, "0.5m 内实体引擎端到端输出绝不能为 L0 安全"


def test_vehicle_body_roi_motion_detection_and_recording():
    """测试车身区域 (门板/侧围) 发生运动扰动时触发 VEHICLE_COLLISION、L2 告警、自动录像与 VLM 复核"""
    engine = CampingGuardEngine()
    engine.activate()

    # 构造一张黑色基准帧
    base_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 预热背景减除器 10 帧并重置冷却计时
    for _ in range(10):
        engine.process_tick({1: base_frame}, raw_detections=[])
    for r in engine.zone3_daemon.rois:
        engine.zone3_daemon.trigger_counters[r.roi_id] = 0
        engine.zone3_daemon.last_disturbance_time[r.roi_id] = 0.0

    # 模拟左侧门板区域发生剧烈运动 (例如邻车开门摆动撞击或树木枝条砸向门板)
    impact_frame = base_frame.copy()
    impact_frame[250:400, 150:450] = 255 # 剧烈白色前景运动

    all_disturbances = []
    collision_tick_res = None
    # 连续送入 3 帧确认 (触发 confirm_frames=2)
    for _ in range(3):
        res = engine.process_tick({1: impact_frame}, raw_detections=[])
        if engine.active_roi_disturbances:
            all_disturbances.extend(engine.active_roi_disturbances)
            collision_tick_res = res

    # 验证车身运动扰动被检出
    assert len(all_disturbances) > 0, "车身门板防碰撞 ROI 应当检出运动扰动"
    assert any(d.get("roi_id") == "body_flank_l" or d.get("roi_type") == "body" for d in all_disturbances), "应当包含车身门板防碰撞 ROI 扰动"

    # 验证触发帧的威胁等级与动作类型
    assert collision_tick_res is not None
    assert collision_tick_res["threat_level"] == ThreatLevel.L2_WARNING.value
    assert collision_tick_res["action_type"] == ActionType.VEHICLE_COLLISION.value
    assert "车身" in collision_tick_res["description"]

    # 验证自动触发了事件录像切片
    assert len(engine.recorder.pending_clips) > 0, "必须自动触发录像切片锁定"
    last_clip = list(engine.recorder.pending_clips.values())[-1]
    assert last_clip.threat_level_str == ThreatLevel.L2_WARNING.value

    # 验证 VLM 语义复核被唤醒并生成报告
    assert collision_tick_res["vlm_summary"] is not None
    assert len(collision_tick_res["vlm_summary"]) > 0


def test_vehicle_collision_fused_with_imu_escalates_to_l3():
    """测试车身区域运动碰撞伴随 IMU 震动冲击峰值时，时序联合确认为 L3 紧急高危"""
    engine = CampingGuardEngine()
    engine.activate()
    now = time.time()

    # 模拟在门板区域检出车身扰动 (mock zone3_daemon.process_frames)
    mock_disturbance = [{
        "roi_id": "body_flank_l",
        "name": "左侧车身门板防碰撞区",
        "camera_id": 1,
        "ratio": 0.25,
        "timestamp": now,
        "roi_type": "body"
    }]
    engine.zone3_daemon.process_frames = lambda frames, t: mock_disturbance

    # 向整车总线注入 IMU 震动冲击事件 (例如车门真实发生金属撞击碰撞)
    imu_event = ContactSensorEvent(
        timestamp=now,
        event_type="IMU_VIBRATION_PEAK",
        intensity=4.8,
        extra_info={"axis": "lateral_y", "peak_g": 3.2}
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tick_res = engine.process_tick({1: dummy_frame}, raw_detections=[], bus_events=[imu_event])

    # 验证升级为 L3 紧急
    assert tick_res["threat_level"] == ThreatLevel.L3_CRITICAL.value
    assert tick_res["action_type"] == ActionType.VEHICLE_COLLISION.value
    assert "车身碰撞确立" in tick_res["description"]
    assert "IMU" in tick_res["description"] or "震动" in tick_res["description"]


def test_vlm_reviewer_handles_vehicle_collision():
    """测试 VLM 多模态大模型对车身碰撞与非人员物理破坏场景的语义理解能力"""
    reviewer = AsyncVLMReviewer()

    # 1. 验证唤醒门控逻辑
    should_trigger = reviewer.should_trigger_vlm(
        category_conf=1.0,
        is_unknown_obstacle=False,
        fast_path_threat=ThreatLevel.L2_WARNING,
        action_type=ActionType.VEHICLE_COLLISION,
        has_contact_signal=False,
        is_zone3_roi_event=True
    )
    assert should_trigger is True, "VEHICLE_COLLISION 必须通过门控唤醒 VLM"

    # 2. 验证 MockVLMBackend 研判输出
    backend = MockVLMBackend()
    from camping_guard.types import ThreatEventRecord, OperatingMode
    event_no_contact = ThreatEventRecord(
        event_id="EVT_COLLISION_01",
        timestamp=time.time(),
        threat_level=ThreatLevel.L2_WARNING,
        operating_mode=OperatingMode.IN_CAR,
        action_type=ActionType.VEHICLE_COLLISION,
        description="车身区域运动扰动"
    )
    res1 = backend.infer(event_no_contact)
    assert res1["action_category"] == "VEHICLE_COLLISION"
    assert "车身" in res1["summary_text"] or "异常运动" in res1["summary_text"]

    event_with_contact = ThreatEventRecord(
        event_id="EVT_COLLISION_02",
        timestamp=time.time(),
        threat_level=ThreatLevel.L3_CRITICAL,
        operating_mode=OperatingMode.IN_CAR,
        action_type=ActionType.VEHICLE_COLLISION,
        description="车身碰撞确立",
        has_contact_signal=True
    )
    res2 = backend.infer(event_with_contact)
    assert res2["threat_escalation"] == "CONFIRM_OR_UPGRADE"
    assert "撞击" in res2["summary_text"] or "冲击" in res2["summary_text"]
