"""
Phase 3 专项测试套件 (tests/test_phase3.py)
验证 RTMPose 姿态估计管线、骨骼点几何动作判定 (贴窗窥探/拉门/踢踹)、多模态联合确认与猛兽冲刺
"""

import time
import numpy as np
import pytest

from camping_guard.perception.pose_estimator import RTMPoseEstimator, KEYPOINT_NAMES
from camping_guard.analysis.pose_action import PoseActionClassifier
from camping_guard.types import TrackedEntity, BoundingBox, VehicleZone, ActionType, ThreatLevel
from camping_guard.config import GuardConfig
from camping_guard.engine import CampingGuardEngine

def test_rtmpose_crop_warp_and_simcc_decoding():
    """测试 RTMPose 仿射自适应裁切与 SimCC 1D 坐标分类解码"""
    estimator = RTMPoseEstimator()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = (200, 150, 80, 180) # (x, y, w, h)

    # 1. 仿射裁剪
    crop_img, inv_matrix = estimator.crop_and_warp(dummy_frame, bbox)
    assert crop_img.shape == (256, 192, 3) # 严格长宽比 192x256

    # 2. 模拟 SimCC 峰值输出
    # simcc_x: [17, 384], simcc_y: [17, 512]
    simcc_x = np.zeros((17, 384), dtype=np.float32)
    simcc_y = np.zeros((17, 512), dtype=np.float32)

    # 假设鼻子 (0) 在裁剪图中心 (x=96, y=128), 乘以 split_ratio 2.0 -> idx_x=192, idx_y=256
    simcc_x[0, 192] = 0.95
    simcc_y[0, 256] = 0.95

    kpts = estimator.decode_simcc(simcc_x, simcc_y, inv_matrix)
    assert "nose" in kpts
    nose_x, nose_y, conf = kpts["nose"]
    assert conf >= 0.90
    # 验证还原回全图坐标接近 bbox 中心 (240, 240)
    assert abs(nose_x - 240.0) < 5.0
    assert abs(nose_y - 240.0) < 5.0

def test_keypoint_driven_window_peeking():
    """测试基于脊柱倾角与头部距离的贴窗窥探判定 (前倾角>25° 且持续>=3s)"""
    classifier = PoseActionClassifier()
    now = time.time()

    # 1. 直立姿态 (脊柱倾角 ~= 0°) -> 正常经过
    straight_kpts = {
        "nose": (200.0, 260.0, 0.9),
        "left_shoulder": (180.0, 280.0, 0.9),
        "right_shoulder": (220.0, 280.0, 0.9),
        "left_hip": (180.0, 360.0, 0.9),
        "right_hip": (220.0, 360.0, 0.9)
    }
    target_straight = TrackedEntity(
        track_id=1, category='person', confidence=0.9,
        bbox=BoundingBox(170, 250, 60, 150),
        ground_dist=0.4, # <= 0.5m
        zone=VehicleZone.ZONE_2_INNER,
        keypoints=straight_kpts
    )
    act1, lvl1, desc1 = classifier.classify_action(target_straight, now)
    assert act1 != ActionType.WINDOW_PEEKING

    # 2. 上身显著前倾姿态 (中肩点 x=260, 中髋点 x=180, dx=80, dy=80 -> angle = 45° > 25°)
    peeking_kpts = {
        "nose": (280.0, 260.0, 0.95),
        "left_shoulder": (240.0, 280.0, 0.95),
        "right_shoulder": (280.0, 280.0, 0.95), # 中肩点 (260, 280)
        "left_hip": (160.0, 360.0, 0.95),
        "right_hip": (200.0, 360.0, 0.95)       # 中髋点 (180, 360)
    }
    target_peeking = TrackedEntity(
        track_id=2, category='person', confidence=0.9,
        bbox=BoundingBox(160, 250, 120, 150),
        ground_dist=0.35, # 贴窗 <= 0.5m
        zone=VehicleZone.ZONE_2_INNER,
        keypoints=peeking_kpts
    )

    # 刚开始前倾 (< 3.0s) -> L1 提示
    act2, lvl2, desc2 = classifier.classify_action(target_peeking, now)
    assert act2 == ActionType.WINDOW_PEEKING
    assert lvl2 == ThreatLevel.L1_NOTICE

    # 持续前倾超过 3.0 秒 -> 升级 L2 预警
    act3, lvl3, desc3 = classifier.classify_action(target_peeking, now + 3.2)
    assert act3 == ActionType.WINDOW_PEEKING
    assert lvl3 == ThreatLevel.L2_WARNING
    assert "上身前倾>25°" in desc3

def test_keypoint_driven_door_handle_pull_and_l3_fusion():
    """测试手腕进入门把手 ROI 判定与 CAN 硬件信号时序联合确认升 L3"""
    engine = CampingGuardEngine()
    engine.activate()
    engine.vehicle_bus.uwb_key_pos = None # 陌生人无钥匙
    now = time.time()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 左前门把手 ROI 在 [(160, 260), (220, 260), (220, 310), (160, 310)]
    # 模拟手腕关键点进入门把手区域 (x=190, y=285)
    handle_pull_kpts = {
        "left_wrist": (190.0, 285.0, 0.95),
        "right_wrist": (195.0, 290.0, 0.92)
    }

    det = [{
        'track_id': 101, 'category': 'person', 'conf': 0.92,
        'bbox': (150, 250, 70, 150)
    }]

    # 1. 模拟手部持续停留超过 1.5 秒 (纯视觉 -> L2 预警)
    engine.pose_classifier.action_states[101] = {
        "action": ActionType.DOOR_HANDLE_PULL,
        "start_time": now - 1.8,
        "frame_count": 25
    }

    out_visual = engine.process_tick({1: dummy_frame}, det)
    assert out_visual["threat_level"] == ThreatLevel.L2_WARNING.value
    assert "暂无车身接触信号" in out_visual["description"]

    # 2. 注入门把手拉动 CAN 硬件信号 -> 秒升 L3 紧急高危
    engine.vehicle_bus.inject_door_handle_pull("handle_fl")
    out_fusion = engine.process_tick({1: dummy_frame}, det)
    assert out_fusion["threat_level"] == ThreatLevel.L3_CRITICAL.value
    assert "拉门撬锁确立" in out_fusion["description"]

def test_keypoint_driven_kicking_and_imu_fusion():
    """测试脚踝高速运动与车身 IMU 震动峰值联合确认为 L3"""
    engine = CampingGuardEngine()
    engine.activate()
    engine.vehicle_bus.uwb_key_pos = None
    now = time.time()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    det = [{
        'track_id': 202, 'category': 'person', 'conf': 0.90,
        'bbox': (200, 320, 60, 140)
    }]

    # 模拟脚踝向车身踢出的速度序列
    engine.pose_classifier.action_states[202] = {
        "action": ActionType.INTENTIONAL_KICK,
        "start_time": now,
        "frame_count": 5
    }

    # 纯视觉踢踹 -> L2 预警
    out_kick = engine.process_tick({1: dummy_frame}, det)
    assert out_kick["threat_level"] == ThreatLevel.L2_WARNING.value

    # 注入 IMU 震动峰值 -> 联合确认升 L3
    engine.vehicle_bus.inject_vibration_peak(8.8)
    out_kick_fusion = engine.process_tick({1: dummy_frame}, det)
    assert out_kick_fusion["threat_level"] == ThreatLevel.L3_CRITICAL.value
    assert "蓄意踢踹确立" in out_kick_fusion["description"]

def test_beast_sprint_multi_frame_l3_confirmation():
    """测试大型猛兽高速径向逼近 (连续 3 帧确认 -> L3 紧急)"""
    engine = CampingGuardEngine()
    engine.activate()
    now = time.time()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 模拟大型猛兽向车身高速逼近 (y 逐帧增加，由外围 Zone 1 冲刺进入核心 Zone 2 并连续维持 3 帧以上)
    out_f = None
    for i in range(6):
        beast_det = [{
            'track_id': 303, 'category': 'large_beast', 'conf': 0.85,
            'bbox': (240, 200 + i * 35, 80, 80)
        }]
        time.sleep(0.04) # 模拟真实帧间时间差以计算物理速度 (冲刺速度 > 1.2 m/s)
        out_f = engine.process_tick({0: dummy_frame}, beast_det)

    assert out_f["threat_level"] == ThreatLevel.L3_CRITICAL.value
    assert "大型猛兽" in out_f["description"]

def test_engine_top_down_rtmpose_pipeline():
    """测试 Top-Down 功耗阶梯机制：远处 Zone 1 不跑姿态，进入 Zone 2 自动唤醒 RTMPose"""
    engine = CampingGuardEngine()
    engine.activate()
    # 模拟陌生人无车钥匙
    engine.vehicle_bus.uwb_key_pos = None
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. 远端 Zone 1 目标 (接地点 y=230, dist ~ 5.5m, 位于 Zone 1 标定多边形内)
    det_z1 = [{'track_id': 10, 'category': 'person', 'conf': 0.88, 'bbox': (200, 150, 40, 80)}]
    engine.process_tick({0: dummy_frame}, det_z1)
    target_z1 = engine.active_targets[0]
    assert target_z1.zone == VehicleZone.ZONE_1_OUTER
    # S1 远端节能，未触发骨骼点提取
    assert target_z1.keypoints is None

    # 2. 目标进入 Zone 2 核心研判区 (y=300, dist ~ 1.5m)
    det_z2 = [{'track_id': 10, 'category': 'person', 'conf': 0.88, 'bbox': (200, 300, 60, 120)}]
    engine.process_tick({0: dummy_frame}, det_z2)
    target_z2 = engine.active_targets[0]
    assert target_z2.zone in [VehicleZone.ZONE_2_INNER, VehicleZone.ZONE_3_CONTACT]
    # S2 自动唤醒 RTMPose，成功注入 17 个骨骼点！
    assert target_z2.keypoints is not None
    assert "left_wrist" in target_z2.keypoints
    assert len(target_z2.keypoints) == 17

if __name__ == "__main__":
    pytest.main(["-v", __file__])
