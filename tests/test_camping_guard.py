"""
露营守卫算法系统 - 核心逻辑与边界工况单元测试套件
验证状态流转、中继攻击防御、Zone3常驻ROI、多模态联合确认、火情监控与相机遮挡降级
"""

import time
import numpy as np
import pytest

from camping_guard.engine import CampingGuardEngine
from camping_guard.types import GuardState, ThreatLevel, OperatingMode, VehicleZone, ActionType
from camping_guard.config import GuardConfig

def test_system_activation():
    """测试系统激活条件与模式判定"""
    engine = CampingGuardEngine()

    # 1. 满足所有条件激活 (默认 P档, EPB, 门锁, 开关开, 座椅有人)
    success, msg = engine.activate()
    assert success is True
    assert engine.state_machine.state == GuardState.S0_STANDBY
    assert engine.state_machine.operating_mode == OperatingMode.IN_CAR

    # 2. 模拟车门未关好
    engine2 = CampingGuardEngine()
    engine2.vehicle_bus.all_doors_locked = False
    success2, msg2 = engine2.activate()
    assert success2 is False
    assert "门/后备箱未完全锁闭" in msg2

def test_relay_attack_defense():
    """测试退出机制中的中继攻击防御与非法越权拦截 (第一章.2)"""
    engine = CampingGuardEngine()
    engine.activate()

    # 钥匙离车遥远 (例如 > 5.0m)
    now = time.time()
    can_exit, new_threat, desc = engine.state_machine.handle_unlock_request(
        unlock_source="REMOTE_BLE_RELAY",
        uwb_distance_to_car=6.2,
        current_time=now
    )

    # 核心验收点：系统坚决不退出守卫，并且升级至 L3 紧急高危！
    assert can_exit is False
    assert engine.state_machine.is_active is True
    assert new_threat == ThreatLevel.L3_CRITICAL
    assert "中继攻击" in desc
    assert engine.state_machine.state == GuardState.S2_ANALYSIS

def test_normal_exit_with_uwb():
    """测试合法钥匙在场下的正常安全退出"""
    engine = CampingGuardEngine()
    engine.activate()

    now = time.time()
    can_exit, new_threat, desc = engine.state_machine.handle_unlock_request(
        unlock_source="PHYSICAL_KEY",
        uwb_distance_to_car=0.8, # <= 1.5m
        current_time=now
    )

    assert can_exit is True
    assert engine.state_machine.is_active is False
    assert new_threat is None

def test_zone3_resident_daemon():
    """测试阶段 0.5 Zone 3 ROI 常驻扰动与直跳 S2"""
    engine = CampingGuardEngine()
    engine.activate()
    assert engine.state_machine.state == GuardState.S0_STANDBY

    # 构造触发画面: 相机 1 (左侧) 门把手 ROI 发生大面积像素变化
    h, w = 480, 640
    frame_bg = np.zeros((h, w, 3), dtype=np.uint8)
    frame_disturbed = np.zeros((h, w, 3), dtype=np.uint8)
    # 在门把手 ROI (160~220, 260~310) 区域填充白色高亮
    frame_disturbed[260:310, 160:220] = 255

    # 暖机前几帧
    now = time.time()
    for _ in range(5):
        engine.zone3_daemon.process_frames({1: frame_bg}, now)

    # 注入多帧扰动
    triggered_events = []
    for i in range(4):
        events = engine.zone3_daemon.process_frames({1: frame_disturbed}, now + i * 0.1)
        triggered_events.extend(events)

    assert len(triggered_events) > 0
    assert triggered_events[0]["roi_id"] == "handle_fl"

def test_multimodal_joint_confirmation():
    """测试多源时序联合确认 (视觉单独 L2 vs 联合确认升 L3)"""
    engine = CampingGuardEngine()
    engine.activate()
    # 模拟陌生人无钥匙在场 (UWB 钥匙不在现场)
    engine.vehicle_bus.uwb_key_pos = None

    now = time.time()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 场景 A: 仅有视觉手部拉门动作，无 CAN 信号 -> L2 预警
    # 模拟在门把手附近 (dist = 0.44m <= 0.8m)
    det = [{
        'track_id': 101,
        'category': 'person',
        'conf': 0.85,
        'bbox': (180, 345, 60, 120)
    }]

    # 持续执行 1.8 秒以满足视觉动作阈值
    engine.pose_classifier.action_states[101] = {
        "action": ActionType.DOOR_HANDLE_PULL,
        "start_time": now - 1.8,
        "frame_count": 20
    }

    out_A = engine.process_tick({1: dummy_frame}, det)
    assert out_A["threat_level"] == ThreatLevel.L2_WARNING.value
    assert "暂无车身接触信号" in out_A["description"]

    # 场景 B: 注入门把手被拉动 CAN 硬件事件 -> 毫秒级联合确认升 L3
    engine.vehicle_bus.inject_door_handle_pull("handle_fl")
    out_B = engine.process_tick({1: dummy_frame}, det)
    assert out_B["threat_level"] == ThreatLevel.L3_CRITICAL.value
    assert "联合确认" in out_B["description"]

def test_campfire_monitoring():
    """测试露营火情专用研判 (稳定火源 vs 失控蔓延)"""
    engine = CampingGuardEngine()
    engine.activate()

    now = time.time()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. 远距离稳定火源 (有人在场) -> L0
    fire_det_far = [
        {'track_id': 201, 'category': 'fire', 'conf': 0.90, 'bbox': (300, 150, 40, 40)},
        {'track_id': 202, 'category': 'person', 'conf': 0.88, 'bbox': (320, 150, 30, 80)}
    ]
    out_1 = engine.process_tick({1: dummy_frame}, fire_det_far)
    assert out_1["threat_level"] in [ThreatLevel.L0_SAFE.value, ThreatLevel.L1_NOTICE.value]

    # 2. 火焰失控蔓延逼近车体 (< 3m 且面积增大) -> L3，且动作不含驱逐
    # 模拟火焰逼近画面下方 (近距离)
    fire_det_close = [
        {'track_id': 201, 'category': 'fire', 'conf': 0.95, 'bbox': (250, 380, 150, 100)}
    ]
    # 手动触发面积扩大模拟
    engine.fire_monitor.fire_records[201] = {
        "first_seen": now - 60,
        "last_seen": now,
        "initial_area": 1000,
        "last_area": 15000,
        "last_person_seen_time": 0.0
    }

    out_2 = engine.process_tick({1: dummy_frame}, fire_det_close)
    assert out_2["threat_level"] == ThreatLevel.L3_CRITICAL.value
    assert "失控火源蔓延" in out_2["description"]
    # 验证响应不包含车外驱逐爆闪 (火情响应为撤离提醒)
    for act in out_2["actions"]:
        assert "爆闪" not in act

def test_camera_occlusion_degradation():
    """测试相机遮挡故障降级 (单路 > 5s 报 L2, 双路报 L3)"""
    engine = CampingGuardEngine()
    engine.activate()

    now = time.time()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 单路相机 (cam 2) 刚被遮挡 1 秒
    engine.state_machine.camera_occlusion_starts[2] = now - 1.0
    out_1 = engine.process_tick({1: dummy_frame}, [], occluded_cams=[2])
    assert out_1["threat_level"] == ThreatLevel.L0_SAFE.value

    # 单路相机被遮挡超 5 秒 -> 触发 L2
    engine.state_machine.camera_occlusion_starts[2] = now - 5.5
    out_2 = engine.process_tick({1: dummy_frame}, [], occluded_cams=[2])
    assert out_2["threat_level"] == ThreatLevel.L2_WARNING.value
    assert "遮挡超过 5 秒" in out_2["description"]

    # 两路相机同时被遮挡 -> 升级 L3 紧急故障
    engine.state_machine.camera_occlusion_starts[1] = now - 6.0
    engine.state_machine.camera_occlusion_starts[2] = now - 6.0
    out_3 = engine.process_tick({1: dummy_frame}, [], occluded_cams=[1, 2])
    assert out_3["threat_level"] == ThreatLevel.L3_CRITICAL.value
    assert "多路相机" in out_3["description"]

if __name__ == "__main__":
    pytest.main(["-v", __file__])
