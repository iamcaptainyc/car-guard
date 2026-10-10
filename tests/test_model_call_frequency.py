"""
端到端小模型调用频次与帧间缓存持久化测试 (tests/test_model_call_frequency.py)
验证优化后:
1. STrack 跨帧持久化语义特征 (ReID、FaceState、Keypoints 等)
2. OSNet-ReID 仅在目标首次建档时执行 1 次，后续帧自动复用
3. RTMPose 在 Zone 2 维持 10 FPS 研判限频 (100ms 刷新率)，中间帧复用骨骼点
4. SCRFD-500M 遵循 retry_interval_sec 机会主义重试冷却，杜绝每帧盲目调用
"""

import numpy as np
import time
from unittest.mock import MagicMock

from camping_guard.engine import CampingGuardEngine
from camping_guard.config import GuardConfig
from camping_guard.types import BoundingBox, VehicleZone, FaceIDState, OperatingMode
from camping_guard.perception.bytetrack import ByteTracker, STrack

def test_bytetrack_semantic_persistence():
    """验证 ByteTracker 的 STrack 跨帧更新能完整保留语义属性"""
    tracker = ByteTracker()
    now = 1000.0

    # 帧 1: 首次检测到目标
    dets_f1 = [{'bbox': (100, 100, 50, 150), 'conf': 0.9, 'category': 'person'}]
    entities_f1 = tracker.update(dets_f1, current_time=now)
    assert len(entities_f1) == 1
    t1 = entities_f1[0]
    tid = t1.track_id

    # 业务层注入 ReID 特征与人脸状态
    mock_reid = np.random.randn(512).astype(np.float32)
    st = tracker.get_track(tid)
    assert st is not None
    st.reid_feature = mock_reid
    st.face_state = FaceIDState.UNAVAILABLE
    st.keypoints = {"nose": (125.0, 120.0)}

    # 帧 2: 同一目标移动微小位移 (33ms 之后)
    dets_f2 = [{'bbox': (102, 101, 50, 150), 'conf': 0.9, 'category': 'person'}]
    entities_f2 = tracker.update(dets_f2, current_time=now + 0.033)
    assert len(entities_f2) == 1
    t2 = entities_f2[0]
    assert t2.track_id == tid

    # 验证 TrackedEntity 成功继承 STrack 持久化属性
    assert t2.reid_feature is not None
    assert np.array_equal(t2.reid_feature, mock_reid)
    assert t2.face_state == FaceIDState.UNAVAILABLE
    assert t2.keypoints == {"nose": (125.0, 120.0)}

def test_engine_reid_single_call_per_track():
    """验证多帧连续输入下，OSNet-ReID 仅调用 1 次建档"""
    engine = CampingGuardEngine(use_bytetrack=True)
    engine.vehicle_bus.uwb_key_pos = None
    engine.state_machine.operating_mode = OperatingMode.OUT_CAR
    engine.state_machine.is_system_active = True

    # Mock ReID 提取器并统计调用次数
    reid_call_count = 0
    def mock_extract(frame, bbox):
        nonlocal reid_call_count
        reid_call_count += 1
        feat = np.ones(512, dtype=np.float32)
        return feat / np.linalg.norm(feat)

    engine.reid_extractor = MagicMock()
    engine.reid_extractor.extract_from_frame.side_effect = mock_extract

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    now = 1000.0

    # 模拟连续 10 帧 (30 FPS, 每帧 33ms) 检测到同一个人
    for i in range(10):
        dets = [{'bbox': (200 + i, 150, 80, 200), 'conf': 0.95, 'category': 'person'}]
        engine.process_tick(
            camera_frames={0: dummy_frame},
            raw_detections=dets,
            current_time=now + i * 0.033
        )

    # 核心断言: 10 帧中 ReID 应该只调用 1 次！
    assert reid_call_count == 1, f"ReID 预期调用 1 次，实际调用了 {reid_call_count} 次！"

def test_engine_rtmpose_10fps_throttling():
    """验证在 Zone 2 核心研判区，RTMPose 被平滑限频至 10 FPS (100ms 刷新率)"""
    engine = CampingGuardEngine(use_bytetrack=True)
    engine.vehicle_bus.uwb_key_pos = None
    engine.state_machine.operating_mode = OperatingMode.OUT_CAR
    engine.state_machine.is_system_active = True

    pose_call_count = 0
    def mock_estimate(frame, boxes):
        nonlocal pose_call_count
        pose_call_count += 1
        return [{"nose": (300.0, 200.0), "left_wrist": (280.0, 250.0)}]

    engine.pose_estimator = MagicMock()
    engine.pose_estimator.estimate_keypoints.side_effect = mock_estimate

    # 确保距离估算器将目标映射至 Zone 2 (<= 3.0m)
    engine.distance_estimator.estimate_ground_distance = MagicMock(return_value=1.8)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    now = 1000.0

    # 模拟 10 帧输入 (总时长 300ms, 间隔 33ms)
    # 在 100ms 限频下，0ms (帧0), 100ms (帧3), 200ms (帧6), 300ms (帧9) 应该触发，共 4 次
    for i in range(10):
        t_now = now + i * 0.0333
        dets = [{'bbox': (200, 150, 80, 200), 'conf': 0.95, 'category': 'person'}]
        engine.process_tick(
            camera_frames={0: dummy_frame},
            raw_detections=dets,
            current_time=t_now
        )
        # 中间未触发推理的帧，也能通过缓存复用拿到 keypoints
        target = engine.active_targets[0]
        assert target.keypoints is not None, f"第 {i} 帧未能复用 keypoints 缓存！"

    # 核心断言: 10 帧中 RTMPose 不得超过 4 次 (原先为 10 次)
    assert pose_call_count <= 4, f"RTMPose 未能限频至 10 FPS，调用了 {pose_call_count} 次！"

def test_faceid_scrfd_cooldown_throttling():
    """验证 FaceID SCRFD 在无法检出人脸或未匹配时遵循 retry_interval_sec 限频"""
    engine = CampingGuardEngine(use_bytetrack=True)
    engine.vehicle_bus.uwb_key_pos = None
    engine.state_machine.operating_mode = OperatingMode.OUT_CAR
    engine.state_machine.is_system_active = True

    scrfd_call_count = 0
    mock_scrfd = MagicMock()
    def mock_detect(crop):
        nonlocal scrfd_call_count
        scrfd_call_count += 1
        # 模拟背身：未检测出人脸
        return np.empty((0, 5)), np.empty((0, 5, 2))

    mock_scrfd.detect.side_effect = mock_detect
    engine.faceid_pipeline.detector = mock_scrfd
    engine.faceid_pipeline.retry_interval_sec = 0.5 # 0.5s 冷却

    # 目标进入 4m 内 (距离 2.5m 满足 faceid 探测门禁)
    engine.distance_estimator.estimate_ground_distance = MagicMock(return_value=2.5)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    now = 1000.0

    # 模拟 30 帧输入 (1 秒时间跨度，33ms 间隔)
    for i in range(30):
        t_now = now + i * 0.0333
        dets = [{'bbox': (200, 150, 80, 200), 'conf': 0.95, 'category': 'person'}]
        engine.process_tick(
            camera_frames={0: dummy_frame},
            raw_detections=dets,
            current_time=t_now
        )

    # 在 1 秒跨度下，0.5s 冷却应仅允许触发约 2~3 次 SCRFD (原先 30 帧会触发 30 次！)
    assert scrfd_call_count <= 3, f"SCRFD 冷却失效，30 帧内调用了 {scrfd_call_count} 次！"

def test_occlusion_reid_track_recovery():
    """验证目标被偶尔遮挡、IoU 失效重现后，通过 ReID 维持原有 track_id 不跳变"""
    engine = CampingGuardEngine(use_bytetrack=True)
    engine.vehicle_bus.uwb_key_pos = None
    engine.state_machine.operating_mode = OperatingMode.OUT_CAR
    engine.state_machine.is_system_active = True

    # 模拟一致的 ReID 512 维特征
    fixed_feat = np.ones(512, dtype=np.float32)
    fixed_feat /= np.linalg.norm(fixed_feat)

    engine.reid_extractor = MagicMock()
    engine.reid_extractor.extract_from_frame.return_value = fixed_feat

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    now = 1000.0

    # 阶段 1: 目标在场 5 帧 (位置 x=100)
    for i in range(5):
        dets = [{'bbox': (100, 100, 50, 150), 'conf': 0.95, 'category': 'person'}]
        engine.process_tick(
            camera_frames={0: dummy_frame},
            raw_detections=dets,
            current_time=now + i * 0.033
        )
    assert len(engine.active_targets) == 1
    init_tid = engine.active_targets[0].track_id

    # 阶段 2: 目标被车辆/立柱遮挡 10 帧 (画面中无检测框)
    for i in range(5, 15):
        engine.process_tick(
            camera_frames={0: dummy_frame},
            raw_detections=[],
            current_time=now + i * 0.033
        )
    assert len(engine.active_targets) == 0

    # 阶段 3: 目标从遮挡后走出，出现在新位置 (x=250，与原有位置 IoU 为 0)
    dets_reappear = [{'bbox': (250, 100, 50, 150), 'conf': 0.95, 'category': 'person'}]
    engine.process_tick(
        camera_frames={0: dummy_frame},
        raw_detections=dets_reappear,
        current_time=now + 15 * 0.033
    )

    # 核心断言: 重现后目标必须通过 ReID 恢复初始 ID，绝对不得变成新 ID！
    assert len(engine.active_targets) == 1
    reappeared_tid = engine.active_targets[0].track_id
    assert reappeared_tid == init_tid, f"遮挡后 ID 发生跳变！原 ID={init_tid}, 重现后 ID={reappeared_tid}"

