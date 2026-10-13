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

def test_faceid_empty_whitelist_zero_calls():
    """验证当白名单库为空时，FaceID 短路退出，SCRFD-500M 调用次数为 0"""
    engine = CampingGuardEngine(use_bytetrack=True)
    engine.vehicle_bus.uwb_key_pos = None
    engine.state_machine.operating_mode = OperatingMode.OUT_CAR
    engine.state_machine.is_system_active = True

    scrfd_call_count = 0
    mock_scrfd = MagicMock()
    def mock_detect(crop):
        nonlocal scrfd_call_count
        scrfd_call_count += 1
        return np.empty((0, 5)), np.empty((0, 5, 2))

    mock_scrfd.detect.side_effect = mock_detect
    engine.faceid_pipeline.detector = mock_scrfd
    engine.faceid_pipeline.whitelist_db.clear()
    engine.distance_estimator.estimate_ground_distance = MagicMock(return_value=2.5)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    now = 1000.0
    for i in range(10):
        dets = [{'bbox': (200, 150, 80, 200), 'conf': 0.95, 'category': 'person'}]
        engine.process_tick(
            camera_frames={0: dummy_frame},
            raw_detections=dets,
            current_time=now + i * 0.0333
        )
    assert scrfd_call_count == 0, f"空人脸白名单库下 SCRFD 应短路 0 调用，实际调用 {scrfd_call_count} 次"


def test_faceid_scrfd_cooldown_throttling():
    """验证已注册白名单时，FaceID SCRFD 在无法检出人脸或未匹配时遵循 retry_interval_sec 限频"""
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
    engine.faceid_pipeline.register_whitelist_feature("owner_1", np.ones(512, dtype=np.float32))
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
    assert 1 <= scrfd_call_count <= 3, f"SCRFD 冷却异常，30 帧内调用了 {scrfd_call_count} 次！"


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


def test_parallel_ensemble_adaptive_fire_substream():
    """验证双流集成检测器在无火情时按 1/3 降频调用火焰检测，检出火情后立即自动升至 1:1 逐帧追踪"""
    from camping_guard.perception.detector import ParallelEnsembleDetector, VisualFlameDetector

    main_calls = 0
    fire_calls = 0
    should_emit_fire = False

    class DummyMainDet:
        model_path = "yolo11n.engine"
        session = None
        last_gpu_compute_ms = 5.0
        last_preprocess_ms = 1.0
        last_postprocess_ms = 1.0
        def detect(self, frame, conf_threshold=None):
            nonlocal main_calls
            main_calls += 1
            return [{'bbox': (10, 10, 50, 50), 'conf': 0.9, 'category': 'person'}]

    class DummyFireDet(VisualFlameDetector):
        model_path = "yolov8n_fire.engine"
        session = None
        last_gpu_compute_ms = 4.0
        last_preprocess_ms = 1.0
        last_postprocess_ms = 0.5
        def detect(self, frame, conf_threshold=None):
            nonlocal fire_calls
            fire_calls += 1
            if should_emit_fire:
                return [{'bbox': (200, 200, 60, 60), 'conf': 0.88, 'category': 'fire'}]
            return []

    ensemble = ParallelEnsembleDetector(DummyMainDet(), DummyFireDet(), fire_detect_interval=3)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 阶段 1: 无火情连续 9 帧 -> 主检测调用 9 次，火焰子流仅在第 1, 4, 7 帧调用 (共 3 次)
    for _ in range(9):
        ensemble.detect(dummy_frame)
    assert main_calls == 9
    assert fire_calls == 3, f"无火情 9 帧预期调用 3 次火焰检测，实际调用 {fire_calls} 次"

    # 阶段 2: 后续帧检出火情后，应立即切回 1:1 每帧调用
    should_emit_fire = True
    for _ in range(3):
        res = ensemble.detect(dummy_frame)
        assert any(d['category'] == 'fire' for d in res)
    assert fire_calls == 3 + 3


def test_vlm_cooldown_and_threat_escalation_breakthrough():
    """验证 VLM 30s 冷却防风暴机制以及 L2->L3 威胁升级即时破冰重评"""
    from camping_guard.review.vlm_reviewer import AsyncVLMReviewer
    from camping_guard.types import ThreatEventRecord, ThreatLevel, ActionType, GuardState, OperatingMode

    reviewer = AsyncVLMReviewer(enable_worker_thread=False, vlm_cooldown_sec=30.0)
    reviewer._is_running = True  # 模拟异步工作流开启但不自动消费队列，精确校验入队与冷却门禁

    t0 = 2000.0
    evt_l2 = ThreatEventRecord(
        event_id="EVT_1",
        timestamp=t0,
        operating_mode=OperatingMode.OUT_CAR,
        description="猛兽进入防区",
        threat_level=ThreatLevel.L2_WARNING,
        target_id=1,
        target_category="large_beast",
        action_type=ActionType.BEAST_PROXIMITY,
        target_distance=1.2,
        target_dwell_time=5.0,
        has_contact_signal=False,
        person_session_id="P_001"
    )

    # 首次提交应入队成功
    reviewer.submit_review_async(evt_l2)
    assert reviewer.total_submitted_tasks == 1
    assert reviewer.cooldown_skipped_count == 0

    # 模拟后台推理完成并写入真实战报缓存
    dedup_key = reviewer._get_dedup_key(evt_l2)
    reviewer._target_cooldown_map[dedup_key]["in_flight"] = False
    reviewer._target_cooldown_map[dedup_key]["last_real_result"] = {
        "summary_text": "一只美洲黑熊在草地徘徊",
        "final_threat": ThreatLevel.L2_WARNING,
    }
    while not reviewer.task_queue.empty():
        reviewer.task_queue.get_nowait()

    # 8 秒后同一目标同一 L2 动作仍在 30s 冷却期内 -> 应跳过提交并复用上一轮真实 VLM 结论
    evt_l2_repeat = ThreatEventRecord(
        event_id="EVT_1_REPEAT",
        timestamp=t0 + 8.0,
        operating_mode=OperatingMode.OUT_CAR,
        description="猛兽进入防区",
        threat_level=ThreatLevel.L2_WARNING,
        target_id=1,
        target_category="large_beast",
        action_type=ActionType.BEAST_PROXIMITY,
        target_distance=1.1,
        target_dwell_time=13.0,
        has_contact_signal=False,
        person_session_id="P_001"
    )
    cached_res = reviewer.review_event_async(evt_l2_repeat)
    summary = reviewer.get_audit_summary()
    assert summary["cooldown_skipped_count"] == 1
    assert reviewer.total_submitted_tasks == 1
    assert "美洲黑熊" in cached_res["summary_text"]

    # 12 秒后威胁等级从 L2 升级为 L3_CRITICAL (或动作突变为猛兽冲刺) -> 立即破冰放行入队！
    evt_l3 = ThreatEventRecord(
        event_id="EVT_2_ESCALATED",
        timestamp=t0 + 12.0,
        operating_mode=OperatingMode.OUT_CAR,
        description="猛兽进入防区",
        threat_level=ThreatLevel.L3_CRITICAL,
        target_id=1,
        target_category="large_beast",
        action_type=ActionType.BEAST_SPRINT,
        target_distance=0.3,
        target_dwell_time=17.0,
        has_contact_signal=True,
        person_session_id="P_001"
    )
    reviewer.submit_review_async(evt_l3)
    assert reviewer.total_submitted_tasks == 2



def test_bytetrack_kalman_dynamic_dt_scale():
    """验证 KalmanFilter2D.predict 在跳帧 (dt_scale > 1) 时按真实时间跨度成比例预测位移"""
    from camping_guard.perception.bytetrack import KalmanFilter2D

    kf = KalmanFilter2D()
    mean, cov = kf.initiate(np.array([100.0, 100.0, 50.0, 100.0]))
    # 注入水平恒定速度 vx = 10 px/frame
    mean[4] = 10.0

    m1, _ = kf.predict(mean.copy(), cov.copy(), dt_scale=1.0)
    m3, _ = kf.predict(mean.copy(), cov.copy(), dt_scale=3.0)

    assert np.isclose(m1[0] - 100.0, 10.0)
    assert np.isclose(m3[0] - 100.0, 30.0)


def test_profiler_3stage_and_json_diagnostic_export(tmp_path):
    """验证 PipelineProfiler 的三段式计时、VLM 并发争抢对比、阶段分解及 JSON 报告导出"""
    from camping_guard.profiler import PipelineProfiler

    prof = PipelineProfiler()
    prof.set_runtime_context(video_name="bear_test.mp4", skipped_frames=2, output_written_frames=100)

    # 帧 1: VLM 空闲
    prof.start_frame(1)
    with prof.profile_stage("Step1_2_3_Detect_Track"):
        prof.record_model_call(
            "MegaDetector",
            duration_ms=28.0,
            gpu_ms=22.0,
            preprocess_ms=3.5,
            postprocess_ms=2.5
        )
    prof.end_frame(threat_level="L1_NOTICE", target_count=1, vlm_is_inferencing=False)
    prof.record_stage_duration("Render_and_Output", 2.0)

    # 帧 2: VLM 繁忙
    prof.start_frame(2)
    with prof.profile_stage("Step1_2_3_Detect_Track"):
        prof.record_model_call(
            "MegaDetector",
            duration_ms=42.0,
            gpu_ms=34.0,
            preprocess_ms=4.5,
            postprocess_ms=3.5
        )
    prof.end_frame(threat_level="L2_WARNING", target_count=1, vlm_is_inferencing=True)
    prof.record_stage_duration("Render_and_Output", 2.5)

    out_file = tmp_path / "diag_report.json"
    saved_path = prof.export_json_report(out_file)
    assert out_file.is_file()

    import json
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["schema_version"] == "2.0"
    assert data["runtime_context"]["video_name"] == "bear_test.mp4"
    assert data["e2e_summary"]["vlm_idle_e2e"]["count"] == 1
    assert data["e2e_summary"]["vlm_busy_e2e"]["count"] == 1
    assert "MegaDetector" in data["models"]
    md_info = data["models"]["MegaDetector"]
    assert md_info["avg_preprocess_ms"] == 4.0
    assert md_info["avg_gpu_compute_ms"] == 28.0
    assert md_info["avg_postprocess_ms"] == 3.0
    assert "Step1_2_3_Detect_Track" in data["pipeline_stages"]
    assert "Render_and_Output" in data["pipeline_stages"]
    assert len(data["slowest_frames_top10"]) == 2

