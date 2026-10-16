import time
import numpy as np
import pytest
from unittest.mock import MagicMock

from camping_guard.types import ThreatLevel, ActionType, ThreatEventRecord
from camping_guard.evidence.recorder import EvidenceRecorder
from camping_guard.review.vlm_reviewer import AsyncVLMReviewer, MockVLMBackend
from camping_guard.engine import CampingGuardEngine


def test_get_temporal_window_frames_and_discard():
    """测试环形缓冲 5 帧时序重采样提取与丢弃机制"""
    recorder = EvidenceRecorder(storage_dir="tests/output_test_evidence", buffer_duration_sec=10.0)

    # 模拟推入 2 秒 30 FPS 视频流 (共 60 帧)
    for i in range(60):
        t = 100.0 + i * (1.0 / 30.0)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = (i % 255, 0, 0)
        recorder.push_frame(img, current_time=t)

    # 提取 101.5 时刻前后 1.0s 窗口的 5 帧 (5 FPS)
    frames = recorder.get_temporal_window_frames(center_time=101.5, window_sec=1.0, target_fps=5.0, target_count=5)
    assert len(frames) == 5
    assert all(isinstance(f, np.ndarray) for f in frames)
    assert frames[0].shape == (100, 100, 3)

    # 锁存一个试探性切片
    event_id = "EVT_TEST_DISCARD_1"
    recorder.lock_event_clip(
        event_id=event_id,
        threat_level_str="L2",
        trigger_time=101.5
    )
    assert event_id in recorder.pending_clips

    # 丢弃切片测试
    ok = recorder.discard_clip(event_id)
    assert ok is True
    assert event_id not in recorder.pending_clips


def test_roi_vlm_high_risk_retention():
    """测试 ROI 扰动微动被 VLM 复核为高危动作时，继续存证录制"""
    engine = CampingGuardEngine()

    # 预先注入缓冲帧
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        engine.recorder.push_frame(frame, current_time=100.0 + i * 0.033)

    # 模拟 VLM 后端返回高危拉车门判定
    mock_backend = MagicMock()
    mock_backend.infer.return_value = {
        "action_category": "DOOR_HANDLE_TAMPERING",
        "confidence": 0.95,
        "is_human_contact": True,
        "is_false_alarm": False,
        "threat_escalation": "CONFIRM_OR_UPGRADE",
        "summary_text": "检测到外来人员伸手触碰并拉拽右侧车门把手。"
    }
    engine.vlm_reviewer.backend = mock_backend

    # 模拟检测到车门把手 ROI 扰动微动
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    engine.zone3_daemon.process_frames = MagicMock(return_value=[{
        "roi_id": "handle_rf",
        "name": "右前车门把手",
        "roi_type": "handle",
        "diff_ratio": 0.15
    }])

    tick_res = engine.process_tick(
        camera_frames={0: frame},
        current_time=101.0
    )
    engine.vlm_reviewer.wait_for_completion(timeout=2.0)

    # 验证是否启动了试探性录像
    assert engine._last_locked_event_id is not None
    locked_id = engine._last_locked_event_id
    assert locked_id in engine.recorder.pending_clips

    # 验证 VLM 复核回调是否维持了录像
    pending = engine.recorder.pending_clips[locked_id]
    assert pending.event_metadata.get("vlm_verified") is True
    assert pending.event_metadata.get("vlm_action") == "DOOR_HANDLE_TAMPERING"


def test_roi_vlm_false_alarm_discard():
    """测试 ROI 扰动微动被 VLM 复核为环境误报 (树叶/光影) 时，立即丢弃试探性录制切片"""
    engine = CampingGuardEngine()

    # 预先注入缓冲帧
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        engine.recorder.push_frame(frame, current_time=200.0 + i * 0.033)

    # 模拟 VLM 后端判定为误报
    mock_backend = MagicMock()
    mock_backend.infer.return_value = {
        "action_category": "NORMAL_PASS",
        "confidence": 0.90,
        "is_human_contact": False,
        "is_false_alarm": True,
        "threat_escalation": "DOWNGRADE_TO_L1",
        "summary_text": "树叶随风晃动投影在车身表面，无人员接触。"
    }
    engine.vlm_reviewer.backend = mock_backend

    # 模拟检测到车门把手微动
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    engine.zone3_daemon.process_frames = MagicMock(return_value=[{
        "roi_id": "handle_rr",
        "name": "右后车门把手",
        "roi_type": "handle",
        "diff_ratio": 0.08
    }])

    tick_res = engine.process_tick(
        camera_frames={0: frame},
        current_time=201.0
    )
    engine.vlm_reviewer.wait_for_completion(timeout=2.0)

    # 验证被 VLM 判定为误报后，试探性录像切片被丢弃
    assert len(engine.recorder.pending_clips) == 0
    assert engine._last_locked_event_id is None
