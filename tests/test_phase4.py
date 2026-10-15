"""
阶段四单元测试套件：端侧 VLM 异步复核与防篡改存证流水线
验证 VLM 唤醒门控、非阻塞异步队列、“只升不降”权限沙箱、视频拼装落盘与 SHA-256 防篡改哈希链
"""

import time
import json
import shutil
import tempfile
import hashlib
from pathlib import Path
import numpy as np
import cv2
import pytest

from camping_guard.types import ThreatLevel, ActionType, ThreatEventRecord, OperatingMode, ClockSource
from camping_guard.review.vlm_reviewer import AsyncVLMReviewer, MockVLMBackend
from camping_guard.evidence.recorder import EvidenceRecorder


def test_vlm_gating_conditions():
    """测试阶段 4.1：VLM 唤醒门控条件 (1~4 类事件准入与普通非准入)"""
    reviewer = AsyncVLMReviewer(enable_worker_thread=False)

    # 1. 低置信度 (<0.35) 准入
    assert reviewer.should_trigger_vlm(
        category_conf=0.25,
        is_unknown_obstacle=False,
        fast_path_threat=ThreatLevel.L1_NOTICE,
        action_type=ActionType.NORMAL_PASS,
        has_contact_signal=False,
        is_zone3_roi_event=False
    ) is True

    # 2. 未知异形障碍物准入
    assert reviewer.should_trigger_vlm(
        category_conf=0.90,
        is_unknown_obstacle=True,
        fast_path_threat=ThreatLevel.L1_NOTICE,
        action_type=ActionType.NORMAL_PASS,
        has_contact_signal=False,
        is_zone3_roi_event=False
    ) is True

    # 3. 快通路 L2/L3 威胁准入
    assert reviewer.should_trigger_vlm(
        category_conf=0.85,
        is_unknown_obstacle=False,
        fast_path_threat=ThreatLevel.L2_WARNING,
        action_type=ActionType.WINDOW_PEEKING,
        has_contact_signal=False,
        is_zone3_roi_event=False
    ) is True

    # 4. 无接触传感器的疑似装备盗窃准入
    assert reviewer.should_trigger_vlm(
        category_conf=0.80,
        is_unknown_obstacle=False,
        fast_path_threat=ThreatLevel.L1_NOTICE,
        action_type=ActionType.EQUIPMENT_THEFT,
        has_contact_signal=False,
        is_zone3_roi_event=False
    ) is True

    # 5. Zone 3 ROI 扰动准入
    assert reviewer.should_trigger_vlm(
        category_conf=0.90,
        is_unknown_obstacle=False,
        fast_path_threat=ThreatLevel.L1_NOTICE,
        action_type=ActionType.NORMAL_PASS,
        has_contact_signal=False,
        is_zone3_roi_event=True
    ) is True

    # 6. 常规高置信度经过 (L0/L1, 无扰动, 非盗窃) -> 严控算力发热，不唤醒 VLM
    assert reviewer.should_trigger_vlm(
        category_conf=0.88,
        is_unknown_obstacle=False,
        fast_path_threat=ThreatLevel.L1_NOTICE,
        action_type=ActionType.NORMAL_PASS,
        has_contact_signal=False,
        is_zone3_roi_event=False
    ) is False


def test_vlm_async_worker_queue_non_blocking():
    """测试阶段 4.2：VLM 异步多线程执行非阻塞性与回调机制"""
    # 模拟大模型端侧推理延迟 0.08 秒
    mock_backend = MockVLMBackend(simulated_delay_sec=0.08)
    reviewer = AsyncVLMReviewer(backend=mock_backend, enable_worker_thread=True)

    record = ThreatEventRecord(
        event_id="EVT_ASYNC_001",
        timestamp=time.time(),
        threat_level=ThreatLevel.L2_WARNING,
        operating_mode=OperatingMode.IN_CAR,
        action_type=ActionType.WINDOW_PEEKING,
        description="贴窗窥探测试"
    )

    callback_called = []
    def on_complete(result):
        callback_called.append(result)

    # 提交异步任务，主调用必须在 5ms 内即刻返回，绝对禁止阻塞主视频通路
    t_start = time.time()
    task_id = reviewer.submit_review_async(record, callback=on_complete)
    submit_duration = time.time() - t_start

    assert submit_duration < 0.015, f"异步提交耗时过长: {submit_duration}s"
    assert task_id.startswith("VLM_TASK_EVT_ASYNC_001")
    assert len(callback_called) == 0 # 此时后台工作线程尚未执行完

    # 等待后台工作线程执行完毕
    finished = reviewer.wait_for_completion(timeout=1.0)
    assert finished is True
    assert len(callback_called) == 1
    res = callback_called[0]
    assert res["event_id"] == "EVT_ASYNC_001"
    assert len(res["summary_text"]) <= 50
    assert "车窗" in res["summary_text"]
    assert record.vlm_summary is not None

    reviewer.stop()


def test_strict_anti_downgrade_rule():
    """测试阶段 4.3：核心安全沙箱——有接触信号支撑的事件“只升不降铁律”"""
    mock_backend = MockVLMBackend(simulated_delay_sec=0.0)
    # 强制让 VLM 尝试恶意或错误地给出降级建议
    mock_backend.override_response = {
        "action_category": "BENIGN_TOUCH",
        "confidence": 0.99,
        "is_human_contact": False,
        "is_false_alarm": True,
        "threat_escalation": "DOWNGRADE_TO_L1",
        "summary_text": "误判为外部飞鸟碰撞，建议取消警报。"
    }

    reviewer = AsyncVLMReviewer(backend=mock_backend, enable_worker_thread=False)

    # 场景：CAN 信号已确认门把手被机械拉扯，快通路锁定 L3
    contact_record = ThreatEventRecord(
        event_id="EVT_CONTACT_L3",
        timestamp=time.time(),
        threat_level=ThreatLevel.L3_CRITICAL,
        operating_mode=OperatingMode.IN_CAR,
        action_type=ActionType.DOOR_HANDLE_PULL,
        description="车门把手微动开关触发",
        has_contact_signal=True # 具有硬件接触硬信号
    )

    result = reviewer.review_event_sync(contact_record)

    # 核心裁决验证：无论 VLM 输出什么，绝对禁止将 L3 降级！
    assert result["final_threat"] == ThreatLevel.L3_CRITICAL
    assert result["threat_escalation"] == "MAINTAIN_L3_CONTACT_LOCKED"
    assert len(result["summary_text"]) <= 50


def test_pure_visual_l2_false_alarm_downgrade():
    """测试阶段 4.4：纯视觉无接触信号的 L2 事件在环境假阳性下的安全降级"""
    mock_backend = MockVLMBackend(simulated_delay_sec=0.0)
    mock_backend.force_false_alarm = True # 模拟大模型判定为树叶/光影

    reviewer = AsyncVLMReviewer(backend=mock_backend, enable_worker_thread=False)

    # 纯视觉 L2 徘徊/晃动，无任何车身传感器接触信号
    visual_l2_record = ThreatEventRecord(
        event_id="EVT_VISUAL_L2",
        timestamp=time.time(),
        threat_level=ThreatLevel.L2_WARNING,
        operating_mode=OperatingMode.OUT_CAR,
        action_type=ActionType.LINGERING,
        description="车周连续扰动",
        has_contact_signal=False
    )

    result = reviewer.review_event_sync(visual_l2_record)

    # 验证安全降级至 L1 提示，且战报提示为环境假阳性
    assert result["final_threat"] == ThreatLevel.L1_NOTICE
    assert result["threat_escalation"] == "DOWNGRADE_TO_L1_ENVIRONMENT_FALSE_ALARM"
    assert "树叶" in result["summary_text"] or "光影" in result["summary_text"]


def test_evidence_recorder_clip_assembly_and_video_export():
    """测试阶段 4.5：环形缓冲 10s事前+20s事后视频拼装与实际文件写入"""
    temp_dir = tempfile.mkdtemp()
    try:
        recorder = EvidenceRecorder(
            buffer_duration_sec=15.0,
            fps=10.0,
            storage_dir=temp_dir,
            clock_source=ClockSource.GNSS_PPS_SYNC
        )

        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        now = time.time()

        # 1. 模拟推送事前 10 秒的视频帧 (100 帧, 10fps)
        for i in range(100):
            t = now - 10.0 + i * 0.1
            recorder.push_frame(dummy_frame, t)

        # 2. 在 now 时刻触发事件锁存
        pre_sig = recorder.lock_event_clip(
            event_id="EVT_CLIP_TEST_001",
            threat_level_str="L3 紧急",
            trigger_time=now,
            pre_duration_sec=10.0,
            post_duration_sec=2.0 # 测试加速：收集事后 2.0s
        )
        assert len(pre_sig) == 64 # SHA-256 长度

        # 3. 持续推送事后视频帧 (20 帧)
        for i in range(25):
            t = now + i * 0.1
            recorder.push_frame(dummy_frame, t)

        # 验证 clip 已自动完成事后录制并落盘
        assert len(recorder.locked_evidence) == 1
        ev = recorder.locked_evidence[0]
        assert ev["event_id"] == "EVT_CLIP_TEST_001"
        assert ev["total_frames"] >= 100 # 包含事前与事后帧
        assert Path(ev["video_path"]).exists()
        assert Path(ev["video_path"]).stat().st_size > 0 # 实际生成了视频文件
        assert ev["clock_source"] == ClockSource.GNSS_PPS_SYNC.value
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_tamper_evident_sha256_hash_chain():
    """测试阶段 4.6：不可篡改 SHA-256 哈希链上链与防篡改审计校验引擎"""
    temp_dir = tempfile.mkdtemp()
    try:
        recorder = EvidenceRecorder(
            buffer_duration_sec=5.0,
            fps=5.0,
            storage_dir=temp_dir,
            clock_source=ClockSource.GNSS_PPS_SYNC
        )
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # 记录 3 个连续的重要威胁事件
        now = time.time()
        for idx in range(3):
            t = now + idx * 5.0
            recorder.push_frame(dummy_frame, t - 1.0)
            recorder.push_frame(dummy_frame, t)
            recorder.lock_event_clip(
                event_id=f"EVT_CHAIN_{idx:03d}",
                threat_level_str="L3 紧急",
                trigger_time=t,
                pre_duration_sec=2.0,
                post_duration_sec=0.1
            )
            recorder.push_frame(dummy_frame, t + 0.2) # 触发完成

        assert len(recorder.manifest) == 3

        # 1. 验证正常哈希链完整性
        is_valid, report, bad_idx = recorder.verify_hash_chain()
        assert is_valid is True
        assert bad_idx is None
        assert "完整性验证通过" in report

        # 2. 篡改攻击实验 A：恶意攻击者篡改区块 1 的时间戳以制造不在场证明
        manifest_file = Path(temp_dir) / "evidence_manifest.jsonl"
        lines = manifest_file.read_text(encoding='utf-8').splitlines()

        tampered_entry = json.loads(lines[1])
        tampered_entry["payload"]["trigger_time"] += 3600.0 # 篡改时间戳推后 1 小时
        lines[1] = json.dumps(tampered_entry)
        manifest_file.write_text("\n".join(lines) + "\n", encoding='utf-8')

        # 验证审计引擎能否精准识别篡改
        is_valid_after_tamper, tamper_report, tampered_block = recorder.verify_hash_chain()
        assert is_valid_after_tamper is False
        assert tampered_block == 1 # 精确锁定第 1 号区块被篡改
        assert "数据签名被篡改" in tamper_report

        # 3. 篡改攻击实验 B：恶意攻击者删除了区块 1 企图销毁特定事件
        lines_deleted = [lines[0], lines[2]] # 删除了第 1 行
        manifest_file.write_text("\n".join(lines_deleted) + "\n", encoding='utf-8')

        is_valid_del, del_report, del_block = recorder.verify_hash_chain()
        assert is_valid_del is False
        assert del_block == 1
        assert "哈希链断裂" in del_report or "不符" in del_report
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_evidence_recorder_30s_30fps_normal_flow_playback():
    """验证事件录像标准化：严格生成 30 FPS、30秒 (事前10s+事后20s=900帧) 且正常 1.0x 物理流速的视频"""
    temp_dir = tempfile.mkdtemp()
    try:
        recorder = EvidenceRecorder(
            buffer_duration_sec=15.0,
            fps=30.0,
            storage_dir=temp_dir,
            clock_source=ClockSource.GNSS_PPS_SYNC
        )
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "TEST_FRAME", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        base_t = 1000.0

        # 模拟 25 FPS 的输入流持续推流 35 秒 (总计 875 帧输入)
        # 验证自适应重采样引擎能否平滑将 25 FPS 输入标准化为标准的 30 FPS、900 帧 1.0x 流速视频
        for i in range(250): # 事前 10 秒
            t = base_t - 10.0 + i * (1.0 / 25.0)
            recorder.push_frame(dummy_frame, t)

        # 在 base_t 触发事件
        recorder.lock_event_clip(
            event_id="EVT_STD_30S_TEST",
            threat_level_str="L3 紧急",
            trigger_time=base_t,
            pre_duration_sec=10.0,
            post_duration_sec=20.0
        )

        # 持续推送事后 20 秒 (500 帧输入)
        for i in range(505):
            t = base_t + i * (1.0 / 25.0)
            recorder.push_frame(dummy_frame, t)

        assert len(recorder.locked_evidence) == 1
        ev = recorder.locked_evidence[0]
        assert ev["event_id"] == "EVT_STD_30S_TEST"

        video_path = ev["video_path"]
        assert Path(video_path).exists()
        assert ev["total_frames"] == 900 # 30s * 30fps = 900 帧整

        # 使用 OpenCV 读取生成的视频属性核验
        cap = cv2.VideoCapture(video_path)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        assert abs(actual_fps - 30.0) < 0.1 # 容器帧率精确为 30 FPS
        assert frame_count == 900 # 严格 900 帧
        # 实际播放时长 = 900 / 30 = 30.0 秒整，彻底解决倍速快进或抽帧慢放
        playback_duration_sec = frame_count / actual_fps
        assert abs(playback_duration_sec - 30.0) < 0.1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_trt_edgellm_backend_path_probing_and_fallback():
    """测试 TensorRT-Edge-LLM 后端路径推导与离线保底行为 (含驻留时间动态注入)"""
    from camping_guard.review.vlm_reviewer import TRTEdgeLLMOnDemandBackend

    backend = TRTEdgeLLMOnDemandBackend(
        engine_path="non_existent/qwen3.5-2b.engine",
        edgellm_dir="non_existent/build"
    )
    backend._probe_paths()

    # 路径不存在时，安全降级且不抛出未捕获异常
    assert backend._resolved_edgellm_bin is None

    # 测试短时间滞留 (<=10s)
    event_short = ThreatEventRecord(
        event_id="EVT_VLM_TRT_001",
        timestamp=time.time(),
        threat_level=ThreatLevel.L2_WARNING,
        operating_mode=OperatingMode.IN_CAR,
        action_type=ActionType.DOOR_HANDLE_PULL,
        description="拉门把手",
        has_contact_signal=True,
        target_dwell_time=4.5,
        target_distance=0.8
    )
    res_short = backend.infer(event_short)
    assert res_short["action_category"] == "DOOR_HANDLE_TAMPERING"
    assert "徘徊" not in res_short["summary_text"]
    assert res_short["is_human_contact"] is True

    # 测试长时间徘徊 (>10s) 自动融合滞留时间
    event_long = ThreatEventRecord(
        event_id="EVT_VLM_TRT_002",
        timestamp=time.time(),
        threat_level=ThreatLevel.L2_WARNING,
        operating_mode=OperatingMode.IN_CAR,
        action_type=ActionType.WINDOW_PEEKING,
        description="窥视",
        has_contact_signal=False,
        target_dwell_time=15.2,
        target_distance=1.2
    )
    res_long = backend.infer(event_long)
    assert "徘徊" in res_long["summary_text"]
    assert "15.2" in res_long["summary_text"]
    assert len(res_long["summary_text"]) <= 50


def test_trt_edgellm_native_cpp_ipc_mocked(monkeypatch):
    """测试 TensorRT-Edge-LLM C++ 原生轻量二进制 IPC 推理流程 (模拟真实 Orin 运行环境)"""
    from camping_guard.review.vlm_reviewer import TRTEdgeLLMOnDemandBackend
    import subprocess

    temp_dir = tempfile.mkdtemp(prefix="test_edgellm_")
    try:
        mock_bin = Path(temp_dir) / "llm_inference"
        mock_bin.write_text("#!/bin/sh\nexit 0\n")
        mock_engine = Path(temp_dir) / "qwen3.5-2b"
        mock_engine.mkdir()
        (mock_engine / "config.json").touch()
        (mock_engine / "llm.engine").touch()

        backend = TRTEdgeLLMOnDemandBackend(
            engine_path=str(mock_engine),
            edgellm_dir=temp_dir
        )
        backend._resolved_edgellm_bin = mock_bin
        backend._resolved_engine_dir = mock_engine

        # 模拟 C++ 二进制执行并写入合规的 output.json
        def mock_subprocess_run(cmd, **kwargs):
            # 找到 --outputFile 参数后面的路径
            out_idx = cmd.index("--outputFile") + 1
            out_file = Path(cmd[out_idx])
            simulated_vlm_json = {
                "action_category": "CABIN_PEEKING",
                "confidence": 0.96,
                "is_human_contact": False,
                "is_false_alarm": False,
                "threat_escalation": "MAINTAIN",
                "summary_text": "监测到可疑人员在主驾车窗贴近窥探，已自动警戒。"
            }
            out_data = {
                "responses": [
                    {"text": f"```json\n{json.dumps(simulated_vlm_json, ensure_ascii=False)}\n```"}
                ]
            }
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(out_data, f, ensure_ascii=False)

            class DummyCompletedProcess:
                returncode = 0
                stdout = "Inference completed in 180ms"
                stderr = ""
            return DummyCompletedProcess()

        monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        event = ThreatEventRecord(
            event_id="EVT_CPP_001",
            timestamp=time.time(),
            threat_level=ThreatLevel.L2_WARNING,
            operating_mode=OperatingMode.IN_CAR,
            action_type=ActionType.WINDOW_PEEKING,
            description="车窗窥视测试",
            target_dwell_time=12.0,
            target_distance=0.9
        )

        res = backend.infer(event, keyframes=[dummy_frame])
        assert res is not None
        assert res["action_category"] == "CABIN_PEEKING"
        assert res["confidence"] == 0.96
        assert "窥探" in res["summary_text"]
        assert len(res["summary_text"]) <= 50
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_trt_edgellm_split_engine_discovery():
    """测试 VLM 多模态 LLM 语言引擎与 Visual 视觉编码器引擎的分离目录探测与组装"""
    from camping_guard.review.vlm_reviewer import TRTEdgeLLMOnDemandBackend

    temp_dir = tempfile.mkdtemp(prefix="test_split_engine_")
    try:
        model_root = Path(temp_dir) / "qwen3.5-2b"
        llm_dir = model_root / "engines" / "llm"
        llm_dir.mkdir(parents=True)
        (llm_dir / "config.json").touch()
        (llm_dir / "llm.engine").touch()
        visual_dir = model_root / "engines" / "visual"
        visual_dir.mkdir(parents=True)
        (visual_dir / "config.json").touch()
        (visual_dir / "visual.engine").touch()

        # 1. 显式传入模型总根目录 (--model-dir) -> 严格校验并解析出 engines/llm 和 engines/visual
        backend = TRTEdgeLLMOnDemandBackend(model_dir=str(model_root))
        backend._probe_paths()
        assert backend._resolved_engine_dir == llm_dir
        assert backend._resolved_visual_dir == visual_dir

        # 2. 显式传入 llm 目录 (--vlm-engine) -> 关联同级 sibling visual 目录
        backend_sibling = TRTEdgeLLMOnDemandBackend(engine_path=str(llm_dir))
        backend_sibling._probe_paths()
        assert backend_sibling._resolved_engine_dir == llm_dir
        assert backend_sibling._resolved_visual_dir == visual_dir

        # 3. 显式分别指定 visual_engine_path
        custom_vis = Path(temp_dir) / "custom_visual"
        custom_vis.mkdir()
        (custom_vis / "config.json").touch()
        (custom_vis / "visual.engine").touch()
        backend_explicit = TRTEdgeLLMOnDemandBackend(
            engine_path=str(llm_dir),
            visual_engine_path=str(custom_vis)
        )
        backend_explicit._probe_paths()
        assert backend_explicit._resolved_engine_dir == llm_dir
        assert backend_explicit._resolved_visual_dir == custom_vis

        # 4. 传入纯空目录 -> 验证不会误判为引擎目录 (防止 Embedding file not found 崩溃)
        empty_root = Path(temp_dir) / "empty_proj"
        empty_root.mkdir()
        backend_empty = TRTEdgeLLMOnDemandBackend(engine_path=str(empty_root))
        backend_empty._probe_paths()
        assert backend_empty._resolved_engine_dir is None

        # 5. 未显式传入任何参数 -> 完全保持休眠状态，不猜测路径，不调度 C++ 运行时
        backend_none = TRTEdgeLLMOnDemandBackend()
        backend_none._probe_paths()
        assert backend_none._resolved_engine_dir is None
        assert backend_none._resolved_edgellm_bin is None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_evidence_recorder_restart_continuity():
    """测试阶段 4.7：验证 EvidenceRecorder 跨会话/跨进程重启后哈希链不断裂且无缝衔接"""
    temp_dir = tempfile.mkdtemp()
    try:
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        now = time.time()

        # 第 1 次启动：记录事件 0
        rec1 = EvidenceRecorder(buffer_duration_sec=2.0, fps=5.0, storage_dir=temp_dir)
        rec1.push_frame(dummy_frame, now)
        rec1.lock_event_clip("EVT_RUN1_000", "L2 预警", now, pre_duration_sec=0.5, post_duration_sec=0.1)
        rec1.push_frame(dummy_frame, now + 0.2)
        assert len(rec1.manifest) == 1
        h0 = rec1.manifest[0]["block_hash"]
        rec1.flush()

        # 第 2 次启动 (模拟守护进程重启或下一次运行)
        rec2 = EvidenceRecorder(buffer_duration_sec=2.0, fps=5.0, storage_dir=temp_dir)
        assert len(rec2.manifest) == 1
        assert rec2.prev_hash == h0

        # 在新会话中记录事件 1
        t2 = now + 10.0
        rec2.push_frame(dummy_frame, t2)
        rec2.lock_event_clip("EVT_RUN2_001", "L3 紧急", t2, pre_duration_sec=0.5, post_duration_sec=0.1)
        rec2.push_frame(dummy_frame, t2 + 0.2)
        rec2.flush()

        assert len(rec2.manifest) == 2
        assert rec2.manifest[1]["payload"]["block_index"] == 1
        assert rec2.manifest[1]["payload"]["prev_hash"] == h0
        h1 = rec2.manifest[1]["block_hash"]

        # 第 3 次启动：再记录事件 2
        rec3 = EvidenceRecorder(buffer_duration_sec=2.0, fps=5.0, storage_dir=temp_dir)
        assert len(rec3.manifest) == 2
        assert rec3.prev_hash == h1

        t3 = now + 20.0
        rec3.push_frame(dummy_frame, t3)
        rec3.lock_event_clip("EVT_RUN3_002", "L1 提示", t3, pre_duration_sec=0.5, post_duration_sec=0.1)
        rec3.push_frame(dummy_frame, t3 + 0.2)
        rec3.flush()

        assert len(rec3.manifest) == 3
        assert rec3.manifest[2]["payload"]["block_index"] == 2
        assert rec3.manifest[2]["payload"]["prev_hash"] == h1

        # 终极审计验证
        is_valid, msg, bad_idx = rec3.verify_hash_chain()
        assert is_valid is True
        assert bad_idx is None
        assert "完整性验证通过" in msg
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_evidence_recorder_auto_heal_legacy_break():
    """测试阶段 4.8：验证对历史断裂清单 (旧版本跨进程重启遗留的多个 GENESIS_HASH) 自动自愈重签"""
    temp_dir = tempfile.mkdtemp()
    try:
        manifest_file = Path(temp_dir) / "evidence_manifest.jsonl"
        # 构造模拟历史故障断裂数据 (与用户日志完全一致的错误链)
        payload0 = {
            "block_index": 0, "event_id": "EVT_OLD_0", "threat_level": "L2 预警",
            "trigger_time": 1000.0, "clock_source": "GNSS_PPS_SYNC",
            "prev_hash": "GENESIS_BLOCK_HASH_CAMPING_GUARD_2026",
            "video_file_name": "EVT_OLD_0.mp4", "video_file_sha256": "fake0"
        }
        h0 = hashlib.sha256(json.dumps(payload0, sort_keys=True).encode("utf-8")).hexdigest()

        # 区块 1 错误地记录了 GENESIS_HASH (旧版重启缺陷)
        payload1_broken = {
            "block_index": 0, "event_id": "EVT_OLD_1", "threat_level": "L3 紧急",
            "trigger_time": 1050.0, "clock_source": "GNSS_PPS_SYNC",
            "prev_hash": "GENESIS_BLOCK_HASH_CAMPING_GUARD_2026",
            "video_file_name": "EVT_OLD_1.mp4", "video_file_sha256": "fake1"
        }
        h1_broken = hashlib.sha256(json.dumps(payload1_broken, sort_keys=True).encode("utf-8")).hexdigest()

        # 区块 2 链接到了 h1_broken
        payload2 = {
            "block_index": 1, "event_id": "EVT_OLD_2", "threat_level": "L1 提示",
            "trigger_time": 1100.0, "clock_source": "GNSS_PPS_SYNC",
            "prev_hash": h1_broken,
            "video_file_name": "EVT_OLD_2.mp4", "video_file_sha256": "fake2"
        }
        h2 = hashlib.sha256(json.dumps(payload2, sort_keys=True).encode("utf-8")).hexdigest()

        # 写入历史故障文件
        with open(manifest_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"block_hash": h0, "payload": payload0}) + "\n")
            f.write(json.dumps({"block_hash": h1_broken, "payload": payload1_broken}) + "\n")
            f.write(json.dumps({"block_hash": h2, "payload": payload2}) + "\n")

        # 启动 Recorder，应触发自愈修复
        recorder = EvidenceRecorder(storage_dir=temp_dir)
        assert len(recorder.manifest) == 3
        # 验证备份文件已创建
        assert (Path(temp_dir) / "evidence_manifest.jsonl.bak").exists()

        # 验证修复后的区块 1 prev_hash 为区块 0 的哈希
        assert recorder.manifest[1]["payload"]["prev_hash"] == recorder.manifest[0]["block_hash"]
        assert recorder.manifest[1]["payload"]["block_index"] == 1
        # 验证区块 2 prev_hash 为修复后的区块 1 哈希
        assert recorder.manifest[2]["payload"]["prev_hash"] == recorder.manifest[1]["block_hash"]
        assert recorder.manifest[2]["payload"]["block_index"] == 2

        # 验证审计全面通过
        is_valid, msg, bad_idx = recorder.verify_hash_chain()
        assert is_valid is True
        assert bad_idx is None
        assert "完整性验证通过" in msg
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_vlm_beast_gating_and_summary():
    """验证大型猛兽动作类型触发 VLM 唤醒门控并生成专业战报"""
    reviewer = AsyncVLMReviewer(enable_worker_thread=False)

    # 1. 验证 BEAST_PROXIMITY 与 BEAST_SPRINT 触发门控
    assert reviewer.should_trigger_vlm(
        category_conf=0.90,
        is_unknown_obstacle=False,
        fast_path_threat=ThreatLevel.L1_NOTICE,
        action_type=ActionType.BEAST_PROXIMITY,
        has_contact_signal=False
    ) is True

    assert reviewer.should_trigger_vlm(
        category_conf=0.90,
        is_unknown_obstacle=False,
        fast_path_threat=ThreatLevel.L2_WARNING,
        action_type=ActionType.BEAST_SPRINT,
        has_contact_signal=False
    ) is True

    # 2. 验证 Mock 后端生成对应战报
    record_prox = ThreatEventRecord(
        event_id="EVT_BEAST_01",
        timestamp=time.time(),
        threat_level=ThreatLevel.L2_WARNING,
        operating_mode=OperatingMode.IN_CAR,
        action_type=ActionType.BEAST_PROXIMITY,
        description="防区大型猛兽游荡",
        species="brown_bear",
        species_cn="棕熊"
    )
    res_prox = reviewer.review_event_sync(record_prox)
    assert "棕熊" in res_prox["summary_text"]
    assert res_prox["final_threat"] == ThreatLevel.L2_WARNING


def test_vlm_profiler_and_timing_integration():
    """验证 VLM 运行时耗时统计并正确记录到 profiler (标记为异步慢通路并发)"""
    from camping_guard.profiler import PipelineProfiler
    profiler = PipelineProfiler()
    mock_backend = MockVLMBackend(simulated_delay_sec=0.03) # 30ms 延迟
    reviewer = AsyncVLMReviewer(backend=mock_backend, enable_worker_thread=True, profiler=profiler)

    record = ThreatEventRecord(
        event_id="EVT_TIMING_01",
        timestamp=time.time(),
        threat_level=ThreatLevel.L2_WARNING,
        operating_mode=OperatingMode.IN_CAR,
        action_type=ActionType.DOOR_HANDLE_PULL,
        description="门把手拉动触碰",
        has_contact_signal=True
    )

    task_id = reviewer.submit_review_async(record)
    assert task_id.startswith("VLM_TASK_")

    done = reviewer.wait_for_completion(timeout=1.5)
    assert done is True
    reviewer.stop()

    # 验证遥测指标
    assert reviewer.total_completed_reviews >= 1
    assert reviewer.last_inference_ms >= 25.0
    assert len(reviewer.last_summary_text) > 0

    # 验证 profiler 记录
    stats = profiler.model_stats.get("Qwen-VLM")
    assert stats is not None
    assert stats.call_count >= 1
    assert stats.avg_ms >= 25.0
    assert stats.is_async is True

    # 验证汇总报表中展示为 异步并发，而非扭曲同步 E2E
    profiler.start_frame(1)
    profiler.end_frame("L2_WARNING", 1)
    summary_table = profiler.format_summary_table()
    assert "Qwen-VLM" in summary_table
    assert "异步并发" in summary_table


def test_vlm_timing_breakdown_and_aggregates_in_audit_and_json():
    """验证 VLM 性能指标细分剥离 (冷启动/纯推理/前后处理/Token指标) 与 JSON 报表完整导出"""
    from camping_guard.profiler import PipelineProfiler
    from camping_guard.review.vlm_reviewer import TRTEdgeLLMOnDemandBackend

    # 1. 验证 _parse_edgellm_metrics 从官方 profile.json 硬件级微秒 Profile 解析
    sample_profile = {
        "generation": {
            "average_time_per_token_ms": 18.866,
            "average_tokens_per_run": 160.0,
            "generated_tokens": 160,
            "tokens_per_second": 53.0,
            "total_runs": 1
        },
        "multimodal": {
            "average_time_per_token_ms": 0.163,
            "total_audio_tokens": 0,
            "total_image_tokens": 1008,
            "total_multimodal_tokens": 1008,
            "total_runs": 1
        },
        "peak_unified_memory_bytes": 1761693696,
        "peak_unified_memory_mb": 1680.08,
        "prefill": {
            "average_time_per_run_ms": 388.95,
            "average_time_per_token_ms": 0.378,
            "average_tokens_per_run": 1027.0,
            "computed_tokens": 1027,
            "reused_tokens": 0,
            "tokens_per_second": 2640.4,
            "total_runs": 1
        },
        "stages": [
            {
                "average_time_per_run_ms": 164.86,
                "stage_id": "vision_encoder",
                "total_gpu_time_ms": 164.86,
                "total_runs": 1
            },
            {
                "average_time_per_run_ms": 18.98,
                "stage_id": "llm_generation",
                "total_gpu_time_ms": 3018.56,
                "total_runs": 159
            },
            {
                "average_time_per_run_ms": 388.95,
                "stage_id": "llm_prefill",
                "total_gpu_time_ms": 388.95,
                "total_runs": 1
            }
        ]
    }
    sample_out_data = {
        "responses": [
            {
                "finish_reason": "end-of-sequence",
                "text": "测试场景描述"
            }
        ]
    }
    timing_data_prof, metric_data_prof = TRTEdgeLLMOnDemandBackend._parse_edgellm_metrics(
        out_data=sample_out_data,
        stdout_str="",
        proc_elapsed_ms=8200.0,
        profile_data=sample_profile
    )
    assert timing_data_prof["vision_encoder_gpu_ms"] == 164.86
    assert timing_data_prof["llm_prefill_gpu_ms"] == 388.95
    assert timing_data_prof["llm_generation_gpu_ms"] == 3018.56
    assert timing_data_prof["pure_inference_ms"] == round(164.86 + 388.95 + 3018.56, 2)
    assert timing_data_prof["cold_start_load_ms"] == round(8200.0 - timing_data_prof["pure_inference_ms"], 2)
    assert metric_data_prof["tokens_per_second"] == 53.0
    assert metric_data_prof["prompt_tokens"] == 1027
    assert metric_data_prof["ttft_ms"] == round(164.86 + 388.95, 2)
    assert metric_data_prof["generated_tokens"] == 160
    assert metric_data_prof["finish_reason"] == "end-of-sequence"
    assert metric_data_prof["peak_unified_memory_mb"] == 1680.08
    assert metric_data_prof["timing_source"] == "OFFICIAL_EDGELLM_PROFILE_JSON"

    # 2. 验证 _parse_edgellm_metrics 从 C++ 日志正则解析回退
    sample_stdout = (
        "[INFO] Loaded engine in 4520.5 ms\n"
        "[INFO] Context phase latency: 240.0 ms\n"
        "[INFO] Generation phase latency: 2150.2 ms\n"
        "[INFO] Generated 80 tokens\n"
        "[INFO] Tokens per second: 37.2\n"
        "[INFO] Time to first token: 260 ms\n"
    )
    timing_data, metric_data = TRTEdgeLLMOnDemandBackend._parse_edgellm_metrics(
        out_data={},
        stdout_str=sample_stdout,
        proc_elapsed_ms=7100.0
    )
    assert timing_data["cold_start_load_ms"] == 4520.5
    assert timing_data["pure_inference_ms"] == 2150.2
    assert metric_data["generated_tokens"] == 80
    assert metric_data["tokens_per_second"] == 37.2
    assert metric_data["ttft_ms"] == 260.0
    assert metric_data["timing_source"] == "PARSED_FROM_LLM_INFERENCE_LOGS"

    # 3. 验证端到端 Reviewer 审计记录与 JSON 报告聚合
    profiler = PipelineProfiler()
    mock_backend = MockVLMBackend(simulated_delay_sec=0.05)
    reviewer = AsyncVLMReviewer(backend=mock_backend, enable_worker_thread=False, profiler=profiler)

    record = ThreatEventRecord(
        event_id="EVT_METRIC_001",
        timestamp=time.time(),
        threat_level=ThreatLevel.L2_WARNING,
        operating_mode=OperatingMode.IN_CAR,
        action_type=ActionType.DOOR_HANDLE_PULL,
        description="门把手微动测试",
        has_contact_signal=True
    )
    res = reviewer.review_event_sync(record)
    assert "timing_breakdown" in res or len(reviewer.detailed_review_logs) > 0

    audit = reviewer.get_audit_summary()
    assert "timing_aggregates" in audit
    aggr = audit["timing_aggregates"]
    assert aggr["reviews_count"] == 1
    assert "avg_pure_inference_sec" in aggr
    assert "avg_cold_start_load_sec" in aggr
    assert "avg_total_duration_sec" in aggr
    assert "avg_vision_encoder_gpu_ms" in aggr
    assert "avg_llm_prefill_gpu_ms" in aggr
    assert "avg_llm_generation_gpu_ms" in aggr
    assert "avg_ttft_ms" in aggr
    assert "avg_tpot_ms" in aggr
    assert aggr["avg_pure_inference_sec"] >= 0.04

    # 3. 验证 JSON 报告导出
    temp_dir = tempfile.mkdtemp()
    try:
        json_file = Path(temp_dir) / "test_perf_report.json"
        profiler.set_runtime_context(vlm_audit=audit)
        saved_path = profiler.export_json_report(json_file)
        assert Path(saved_path).is_file()

        with open(saved_path, "r", encoding="utf-8") as f:
            report_json = json.load(f)

        assert "vlm_audit_summary" in report_json
        assert "vlm_detailed_timing_breakdown" in report_json
        assert report_json["vlm_detailed_timing_breakdown"]["reviews_count"] == 1
        reviews = report_json["vlm_audit_summary"]["reviews"]
        assert len(reviews) == 1
        assert "timing_breakdown" in reviews[0]
        assert "cold_start_load_sec" in reviews[0]["timing_breakdown"]
        assert "pure_inference_sec" in reviews[0]["timing_breakdown"]
        assert "preprocess_sec" in reviews[0]["timing_breakdown"]
        assert "postprocess_sec" in reviews[0]["timing_breakdown"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_vlm_server_client_inference_and_metrics():
    """验证 TRTEdgeLLMOnDemandBackend 连接常驻守护服务 (vlm_server) 零冷启动推理与指标解析"""
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from camping_guard.review.vlm_reviewer import TRTEdgeLLMOnDemandBackend

    received_requests = []
    shutdown_received = []

    class MockVLMServerHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                resp = json.dumps({"status": "ready", "model_ready": True, "idle_remaining_sec": 59.5})
                self.wfile.write(resp.encode('utf-8'))
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            content_len = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_len) if content_len > 0 else b""
            body_str = body_bytes.decode('utf-8', errors='ignore')

            if self.path == "/generate":
                received_requests.append(body_str)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()

                out_payload = {
                    "action_category": "DOOR_HANDLE_TAMPERING",
                    "confidence": 0.94,
                    "is_human_contact": True,
                    "is_false_alarm": False,
                    "threat_escalation": "CONFIRM_OR_UPGRADE",
                    "summary_text": "复核确认有人在拉扯左前门把手，已锁定存证。"
                }
                resp_json = {
                    "status": "success",
                    "output_text": json.dumps(out_payload, ensure_ascii=False),
                    "finish_reason": "stop",
                    "timing": {
                        "pure_inference_ms": 1150.2,
                        "cold_start_load_ms": 0.0,
                        "vision_encoder_gpu_ms": 112.5,
                        "llm_prefill_gpu_ms": 280.4,
                        "llm_generation_gpu_ms": 757.3,
                        "tokens_per_second": 48.5,
                        "generated_tokens": 42,
                        "prompt_tokens": 850
                    },
                    "inference_metrics": {
                        "prompt_tokens": 850,
                        "generated_tokens": 42,
                        "tokens_per_second": 48.5,
                        "finish_reason": "stop",
                        "timing_source": "PERSISTENT_DAEMON_RUNTIME"
                    }
                }
                self.wfile.write(json.dumps(resp_json, ensure_ascii=False).encode('utf-8'))
            elif self.path == "/shutdown":
                shutdown_received.append(True)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"shutting down"}')
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), MockVLMServerHandler)
    server_port = server.server_port
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        backend = TRTEdgeLLMOnDemandBackend(
            mode="server",
            server_host="127.0.0.1",
            server_port=server_port,
            timeout=5.0
        )

        record = ThreatEventRecord(
            event_id="EVT_SRV_001",
            timestamp=time.time(),
            threat_level=ThreatLevel.L2_WARNING,
            operating_mode=OperatingMode.IN_CAR,
            action_type=ActionType.DOOR_HANDLE_PULL,
            description="门把手微动测试",
            has_contact_signal=True
        )

        # 模拟 100x100 关键帧
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # 1. 验证通过常驻服务获得推理结果
        result = backend.infer(record, keyframes=[dummy_frame])
        assert result is not None
        assert result["action_category"] == "DOOR_HANDLE_TAMPERING"
        assert result["backend_source"] == "TRT-Edge-LLM (vlm_server 常驻持久化服务)"
        assert len(received_requests) == 1
        assert "拉门撬锁" in received_requests[0]

        # 2. 验证硬件细分指标与零冷启动
        tb = result["timing_breakdown"]
        assert tb["pure_inference_ms"] == 1150.2
        assert tb["cold_start_load_ms"] == 0.0 # 常驻状态 0 冷启动！
        assert tb["vision_encoder_gpu_ms"] == 112.5
        assert tb["llm_prefill_gpu_ms"] == 280.4
        assert tb["llm_generation_gpu_ms"] == 757.3

        metrics = result["inference_metrics"]
        assert metrics["tokens_per_second"] == 48.5
        assert metrics["prompt_tokens"] == 850
        assert metrics["generated_tokens"] == 42
        assert metrics["finish_reason"] == "stop"

        # 3. 验证优雅退出指令发送
        backend.shutdown()
        assert len(shutdown_received) == 1
    finally:
        server.shutdown()
        server.server_close()


def test_vlm_mode_fallback_and_selection():
    """验证 --vlm-mode (auto / server / subprocess) 分流与平滑自动降级策略"""
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from camping_guard.review.vlm_reviewer import TRTEdgeLLMOnDemandBackend

    # 1. auto 模式在无 server 环境自动平滑降级至规则保底 (且保留完整战报生成)
    backend_auto = TRTEdgeLLMOnDemandBackend(
        mode="auto",
        server_host="127.0.0.1",
        server_port=65432, # 未监听的无效端口
        timeout=1.0
    )
    record = ThreatEventRecord(
        event_id="EVT_AUTO_FALLBACK",
        timestamp=time.time(),
        threat_level=ThreatLevel.L2_WARNING,
        operating_mode=OperatingMode.IN_CAR,
        action_type=ActionType.WINDOW_PEEKING,
        description="窥探测试",
        has_contact_signal=False
    )
    res_auto = backend_auto.infer(record)
    assert res_auto is not None
    assert "车窗" in res_auto["summary_text"]
    assert "FALLBACK_RULE_MOCK" in res_auto["backend_source"]

    # 2. subprocess 模式：即使指定了 server 端口也不连接 server，直接走一次性子进程/离线降级
    backend_sub = TRTEdgeLLMOnDemandBackend(
        mode="subprocess",
        server_host="127.0.0.1",
        server_port=65432,
        timeout=1.0
    )
    res_sub = backend_sub.infer(record)
    assert res_sub is not None
    assert "车窗" in res_sub["summary_text"]

    # 3. server 模式：服务不可用时显式标记强制 server 失败回退
    backend_srv = TRTEdgeLLMOnDemandBackend(
        mode="server",
        server_host="127.0.0.1",
        server_port=65432,
        timeout=1.0
    )
    res_srv = backend_srv.infer(record)
    assert res_srv is not None
    assert "FALLBACK_RULE_MOCK (强制server模式不可用回退)" in res_srv["backend_source"]


def test_trt_edgellm_inprocess_so_mocked(monkeypatch):
    """测试 TensorRT-Edge-LLM 原生 libvlm_engine.so C-API 进程内直调流程 (首帧 5s 加载，后续 0 冷启动 3.5s 推理)"""
    import ctypes
    from camping_guard.review.vlm_reviewer import TRTEdgeLLMOnDemandBackend

    temp_dir = tempfile.mkdtemp(prefix="test_vlm_so_")
    try:
        mock_so = Path(temp_dir) / "libvlm_engine.so"
        mock_so.touch()

        mock_engine = Path(temp_dir) / "qwen3.5-2b"
        mock_engine.mkdir()
        (mock_engine / "config.json").touch()
        (mock_engine / "llm.engine").touch()

        simulated_response = json.dumps({
            "action_category": "CABIN_PEEKING",
            "confidence": 0.98,
            "is_human_contact": False,
            "is_false_alarm": False,
            "threat_escalation": "MAINTAIN",
            "summary_text": "在主驾驶车窗处发现人员近距离窥探，已自动提高警戒级别。"
        }, ensure_ascii=False).encode('utf-8')

        created_handles = []
        destroyed_handles = []
        generated_calls = []

        class DummyCtypesFunc:
            def __init__(self, name):
                self.name = name
                self.argtypes = []
                self.restype = None

            def __call__(self, *args):
                if self.name == "vlm_create":
                    handle = 0x12345678
                    created_handles.append(handle)
                    return handle
                elif self.name == "vlm_generate":
                    generated_calls.append(args)
                    return simulated_response
                elif self.name == "vlm_get_cold_start_ms":
                    return 5120.0
                elif self.name == "vlm_get_last_inference_ms":
                    return 3450.0
                elif self.name == "vlm_get_last_generated_tokens":
                    return 38
                elif self.name == "vlm_destroy":
                    destroyed_handles.append(args[0])
                    return None
                return None

        class DummyCDLL:
            def __init__(self, path):
                self.path = path
                self.vlm_create = DummyCtypesFunc("vlm_create")
                self.vlm_generate = DummyCtypesFunc("vlm_generate")
                self.vlm_get_cold_start_ms = DummyCtypesFunc("vlm_get_cold_start_ms")
                self.vlm_get_last_inference_ms = DummyCtypesFunc("vlm_get_last_inference_ms")
                self.vlm_get_last_generated_tokens = DummyCtypesFunc("vlm_get_last_generated_tokens")
                self.vlm_destroy = DummyCtypesFunc("vlm_destroy")

        monkeypatch.setattr(ctypes, "CDLL", lambda p: DummyCDLL(p))

        backend = TRTEdgeLLMOnDemandBackend(
            engine_path=str(mock_engine),
            engine_so_path=str(mock_so),
            mode="auto"
        )

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        event = ThreatEventRecord(
            event_id="EVT_SO_001",
            timestamp=time.time(),
            threat_level=ThreatLevel.L2_WARNING,
            operating_mode=OperatingMode.IN_CAR,
            action_type=ActionType.WINDOW_PEEKING,
            description="车窗窥视测试",
            target_dwell_time=8.0,
            target_distance=0.9
        )

        # 1. 首次推理：触发冷启动加载 (5.12s)
        res1 = backend.infer(event, keyframes=[dummy_frame])
        assert res1 is not None
        assert res1["action_category"] == "CABIN_PEEKING"
        assert "窥探" in res1["summary_text"]
        assert "libvlm_engine.so" in res1["backend_source"]
        assert len(created_handles) == 1
        assert len(generated_calls) == 1
        t1 = res1["timing_breakdown"]
        assert t1["cold_start_load_ms"] == 5120.0
        assert t1["pure_inference_ms"] == 3450.0
        assert t1["vision_encoder_gpu_ms"] > 0
        assert t1["llm_prefill_gpu_ms"] > 0
        assert res1["inference_metrics"]["ttft_ms"] is not None
        assert res1["inference_metrics"]["ttft_ms"] > 0
        assert res1["inference_metrics"]["tpot_ms"] is not None
        assert res1["inference_metrics"]["tpot_ms"] > 0

        # 2. 第二次推理：0 冷启动常驻直调 (cold_start_load_ms == 0.0)
        res2 = backend.infer(event, keyframes=[dummy_frame])
        assert res2 is not None
        assert len(created_handles) == 1
        assert len(generated_calls) == 2
        t2 = res2["timing_breakdown"]
        assert t2["cold_start_load_ms"] == 0.0
        assert t2["pure_inference_ms"] == 3450.0

        # 3. 验证显存释放与优雅析构
        backend.shutdown()
        assert len(destroyed_handles) == 1
        assert destroyed_handles[0] == 0x12345678
        assert backend._inprocess_handle is None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)







