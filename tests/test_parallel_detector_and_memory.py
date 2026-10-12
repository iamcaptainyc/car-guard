"""
单元测试：Orin 多模型双流并发集成检测器 (ParallelEnsembleDetector) 与 EvidenceRecorder 内存泄漏根治
验证：
1. ParallelEnsembleDetector 双流并发调度与多类别安全合并 (人/车/动物 + 火灾/浓烟)
2. 并发耗时特性：max(t_main, t_fire) 近似并发执行
3. EvidenceRecorder 事件切片完成后原始帧清空与内存回收 (杜绝 OOM-Killer)
4. EvidenceRecorder 环形缓冲与切片帧的轻量 JPEG 压缩存储有效性
"""

import time
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest

from camping_guard.perception.detector import BaseObjectDetector, ParallelEnsembleDetector, SyntheticDetector, VisualFlameDetector
from camping_guard.evidence.recorder import EvidenceRecorder
from camping_guard.types import ClockSource


class SimulatedSlowDetector(BaseObjectDetector):
    """模拟具有一定推理延迟的检测器，用于测试并发加速"""
    def __init__(self, delay_sec: float, detections: list, name: str = "slow_det"):
        self.delay_sec = delay_sec
        self.detections = detections
        self.name = name
        self.model_path = f"models/{name}.engine"

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.25):
        time.sleep(self.delay_sec)
        return list(self.detections)


def test_parallel_ensemble_detector_concurrency_and_merge():
    """测试多模型并发检测：两个耗时模型并发执行，总耗时远小于串行相加"""
    # 模拟主检测器 (检出人员与车辆，延迟 50ms)
    main_dets = [
        {"bbox": (100, 100, 50, 120), "category": "person", "conf": 0.88},
        {"bbox": (300, 200, 150, 100), "category": "car", "conf": 0.92}
    ]
    main_det = SimulatedSlowDetector(delay_sec=0.05, detections=main_dets, name="main_yolo")

    # 模拟火灾/烟雾辅助检测器 (检出火源，延迟 45ms)
    fire_dets = [
        {"bbox": (500, 300, 60, 80), "category": "fire", "conf": 0.85}
    ]
    fire_det = SimulatedSlowDetector(delay_sec=0.045, detections=fire_dets, name="fire_det")

    ensemble = ParallelEnsembleDetector(main_detector=main_det, fire_detector=fire_det)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    t0 = time.perf_counter()
    results = ensemble.detect(dummy_frame)
    elapsed = time.perf_counter() - t0

    # 1. 验证目标合并正确性 (包含主检测器的人、车，以及火灾检测器的火)
    categories = [d["category"] for d in results]
    assert "person" in categories
    assert "car" in categories
    assert "fire" in categories
    assert len(results) == 3

    # 2. 验证并发加速：串行耗时约 95ms (50ms + 45ms)，并发耗时应接近 50ms (允许轻量线程开销，< 80ms)
    assert elapsed < 0.085, f"双流并发未生效，耗时过长: {elapsed * 1000:.1f}ms (预期接近 50ms，远小于串行 95ms)"

    # 3. 验证性能指标记录
    assert ensemble.last_main_ms >= 40.0
    assert ensemble.last_fire_ms >= 35.0
    assert ensemble.last_parallel_ms < 85.0

    ensemble.close()


def test_parallel_ensemble_fallback_without_fire_detector():
    """测试未挂载火灾检测器时，优雅降级单模型推理"""
    main_dets = [{"bbox": (50, 50, 30, 80), "category": "person", "conf": 0.90}]
    main_det = SyntheticDetector()
    main_det.set_injected_detections(main_dets)

    ensemble = ParallelEnsembleDetector(main_detector=main_det, fire_detector=None)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    results = ensemble.detect(dummy_frame)
    assert len(results) == 1
    assert results[0]["category"] == "person"
    ensemble.close()


def test_evidence_recorder_memory_leak_freed_after_finalize():
    """测试 EvidenceRecorder 内存泄漏彻底根治：切片落盘后帧内存被即时清空释放"""
    temp_dir = tempfile.mkdtemp()
    try:
        recorder = EvidenceRecorder(
            buffer_duration_sec=3.0,
            fps=10.0,
            storage_dir=temp_dir,
            clock_source=ClockSource.GNSS_PPS_SYNC,
            compress_buffer=True
        )

        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        now = time.time()

        # 写入事前帧
        for i in range(30):
            recorder.push_frame(dummy_frame, now - 3.0 + i * 0.1)

        # 触发切片锁定
        recorder.lock_event_clip(
            event_id="EVT_MEM_TEST_001",
            threat_level_str="L3 紧急",
            trigger_time=now,
            pre_duration_sec=2.0,
            post_duration_sec=1.0
        )

        pending = recorder.pending_clips.get("EVT_MEM_TEST_001")
        assert pending is not None
        assert len(pending.pre_frames) > 0

        # 推送事后帧以满足触发条件
        for i in range(15):
            recorder.push_frame(dummy_frame, now + i * 0.1)

        # 验证已成功封装落盘
        assert len(recorder.locked_evidence) == 1
        ev = recorder.locked_evidence[0]
        assert ev["event_id"] == "EVT_MEM_TEST_001"
        assert Path(ev["video_path"]).exists()

        # 核心内存泄漏验证：已完成切片的 pre_frames 与 post_frames 必须被彻底清空！
        assert len(pending.pre_frames) == 0, "pre_frames 必须清空以释放 NumPy 内存"
        assert len(pending.post_frames) == 0, "post_frames 必须清空以释放 NumPy 内存"
        assert pending.is_finalized is True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_evidence_recorder_lightweight_jpeg_compression():
    """测试 EvidenceRecorder 轻量 JPEG 缓存能够大幅缩减环形缓冲内存"""
    temp_dir = tempfile.mkdtemp()
    try:
        recorder = EvidenceRecorder(
            buffer_duration_sec=2.0,
            fps=10.0,
            storage_dir=temp_dir,
            compress_buffer=True
        )

        # 模拟 1280x720 高清彩色帧 (单帧未压缩约为 2.76 MB)
        h, w = 720, 1280
        test_frame = np.full((h, w, 3), 128, dtype=np.uint8)
        raw_nbytes = test_frame.nbytes # 2,764,800 bytes

        now = time.time()
        recorder.push_frame(test_frame, now)

        stored_t, stored_data = recorder.ring_buffer[-1]
        # 验证存储的数据是压缩的 1D uint8 字节流
        assert stored_data.ndim == 1, "轻量模式下必须存储压缩后的 1D JPEG 字节"
        compressed_nbytes = stored_data.nbytes

        # 压缩率应超过 90% (即压缩后小于未压缩大小的 10%)
        compression_ratio = compressed_nbytes / raw_nbytes
        assert compression_ratio < 0.10, f"压缩率未达预期: {compression_ratio:.2%}"

        # 验证切片仍能正常解压并导出有效视频
        recorder.lock_event_clip(
            event_id="EVT_JPEG_001",
            threat_level_str="L2 预警",
            trigger_time=now,
            pre_duration_sec=1.0,
            post_duration_sec=0.2
        )
        recorder.push_frame(test_frame, now + 0.3)

        assert len(recorder.locked_evidence) == 1
        video_p = Path(recorder.locked_evidence[0]["video_path"])
        assert video_p.exists()
        assert video_p.stat().st_size > 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_onnx_object_detector_detect_cls_mapping():
    """测试 ONNXObjectDetector.detect 预测分支中的类别映射逻辑，确保无未定义变量"""
    from camping_guard.perception.detector import ONNXObjectDetector

    det = ONNXObjectDetector.__new__(ONNXObjectDetector)
    det.input_size = (640, 640)
    det.conf_threshold = 0.25
    det.input_name = "images"
    det.backend = "mock"
    det.animal_classifier = None
    det.custom_cls_map = None
    det.is_megadetector = False
    det.is_fire_model = False
    det.coco_to_guard_map = {0: 'person', 2: 'car'}
    det.megadetector_to_guard_map = {0: 'animal', 1: 'person', 2: 'car'}
    det.fisheye_to_guard_map = {3: 'person', 2: 'car'}
    det.fire_to_guard_map = {0: 'fire', 1: 'smoke'}

    class MockSession:
        def __init__(self, num_classes=80):
            self.num_classes = num_classes
        def run(self, output_names, input_feed):
            num_feats = 4 + self.num_classes
            preds = np.zeros((1, num_feats, 1000), dtype=np.float32)
            preds[0, 0:4, 0] = [320, 320, 100, 100]
            preds[0, 4, 0] = 0.95
            return [preds]

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. 测试 COCO 80 类模式 (YOLO)
    det.session = MockSession(num_classes=80)
    res_coco = det.detect(dummy_frame)
    assert len(res_coco) == 1
    assert res_coco[0]["category"] == "person"

    # 2. 测试 MegaDetector 3 类模式
    det.session = MockSession(num_classes=3)
    det.is_megadetector = True
    res_md = det.detect(dummy_frame)
    assert len(res_md) == 1
    assert res_md[0]["category"] == "small_animal"

    # 3. 测试 Fire 2 类模式
    det.is_megadetector = False
    det.is_fire_model = True
    det.session = MockSession(num_classes=2)
    res_fire = det.detect(dummy_frame)
    assert len(res_fire) == 1
    assert res_fire[0]["category"] == "fire"


def test_visual_flame_detector_and_ensemble():
    """测试内置自适应视觉火焰检测器 (VisualFlameDetector) 零权重秒级检出与多流并发集成"""
    v_flame = VisualFlameDetector(min_area=30, conf_threshold=0.25)

    # 1. 负样本测试：全黑画面与白昼日光画面不应误报火焰
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert len(v_flame.detect(black_frame)) == 0

    white_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    assert len(v_flame.detect(white_frame)) == 0

    green_lawn = np.full((480, 640, 3), (30, 120, 30), dtype=np.uint8)
    assert len(v_flame.detect(green_lawn)) == 0

    # 2. 正样本测试：夜幕营地中燃烧的橙红营火 (BGR: B=20, G=100, R=240)
    fire_frame = np.full((480, 640, 3), (20, 25, 20), dtype=np.uint8)
    # 在画面中心绘制一块 80x60 的火焰区域
    fire_frame[200:260, 280:360] = (25, 110, 245)

    dets = v_flame.detect(fire_frame)
    assert len(dets) >= 1
    fire_det = dets[0]
    assert fire_det["category"] == "fire"
    assert fire_det["conf"] >= 0.60

    bx, by, bw, bh = fire_det["bbox"]
    # 验证检出框大致覆盖注入的火焰坐标 (允许缩放与形态学边界微差)
    assert 260 <= bx <= 300
    assert 180 <= by <= 220
    assert bw >= 50 and bh >= 40

    # 3. 验证与主检测器 (SyntheticDetector) 构建 ParallelEnsembleDetector 双流并发集成
    main_det = SyntheticDetector()
    main_det.set_injected_detections([
        {"bbox": (100, 100, 50, 100), "category": "person", "conf": 0.92}
    ])

    ensemble = ParallelEnsembleDetector(main_detector=main_det, fire_detector=v_flame)
    combined = ensemble.detect(fire_frame)

    cats = [d["category"] for d in combined]
    assert "person" in cats
    assert "fire" in cats
    assert len(combined) == 2
    ensemble.close()

