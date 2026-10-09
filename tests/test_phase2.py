"""
Phase 2 专项测试套件 (tests/test_phase2.py)
验证 Zone 3 ROI 标定与持久化、ByteTrack 双重关联与遮挡恢复、Detector 与多路流服务
"""

import time
import os
import numpy as np
import pytest

from camping_guard.tools.roi_calibrator import ROICalibrator
from camping_guard.types import Zone3ROI, ThreatLevel, VehicleZone
from camping_guard.perception.bytetrack import ByteTracker, STrack
from camping_guard.perception.detector import SyntheticDetector, ONNXObjectDetector
from camping_guard.perception.video_source import MultiCameraStreamManager
from camping_guard.engine import CampingGuardEngine

def test_roi_calibrator_io_and_validation(tmp_path):
    """测试 Zone 3 ROI 标定文件保存、加载与几何合法性校验"""
    tmp_json = str(tmp_path / "test_rois.json")
    calibrator = ROICalibrator(config_path=tmp_json)

    # 1. 构造测试 ROI 并保存
    sample_rois = [
        Zone3ROI(
            roi_id="test_handle",
            name="测试把手",
            camera_id=1,
            polygon=[(100, 100), (200, 100), (200, 180), (100, 180)],
            threshold_ratio=0.15,
            confirm_frames=3
        )
    ]
    ok_save = calibrator.save_rois(sample_rois)
    assert ok_save is True
    assert os.path.exists(tmp_json)

    # 2. 重新载入并比对
    loaded_rois = calibrator.load_rois()
    assert len(loaded_rois) == 1
    r = loaded_rois[0]
    assert r.roi_id == "test_handle"
    assert r.name == "测试把手"
    assert r.threshold_ratio == 0.15
    assert len(r.polygon) == 4

    # 3. 几何合法性校验
    is_valid, msg = calibrator.validate_roi(r, frame_shape=(480, 640))
    assert is_valid is True

    # 4. 测试非法多边形 (< 3 点)
    invalid_roi = Zone3ROI(roi_id="bad", name="非法", camera_id=1, polygon=[(10, 10), (20, 20)])
    is_valid2, msg2 = calibrator.validate_roi(invalid_roi)
    assert is_valid2 is False
    assert "顶点数不足" in msg2

def test_bytetrack_association_and_occlusion_recovery():
    """测试 ByteTrack 双重关联、卡尔曼状态更新与低置信度遮挡恢复"""
    tracker = ByteTracker(high_thresh=0.5, low_thresh=0.15, max_time_lost=10)
    now = time.time()

    # 第 1 帧: 高置信度检测出现 (目标首次建立轨迹)
    dets_f1 = [{'bbox': (100, 100, 50, 100), 'conf': 0.88, 'category': 'person'}]
    t1 = tracker.update(dets_f1, now)
    assert len(t1) == 1
    track_id_1 = t1[0].track_id

    # 第 2 帧: 目标位移并保持高置信度
    dets_f2 = [{'bbox': (105, 102, 50, 100), 'conf': 0.85, 'category': 'person'}]
    t2 = tracker.update(dets_f2, now + 0.1)
    assert len(t2) == 1
    assert t2[0].track_id == track_id_1 # ID 连续保持

    # 第 3 帧: 模拟近车严重遮挡/反光，置信度暴跌至 0.25 (常规跟踪器会丢失，ByteTrack 第二级关联恢复)
    dets_f3 = [{'bbox': (110, 104, 50, 100), 'conf': 0.25, 'category': 'person'}]
    t3 = tracker.update(dets_f3, now + 0.2)
    assert len(t3) == 1
    assert t3[0].track_id == track_id_1 # ByteTrack 成功凭借低置信度匹配保留了轨迹！

    # 第 4 帧: 目标暂时完全被遮挡丢失 1 帧
    t4 = tracker.update([], now + 0.3)
    assert len(t4) == 0 # 当前无活跃，转为 Lost 状态

    # 第 5 帧: 目标重现，置信度恢复
    dets_f5 = [{'bbox': (118, 106, 50, 100), 'conf': 0.80, 'category': 'person'}]
    t5 = tracker.update(dets_f5, now + 0.4)
    assert len(t5) == 1
    assert t5[0].track_id == track_id_1 # 成功从 Lost 状态寻回并复用同一 ID

def test_bytetrack_integrated_with_engine():
    """测试 ByteTracker 集成至 CampingGuardEngine 的端到端流程"""
    engine = CampingGuardEngine(use_bytetrack=True)
    engine.activate()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    now = time.time()
    # 模拟在 Zone 2 移动的人员检测框
    for i in range(5):
        dets = [{'bbox': (200 + i * 2, 280, 60, 120), 'conf': 0.88, 'category': 'person'}]
        out = engine.process_tick({1: dummy_frame}, dets)

    assert len(engine.active_targets) == 1
    t = engine.active_targets[0]
    assert t.zone == VehicleZone.ZONE_2_INNER
    assert engine.bytetracker is not None

def test_detector_interface_and_synthetic():
    """测试 Detector 抽象与 SyntheticDetector 自动挂接"""
    synth = SyntheticDetector()
    synth.set_injected_detections([
        {'bbox': (150, 180, 45, 90), 'category': 'person', 'conf': 0.90}
    ])

    engine = CampingGuardEngine(detector=synth, use_bytetrack=True)
    engine.activate()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 不显式传入 raw_detections，由引擎自动调用 detector.detect()
    out = engine.process_tick({1: dummy_frame}, raw_detections=None)
    assert len(engine.active_targets) == 1
    assert engine.active_targets[0].category == 'person'

def test_onnx_detector_helpers():
    """测试 ONNXObjectDetector 的 Letterbox 预处理与 NMS 工具函数"""
    detector = ONNXObjectDetector(model_path="non_existent.onnx")
    assert detector.session is None

    # 测试 letterbox
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    lb_img, r, (dw, dh) = detector.letterbox(img, new_shape=(640, 640))
    assert lb_img.shape == (640, 640, 3)

    # 测试 NMS
    dets = [
        {'bbox': (100, 100, 50, 50), 'conf': 0.9, 'category': 'person'},
        {'bbox': (102, 102, 48, 48), 'conf': 0.8, 'category': 'person'}, # 重叠框
        {'bbox': (300, 300, 50, 50), 'conf': 0.85, 'category': 'car'}
    ]
    nms_res = detector._apply_nms(dets, iou_threshold=0.5)
    assert len(nms_res) == 2 # 重叠框被抑制

def test_multicamera_stream_manager():
    """测试多路环视流管理与目标 FPS 降采样"""
    manager = MultiCameraStreamManager() # 默认合成流模式
    assert manager.synthetic_enabled is True

    # 测试 S0 待机低帧率模式读取 (2 fps)
    frames_s0 = manager.fetch_frames(target_fps=2.0)
    assert len(frames_s0) == 4
    for cid in [0, 1, 2, 3]:
        assert cid in frames_s0
        assert frames_s0[cid].shape == (480, 640, 3)

    # 紧随其后的即时调用应被 min_interval 频控过滤 (返回空或不变)
    frames_immediate = manager.fetch_frames(target_fps=2.0)
    assert len(frames_immediate) == 0

    manager.release_all()

if __name__ == "__main__":
    pytest.main(["-v", __file__])
