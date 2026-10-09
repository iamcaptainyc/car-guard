"""
TensorRT Engine 与 Jetson Orin 适配专项单元测试套件 (tests/test_tensorrt_engine.py)
验证:
1. Jetson 硬件自检与信息探测函数
2. TensorMeta 与 MockTRTInferenceSession 接口对齐
3. 各感知模型 (YOLO / RTMPose / OSNet / FaceID) 对 TensorRT engine 与 use_trt 参数的兼容性
4. CampingGuardEngine 在 use_tensorrt=True 模式下的初始化与流水线运行
"""

import os
import numpy as np
import pytest

from camping_guard.perception.trt_engine import (
    is_jetson,
    get_jetson_info,
    TensorMeta,
    MockTRTInferenceSession,
    load_inference_backend
)
from camping_guard.perception.detector import ONNXObjectDetector, TRTObjectDetector
from camping_guard.perception.pose_estimator import RTMPoseEstimator
from camping_guard.perception.reid import OSNetReIDExtractor
from camping_guard.perception.faceid import SCRFDFaceDetector, ArcFaceRecognizer, FaceIDPipeline
from camping_guard.engine import CampingGuardEngine
from camping_guard.types import VehicleZone, ThreatLevel


def test_jetson_hardware_detect():
    """测试 Jetson 硬件状态获取接口无异常且结构完整"""
    info = get_jetson_info()
    assert isinstance(info, dict)
    assert "is_jetson" in info
    assert "model" in info
    assert "tensorrt_ver" in info
    # is_jetson 返回 bool
    assert isinstance(is_jetson(), bool)


def test_mock_trt_inference_session_api():
    """测试 TensorRT 推理会话的接口完全兼容 ONNXRuntime (get_inputs, get_outputs, run)"""
    mock_session = MockTRTInferenceSession("dummy_model.engine")
    
    # 验证 get_inputs
    inputs = mock_session.get_inputs()
    assert len(inputs) >= 1
    assert inputs[0].name == "images"
    assert inputs[0].shape == [1, 3, 640, 640]

    # 验证 get_outputs
    outputs = mock_session.get_outputs()
    assert len(outputs) >= 1
    assert outputs[0].name == "output0"

    # 验证 run 前向
    dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
    outs = mock_session.run(None, {"images": dummy_input})
    assert isinstance(outs, list)
    assert len(outs) >= 1
    assert outs[0].shape == (1, 84, 8400)


def test_detector_with_trt_flag():
    """测试 ONNXObjectDetector 与 TRTObjectDetector 支持 use_trt 参数与平稳降级"""
    model_path = "models/yolo/yolo11n.onnx"
    # 即使传入不存在的 .engine，也能正常实例化而不抛崩溃异常
    det_engine = ONNXObjectDetector("models/yolo/yolo11n.engine", use_trt=True)
    assert det_engine is not None
    # 传入存在模型且开启 use_trt
    det = ONNXObjectDetector(model_path, use_trt=True)
    assert det is not None
    assert det.backend in ["tensorrt", "onnxruntime", "unknown"]

    trt_det = TRTObjectDetector(model_path)
    assert trt_det is not None


def test_pose_estimator_with_trt_flag():
    """测试 RTMPoseEstimator 支持 use_trt 参数与 .engine 路径"""
    estimator = RTMPoseEstimator("models/pose/rtmpose-s.engine", use_trt=True)
    assert estimator is not None
    assert estimator.backend in ["tensorrt", "onnxruntime", "unknown"]


def test_reid_extractor_with_trt_flag():
    """测试 OSNetReIDExtractor 支持 use_trt 参数与 .engine 路径"""
    reid = OSNetReIDExtractor("models/reid/osnet_x0_25.engine", use_trt=True)
    assert reid is not None
    # 验证裁剪特征提取不崩溃
    dummy_crop = np.zeros((200, 100, 3), dtype=np.uint8)
    feat = reid.extract_crop_feature(dummy_crop)
    assert feat.shape == (512,)


def test_faceid_pipeline_with_trt_flag():
    """测试 FaceIDPipeline 挂载 use_trt=True 并成功流转"""
    pipeline = FaceIDPipeline(use_trt=True)
    assert pipeline is not None
    assert pipeline.detector is not None
    assert pipeline.recognizer is not None


def test_engine_full_pipeline_with_use_tensorrt():
    """测试 CampingGuardEngine 在 use_tensorrt=True 模式下完整运行 process_tick"""
    engine = CampingGuardEngine(use_tensorrt=True)
    assert engine.use_tensorrt is True
    success, msg = engine.activate()
    assert success is True

    # 发送一帧进行调度，验证端到端不报错
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = engine.process_tick({1: dummy_frame})
    assert "threat_level" in res
    assert "state" in res
    assert res["threat_level"] == ThreatLevel.L0_SAFE.value
