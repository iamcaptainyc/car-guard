"""
车辆与多相机统一标定 JSON 解耦测试 (test_vehicle_calib.py)
验证 vehicle_calib.json 的解析、相机外参表、Zone 3 ROIs 与阈值自动注入
"""

import os
import json
import pytest
import numpy as np

from camping_guard.config import GuardConfig, load_vehicle_calibration
from camping_guard.types import VehicleCalibration, CameraCalib, VehicleZone
from camping_guard.perception.distance import GroundDistanceEstimator
from camping_guard.engine import CampingGuardEngine

def test_load_vehicle_calib_from_json():
    """测试从 vehicle_calib.json 载入完整标定对象"""
    calib = load_vehicle_calibration("vehicle_calib.json")
    assert isinstance(calib, VehicleCalibration)

    # 1. 验证车辆信息
    assert calib.vehicle_info.model_name == "SUV_Camping_Guard_Edition"
    assert "8155" in calib.vehicle_info.vin
    assert calib.vehicle_info.wheelbase_m == 2.95

    # 2. 验证多相机外参表
    assert 0 in calib.cameras
    assert 1 in calib.cameras
    assert 2 in calib.cameras
    assert 3 in calib.cameras

    front_cam = calib.cameras[0]
    left_cam = calib.cameras[1]
    rear_cam = calib.cameras[3]

    assert 0.5 <= front_cam.camera_height_m <= 2.5
    assert -60.0 <= front_cam.camera_pitch_deg <= 0.0

    assert 0.5 <= left_cam.camera_height_m <= 2.5
    assert -60.0 <= left_cam.camera_pitch_deg <= 0.0

    assert 0.5 <= rear_cam.camera_height_m <= 2.5
    assert -60.0 <= rear_cam.camera_pitch_deg <= 0.0

    # 3. 验证 Zone 3 ROIs
    assert len(calib.zone3_rois) >= 5
    roi_ids = [r.roi_id for r in calib.zone3_rois]
    assert "handle_fl" in roi_ids
    assert "handle_rl" in roi_ids
    assert "charge_port" in roi_ids
    assert "trunk_lock" in roi_ids
    assert "roof_rack" in roi_ids

    # 验证后视相机专属 ROI
    rear_rois = calib.get_rois_for_camera(3)
    rear_roi_ids = [r.roi_id for r in rear_rois]
    assert "trunk_lock" in rear_roi_ids
    assert "roof_rack" in rear_roi_ids

    # 4. 验证阈值注入
    assert "zone_far_dist_m" in calib.thresholds
    assert calib.thresholds["zone_far_dist_m"] == 6.0


def test_distance_estimator_with_cameras_calib():
    """测试不同通道相机外参注入后的差异化测距补偿"""
    calib = load_vehicle_calibration("vehicle_calib.json")
    estimator = GroundDistanceEstimator(cameras_calib=calib.cameras)

    frame_shape = (480, 640)
    test_pixel = (320, 360) # 画面中下部接地点

    # 比较 Camera 1 (height=1.2m) vs Camera 0 (height=0.85m)
    dist_cam1 = estimator.estimate_ground_distance(test_pixel, frame_shape, camera_id=1)
    dist_cam0 = estimator.estimate_ground_distance(test_pixel, frame_shape, camera_id=0)

    # 不同朝向与俯仰外参的相机，相同像素位置计算出的物理接地距离具备真实差异化几何补偿
    assert dist_cam0 != dist_cam1
    assert 0.1 <= dist_cam0 <= 15.0
    assert 0.1 <= dist_cam1 <= 15.0


def test_engine_auto_injection_from_json():
    """测试主引擎从 JSON 自动完成各子模块的解耦装配"""
    engine = CampingGuardEngine(config_path="vehicle_calib.json")

    # 验证标定对象已绑定
    assert engine.calib is not None
    assert engine.calib.vehicle_info.model_name == "SUV_Camping_Guard_Edition"

    # 验证 Zone3 常驻守护进程的 ROIs 来自 JSON
    assert len(engine.zone3_daemon.rois) == len(engine.calib.zone3_rois)

    # 验证测距仪绑定了 4 路相机外参
    assert len(engine.distance_estimator.cameras_calib) == 4
    h, pitch = engine.distance_estimator.get_camera_params(0)
    assert h == 0.85
    assert pitch == -12.0


def test_calib_fallback_on_missing_file():
    """测试当配置文件路径不存在时的平稳降级"""
    calib = load_vehicle_calibration("non_existent_file.json")
    assert calib is not None
    assert len(calib.cameras) == 4
    assert len(calib.zone3_rois) >= 5
