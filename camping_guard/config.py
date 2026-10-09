"""
智能汽车露营守卫系统 - 全局配置与参数阈值规范
基于 camping_guard_algorithm_flow_revised_1.md
"""

import os
import json
from typing import List, Dict, Union, Optional
from camping_guard.types import Zone3ROI, CameraCalib, VehicleInfo, VehicleCalibration


class GuardConfig:
    # ================= 1. 防区距离与边界迟滞 (第二章) =================
    ZONE_FAR_DIST: float = 6.0         # Zone 0 远场边界 (> 6.0m 忽略)
    ZONE_OUTER_DIST: float = 3.0       # Zone 1 外围初筛边界 (3.0m ~ 6.0m)
    ZONE_INNER_DIST: float = 0.5       # Zone 2 核心研判边界 (0.5m ~ 3.0m)
    ZONE_CONTACT_DIST: float = 0.5     # Zone 3 接触区 (0 ~ 0.5m，改用固定 ROI)
    HYSTERESIS_BUFFER: float = 0.3     # 距离分档迟滞区间 (±0.3m)

    # ================= 2. 状态机流转与延迟配置 (第五章) =================
    S0_MOTION_CONFIRM_SEC: float = 0.5 # S0->S1 运动检测持续确认时长 (防单帧噪声)
    S1_TO_S0_TIMEOUT_SEC: float = 15.0 # S1->S0 无目标/仅白名单超时回落时长
    S2_TO_S1_TIMEOUT_SEC: float = 10.0 # S2->S1 Zone 2/3 无目标且无接触事件超时回落
    EVENT_COOLDOWN_SEC: float = 60.0   # 发生 L2/L3 后至少保持 S1 的冷却时长
    SAME_TRACE_DEDUP_SEC: float = 60.0 # 同一 Trace 同一等级事件推送去重冷却

    # ================= 3. 滞留时间与速度判定规则 (第四章) =================
    DWELL_TIME_Z1_L2_SEC: float = 30.0 # 陌生人在 Zone 1 滞留超过 30s 升 L2
    DWELL_TIME_Z2_L2_SEC: float = 10.0 # 陌生人在 Zone 2 滞留超过 10s 升 L2
    DWELL_RESET_WINDOW_SEC: float = 10.0 # 短暂离开防区 10s 内重新进入，累计计时不清零

    RADIAL_SPEED_WAKE_S2: float = 0.8  # 径向逼近速度 > 0.8 m/s 唤醒 S2
    RADIAL_SPEED_HUMAN_L2: float = 1.5 # 快速逼近 (人 > 1.5 m/s 且进入 Zone 2) 升 L2
    RADIAL_SPEED_BEAST_L3: float = 1.2 # 猛兽冲刺 (> 1.2 m/s，连续 3 帧且进入 Zone 2) 升 L3

    # ================= 4. 白名单鉴权规则 (第一、四章) =================
    UWB_DISTANCE_THRESHOLD: float = 1.2 # UWB 钥匙空间对齐阈值 (<= 1.2m)
    UWB_UNLOCK_EXIT_DIST: float = 1.5   # 正常退出解锁必须 UWB 在 1.5m 内 (防中继攻击)
    WHITE_TRACK_TIMEOUT_SEC: float = 3.0 # 白名单绑定失效超时 (遮挡/丢失超 3s)

    # ================= 5. 动作与时序阈值 (第二、四、六章) =================
    WINDOW_PEEK_DISTANCE: float = 0.5  # 贴窗窥探头部与车窗距离 (<= 0.5m)
    WINDOW_PEEK_DURATION_SEC: float = 3.0 # 贴窗窥探持续时长 (>= 3.0s)
    DOOR_PULL_DURATION_SEC: float = 1.5   # 拉门撬锁手部持续动作时长 (>= 1.5s)
    THEFT_DURATION_SEC: float = 2.0       # 装备盗窃手部停留时长 (>= 2.0s)
    CONTACT_FUSION_WINDOW_SEC: float = 2.0 # 视觉与 CAN/IMU 联合确认时间窗口 (±2.0s)

    # ================= 6. 露营火情研判规则 (第六章) =================
    CAMPFIRE_UNATTENDED_L2_SEC: float = 300.0 # 稳定火源无人看管超过 5 分钟 (300s) 升 L2
    CAMPFIRE_SPREAD_L3_DIST: float = 3.0      # 火焰蔓延逼近车体 < 3m 升 L3

    # ================= 7. 故障与降级阈值 (第一章) =================
    CAMERA_OCCLUSION_L2_SEC: float = 5.0 # 单路相机遮挡持续超过 5s 报 L2

    # ================= 8. Zone 3 敏感部件固定 ROI 预标定 (640x480 分辨率基准) =================
    @staticmethod
    def get_default_rois(config_path: str = "vehicle_calib.json") -> List[Zone3ROI]:
        """优先从 vehicle_calib.json 或 roi_config.json 载入，若不存在则回退至硬编码预置坐标"""
        if os.path.exists(config_path):
            calib = load_vehicle_calibration(config_path)
            if calib.zone3_rois:
                return calib.zone3_rois

        if os.path.exists("roi_config.json"):
            from camping_guard.tools.roi_calibrator import ROICalibrator
            calibrator = ROICalibrator("roi_config.json")
            rois = calibrator.load_rois()
            if rois:
                return rois

        return [
            Zone3ROI(
                roi_id="handle_fl",
                name="左前门把手",
                camera_id=1, # 左侧相机
                polygon=[(160, 260), (220, 260), (220, 310), (160, 310)],
                threshold_ratio=0.10,
                confirm_frames=2
            ),
            Zone3ROI(
                roi_id="handle_rl",
                name="左后门把手",
                camera_id=1, # 左侧相机
                polygon=[(380, 260), (440, 260), (440, 310), (380, 310)],
                threshold_ratio=0.10,
                confirm_frames=2
            ),
            Zone3ROI(
                roi_id="charge_port",
                name="快慢充电口",
                camera_id=1, # 左后侧相机
                polygon=[(520, 240), (580, 240), (580, 320), (520, 320)],
                threshold_ratio=0.12,
                confirm_frames=2
            ),
            Zone3ROI(
                roi_id="trunk_lock",
                name="后备箱开关与锁扣",
                camera_id=3, # 后视相机
                polygon=[(270, 280), (370, 280), (370, 350), (270, 350)],
                threshold_ratio=0.12,
                confirm_frames=2
            ),
            Zone3ROI(
                roi_id="roof_rack",
                name="车顶行李架与外挂自行车",
                camera_id=3, # 后视相机
                polygon=[(180, 80), (460, 80), (460, 160), (180, 160)],
                threshold_ratio=0.15,
                confirm_frames=2,
                roi_type="part"
            ),
            # 车身核心防碰撞监视区 (监测邻车开门碰撞、倒木、外物砸击)
            Zone3ROI(
                roi_id="body_flank_l",
                name="左侧车身门板防碰撞区",
                camera_id=1, # 左侧鱼眼
                polygon=[(120, 240), (580, 240), (600, 440), (100, 440)],
                threshold_ratio=0.08,
                confirm_frames=2,
                roi_type="body"
            ),
            Zone3ROI(
                roi_id="body_flank_r",
                name="右侧车身门板防碰撞区",
                camera_id=2, # 右侧鱼眼
                polygon=[(120, 240), (580, 240), (600, 440), (100, 440)],
                threshold_ratio=0.08,
                confirm_frames=2,
                roi_type="body"
            ),
            Zone3ROI(
                roi_id="body_rear_bumper",
                name="后保险杠与尾门防撞区",
                camera_id=3, # 后视相机
                polygon=[(150, 320), (490, 320), (520, 440), (120, 440)],
                threshold_ratio=0.08,
                confirm_frames=2,
                roi_type="body"
            )
        ]


def apply_calibration_to_guard_config(calib: VehicleCalibration):
    """将标定文件中的全局防区与时间阈值同步注入至 GuardConfig"""
    t = calib.thresholds
    if not t:
        return
    if "zone_far_dist_m" in t:
        GuardConfig.ZONE_FAR_DIST = float(t["zone_far_dist_m"])
    if "zone_outer_dist_m" in t:
        GuardConfig.ZONE_OUTER_DIST = float(t["zone_outer_dist_m"])
    if "zone_inner_dist_m" in t:
        GuardConfig.ZONE_INNER_DIST = float(t["zone_inner_dist_m"])
    if "zone_contact_dist_m" in t:
        GuardConfig.ZONE_CONTACT_DIST = float(t["zone_contact_dist_m"])
    else:
        GuardConfig.ZONE_CONTACT_DIST = 0.3
    if "hysteresis_buffer_m" in t:
        GuardConfig.HYSTERESIS_BUFFER = float(t["hysteresis_buffer_m"])
    if "door_pull_min_sec" in t:
        GuardConfig.DOOR_PULL_DURATION_SEC = float(t["door_pull_min_sec"])
    if "window_peek_min_sec" in t:
        GuardConfig.WINDOW_PEEK_DURATION_SEC = float(t["window_peek_min_sec"])
    if "theft_min_sec" in t:
        GuardConfig.THEFT_DURATION_SEC = float(t["theft_min_sec"])
    if "contact_fusion_window_sec" in t:
        GuardConfig.CONTACT_FUSION_WINDOW_SEC = float(t["contact_fusion_window_sec"])


def load_vehicle_calibration(config_path_or_dict: Union[str, dict, None] = "vehicle_calib.json") -> VehicleCalibration:
    """
    载入车辆标定参数 JSON (相机外参、几何尺寸、固定 ROI、安全防区阈值)
    若文件不存在，则自动降级回退至 GuardConfig 内置默认工业基准
    """
    data = None
    if isinstance(config_path_or_dict, dict):
        data = config_path_or_dict
    elif isinstance(config_path_or_dict, str):
        if os.path.exists(config_path_or_dict):
            with open(config_path_or_dict, "r", encoding="utf-8") as f:
                data = json.load(f)

    if data is None:
        # 默认基准标定对象 (符合行业典型 4 路环视 AVM 标定外参)
        default_cams = {
            0: CameraCalib(
                camera_id=0, name="front_fisheye",
                install_pos_m=(2.45, 0.0, 0.75), camera_height_m=0.75, camera_pitch_deg=-32.0,
                camera_yaw_deg=0.0, camera_roll_deg=0.0, fov_deg=190.0, resolution=(640, 480),
                vehicle_body_side="front",
                zone1_polygon=[(64, 168), (576, 168), (627, 456), (13, 456)],
                zone2_polygon=[(140, 240), (500, 240), (588, 456), (52, 456)]
            ),
            1: CameraCalib(
                camera_id=1, name="left_mirror_fisheye",
                install_pos_m=(0.85, 1.02, 1.15), camera_height_m=1.15, camera_pitch_deg=-38.0,
                camera_yaw_deg=90.0, camera_roll_deg=0.0, fov_deg=190.0, resolution=(640, 480),
                vehicle_body_side="left",
                zone1_polygon=[(40, 140), (600, 140), (620, 360), (20, 360)],
                zone2_polygon=[(80, 200), (560, 200), (600, 440), (40, 440)]
            ),
            2: CameraCalib(
                camera_id=2, name="right_mirror_fisheye",
                install_pos_m=(0.85, -1.02, 1.15), camera_height_m=1.15, camera_pitch_deg=-38.0,
                camera_yaw_deg=-90.0, camera_roll_deg=0.0, fov_deg=190.0, resolution=(640, 480),
                vehicle_body_side="right",
                zone1_polygon=[(40, 140), (600, 140), (620, 360), (20, 360)],
                zone2_polygon=[(80, 200), (560, 200), (600, 440), (40, 440)]
            ),
            3: CameraCalib(
                camera_id=3, name="rear_fisheye",
                install_pos_m=(-2.45, 0.0, 0.95), camera_height_m=0.95, camera_pitch_deg=-35.0,
                camera_yaw_deg=180.0, camera_roll_deg=0.0, fov_deg=190.0, resolution=(640, 480),
                vehicle_body_side="rear",
                zone1_polygon=[(64, 168), (576, 168), (627, 456), (13, 456)],
                zone2_polygon=[(140, 240), (500, 240), (588, 456), (52, 456)]
            ),
        }
        calib = VehicleCalibration(
            vehicle_info=VehicleInfo(model_name="Default_Vehicle", vin="VIN_DEFAULT"),
            cameras=default_cams,
            zone3_rois=GuardConfig.get_default_rois(""),
            thresholds={
                "zone_far_dist_m": GuardConfig.ZONE_FAR_DIST,
                "zone_outer_dist_m": GuardConfig.ZONE_OUTER_DIST,
                "zone_inner_dist_m": GuardConfig.ZONE_INNER_DIST,
                "hysteresis_buffer_m": GuardConfig.HYSTERESIS_BUFFER,
                "door_pull_min_sec": GuardConfig.DOOR_PULL_DURATION_SEC,
                "window_peek_min_sec": GuardConfig.WINDOW_PEEK_DURATION_SEC,
                "theft_min_sec": GuardConfig.THEFT_DURATION_SEC,
                "contact_fusion_window_sec": GuardConfig.CONTACT_FUSION_WINDOW_SEC,
            }
        )
        VehicleCalibration.apply_to_guard_config = lambda self: apply_calibration_to_guard_config(self)
        return calib

    # 1. 解析车辆信息
    v_raw = data.get("vehicle_info", {})
    v_info = VehicleInfo(
        model_name=v_raw.get("model_name", "Unknown_Model"),
        vin=v_raw.get("vin", "UNKNOWN_VIN"),
        wheelbase_m=float(v_raw.get("wheelbase_m", 2.95)),
        width_m=float(v_raw.get("width_m", 1.98)),
        length_m=float(v_raw.get("length_m", 4.98))
    )

    # 2. 解析多相机通道外参表
    cams_raw = data.get("cameras", {})
    cameras: Dict[int, CameraCalib] = {}
    for cid_str, cinfo in cams_raw.items():
        cid = int(cinfo.get("camera_id", cid_str))
        def_pos = (2.45, 0.0, 0.75) if cid == 0 else ((0.85, 1.02, 1.15) if cid == 1 else ((0.85, -1.02, 1.15) if cid == 2 else (-2.45, 0.0, 0.95)))
        def_yaw = 0.0 if cid == 0 else (90.0 if cid == 1 else (-90.0 if cid == 2 else 180.0))
        def_pitch = -32.0 if cid == 0 else (-38.0 if cid in (1, 2) else -35.0)
        def_side = "front" if cid == 0 else ("left" if cid == 1 else ("right" if cid == 2 else "rear"))

        cameras[cid] = CameraCalib(
            camera_id=cid,
            name=cinfo.get("name", f"camera_{cid}"),
            camera_height_m=float(cinfo.get("camera_height_m", def_pos[2])),
            camera_pitch_deg=float(cinfo.get("camera_pitch_deg", def_pitch)),
            fov_deg=float(cinfo.get("fov_deg", 190.0)),
            resolution=tuple(cinfo.get("resolution", [640, 480])),
            install_pos_m=tuple(cinfo.get("install_pos_m", def_pos)),
            camera_yaw_deg=float(cinfo.get("camera_yaw_deg", def_yaw)),
            camera_roll_deg=float(cinfo.get("camera_roll_deg", 0.0)),
            vehicle_body_side=cinfo.get("vehicle_body_side", def_side),
            zone1_polygon=[tuple(p) for p in cinfo.get("zone1_polygon", [])] if cinfo.get("zone1_polygon") else None,
            zone2_polygon=[tuple(p) for p in cinfo.get("zone2_polygon", [])] if cinfo.get("zone2_polygon") else None
        )

    # 3. 解析 Zone 3 车身敏感部件 ROI 多边形
    rois_raw = data.get("zone3_rois", [])
    zone3_rois: List[Zone3ROI] = []
    for r in rois_raw:
        pts = [tuple(p) for p in r.get("polygon", [])]
        zone3_rois.append(Zone3ROI(
            roi_id=r.get("roi_id", "custom_roi"),
            name=r.get("name", "自定义敏感区域"),
            camera_id=int(r.get("camera_id", 1)),
            polygon=pts,
            threshold_ratio=float(r.get("threshold_ratio", 0.12)),
            confirm_frames=int(r.get("confirm_frames", 2)),
            roi_type=r.get("roi_type", "part")
        ))

    # 4. 解析阈值字典
    thresholds = data.get("thresholds", {})

    calib = VehicleCalibration(
        vehicle_info=v_info,
        cameras=cameras,
        zone3_rois=zone3_rois,
        thresholds=thresholds
    )
    VehicleCalibration.apply_to_guard_config = lambda self: apply_calibration_to_guard_config(self)
    return calib

