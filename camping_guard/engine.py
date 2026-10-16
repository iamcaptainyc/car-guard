"""
露营守卫系统端到端核心协同引擎 (CampingGuardEngine)
聚合感知、常驻 ROI 守护、多源时序联合确认、异步 VLM 复核与整车执行联动
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from camping_guard.types import (
    GuardState, ThreatLevel, OperatingMode, VehicleZone,
    TrackedEntity, BoundingBox, ThreatEventRecord, ActionType,
    VehicleCalibration, ContactSensorEvent
)
from camping_guard.config import GuardConfig, load_vehicle_calibration

from camping_guard.state_machine import GuardStateMachine
from camping_guard.zone3_daemon import Zone3ResidentDaemon
from camping_guard.perception.distance import GroundDistanceEstimator, ZoneHysteresisFilter
from camping_guard.perception.whitelist import WhitelistManager
from camping_guard.perception.tracker import TargetTrackManager
from camping_guard.perception.bytetrack import ByteTracker
from camping_guard.perception.reid import OSNetReIDExtractor, PersonSessionManager
from camping_guard.perception.faceid import FaceIDPipeline
from camping_guard.perception.detector import BaseObjectDetector
from camping_guard.perception.video_source import MultiCameraStreamManager
from camping_guard.perception.pose_estimator import BasePoseEstimator, RTMPoseEstimator
from camping_guard.analysis.pose_action import PoseActionClassifier
from camping_guard.analysis.fusion import MultiSourceFusionEngine
from camping_guard.analysis.fire_monitor import CampfireMonitor
from camping_guard.review.vlm_reviewer import AsyncVLMReviewer
from camping_guard.evidence.recorder import EvidenceRecorder
from camping_guard.vehicle.mock_bus import MockVehicleBus
from camping_guard.vehicle.actuator import VehicleActuator
from camping_guard.profiler import PipelineProfiler

class CampingGuardEngine:
    def __init__(
        self,
        config_path: Optional[str] = "vehicle_calib.json",
        detector: Optional[BaseObjectDetector] = None,
        use_bytetrack: bool = False,
        stream_manager: Optional[MultiCameraStreamManager] = None,
        pose_estimator: Optional[BasePoseEstimator] = None,
        distance_estimator: Optional[GroundDistanceEstimator] = None,
        calib: Optional[VehicleCalibration] = None,
        profiler: Optional[PipelineProfiler] = None,
        use_tensorrt: bool = False,
        animal_classifier: Optional[Any] = None,
        vlm_throttle_enabled: bool = True,
        vlm_active_fps: float = 5.0
    ):
        # 0. 载入整车标定参数配置 (解耦 JSON 配置形式)
        if calib is not None:
            self.calib = calib
        elif config_path is not None:
            self.calib = load_vehicle_calibration(config_path)
        else:
            self.calib = load_vehicle_calibration()

        if hasattr(self.calib, "apply_to_guard_config"):
            self.calib.apply_to_guard_config()

        # 0.5 性能与模型耗时剖析器初始化
        self.profiler = profiler or PipelineProfiler()
        self._frame_count = 0

        # 检测 Jetson Orin 硬件平台自适应
        from camping_guard.perception.trt_engine import is_jetson
        self.use_tensorrt = use_tensorrt or is_jetson()

        # 1. 核心子系统实例化
        self.state_machine = GuardStateMachine()
        self.zone3_daemon = Zone3ResidentDaemon(
            rois=self.calib.zone3_rois,
            on_disturbance_callback=self._on_zone3_roi_disturbed
        )
        if distance_estimator is not None:
            self.distance_estimator = distance_estimator
            if self.calib and not self.distance_estimator.cameras_calib:
                self.distance_estimator.cameras_calib = self.calib.cameras
        else:
            self.distance_estimator = GroundDistanceEstimator(cameras_calib=self.calib.cameras)

        self.zone_filter = ZoneHysteresisFilter()
        reid_cand = "models/reid/osnet_x0_25.engine"
        reid_path = reid_cand if (self.use_tensorrt and os.path.exists(reid_cand)) else "models/reid/osnet_x0_25.onnx"
        self.reid_extractor = OSNetReIDExtractor(model_path=reid_path, use_trt=self.use_tensorrt)
        self.session_manager = PersonSessionManager()
        self.faceid_pipeline = FaceIDPipeline(use_trt=self.use_tensorrt)
        self.faceid_pipeline.profiler = self.profiler

        self.whitelist_mgr = WhitelistManager(
            faceid_pipeline=self.faceid_pipeline,
            session_manager=self.session_manager
        )
        self.track_mgr = TargetTrackManager(
            reid_extractor=self.reid_extractor,
            session_manager=self.session_manager
        )
        self.track_mgr.profiler = self.profiler

        self.bytetracker = ByteTracker(profiler=self.profiler) if use_bytetrack else None
        self.detector = detector
        self.stream_manager = stream_manager
        self.pose_estimator = pose_estimator or RTMPoseEstimator(use_trt=self.use_tensorrt)
        self.pose_classifier = PoseActionClassifier()
        self.fusion_engine = MultiSourceFusionEngine()
        self.fire_monitor = CampfireMonitor()
        self.vlm_reviewer = AsyncVLMReviewer(profiler=self.profiler)
        self.recorder = EvidenceRecorder(fps=30.0, buffer_duration_sec=15.0, async_encoding=True)
        self.vehicle_bus = MockVehicleBus()
        self.actuator = VehicleActuator()

        self.animal_classifier = animal_classifier
        if self.animal_classifier is None:
            try:
                from camping_guard.perception.animal_classifier import AnimalClassifier
                self.animal_classifier = AnimalClassifier(use_trt=self.use_tensorrt)
            except Exception:
                pass

        if self.detector is not None and getattr(self.detector, 'animal_classifier', None) is None:
            if hasattr(self.detector, 'animal_classifier'):
                self.detector.animal_classifier = self.animal_classifier
        if self.animal_classifier is None and self.detector is not None:
            self.animal_classifier = getattr(self.detector, 'animal_classifier', None)

        # 若启用了 ByteTracker 且挂载了二级动物分类器，开启检测器延迟分类模式 (defer_animal_cls=True)，
        # 将二级细分类解耦至 ByteTrack 轨迹关联之后按需触发，彻底消除同目标多框与跨帧重复推理
        if self.bytetracker is not None and self.animal_classifier is not None and self.detector is not None:
            if hasattr(self.detector, 'defer_animal_cls'):
                self.detector.defer_animal_cls = True

        # 注册模型元数据至 profiler (保持 key 与执行期 profile_model("...") 完全一致)
        if self.detector:
            self.detector.profiler = self.profiler
        if self.detector and hasattr(self.detector, 'model_path'):
            if getattr(self.detector, 'is_ensemble', False):
                det_label = "Parallel-Ensemble-Det"
                desc = "Orin双流并发目标检测 (主检测器+火灾检测器聚合)"
                self.profiler.register_model_metadata(det_label, self.detector.model_path, desc)

                # 分别注册通用检测主流与火灾检测副流模型元数据
                main_det = getattr(self.detector, 'main_detector', None)
                if main_det and hasattr(main_det, 'model_path'):
                    m_lbl = getattr(main_det, 'profiler_label', 'General-Det')
                    self.profiler.register_model_metadata(
                        m_lbl,
                        main_det.model_path,
                        "通用人车动物目标检测 (并发主流)",
                        is_substream=True
                    )
                fire_det = getattr(self.detector, 'fire_detector', None)
                if fire_det and hasattr(fire_det, 'model_path'):
                    f_lbl = getattr(fire_det, 'profiler_label', 'FireSmoke-Det')
                    self.profiler.register_model_metadata(
                        f_lbl,
                        fire_det.model_path,
                        "火灾浓烟专用目标检测 (并发副流)",
                        is_substream=True
                    )
            elif getattr(self.detector, 'is_megadetector', False):
                det_label = "MegaDetector"
                desc = "S1全图目标检测初筛 (640x480)"
                self.profiler.register_model_metadata(det_label, self.detector.model_path, desc)
            else:
                det_label = getattr(self.detector, 'profiler_label', 'YOLO11n-Det')
                desc = "S1全图目标检测初筛 (640x480)"
                self.profiler.register_model_metadata(det_label, self.detector.model_path, desc)
        if self.animal_classifier and getattr(self.animal_classifier, 'model_path', None) and getattr(self.animal_classifier, 'session', None) is not None:
            self.profiler.register_model_metadata("MobileNetV3-AnimalCls", self.animal_classifier.model_path, "MegaDetector二级动物细分类 (224x224)")
        if self.pose_estimator and hasattr(self.pose_estimator, 'model_path'):
            self.profiler.register_model_metadata("RTMPose-s", self.pose_estimator.model_path, "Zone2/3人员姿态与17骨骼点提取")
        if self.reid_extractor and hasattr(self.reid_extractor, 'model_path'):
            self.profiler.register_model_metadata("OSNet-ReID", self.reid_extractor.model_path, "人员跨时空重识别512维特征建档")
        if self.faceid_pipeline and hasattr(self.faceid_pipeline, 'detector') and hasattr(self.faceid_pipeline.detector, 'model_path'):
            self.profiler.register_model_metadata("SCRFD-500M", self.faceid_pipeline.detector.model_path, "4m以内人员上半身人脸定位初筛")
        if self.faceid_pipeline and hasattr(self.faceid_pipeline, 'recognizer') and hasattr(self.faceid_pipeline.recognizer, 'model_path'):
            self.profiler.register_model_metadata("ArcFace-MobileFaceNet", self.faceid_pipeline.recognizer.model_path, "FQA质检通过后白名单比对")

        # 注册 Qwen-VLM 慢通路模型元信息
        vlm_model_desc = "Qwen2.5-VL / Qwen3.5-VLM"
        backend_obj = getattr(self.vlm_reviewer, "backend", None)
        if backend_obj:
            if getattr(backend_obj, "_resolved_engine_dir", None):
                vlm_model_desc = str(backend_obj._resolved_engine_dir)
            elif getattr(backend_obj, "model_dir_arg", None):
                vlm_model_desc = str(backend_obj.model_dir_arg)
        self.profiler.register_model_metadata("Qwen-VLM", vlm_model_desc, "慢通路异步端侧多模态复核与中文战报生成", is_async=True)

        # 运行指标与统计
        self.current_threat_level: ThreatLevel = ThreatLevel.L0_SAFE
        self.latest_event_description: str = "系统正常运行"
        self.executed_actions: List[str] = []
        self.active_targets: List[TrackedEntity] = []
        self.active_roi_disturbances: List[dict] = []
        self.historical_threat_events: List[ThreatEventRecord] = [] # 全流程高危威胁事件审计档案
        self.vlm_summaries: List[dict] = []

        # 标志位与事件锁存防抖状态
        self._zone3_triggered_this_frame = False
        self._last_locked_event_id: Optional[str] = None
        self._last_locked_time: float = 0.0

        # 慢通路并发动态自适应让步调度 (方案 A: Dynamic Yield / Adaptive Throttling)
        self.vlm_throttle_enabled = vlm_throttle_enabled
        self.vlm_active_fps = vlm_active_fps
        self._last_detection_time: float = 0.0
        self._last_raw_detections: List[dict] = []
        self.is_vlm_throttling_now: bool = False

    def is_vlm_active(self) -> bool:
        """查询慢通路端侧大模型当前是否正在执行生成/复核 (纯进程内状态，零耗时查询)"""
        if self.vlm_reviewer is not None and getattr(self.vlm_reviewer, "is_inferencing", False):
            return True
        return False

    def _on_zone3_roi_disturbed(self, roi_id: str, camera_id: int):
        """Zone 3 常驻守护进程回调，强制拉起 S2"""
        self._zone3_triggered_this_frame = True

    def activate(self) -> Tuple[bool, str]:
        """依据整车状态激活露营守卫"""
        bus = self.vehicle_bus
        success, msg = self.state_machine.activate_system(
            gear=bus.gear,
            epb_engaged=bus.epb_engaged,
            guard_switch_on=bus.guard_switch_on,
            all_doors_locked=bus.all_doors_locked,
            seat_occupied=bus.seat_occupied,
            uwb_key_inside=(bus.uwb_key_pos is not None)
        )
        return success, msg

    def process_tick(
        self,
        camera_frames: Dict[int, np.ndarray],
        raw_detections: Optional[List[dict]] = None,
        occluded_cams: Optional[List[int]] = None,
        bus_events: Optional[List[ContactSensorEvent]] = None,
        current_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        每帧调度周期主处理
        :param camera_frames: {cam_id: np.ndarray}
        :param raw_detections: [{ 'track_id': int, 'category': str, 'conf': float, 'bbox': (x,y,w,h) }]
        :param occluded_cams: 被遮挡的相机 ID 列表
        :param bus_events: 显式传入的车身物理传感器事件 (CAN/IMU/UWB)
        :param current_time: 帧物理时间戳 (秒，若为 None 则自动取系统挂钟 time.time())
        """
        current_time = time.time() if current_time is None else current_time
        self._zone3_triggered_this_frame = False
        occluded_cams = occluded_cams or []

        # 性能剖析启动当帧采样
        self._frame_count += 1
        self.profiler.start_frame(self._frame_count)

        # 兼容用户直接传入单个 numpy ndarray 的情况
        if isinstance(camera_frames, np.ndarray):
            camera_frames = {0: camera_frames}

        # 0. 自动获取视频流 (若未显式传入且配置了 StreamManager)
        if not camera_frames and self.stream_manager is not None:
            fps_target = 2.0 if self.state_machine.state == GuardState.S0_STANDBY else 15.0
            camera_frames = self.stream_manager.fetch_frames(target_fps=fps_target)

        # 1. 录像环形缓冲持续推流 (单路测试时自动对齐传入相机，多路时以 0/1 为主)
        if len(camera_frames) == 1:
            primary_cam_id = list(camera_frames.keys())[0]
        else:
            primary_cam_id = 1 if 1 in camera_frames else (0 if 0 in camera_frames else (list(camera_frames.keys())[0] if camera_frames else 1))
        primary_frame = camera_frames.get(primary_cam_id)
        if primary_frame is not None:
            with self.profiler.profile_stage("Recorder_RingBuffer"):
                self.recorder.push_frame(primary_frame, current_time)

        # 2. 消费车端总线事件 (CAN / IMU / UWB)
        pulled_events = self.vehicle_bus.pull_pending_events()
        all_bus_events = (bus_events or []) + pulled_events
        for bev in all_bus_events:
            self.fusion_engine.record_sensor_event(bev)

        # 3. 常驻车身固定 ROI 扰动检测 (阶段 0.5 跨状态并行运行)
        with self.profiler.profile_stage("Body_ROI_Daemon"):
            self.active_roi_disturbances = self.zone3_daemon.process_frames(camera_frames, current_time)

        # 4. 相机遮挡故障降级检测 (第一章)
        occl_threat, occl_desc = self.state_machine.update_camera_occlusion(occluded_cams, current_time)

        # 5. 目标检测 (若未显式传入 detections 且配置了 Detector)
        with self.profiler.profile_stage("Object_Detection"):
            vlm_busy = self.is_vlm_active()
            should_throttle = False
            if vlm_busy and self.vlm_throttle_enabled:
                # 动态自适应降频让步机制 (Dynamic Throttling / Yield - 方案 A):
                # 当大模型在 GPU 上进行自回归解码时，将快通路 YOLO 动态降频至 vlm_active_fps (如 5 FPS)
                # 释放 80%+ 的 SM 计算时间片给大模型，同时依靠 ByteTrack 卡尔曼滤波预测维持丝滑目标轨迹
                min_interval = 1.0 / max(0.1, self.vlm_active_fps)
                if (current_time - self._last_detection_time) < min_interval:
                    should_throttle = True

            self.is_vlm_throttling_now = should_throttle

            if not should_throttle and (raw_detections is None or len(raw_detections) == 0) and self.detector is not None:
                if primary_frame is not None:
                    det_sess = getattr(self.detector, 'session', None)
                    if getattr(self.detector, 'is_ensemble', False):
                        det_label = "Parallel-Ensemble-Det"
                    elif getattr(self.detector, 'is_megadetector', False):
                        det_label = "MegaDetector"
                    else:
                        det_label = getattr(self.detector, 'profiler_label', 'YOLO11n-Det')
                    with self.profiler.profile_model(det_label, condition="S1全图目标检测 (640x480)", session=det_sess):
                        raw_detections = self.detector.detect(primary_frame)
                    self._last_detection_time = current_time
                    self._last_raw_detections = raw_detections
            elif should_throttle:
                # 动态降频让步帧：跳过重量级 GPU 前向推理，复用上一有效检测候选传入 ByteTrack，
                # 配合卡尔曼滤波在 CPU 上微秒级更新，既不造成目标漂移/丢失，又彻底释放 GPU SM！
                raw_detections = list(self._last_raw_detections)

            if raw_detections is None:
                raw_detections = []

        frame_shape = (480, 640)
        if primary_frame is not None:
            frame_shape = primary_frame.shape[:2]

        # 6. 目标追踪与防区测距
        with self.profiler.profile_stage("Tracking_and_ReID"):
            if self.bytetracker is not None:
                # 使用车载轻量 ByteTrack 引擎 (内嵌 Deep-ByteTrack 遮挡重识别外观召回)
                tracked_entities = self.bytetracker.update(
                    raw_detections, current_time,
                    frame=primary_frame,
                    reid_extractor=self.reid_extractor
                )
                for t in tracked_entities:
                    cam_id = getattr(t, 'camera_id', primary_cam_id)
                    dist = self.distance_estimator.estimate_ground_distance(t.bbox.bottom_center, frame_shape, camera_id=cam_id)
                    zone = self.zone_filter.update_zone(t.track_id, dist)
                    vr = self.distance_estimator.update_radial_velocity(t.track_id, dist, current_time)
                    t.ground_dist = dist
                    t.zone = zone
                    t.radial_speed = vr
                    if not t.track_uid:
                        t.track_uid = f"cam{cam_id}_trk{t.track_id}"

                    st = self.bytetracker.get_track(t.track_id)
                    if st is not None:
                        st.ground_dist = dist
                        st.zone = zone
                        st.radial_speed = vr
                        st.track_uid = t.track_uid

                    # 二级动物物种细分类 (质量门控 + 多帧滑动证据融合 + 猛兽破局纠偏 + 自适应分频)
                    if t.category in ['animal', 'small_animal', 'large_beast'] and self.animal_classifier is not None and primary_frame is not None:
                        bx, by, bw, bh = int(t.bbox.x), int(t.bbox.y), int(t.bbox.w), int(t.bbox.h)
                        area = float(bw * bh)
                        min_dim = min(bw, bh)
                        last_cls_t = getattr(st, 'last_animal_cls_time', 0.0) if st else 0.0
                        cls_obs = getattr(st, 'cls_observations', []) if st else []
                        cls_stable = getattr(st, 'cls_stable', False) if st else False
                        stable_area = getattr(st, 'stable_bbox_area', 0.0) if st else 0.0
                        time_since_last = current_time - last_cls_t

                        need_animal_cls = (
                            len(cls_obs) == 0
                            or not t.species
                            or t.category == 'animal'
                            or (cls_stable and stable_area > 0 and (area >= 1.5 * stable_area) and (time_since_last >= 0.5))
                            or (not cls_stable and (time_since_last >= 0.6))
                            or (cls_stable and (time_since_last >= 3.0))
                        )
                        if need_animal_cls:
                            fh, fw = primary_frame.shape[:2]
                            cx1, cy1 = max(0, min(fw - 1, bx)), max(0, min(fh - 1, by))
                            cx2, cy2 = max(cx1 + 1, min(fw, bx + bw)), max(cy1 + 1, min(fh, by + bh))
                            crop = primary_frame[cy1:cy2, cx1:cx2]
                            if crop.size > 0:
                                cls_sess = getattr(self.animal_classifier, 'session', None)
                                with self.profiler.profile_model("MobileNetV3-AnimalCls", condition=f"ID:{t.track_id} 动物物种细分类 (224x224)", session=cls_sess):
                                    cat, cls_conf, sp_en, sp_cn = self.animal_classifier.classify(crop)

                                # 1. 画质与尺寸前置门控 (Quality Gating)
                                # 极小或边缘模糊裁剪块 (边长<40 或面积<1600) 权重降权，且严禁直接稳定锁存
                                is_low_quality = (min_dim < 40 or area < 1600)
                                q_size = min(1.0, (area ** 0.5) / 160.0)
                                obs_weight = float(cls_conf) * (0.2 if is_low_quality else q_size)

                                new_obs = {
                                    'cat': cat,
                                    'cls_conf': float(cls_conf),
                                    'sp_en': sp_en,
                                    'sp_cn': sp_cn,
                                    'weight': obs_weight,
                                    'area': area,
                                    'time': current_time,
                                    'is_high_quality': not is_low_quality
                                }

                                if st is not None:
                                    st.last_animal_cls_time = current_time
                                    st.animal_cls_count = getattr(st, 'animal_cls_count', 0) + 1
                                    if not hasattr(st, 'cls_observations') or st.cls_observations is None:
                                        st.cls_observations = []

                                    # 2. 安全优先：猛兽确信破局覆盖 (Breakthrough Override)
                                    # 当捕捉到高置信度大型猛兽时 (conf >= 0.55 且非模糊小图)，立即破除过往小动物观察并拉响警报
                                    if cat == 'large_beast' and cls_conf >= 0.55 and not is_low_quality:
                                        st.confirmed_large_beast = True
                                        st.category = 'large_beast'
                                        st.species = sp_en
                                        st.species_cn = sp_cn
                                        st.cls_observations = [o for o in st.cls_observations if o['cat'] == 'large_beast']
                                        st.cls_observations.append(new_obs)
                                        if cls_conf >= 0.80:
                                            st.cls_stable = True
                                            st.stable_bbox_area = area
                                    else:
                                        st.cls_observations.append(new_obs)
                                        if len(st.cls_observations) > 5:
                                            st.cls_observations.pop(0)

                                        # 3. 多帧滑动证据加权融合 (Multi-Frame Rolling Evidence Fusion)
                                        beast_weight = sum(o['weight'] for o in st.cls_observations if o['cat'] == 'large_beast')
                                        small_weight = sum(o['weight'] for o in st.cls_observations if o['cat'] == 'small_animal')
                                        total_weight = beast_weight + small_weight

                                        if total_weight > 0:
                                            if beast_weight > small_weight:
                                                winning_cat = 'large_beast'
                                            else:
                                                winning_cat = 'small_animal'

                                            sp_weights = {}
                                            for o in st.cls_observations:
                                                if o['cat'] == winning_cat and o['sp_en'] and o['sp_en'] not in ['unknown', 'unclassified_animal']:
                                                    key = (o['sp_en'], o['sp_cn'])
                                                    sp_weights[key] = sp_weights.get(key, 0.0) + o['weight']

                                            if sp_weights:
                                                best_sp_en, best_sp_cn = max(sp_weights.items(), key=lambda x: x[1])[0]
                                            else:
                                                best_sp_en = sp_en
                                                best_sp_cn = sp_cn

                                            st.category = winning_cat
                                            st.species = best_sp_en
                                            st.species_cn = best_sp_cn

                                            if winning_cat == 'large_beast':
                                                st.confirmed_large_beast = True
                                            else:
                                                # 动态自愈纠偏：若早前模糊误报为猛兽，但后续积累了 >= 2 个高质量小动物证据且权重占比 >= 80%
                                                hq_small_count = sum(1 for o in st.cls_observations if o['is_high_quality'] and o['cat'] == 'small_animal')
                                                if hq_small_count >= 2 and (small_weight / total_weight) >= 0.80:
                                                    st.confirmed_large_beast = False

                                            # 4. 稳定态收敛判定 (Convergence Gate)
                                            hq_obs_count = sum(1 for o in st.cls_observations if o['is_high_quality'])
                                            winning_ratio = max(beast_weight, small_weight) / total_weight
                                            if (not is_low_quality and cls_conf >= 0.80) or (hq_obs_count >= 2 and winning_ratio >= 0.70):
                                                st.cls_stable = True
                                                st.stable_bbox_area = area

                                    t.category = st.category
                                    t.species = st.species
                                    t.species_cn = st.species_cn
                                else:
                                    t.category = cat
                                    t.species = sp_en
                                    t.species_cn = sp_cn
                    elif t.category == 'animal' and self.animal_classifier is None:
                        t.category = getattr(t, 'fallback_category', 'small_animal')
                        if st is not None:
                            st.category = t.category

                    # 人员 ReID 特征建档 (仅在首次入场且无特征时建档 1 次)
                    if t.category == 'person' and self.reid_extractor and primary_frame is not None and t.reid_feature is None:
                        reid_sess = getattr(self.reid_extractor, 'session', None)
                        with self.profiler.profile_model("OSNet-ReID", condition=f"ID:{t.track_id} ByteTrack建档", session=reid_sess):
                            t.reid_feature = self.reid_extractor.extract_from_frame(primary_frame, t.bbox)

                    if t.category == 'person' and t.reid_feature is not None and self.session_manager:
                        sess, is_new = self.session_manager.match_or_create_session(
                            t.track_uid, t.reid_feature, current_time, quality_score=t.confidence, camera_id=cam_id, track_id=t.track_id
                        )
                        t.person_session_id = sess.session_id
                        if not is_new:
                            t.cumulative_dwell_time = max(t.cumulative_dwell_time, sess.cumulative_dwell_time)
                            # 跨长时遮挡缝合根轨迹 ID，保证目标被遮挡重现后 ID 绝对稳定不变
                            if sess.root_track_id is not None and sess.root_track_id != t.track_id:
                                old_id = t.track_id
                                root_id = sess.root_track_id
                                self.bytetracker.reassign_track_id(old_id, root_id)
                                self.distance_estimator.reassign_track_id(old_id, root_id)
                                self.zone_filter.reassign_track_id(old_id, root_id)
                                self.pose_classifier.reassign_track_id(old_id, root_id)
                                t.track_id = root_id
                                t.track_uid = f"cam{cam_id}_trk{root_id}"
                                st = self.bytetracker.get_track(root_id)

                        if sess.is_whitelisted and not t.is_whitelisted:
                            t.is_whitelisted = True
                            t.face_state = FaceIDState.VERIFIED
                            t.face_id_name = sess.whitelist_name

                    # 持久化同步写回底层 STrack 缓存，杜绝下帧重复推理
                    if st is not None:
                        st.reid_feature = t.reid_feature
                        st.person_session_id = t.person_session_id
                        st.is_whitelisted = t.is_whitelisted
                        st.face_state = t.face_state
                        st.face_id_name = t.face_id_name
                self.active_targets = tracked_entities
            else:
                # 使用基础 TrackManager 流程 (向下兼容)
                processed_dets = []
                for d in raw_detections:
                    tid = d.get('track_id', 1)
                    bx, by, bw, bh = d['bbox']
                    bbox_obj = BoundingBox(bx, by, bw, bh)
                    cam_id = d.get('camera_id', primary_cam_id)

                    dist = self.distance_estimator.estimate_ground_distance(bbox_obj.bottom_center, frame_shape, camera_id=cam_id)
                    zone = self.zone_filter.update_zone(tid, dist)
                    vr = self.distance_estimator.update_radial_velocity(tid, dist, current_time)


                    processed_dets.append({
                        'track_id': tid,
                        'category': d.get('category', 'person'),
                        'conf': d.get('conf', 0.8),
                        'bbox': bbox_obj,
                        'dist': dist,
                        'radial_speed': vr,
                        'zone': zone,
                        'species': d.get('species'),
                        'species_cn': d.get('species_cn')
                    })
                self.active_targets = self.track_mgr.update_tracks(
                    processed_dets,
                    current_time,
                    frame=primary_frame,
                    camera_id=primary_cam_id
                )

        # 定期清理已彻底移除轨迹的历史字典，同时保留 lost_stracks 临时遮挡轨迹上下文
        retain_ids = set(t.track_id for t in self.active_targets)
        if self.bytetracker is not None:
            for lst in self.bytetracker.lost_stracks:
                retain_ids.add(lst.track_id)
        self.distance_estimator.clean_stale_tracks(retain_ids)
        if hasattr(self.zone_filter, 'clean_stale_tracks'):
            self.zone_filter.clean_stale_tracks(retain_ids)
        if hasattr(self.pose_classifier, 'clean_stale_tracks') and self._frame_count % 30 == 0:
            self.pose_classifier.clean_stale_tracks(retain_ids)
        if hasattr(self.faceid_pipeline, 'clean_stale_tracks') and self._frame_count % 30 == 0:
            self.faceid_pipeline.clean_stale_tracks(retain_ids)

        # 7. 白名单双通道排查 (步骤 1.2, 结合 UWB, FaceID 深度模型与 ReID 全局会话)
        with self.profiler.profile_stage("Whitelist_and_FaceID"):
            self.whitelist_mgr.evaluate_whitelist(
                targets=self.active_targets,
                uwb_key_pos=self.vehicle_bus.uwb_key_pos,
                face_id_results=None,
                current_time=current_time,
                frame=primary_frame
            )
            if self.bytetracker is not None:
                for t in self.active_targets:
                    st = self.bytetracker.get_track(t.track_id)
                    if st is not None:
                        st.is_whitelisted = t.is_whitelisted
                        st.face_state = t.face_state
                        st.face_id_name = t.face_id_name

        # 8. 统计防区状态用于状态机调度
        has_zone1 = any(t.zone == VehicleZone.ZONE_1_OUTER and not t.is_whitelisted for t in self.active_targets)
        has_zone2 = any(t.zone == VehicleZone.ZONE_2_INNER and not t.is_whitelisted for t in self.active_targets)
        min_radial_speed = min([t.radial_speed for t in self.active_targets], default=0.0)
        has_fire_smoke = any(t.category in ['fire', 'smoke'] for t in self.active_targets)
        has_contact_event = any(e.event_type in ["IMU_VIBRATION_PEAK", "DOOR_HANDLE_PULL", "CHARGING_TAMPER"] for e in all_bus_events)

        # 9. 状态机推进 (S0 / S1 / S2)
        self.state_machine.step(
            has_motion_s0=(len(self.active_targets) > 0),
            has_zone1_targets=has_zone1,
            has_zone2_targets=has_zone2,
            has_zone2_3_targets=has_zone2,
            max_radial_speed=min_radial_speed,
            has_fire_smoke=has_fire_smoke,
            has_contact_event=has_contact_event,
            is_zone3_disturbed=(len(self.active_roi_disturbances) > 0 or self._zone3_triggered_this_frame),
            is_body_roi_disturbed=(len(self.active_roi_disturbances) > 0 or self._zone3_triggered_this_frame),
            current_time=current_time
        )

        # 10. 深度研判 (S2) 与多源时序联合确认 (阶段 2)
        highest_threat = ThreatLevel.L0_SAFE
        highest_action = ActionType.NORMAL_PASS
        threat_description = "安全免扰"
        has_contact_confirmed = False
        target_for_threat = None

        with self.profiler.profile_stage("Pose_and_Action_Fusion"):
            # 火情研判通道
            fire_targets = [t for t in self.active_targets if t.category in ['fire', 'smoke']]
            person_targets = [t for t in self.active_targets if t.category == 'person']
            if fire_targets:
                fire_threat, fire_desc = self.fire_monitor.update_fire_status(fire_targets, person_targets, current_time)
                threat_rank = {ThreatLevel.L0_SAFE: 0, ThreatLevel.L1_NOTICE: 1, ThreatLevel.L2_WARNING: 2, ThreatLevel.L3_CRITICAL: 3}
                if threat_rank.get(fire_threat, 0) > threat_rank.get(highest_threat, 0):
                    highest_threat = fire_threat
                    highest_action = ActionType.FIRE_SMOKE
                    threat_description = fire_desc
                    target_for_threat = fire_targets[0]

            # 人员与动物研判通道
            handle_polys = [r.polygon for r in self.zone3_daemon.rois if "handle" in r.roi_id]
            window_polys = [r.polygon for r in self.zone3_daemon.rois if "window" in r.roi_id]

            for target in self.active_targets:
                if target.is_whitelisted:
                    continue

                # 若目标进入核心研判区 (Zone 2) 且为人员，按需触发 RTMPose 局部切片骨骼点提取 (Top-Down 节能范式)
                if target.category == 'person' and target.zone == VehicleZone.ZONE_2_INNER:
                    st = self.bytetracker.get_track(target.track_id) if self.bytetracker else None
                    last_pose_t = getattr(st, 'last_pose_time', 0.0) if st else 0.0
                    # 姿态时序限频：在 Zone 2 维持 10 FPS (100ms 间隔) 研判刷新率，避免 30 FPS 饱和挤占边缘算力
                    need_pose_infer = (current_time - last_pose_t >= 0.10) or (target.keypoints is None)

                    if need_pose_infer and primary_frame is not None and self.pose_estimator is not None:
                        bx = int(target.bbox.x if hasattr(target.bbox, 'x') else target.bbox[0])
                        by = int(target.bbox.y if hasattr(target.bbox, 'y') else target.bbox[1])
                        bw = int(target.bbox.w if hasattr(target.bbox, 'w') else target.bbox[2])
                        bh = int(target.bbox.h if hasattr(target.bbox, 'h') else target.bbox[3])
                        pose_sess = getattr(self.pose_estimator, 'session', None)
                        with self.profiler.profile_model("RTMPose-s", condition=f"ID:{target.track_id} Zone 2 姿态骨骼提取 (10FPS)", session=pose_sess):
                            kpts_list = self.pose_estimator.estimate_keypoints(primary_frame, [(bx, by, bw, bh)])
                        if kpts_list:
                            target.keypoints = kpts_list[0]
                        if st is not None:
                            st.keypoints = target.keypoints
                            st.last_pose_time = current_time
                    else:
                        # 复用上一帧提取的骨骼点，保持动作分类器时序连贯性
                        if st is not None and st.keypoints is not None:
                            target.keypoints = st.keypoints
                elif target.category == 'person' and target.zone != VehicleZone.ZONE_2_INNER:
                    # 退出 Zone 2 时及时清理姿态关键点，避免远距离依然残留陈旧姿态
                    target.keypoints = None
                    if self.bytetracker:
                        st = self.bytetracker.get_track(target.track_id)
                        if st is not None:
                            st.keypoints = None

                # 动作与姿态分类 (传入门把手与车窗 ROI 空间多边形)
                action, visual_threat, v_desc = self.pose_classifier.classify_action(
                    target, current_time,
                    handle_roi_polygons=handle_polys,
                    window_roi_polygons=window_polys
                )
                # 融合整车接触信号仲裁
                final_threat, contact_flag, fusion_desc = self.fusion_engine.arbitrate_threat(
                    target, action, visual_threat, current_time
                )

                # 更新最高等级
                threat_rank = {ThreatLevel.L0_SAFE: 0, ThreatLevel.L1_NOTICE: 1, ThreatLevel.L2_WARNING: 2, ThreatLevel.L3_CRITICAL: 3}
                if threat_rank[final_threat] > threat_rank[highest_threat]:
                    highest_threat = final_threat
                    highest_action = action
                    threat_description = fusion_desc
                    has_contact_confirmed = contact_flag
                    target_for_threat = target

            # 叠加相机故障威胁
            if occl_threat is not None:
                threat_rank = {ThreatLevel.L0_SAFE: 0, ThreatLevel.L1_NOTICE: 1, ThreatLevel.L2_WARNING: 2, ThreatLevel.L3_CRITICAL: 3}
                if threat_rank[occl_threat] > threat_rank[highest_threat]:
                    highest_threat = occl_threat
                    threat_description = occl_desc

            # 叠加车身防碰撞区与敏感部件 ROI 运动扰动研判 (邻车开门碰撞、树木倒塌、外物砸击、手部触碰)
            if len(self.active_roi_disturbances) > 0:
                # 优先检查是否存在车身大面积防碰撞监视区扰动 (邻车开门碰撞/倒木/外物砸车)
                body_dists = [d for d in self.active_roi_disturbances if d.get("roi_type") == "body" or "碰撞" in d.get("name", "") or "门板" in d.get("name", "")]
                if body_dists:
                    first_dist = body_dists[0]
                    collision_action = ActionType.VEHICLE_COLLISION
                else:
                    first_dist = self.active_roi_disturbances[0]
                    roi_id = first_dist.get("roi_id", "")
                    if "handle" in roi_id:
                        collision_action = ActionType.DOOR_HANDLE_PULL
                    elif "rack" in roi_id or "charge" in roi_id:
                        collision_action = ActionType.EQUIPMENT_THEFT
                    else:
                        collision_action = ActionType.VEHICLE_COLLISION

                # 融合整车 CAN/IMU 硬件信号 (若伴随 IMU 震动峰值直接升 L3)
                collision_threat, collision_contact, collision_desc = self.fusion_engine.arbitrate_threat(
                    target=None,
                    visual_action=collision_action,
                    visual_threat=ThreatLevel.L2_WARNING,
                    current_time=current_time
                )

                threat_rank = {ThreatLevel.L0_SAFE: 0, ThreatLevel.L1_NOTICE: 1, ThreatLevel.L2_WARNING: 2, ThreatLevel.L3_CRITICAL: 3}
                if threat_rank[collision_threat] >= threat_rank[highest_threat]:
                    highest_threat = collision_threat
                    highest_action = collision_action
                    threat_description = collision_desc
                    has_contact_confirmed = has_contact_confirmed or collision_contact

            # 极近距离 (<0.5m) 绝对兜底：任何目标进入 0.5m 内绝不可判定为 L0 安全
            threat_rank = {ThreatLevel.L0_SAFE: 0, ThreatLevel.L1_NOTICE: 1, ThreatLevel.L2_WARNING: 2, ThreatLevel.L3_CRITICAL: 3}
            for t in self.active_targets:
                if not t.is_whitelisted and t.ground_dist <= 0.5:
                    if threat_rank[highest_threat] < threat_rank[ThreatLevel.L1_NOTICE]:
                        highest_threat = ThreatLevel.L1_NOTICE
                        threat_description = f"目标在车身极近距离 (<0.5m)，进入高警戒状态"
                        target_for_threat = t

        self.current_threat_level = highest_threat
        self.latest_event_description = threat_description

        # 提取目标全生命周期画像 (类别、ID、空间距离、关键滞留时长)
        tgt_cat = target_for_threat.category if target_for_threat else "unknown"
        tgt_species = getattr(target_for_threat, 'species', None)
        tgt_species_cn = getattr(target_for_threat, 'species_cn', None)
        tgt_dist = target_for_threat.ground_dist if target_for_threat else 0.0
        tgt_dwell = target_for_threat.cumulative_dwell_time if target_for_threat else 0.0
        tgt_vr = target_for_threat.radial_speed if target_for_threat else 0.0
        tgt_zone = target_for_threat.zone.value if target_for_threat and hasattr(target_for_threat.zone, 'value') else str(target_for_threat.zone if target_for_threat else "")
        tgt_sess = getattr(target_for_threat, 'person_session_id', None)
        tgt_white = target_for_threat.is_whitelisted if target_for_threat else False
        disturbed_rois = [d['name'] if 'name' in d else str(d) for d in self.active_roi_disturbances]

        # 11. 构建事件记录
        event_record = ThreatEventRecord(
            event_id=f"EVT_{int(current_time * 1000)}",
            timestamp=current_time,
            threat_level=highest_threat,
            operating_mode=self.state_machine.operating_mode,
            action_type=highest_action,
            description=threat_description,
            target_id=target_for_threat.track_id if target_for_threat else None,
            has_contact_signal=has_contact_confirmed,
            target_category=tgt_cat,
            species=tgt_species,
            species_cn=tgt_species_cn,
            target_distance=tgt_dist,
            target_dwell_time=tgt_dwell,
            target_radial_speed=tgt_vr,
            target_zone=tgt_zone,
            person_session_id=tgt_sess,
            is_whitelisted=tgt_white,
            disturbed_rois=disturbed_rois
        )

        # 12. 慢通路异步 VLM 复核 (阶段 2.2 / 阶段 4)
        with self.profiler.profile_stage("VLM_Dispatch"):
            if self.vlm_reviewer.should_trigger_vlm(
                category_conf=target_for_threat.confidence if target_for_threat else 1.0,
                is_unknown_obstacle=False,
                fast_path_threat=highest_threat,
                action_type=highest_action,
                has_contact_signal=has_contact_confirmed,
                is_zone3_roi_event=(len(self.active_roi_disturbances) > 0)
            ):
                kframes = [primary_frame] if primary_frame is not None else []
                vlm_res = self.vlm_reviewer.review_event_async(event_record, keyframes=kframes)
                self.vlm_summaries.append(vlm_res)
                if len(self.vlm_summaries) > 200:
                    self.vlm_summaries = self.vlm_summaries[-200:]
                event_record.vlm_summary = vlm_res.get("summary_text")
                # 若 VLM 经安全沙箱仲裁调整了威胁等级 (如纯视觉 L2 降级为 L1)
                if "final_threat" in vlm_res:
                    # 权限沙箱原则：凡是有车身物理接触信号的已确认高危 (L3)，异步保底绝不可降级
                    if not (event_record.has_contact_signal and highest_threat == ThreatLevel.L3_CRITICAL and vlm_res["final_threat"] != ThreatLevel.L3_CRITICAL):
                        event_record.final_threat_level = vlm_res["final_threat"]
                        highest_threat = vlm_res["final_threat"]

        # 13 & 14. 响应执行器输出动作与存证录像锁存
        with self.profiler.profile_stage("Actuator_and_Evidence_Lock"):
            self.executed_actions = self.actuator.execute_response(
                record=event_record,
                current_time=current_time,
                is_fire_event=(highest_action == ActionType.FIRE_SMOKE)
            )
            event_record.executed_actions = list(self.executed_actions)

            if event_record.threat_level in [ThreatLevel.L2_WARNING, ThreatLevel.L3_CRITICAL] or highest_threat in [ThreatLevel.L2_WARNING, ThreatLevel.L3_CRITICAL]:
                self.state_machine.last_high_threat_time = current_time

                # 存证事件去重与防抖合并机制：
                # 持续高危期间避免在每一帧重复创建独立录像切片，而是复用/延展当前活跃录制切片
                need_new_clip = True
                if self._last_locked_event_id:
                    pending_clip = self.recorder.pending_clips.get(self._last_locked_event_id)
                    if pending_clip and not pending_clip.is_finalized:
                        need_new_clip = False
                        event_record.hash_signature = getattr(self.recorder, 'prev_hash', "")
                        if hasattr(self.recorder, 'update_clip_metadata'):
                            self.recorder.update_clip_metadata(self._last_locked_event_id, {
                                "target_dwell_time_sec": tgt_dwell,
                                "vlm_summary": event_record.vlm_summary or "",
                                "executed_actions": list(self.executed_actions)
                            })
                    elif current_time - self._last_locked_time < 15.0:
                        need_new_clip = False
                        event_record.hash_signature = getattr(self.recorder, 'prev_hash', "")

                if need_new_clip:
                    sig = self.recorder.lock_event_clip(
                        event_id=event_record.event_id,
                        threat_level_str=event_record.threat_level.value,
                        trigger_time=current_time,
                        event_record=event_record
                    )
                    self._last_locked_event_id = event_record.event_id
                    self._last_locked_time = current_time
                    event_record.hash_signature = sig
                    video_filename = f"{event_record.event_id}_{event_record.threat_level.value.replace(' ', '_')}.mp4"
                    event_record.video_clip_path = str(self.recorder.storage_dir / video_filename)
                    self.historical_threat_events.append(event_record)

        # 15. 结束当帧性能统计并生成指标快照 (同步记录当前帧是否与后台 VLM 并发)
        vlm_busy = bool(getattr(self.vlm_reviewer, "is_inferencing", False))
        frame_metrics = self.profiler.end_frame(
            threat_level=highest_threat.value,
            target_count=len(self.active_targets),
            vlm_is_inferencing=vlm_busy
        )

        return {
            "state": self.state_machine.state.value,
            "threat_level": highest_threat.value,
            "action_type": highest_action.value,
            "mode": self.state_machine.operating_mode.value,
            "description": threat_description,
            "actions": self.executed_actions,
            "active_targets_count": len(self.active_targets),
            "vlm_summary": event_record.vlm_summary,
            "vlm_is_inferencing": vlm_busy,
            "vlm_last_duration_ms": getattr(self.vlm_reviewer, "last_inference_ms", 0.0),
            "vlm_last_summary": getattr(self.vlm_reviewer, "last_summary_text", ""),
            "vlm_total_reviews": getattr(self.vlm_reviewer, "total_completed_reviews", 0),
            "metrics": frame_metrics,
            "e2e_ms": frame_metrics.e2e_ms,
            "fps": frame_metrics.fps
        }
