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
        animal_classifier: Optional[Any] = None
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

        self.bytetracker = ByteTracker() if use_bytetrack else None
        self.detector = detector
        self.stream_manager = stream_manager
        self.pose_estimator = pose_estimator or RTMPoseEstimator(use_trt=self.use_tensorrt)
        self.pose_classifier = PoseActionClassifier()
        self.fusion_engine = MultiSourceFusionEngine()
        self.fire_monitor = CampfireMonitor()
        self.vlm_reviewer = AsyncVLMReviewer()
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

        # 注册模型元数据至 profiler (保持 key 与执行期 profile_model("...") 完全一致)
        if self.detector and hasattr(self.detector, 'model_path'):
            det_label = "MegaDetector" if getattr(self.detector, 'is_megadetector', False) else "YOLO11n-Det"
            self.profiler.register_model_metadata(det_label, self.detector.model_path, "S1全图目标检测初筛 (640x480)")
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
            if (raw_detections is None or len(raw_detections) == 0) and self.detector is not None:
                if primary_frame is not None:
                    det_sess = getattr(self.detector, 'session', None)
                    with self.profiler.profile_model("YOLO11n-Det", condition="S1全图目标检测 (640x480)", session=det_sess):
                        raw_detections = self.detector.detect(primary_frame)
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

        active_ids = set(t.track_id for t in self.active_targets)
        self.distance_estimator.clean_stale_tracks(active_ids)

        # 7. 白名单双通道排查 (步骤 1.2, 结合 UWB, FaceID 深度模型与 ReID 全局会话)
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

        # 火情研判通道
        fire_targets = [t for t in self.active_targets if t.category in ['fire', 'smoke']]
        person_targets = [t for t in self.active_targets if t.category == 'person']
        if fire_targets:
            fire_threat, fire_desc = self.fire_monitor.update_fire_status(fire_targets, person_targets, current_time)
            if fire_threat.value > highest_threat.value:
                highest_threat = fire_threat
                highest_action = ActionType.FIRE_SMOKE
                threat_description = fire_desc

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
            event_record.vlm_summary = vlm_res.get("summary_text")
            # 若 VLM 经安全沙箱仲裁调整了威胁等级 (如纯视觉 L2 降级为 L1)
            if "final_threat" in vlm_res:
                event_record.final_threat_level = vlm_res["final_threat"]
                highest_threat = vlm_res["final_threat"]

        # 13. 响应执行器输出动作 (阶段 3)
        self.executed_actions = self.actuator.execute_response(
            record=event_record,
            current_time=current_time,
            is_fire_event=(highest_action == ActionType.FIRE_SMOKE)
        )
        event_record.executed_actions = list(self.executed_actions)

        # 14. 存证与录像锁存 (阶段 4)
        if event_record.threat_level in [ThreatLevel.L2_WARNING, ThreatLevel.L3_CRITICAL] or highest_threat in [ThreatLevel.L2_WARNING, ThreatLevel.L3_CRITICAL]:
            self.state_machine.last_high_threat_time = current_time

            # 存证事件去重与防抖合并机制：
            # 持续高危期间避免在每一帧重复创建独立录像切片，而是复用/延展当前活跃录制切片
            need_new_clip = True
            if self._last_locked_event_id:
                pending_clip = self.recorder.pending_clips.get(self._last_locked_event_id)
                if pending_clip and not pending_clip.is_finalized:
                    # 保持以首个触发报警事件 A 为原点的标准 30s 视窗 (事前 10s + 事后 20s)，
                    # 在事后 20 秒窗口期内的新伴随事件均被该 30s 黄金证据链完整覆盖，避免切片被不可控拉长
                    need_new_clip = False
                    event_record.hash_signature = getattr(self.recorder, 'prev_hash', "")
                    # 持续动态更新当前活跃切片的最新语义 (更新驻留时长、动作与 VLM 战报)
                    if hasattr(self.recorder, 'update_clip_metadata'):
                        self.recorder.update_clip_metadata(self._last_locked_event_id, {
                            "target_dwell_time_sec": tgt_dwell,
                            "vlm_summary": event_record.vlm_summary or "",
                            "executed_actions": list(self.executed_actions)
                        })
                elif current_time - self._last_locked_time < 15.0:
                    # 距离上次事件录像完成不足 15 秒冷却窗口，跳过重复创建
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

        # 15. 结束当帧性能统计并生成指标快照
        frame_metrics = self.profiler.end_frame(
            threat_level=highest_threat.value,
            target_count=len(self.active_targets)
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
            "metrics": frame_metrics,
            "e2e_ms": frame_metrics.e2e_ms,
            "fps": frame_metrics.fps
        }
