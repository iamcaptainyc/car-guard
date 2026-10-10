"""
目标追踪与累计滞留计时器 (第四章、步骤 1.1)
实现目标短期离开 (<=10s) 重新进入后的滞留时间累计，防止来回试探绕过阈值
"""

from typing import Dict, List, Optional, Any
import time
import numpy as np
from camping_guard.types import TrackedEntity, BoundingBox, VehicleZone, FaceIDState
from camping_guard.config import GuardConfig
from camping_guard.perception.reid import OSNetReIDExtractor, PersonSessionManager

class TargetTrackManager:
    def __init__(
        self,
        reid_extractor: Optional[OSNetReIDExtractor] = None,
        session_manager: Optional[PersonSessionManager] = None
    ):
        # 活跃目标字典: {track_id: TrackedEntity}
        self.active_tracks: Dict[int, TrackedEntity] = {}
        # 历史离场目标缓存: {track_id: {"leave_time": float, "cumulative_dwell": float, "zone": VehicleZone}}
        self.departed_tracks: Dict[int, dict] = {}
        self.reid_extractor = reid_extractor
        self.session_manager = session_manager
        self.profiler: Optional[Any] = None

    def update_tracks(
        self,
        raw_detections: List[dict],
        current_time: float,
        frame: Optional[np.ndarray] = None,
        camera_id: int = 0
    ) -> List[TrackedEntity]:
        """
        更新目标跟踪与滞留时间 (结合 ReID 与全局人员会话)
        :param raw_detections: 检测输入列表 [{'track_id': int, 'category': str, 'conf': float, 'bbox': BoundingBox, 'dist': float, 'radial_speed': float, 'zone': VehicleZone}]
        :param current_time: 当前时钟戳
        :param frame: 可选当前全景视频帧 (用于提取 ReID 特征)
        :param camera_id: 当前机位编号
        """
        current_seen_ids = set()
        updated_entities = []

        for det in raw_detections:
            tid = det['track_id']
            current_seen_ids.add(tid)

            if tid in self.active_tracks:
                # 目标持续在场，累加时长
                entity = self.active_tracks[tid]
                dt = current_time - entity.last_seen_time
                if dt <= 0:
                    dt = 1.0 / 25.0
                entity.last_seen_time = current_time
                entity.bbox = det['bbox']
                entity.ground_dist = det['dist']
                entity.radial_speed = det['radial_speed']
                entity.zone = det['zone']
                entity.confidence = det['conf']
                entity.cumulative_dwell_time += dt

                # 全局会话时长同步
                if self.session_manager and entity.track_uid:
                    self.session_manager.update_dwell_time(entity.track_uid, dt)
                    if entity.person_session_id and entity.person_session_id in self.session_manager.sessions:
                        sess = self.session_manager.sessions[entity.person_session_id]
                        entity.cumulative_dwell_time = max(entity.cumulative_dwell_time, sess.cumulative_dwell_time)
                        if sess.is_whitelisted and not entity.is_whitelisted:
                            entity.is_whitelisted = True
                            entity.face_state = FaceIDState.VERIFIED
                            entity.face_id_name = sess.whitelist_name
            else:
                # 新检测目标，检查是否为 10 秒内重入防区的目标
                initial_dwell = 0.0
                if tid in self.departed_tracks:
                    dep_info = self.departed_tracks[tid]
                    if current_time - dep_info["leave_time"] <= GuardConfig.DWELL_RESET_WINDOW_SEC:
                        # 10s 内重入，恢复既有累计滞留时间
                        initial_dwell = dep_info["cumulative_dwell"]
                    del self.departed_tracks[tid]

                track_uid = f"cam{camera_id}_trk{tid}"
                entity = TrackedEntity(
                    track_id=tid,
                    category=det['category'],
                    confidence=det['conf'],
                    bbox=det['bbox'],
                    ground_dist=det['dist'],
                    radial_speed=det['radial_speed'],
                    zone=det['zone'],
                    first_seen_time=current_time,
                    last_seen_time=current_time,
                    cumulative_dwell_time=initial_dwell,
                    track_uid=track_uid
                )

                # 如果是人员目标且启用了 ReID，执行特征提取与全局人员会话关联
                if det['category'] == 'person' and self.reid_extractor and frame is not None:
                    if self.profiler:
                        reid_s = getattr(self.reid_extractor, 'session', None)
                        with self.profiler.profile_model("OSNet-ReID", condition=f"ID:{tid} 人员ReID特征建档", session=reid_s):
                            feat = self.reid_extractor.extract_from_frame(frame, entity.bbox)
                    else:
                        feat = self.reid_extractor.extract_from_frame(frame, entity.bbox)
                    entity.reid_feature = feat
                    if self.session_manager:
                        session, is_new = self.session_manager.match_or_create_session(
                            track_uid=track_uid,
                            feature=feat,
                            current_time=current_time,
                            quality_score=entity.confidence,
                            camera_id=camera_id
                        )
                        entity.person_session_id = session.session_id
                        if not is_new:
                            # 继承跨机位或遮挡重入的历史累计滞留时间 (徘徊判定核心)
                            entity.cumulative_dwell_time = max(entity.cumulative_dwell_time, session.cumulative_dwell_time)
                        if session.is_whitelisted:
                            entity.is_whitelisted = True
                            entity.face_state = FaceIDState.VERIFIED
                            entity.face_id_name = session.whitelist_name

                self.active_tracks[tid] = entity

            updated_entities.append(self.active_tracks[tid])

        # 检查离场目标并归档
        disappeared_ids = [tid for tid in self.active_tracks if tid not in current_seen_ids]
        for tid in disappeared_ids:
            entity = self.active_tracks[tid]
            self.departed_tracks[tid] = {
                "leave_time": current_time,
                "cumulative_dwell": entity.cumulative_dwell_time,
                "zone": entity.zone
            }
            del self.active_tracks[tid]

        # 清理超过 10 秒的已离场目标缓存
        stale_departed = [
            tid for tid, info in self.departed_tracks.items()
            if current_time - info["leave_time"] > GuardConfig.DWELL_RESET_WINDOW_SEC
        ]
        for tid in stale_departed:
            del self.departed_tracks[tid]

        return updated_entities
