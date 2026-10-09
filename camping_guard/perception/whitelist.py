"""
白名单双通道判定与生命周期管理 (步骤 1.2)
通道 A (UWB 空间对齐 <= 1.2m) + 通道 B (FaceID 活体)
包含防借用与身份超时解绑机制
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import time
from camping_guard.config import GuardConfig
from camping_guard.types import TrackedEntity, FaceIDState
from camping_guard.perception.faceid import FaceIDPipeline
from camping_guard.perception.reid import PersonSessionManager

class WhitelistManager:
    def __init__(
        self,
        faceid_pipeline: Optional[FaceIDPipeline] = None,
        session_manager: Optional[PersonSessionManager] = None
    ):
        # 绑定的白名单身份记录: {track_id: {"bind_time": float, "last_verified_time": float, "auth_channel": str}}
        self.whitelisted_tracks: Dict[int, dict] = {}
        self.faceid_pipeline = faceid_pipeline
        self.session_manager = session_manager

    def evaluate_whitelist(
        self,
        targets: List[TrackedEntity],
        uwb_key_pos: Optional[Tuple[float, float]], # (X_uwb, Y_uwb) 车体坐标系下的米
        face_id_results: Optional[Dict[int, dict]], # {track_id: {"is_valid": bool, "similarity": float, "liveness": float}}
        current_time: float,
        frame: Optional[np.ndarray] = None
    ):
        """
        双通道白名单判定主入口 (增强 FaceID 机会主义识别与 ReID 全局会话状态反哺)
        """
        # 1. 清理过期与超时 track (丢失超 3.0s 强制解绑)
        active_ids = set(t.track_id for t in targets)
        stale_ids = []
        for tid, meta in self.whitelisted_tracks.items():
            if tid not in active_ids:
                if current_time - meta["last_verified_time"] > GuardConfig.WHITE_TRACK_TIMEOUT_SEC:
                    stale_ids.append(tid)
            else:
                meta["last_verified_time"] = current_time

        for tid in stale_ids:
            del self.whitelisted_tracks[tid]

        # 2. 如果存在已绑定的 track，同步状态
        for target in targets:
            if target.track_id in self.whitelisted_tracks:
                target.is_whitelisted = True

        # 3. 若 UWB 钥匙在线，尝试空间对齐 (通道 A)
        if uwb_key_pos is not None:
            kx, ky = uwb_key_pos
            candidates_in_range = []

            for target in targets:
                if target.category != 'person':
                    continue
                # 近似估算人体在车身坐标系下的坐标 (假设前方/侧方距离)
                # 使用 ground_dist 进行距离比较
                dist_to_key = abs(target.ground_dist - np.hypot(kx, ky))
                if dist_to_key <= GuardConfig.UWB_DISTANCE_THRESHOLD: # <= 1.2m
                    candidates_in_range.append(target)

            # 多人同处处理: 1.2m 内有多个候选人时，不做 UWB 单独判定，需结合通道 B
            if len(candidates_in_range) == 1:
                target = candidates_in_range[0]
                target.is_whitelisted = True
                self.whitelisted_tracks[target.track_id] = {
                    "bind_time": current_time,
                    "last_verified_time": current_time,
                    "auth_channel": "UWB_CHANNEL_A"
                }
            elif len(candidates_in_range) > 1:
                # 多人混行，降级到通道 B
                for target in candidates_in_range:
                    if face_id_results and target.track_id in face_id_results:
                        finfo = face_id_results[target.track_id]
                        if finfo.get("liveness", 0.0) >= 0.90 and finfo.get("similarity", 0.0) >= 0.78:
                            target.is_whitelisted = True
                            self.whitelisted_tracks[target.track_id] = {
                                "bind_time": current_time,
                                "last_verified_time": current_time,
                                "auth_channel": "FACE_ID_CHANNEL_B"
                            }

        # 4. 单独检测通道 B (人脸识别已录入且距车身 <= 2m)
        if face_id_results:
            for target in targets:
                if target.category == 'person' and not target.is_whitelisted:
                    if target.ground_dist <= 2.0 and target.track_id in face_id_results:
                        finfo = face_id_results[target.track_id]
                        if finfo.get("liveness", 0.0) >= 0.90 and finfo.get("similarity", 0.0) >= 0.78:
                            target.is_whitelisted = True
                            self.whitelisted_tracks[target.track_id] = {
                                "bind_time": current_time,
                                "last_verified_time": current_time,
                                "auth_channel": "FACE_ID_CHANNEL_B"
                            }

        # 5. 基于深度模型的 FaceID 机会主义动态评估与多帧确权
        if self.faceid_pipeline and frame is not None:
            for target in targets:
                if target.category == 'person' and not target.is_whitelisted:
                    if target.ground_dist <= 4.0: # 4米以内允许尝试识别人脸
                        state = self.faceid_pipeline.process_target_face(frame, target, current_time)
                        if state == FaceIDState.VERIFIED:
                            target.is_whitelisted = True
                            self.whitelisted_tracks[target.track_id] = {
                                "bind_time": current_time,
                                "last_verified_time": current_time,
                                "auth_channel": "FACE_ID_DEEP_MODEL"
                            }
                            if self.session_manager and target.track_uid:
                                self.session_manager.mark_session_whitelisted(
                                    target.track_uid, target.face_id_name or "OWNER"
                                )

        # 6. ReID 全局会话状态反哺 (软白名单继承与防反复打扰)
        if self.session_manager:
            for target in targets:
                if target.category == 'person' and not target.is_whitelisted:
                    if target.person_session_id and target.person_session_id in self.session_manager.sessions:
                        sess = self.session_manager.sessions[target.person_session_id]
                        if sess.is_whitelisted:
                            target.is_whitelisted = True
                            target.face_state = FaceIDState.VERIFIED
                            target.face_id_name = sess.whitelist_name
                            self.whitelisted_tracks[target.track_id] = {
                                "bind_time": current_time,
                                "last_verified_time": current_time,
                                "auth_channel": "REID_SESSION_INHERIT"
                            }
