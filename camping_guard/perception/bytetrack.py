"""
车载轻量 ByteTrack 多目标追踪算法引擎 (camping_guard/perception/bytetrack.py)
实现高低置信度双重关联 (ByteTrack 原理)、卡尔曼滤波状态估计与遮挡恢复
并内嵌露营守卫累计滞留计时与 10 秒重入继承机制
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import time

from camping_guard.types import TrackedEntity, BoundingBox, VehicleZone, FaceIDState
from camping_guard.config import GuardConfig

class KalmanFilter2D:
    """针对 2D 目标框的轻量卡尔曼滤波器 [cx, cy, a, h, vcx, vcy, va, vh]"""
    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """measurement: [cx, cy, aspect_ratio, height]"""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = np.linalg.cholesky(projected_cov), True
        kalman_gain = np.linalg.multi_dot((covariance, self._update_mat.T, np.linalg.inv(projected_cov)))
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(kalman_gain, innovation)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


def calc_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """box: (x, y, w, h)"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea if unionArea > 0 else 0.0


class STrack:
    """单条目标轨迹状态"""
    _count = 0

    def __init__(self, bbox_xywh: Tuple[float, float, float, float], score: float, category: str, initial_dwell: float = 0.0, current_time: Optional[float] = None):
        STrack._count += 1
        self.track_id = STrack._count
        self.category = category
        self.confirmed_large_beast: bool = (category == 'large_beast')
        self.score = score
        self.is_activated = False
        self.state = 1 # 1: Tracked, 2: Lost, 3: Removed

        # 转换为 [cx, cy, aspect_ratio, h]
        x, y, w, h = bbox_xywh
        self.mean, self.covariance = None, None
        self.initial_measurement = np.array([x + w / 2.0, y + h / 2.0, w / float(h if h > 0 else 1), h])

        self.kalman = KalmanFilter2D()
        self.mean, self.covariance = self.kalman.initiate(self.initial_measurement)

        self.frame_id = 0
        self.tracklet_len = 0
        now_t = current_time if current_time is not None else time.time()
        self.first_seen_time = now_t
        self.last_seen_time = now_t
        self.cumulative_dwell_time = initial_dwell
        self.ground_dist = 5.0
        self.radial_speed = 0.0
        self.zone = VehicleZone.ZONE_1_OUTER
        self.is_whitelisted = False
        self.track_uid: str = ""
        self.person_session_id: Optional[str] = None
        self.face_state: FaceIDState = FaceIDState.PENDING
        self.face_id_name: Optional[str] = None
        self.reid_feature: Optional[Any] = None
        self.keypoints: Optional[Dict[str, Tuple[float, float]]] = None
        self.association_version: int = 1
        self.species: Optional[str] = None
        self.species_cn: Optional[str] = None
        self.last_animal_cls_time: float = 0.0
        self.animal_cls_count: int = 0
        self.animal_cls_locked: bool = False

    @property
    def tlwh(self) -> Tuple[float, float, float, float]:
        """返回 [top-left-x, top-left-y, width, height]"""
        if self.mean is None:
            return (0, 0, 0, 0)
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return (float(ret[0]), float(ret[1]), float(ret[2]), float(ret[3]))

    def predict(self):
        if self.state != 1:
            self.mean[7] = 0
        self.mean, self.covariance = self.kalman.predict(self.mean, self.covariance)

    def update(self, new_track: 'STrack', frame_id: int, current_time: float):
        self.frame_id = frame_id
        self.tracklet_len += 1
        measurement = new_track.initial_measurement
        self.mean, self.covariance = self.kalman.update(self.mean, self.covariance, measurement)
        self.score = new_track.score
        # 类别更新与猛兽防抖锁存 (Hysteresis Latching)
        if new_track.category == 'large_beast':
            self.confirmed_large_beast = True
            self.category = 'large_beast'
            if getattr(new_track, 'species', None):
                self.species = new_track.species
                if self.species not in ['unknown', 'unclassified_animal']:
                    self.animal_cls_locked = True
            if getattr(new_track, 'species_cn', None):
                self.species_cn = new_track.species_cn
        elif self.confirmed_large_beast:
            # 轨迹已确认为大型猛兽：若因目标背身/遮挡/光照退化为通用 animal 或 small_animal，
            # 自动保持高危猛兽锁存与已知物种，防止威胁等级与 UI 标注跳变闪烁
            if new_track.category in ['animal', 'small_animal']:
                self.category = 'large_beast'
                sp_en = getattr(new_track, 'species', None)
                if sp_en and sp_en not in ['unknown', 'unclassified_animal']:
                    self.species = sp_en
                sp_cn = getattr(new_track, 'species_cn', None)
                if sp_cn and sp_cn not in ['未知动物', '通用动物']:
                    self.species_cn = sp_cn
            else:
                # 若检测确凿变为人员或车辆等完全不相干物理类别，解除猛兽锁存
                self.confirmed_large_beast = False
                self.animal_cls_locked = False
                self.category = new_track.category
                self.species = getattr(new_track, 'species', None)
                self.species_cn = getattr(new_track, 'species_cn', None)
        else:
            if new_track.category == 'animal' and self.category == 'small_animal' and self.species:
                # 保持已建档的二级细分类别与物种，避免被未分类的原始 'animal' 覆盖
                pass
            else:
                self.category = new_track.category
            if getattr(new_track, 'species', None):
                self.species = new_track.species
            if getattr(new_track, 'species_cn', None):
                self.species_cn = new_track.species_cn
        self.state = 1
        self.is_activated = True

        dt = current_time - self.last_seen_time
        if dt <= 0 or dt > 1.0:
            dt = 1.0 / 25.0
        self.last_seen_time = current_time
        self.cumulative_dwell_time += dt

    def mark_lost(self):
        self.state = 2

    def mark_removed(self):
        self.state = 3


class ByteTracker:
    def __init__(self, high_thresh: float = 0.5, low_thresh: float = 0.15, max_time_lost: int = 60):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_time_lost = max_time_lost
        self.frame_id = 0

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        # 离场 10 秒内重入缓存: {track_id: (leave_time, cumulative_dwell)}
        self.departed_history: Dict[int, Tuple[float, float]] = {}

    def update(
        self,
        detections: List[dict],
        current_time: float,
        frame: Optional[np.ndarray] = None,
        reid_extractor: Optional[Any] = None
    ) -> List[TrackedEntity]:
        """
        :param detections: [{'bbox': (x,y,w,h), 'conf': float, 'category': str, 'track_id': opt}]
        :param current_time: 当前秒级时间戳
        :param frame: 原始全景帧 (可选，用于遮挡重识别外观特征提取)
        :param reid_extractor: ReID 特征提取器 (可选，用于外观特征召回)
        :return: 活跃的 TrackedEntity 实体列表
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # 1. 区分高低置信度检测候选
        high_dets = []
        low_dets = []
        for d in detections:
            conf = d.get('conf', 0.5)
            cat = d.get('category', 'person')
            bbox = tuple(d['bbox'])
            st = STrack(bbox, conf, cat, current_time=current_time)
            st.species = d.get('species')
            st.species_cn = d.get('species_cn')
            if cat == 'large_beast' and st.species and st.species not in ['unknown', 'unclassified_animal']:
                st.animal_cls_locked = True
            if conf >= self.high_thresh:
                high_dets.append(st)
            elif conf >= self.low_thresh:
                low_dets.append(st)

        # 2. 对当前追踪目标进行卡尔曼滤波预测
        all_pool = self.tracked_stracks + self.lost_stracks
        for st in all_pool:
            st.predict()

        # 3. 第一次关联：高置信度检测与当前目标池关联
        unmatched_tracks_1, unmatched_dets_1, matches_1 = self._linear_assignment(all_pool, high_dets, iou_thresh=0.7)

        for t_idx, d_idx in matches_1:
            track = all_pool[t_idx]
            det = high_dets[d_idx]
            if track.state == 1:
                track.update(det, self.frame_id, current_time)
                activated_stracks.append(track)
            else:
                track.update(det, self.frame_id, current_time)
                refind_stracks.append(track)

        # 4. 第二次关联 (ByteTrack 核心)：剩余未匹配轨迹与低置信度检测关联 (应对近车遮挡与模糊)
        remaining_tracked = [all_pool[i] for i in unmatched_tracks_1 if all_pool[i].state == 1]
        unmatched_tracks_2, _, matches_2 = self._linear_assignment(remaining_tracked, low_dets, iou_thresh=0.5)

        for t_idx, d_idx in matches_2:
            track = remaining_tracked[t_idx]
            det = low_dets[d_idx]
            track.update(det, self.frame_id, current_time)
            activated_stracks.append(track)

        # 5. 处理剩余未匹配轨迹 -> 标为 Lost
        for t_idx in unmatched_tracks_2:
            track = remaining_tracked[t_idx]
            track.mark_lost()
            lost_stracks.append(track)

        # 5.5 (Deep-ByteTrack 核心): 针对未通过 IoU 匹配的高置信度人员检测，通过 ReID 外观特征从 lost_stracks 中精准召回
        matched_reid_dets = set()
        if frame is not None and reid_extractor is not None and len(unmatched_dets_1) > 0 and len(self.lost_stracks) > 0:
            lost_candidates = [t for t in self.lost_stracks if t.category == 'person' and t.reid_feature is not None]
            if len(lost_candidates) > 0:
                for d_idx in unmatched_dets_1:
                    det = high_dets[d_idx]
                    if det.category != 'person':
                        continue
                    bx, by, bw, bh = det.tlwh
                    crop_bbox = BoundingBox(int(bx), int(by), int(bw), int(bh))
                    feat = reid_extractor.extract_from_frame(frame, crop_bbox)
                    if feat is None or len(feat) == 0:
                        continue
                    det.reid_feature = feat

                    best_sim = -1.0
                    best_lost_trk = None
                    for lost_trk in lost_candidates:
                        sim = float(np.dot(feat, lost_trk.reid_feature))
                        if sim > best_sim:
                            best_sim = sim
                            best_lost_trk = lost_trk

                    # 跨遮挡外观相似度门禁 (>= 0.65 判定为同一目标被遮挡重现)
                    if best_lost_trk is not None and best_sim >= 0.65:
                        # 核心修复: 目标被遮挡重现后，位置可能发生显著空间跳跃，
                        # 必须将卡尔曼滤波器重置/对齐至当前最新检测框测量值，
                        # 杜绝 Kalman 平滑滤波导致的目标框滞后、悬空漂移与关键点/画面延迟不匹配
                        best_lost_trk.mean, best_lost_trk.covariance = best_lost_trk.kalman.initiate(det.initial_measurement)
                        best_lost_trk.score = det.score
                        best_lost_trk.category = det.category
                        best_lost_trk.state = 1
                        best_lost_trk.is_activated = True
                        best_lost_trk.frame_id = self.frame_id
                        best_lost_trk.last_seen_time = current_time
                        best_lost_trk.reid_feature = feat
                        refind_stracks.append(best_lost_trk)
                        lost_candidates.remove(best_lost_trk)
                        matched_reid_dets.add(d_idx)

        # 6. 处理剩余未匹配的高置信度检测 -> 初始化为新目标
        unmatched_dets_final = [i for i in unmatched_dets_1 if i not in matched_reid_dets]
        for d_idx in unmatched_dets_final:
            det = high_dets[d_idx]
            # 继承或赋予新 ID
            det.frame_id = self.frame_id
            det.is_activated = True
            det.first_seen_time = current_time
            det.last_seen_time = current_time
            activated_stracks.append(det)

        # 7. 更新内部轨迹集合与清理超时 Lost 轨迹
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 1] + activated_stracks + refind_stracks
        # 去重
        self.tracked_stracks = list({t.track_id: t for t in self.tracked_stracks}.values())

        for t in self.lost_stracks:
            if self.frame_id - t.frame_id > self.max_time_lost:
                t.mark_removed()
                removed_stracks.append(t)
                # 记录离场时间用于 10 秒内重入累计
                self.departed_history[t.track_id] = (current_time, t.cumulative_dwell_time)
            else:
                lost_stracks.append(t)

        self.lost_stracks = list({t.track_id: t for t in lost_stracks if t.state == 2}.values())
        self.removed_stracks.extend(removed_stracks)

        # 清理超过 10 秒的 departed 缓存
        stale_dep = [tid for tid, info in self.departed_history.items() if current_time - info[0] > GuardConfig.DWELL_RESET_WINDOW_SEC]
        for tid in stale_dep:
            del self.departed_history[tid]

        # 8. 构造输出活跃目标实体
        output_entities = []
        for st in self.tracked_stracks:
            tx, ty, tw, th = st.tlwh
            # 约束数值在正常区间
            tx, ty = max(0, int(tx)), max(0, int(ty))
            tw, th = max(10, int(tw)), max(10, int(th))

            vx = float(st.mean[4]) if st.mean is not None and len(st.mean) > 4 else 0.0
            vy = float(st.mean[5]) if st.mean is not None and len(st.mean) > 5 else 0.0

            entity = TrackedEntity(
                track_id=st.track_id,
                category=st.category,
                confidence=st.score,
                bbox=BoundingBox(tx, ty, tw, th),
                ground_dist=st.ground_dist,
                radial_speed=st.radial_speed,
                zone=st.zone,
                is_whitelisted=st.is_whitelisted,
                first_seen_time=st.first_seen_time,
                last_seen_time=st.last_seen_time,
                cumulative_dwell_time=st.cumulative_dwell_time,
                keypoints=st.keypoints,
                track_uid=st.track_uid or f"cam1_trk{st.track_id}",
                person_session_id=st.person_session_id,
                face_state=st.face_state,
                face_id_name=st.face_id_name,
                reid_feature=st.reid_feature,
                association_version=st.association_version,
                velocity_pixel=(vx, vy),
                species=getattr(st, 'species', None),
                species_cn=getattr(st, 'species_cn', None)
            )
            output_entities.append(entity)

        return output_entities

    def get_track(self, track_id: int) -> Optional[STrack]:
        """获取底层 STrack 轨迹对象，用于业务层状态双向同步与结果持久化"""
        for st in self.tracked_stracks:
            if st.track_id == track_id:
                return st
        for st in self.lost_stracks:
            if st.track_id == track_id:
                return st
        return None

    def reassign_track_id(self, current_id: int, target_id: int):
        """
        跨遮挡/长时时空穿透 ReID 关联重识别: 将新分配的临时轨迹 ID 缝合回历史根轨迹 ID
        """
        if current_id == target_id:
            return

        # 1. 清理 lost_stracks 或 removed_stracks 中残留的原 target_id 孤儿轨迹
        self.lost_stracks = [s for s in self.lost_stracks if s.track_id != target_id]
        self.removed_stracks = [s for s in self.removed_stracks if s.track_id != target_id]

        # 2. 将当前活跃轨迹的 ID 变更回 target_id
        for st in self.tracked_stracks:
            if st.track_id == current_id:
                st.track_id = target_id
                st.track_uid = f"cam1_trk{target_id}"
                break

    def _linear_assignment(self, tracks: List[STrack], dets: List[STrack], iou_thresh: float) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
        """贪心/IoU 快速关联算法"""
        if len(tracks) == 0 or len(dets) == 0:
            return list(range(len(tracks))), list(range(len(dets))), []

        matches = []
        matched_tracks = set()
        matched_dets = set()

        # 计算两两 IoU
        iou_matrix = np.zeros((len(tracks), len(dets)))
        for i, trk in enumerate(tracks):
            boxA = trk.tlwh
            for j, det in enumerate(dets):
                boxB = det.tlwh
                iou_matrix[i, j] = calc_iou(boxA, boxB)

        # 贪心匹配最大 IoU
        while True:
            max_iou = np.max(iou_matrix) if iou_matrix.size > 0 else 0
            if max_iou < (1.0 - iou_thresh):
                break
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            matches.append((i, j))
            matched_tracks.add(i)
            matched_dets.add(j)
            iou_matrix[i, :] = -1
            iou_matrix[:, j] = -1

        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_dets = [j for j in range(len(dets)) if j not in matched_dets]
        return unmatched_tracks, unmatched_dets, matches
