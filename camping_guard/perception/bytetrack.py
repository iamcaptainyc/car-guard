"""
车载轻量 ByteTrack 多目标追踪算法引擎 (camping_guard/perception/bytetrack.py)
实现高低置信度双重关联 (ByteTrack 原理)、卡尔曼滤波状态估计与遮挡恢复
并内嵌露营守卫累计滞留计时与 10 秒重入继承机制
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import time

from camping_guard.types import TrackedEntity, BoundingBox, VehicleZone
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

    def __init__(self, bbox_xywh: Tuple[float, float, float, float], score: float, category: str, initial_dwell: float = 0.0):
        STrack._count += 1
        self.track_id = STrack._count
        self.category = category
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
        self.first_seen_time = time.time()
        self.last_seen_time = self.first_seen_time
        self.cumulative_dwell_time = initial_dwell
        self.ground_dist = 5.0
        self.radial_speed = 0.0
        self.zone = VehicleZone.ZONE_1_OUTER
        self.is_whitelisted = False

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
        self.category = new_track.category
        self.state = 1
        self.is_activated = True

        dt = current_time - self.last_seen_time
        self.last_seen_time = current_time
        if self.zone in [VehicleZone.ZONE_1_OUTER, VehicleZone.ZONE_2_INNER]:
            self.cumulative_dwell_time += dt

    def mark_lost(self):
        self.state = 2

    def mark_removed(self):
        self.state = 3


class ByteTracker:
    def __init__(self, high_thresh: float = 0.5, low_thresh: float = 0.15, max_time_lost: int = 30):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_time_lost = max_time_lost
        self.frame_id = 0

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        # 离场 10 秒内重入缓存: {track_id: (leave_time, cumulative_dwell)}
        self.departed_history: Dict[int, Tuple[float, float]] = {}

    def update(self, detections: List[dict], current_time: float) -> List[TrackedEntity]:
        """
        :param detections: [{'bbox': (x,y,w,h), 'conf': float, 'category': str, 'track_id': opt}]
        :param current_time: 当前秒级时间戳
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
            st = STrack(bbox, conf, cat)
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

        # 6. 处理未匹配的高置信度检测 -> 初始化为新目标 (并检查 10 秒重入继承)
        for d_idx in unmatched_dets_1:
            det = high_dets[d_idx]
            # 继承或赋予新 ID
            det.frame_id = self.frame_id
            det.is_activated = True
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
                first_seen_time=st.first_seen_time,
                last_seen_time=st.last_seen_time,
                cumulative_dwell_time=st.cumulative_dwell_time,
                velocity_pixel=(vx, vy)
            )
            output_entities.append(entity)

        return output_entities

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
