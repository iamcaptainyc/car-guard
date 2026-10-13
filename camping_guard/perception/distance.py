"""
单目鱼眼接地测距、迟滞防抖与径向速度计算 (第二、四、六章)
实现透视变换/几何接地测距、±0.3m 迟滞区间状态机与速度矢量分解
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
from camping_guard.types import VehicleZone, CameraCalib
from camping_guard.config import GuardConfig


class FisheyeProjector:
    """标准车载鱼眼相机的 3D IPM 几何投影与逆透视解算器"""
    def __init__(
        self,
        w: int,
        h: int,
        yaw_deg: float,
        pitch_deg: float,
        roll_deg: float,
        fov_deg: float,
        cam_pos: Tuple[float, float, float],
        v_len: float = 4.98,
        v_wid: float = 1.98,
        x_span: float = 6.5
    ):
        self.w = w
        self.h = h
        self.cx = w / 2.0
        self.cy = h / 2.0
        self.yaw_deg = yaw_deg
        self.pitch_deg = pitch_deg
        self.roll_deg = roll_deg
        self.fov_deg = fov_deg
        self.cam_pos = list(cam_pos)
        self.half_l = v_len / 2.0
        self.half_w = v_wid / 2.0
        self.x_span = x_span
        self.update_matrices()

    def update_params(self, yaw_deg: float, pitch_deg: float, roll_deg: float, fov_deg: float, cam_z: float, x_span: Optional[float] = None):
        self.yaw_deg = yaw_deg
        self.pitch_deg = pitch_deg
        self.roll_deg = roll_deg
        self.fov_deg = fov_deg
        self.cam_pos[2] = cam_z
        if x_span is not None:
            self.x_span = x_span
        self.update_matrices()

    def update_matrices(self):
        yaw = np.radians(self.yaw_deg)
        pitch = np.radians(self.pitch_deg)
        roll = np.radians(self.roll_deg)
        self.fov_rad = np.radians(self.fov_deg)

        # 1. 光轴 Z_cam
        self.f = np.array([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch)
        ], dtype=np.float64)

        # 2. 相机右轴 X_cam
        r0 = np.array([np.sin(yaw), -np.cos(yaw), 0.0], dtype=np.float64)

        # 3. 相机下轴 Y_cam
        d0 = np.array([
            np.sin(pitch) * np.cos(yaw),
            np.sin(pitch) * np.sin(yaw),
            -np.cos(pitch)
        ], dtype=np.float64)

        # 4. Roll 横滚角变换
        self.r = r0 * np.cos(roll) + d0 * np.sin(roll)
        self.d = -r0 * np.sin(roll) + d0 * np.cos(roll)

        # 5. 等距鱼眼焦距模型 (r = f * theta)
        self.f_px = (self.h * 0.48) / np.radians(self.fov_deg / 2.0 * 0.65)

    def ground_to_pixel(self, xg: float, yg: float, zg: float = 0.0) -> Optional[Tuple[int, int]]:
        """三维物理点 (X, Y, Z) 投影到鱼眼画面像素 (u, v)"""
        v = np.array([xg - self.cam_pos[0], yg - self.cam_pos[1], zg - self.cam_pos[2]], dtype=np.float64)
        xc = float(np.dot(v, self.r))
        yc = float(np.dot(v, self.d))
        zc = float(np.dot(v, self.f))
        r3d = np.hypot(xc, yc)
        theta = np.arctan2(r3d, zc)
        max_theta = max(self.fov_rad / 2.0, np.radians(115.0))
        if theta > max_theta:
            return None
        phi = np.arctan2(yc, xc)
        rn = self.f_px * theta
        u = self.cx + rn * np.cos(phi)
        v_px = self.cy + rn * np.sin(phi)
        return int(round(u)), int(round(v_px))

    def pixel_to_ground(self, u: float, v: float) -> Tuple[Optional[float], Optional[float], float]:
        """逆透视 IPM：将画面像素 (u, v) 反算为物理地面 (Xg, Yg) 与距车身净距"""
        dx = u - self.cx
        dy = v - self.cy
        rn = np.hypot(dx, dy)
        theta = rn / self.f_px
        if theta >= self.fov_rad / 2.0:
            return None, None, 15.0
        phi = np.arctan2(dy, dx)

        xc = np.sin(theta) * np.cos(phi)
        yc = np.sin(theta) * np.sin(phi)
        zc = np.cos(theta)

        d_veh = xc * self.r + yc * self.d + zc * self.f
        if d_veh[2] >= -0.01:
            return None, None, 15.0

        t = -self.cam_pos[2] / d_veh[2]
        xg = self.cam_pos[0] + t * d_veh[0]
        yg = self.cam_pos[1] + t * d_veh[1]

        dist_x = max(0.0, abs(xg) - self.half_l)
        dist_y = max(0.0, abs(yg) - self.half_w)
        dist = np.hypot(dist_x, dist_y)
        return float(xg), float(yg), float(dist)


class GroundDistanceEstimator:
    def __init__(
        self,
        camera_height: float = 1.2,
        camera_pitch: float = -15.0,
        cameras_calib: Optional[Dict[int, CameraCalib]] = None,
        vehicle_length_m: float = 4.98,
        vehicle_width_m: float = 1.98
    ):
        """
        :param camera_height: 默认相机离地安装高度 (米)
        :param camera_pitch: 默认俯仰角 (度, 负数为向下看)
        :param cameras_calib: 从 vehicle_calib.json 解析的多相机外参字典 {cam_id: CameraCalib}
        :param vehicle_length_m: 车辆物理长度 (米)
        :param vehicle_width_m: 车辆物理宽度 (米)
        """
        self.camera_height = camera_height
        self.camera_pitch_rad = np.radians(camera_pitch)
        self.cameras_calib: Dict[int, CameraCalib] = cameras_calib or {}
        self.vehicle_length_m = vehicle_length_m
        self.vehicle_width_m = vehicle_width_m
        self.half_l = vehicle_length_m / 2.0
        self.half_w = vehicle_width_m / 2.0

        # 3D 鱼眼投影器缓存池 {cam_id: FisheyeProjector}
        self.projectors: Dict[int, FisheyeProjector] = {}
        self._init_projectors()

        # 历史目标距离缓存，用于径向速度滤波: {track_id: [(time, dist), ...]}
        self.history_records: Dict[int, list] = {}

    def _init_projectors(self):
        """基于已知外参表批量构建物理 3D 逆透视投影器"""
        for cam_id, calib in self.cameras_calib.items():
            self._build_projector_for_cam(cam_id, calib)

    def _build_projector_for_cam(self, cam_id: int, calib: Optional[CameraCalib] = None, w: int = 640, h: int = 480) -> FisheyeProjector:
        if calib is not None:
            res = getattr(calib, 'resolution', (w, h))
            res_w, res_h = res[0], res[1]
            inst = getattr(calib, 'install_pos_m', None)
            h_cam = float(calib.camera_height_m)
            side = getattr(calib, 'vehicle_body_side', 'front')
            if not inst or len(inst) < 3:
                if side == 'left' or cam_id == 1:
                    inst = (0.85, self.half_w + 0.03, h_cam)
                elif side == 'right' or cam_id == 2:
                    inst = (0.85, -self.half_w - 0.03, h_cam)
                elif side == 'rear' or cam_id == 3:
                    inst = (-self.half_l + 0.05, 0.0, h_cam)
                else:
                    inst = (self.half_l - 0.05, 0.0, h_cam)

            yaw = float(getattr(calib, 'camera_yaw_deg', 0.0))
            pitch = float(calib.camera_pitch_deg)
            roll = float(getattr(calib, 'camera_roll_deg', 0.0))
            fov = float(getattr(calib, 'fov_deg', 190.0))
        else:
            res_w, res_h = w, h
            h_cam = self.camera_height
            pitch = float(np.degrees(self.camera_pitch_rad))
            roll = 0.0
            fov = 190.0
            if cam_id == 1:
                inst = (0.85, self.half_w + 0.03, h_cam)
                yaw = 105.0
            elif cam_id == 2:
                inst = (0.85, -self.half_w - 0.03, h_cam)
                yaw = -90.0
            elif cam_id == 3:
                inst = (-self.half_l + 0.05, 0.0, h_cam)
                yaw = 180.0
            else:
                inst = (self.half_l - 0.05, 0.0, h_cam)
                yaw = 0.0

        p = FisheyeProjector(
            w=res_w,
            h=res_h,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            fov_deg=fov,
            cam_pos=inst,
            v_len=self.vehicle_length_m,
            v_wid=self.vehicle_width_m
        )
        self.projectors[cam_id] = p
        return p

    def _get_projector(self, camera_id: int, w: int = 640, h: int = 480) -> FisheyeProjector:
        if camera_id in self.projectors:
            return self.projectors[camera_id]
        calib = self.cameras_calib.get(camera_id)
        return self._build_projector_for_cam(camera_id, calib, w, h)

    def get_camera_params(self, camera_id: int) -> Tuple[float, float]:
        """获取指定通道相机的安装高度(m)与俯仰角(deg)"""
        if camera_id in self.cameras_calib:
            calib = self.cameras_calib[camera_id]
            return calib.camera_height_m, calib.camera_pitch_deg
        return self.camera_height, float(np.degrees(self.camera_pitch_rad))

    def estimate_ground_distance(
        self,
        bottom_center_pixel: Tuple[int, int],
        frame_shape: Tuple[int, int],
        camera_id: int = 1
    ) -> float:
        """
        根据检测框底边中点 (接地点) 几何估算与车身水平物理距离 (米)
        1. 采用高保真 3D 鱼眼射线-地面相交逆透视 (IPM) 几何物理模型
        2. 与视觉标定多边形 (Zone 2 / Zone 1) 进行严格互锁一致性校准：
           - 目标脚点位于 Zone 2 标定多边形内 -> 物理距离严格落在 [0.1, 2.9]m (对应 Zone 2 核心研判区)
           - 目标脚点位于 Zone 1 标定多边形内 -> 物理距离严格落在 [3.1, 5.9]m (对应 Zone 1 一级初筛区)
           - 目标脚点位于防区外远场 -> 物理距离输出真实远场距离 (>= 6.0m)
        """
        u, v = bottom_center_pixel
        h, w = frame_shape[:2]

        calib = self.cameras_calib.get(camera_id)
        calib_w, calib_h = (calib.resolution if calib and hasattr(calib, 'resolution') else (640, 480))

        # 归一化缩放至标定基准空间 (通常为 640x480)
        scale_x = float(w) / float(calib_w)
        scale_y = float(h) / float(calib_h)
        u_calib = u / max(scale_x, 1e-4)
        v_calib = v / max(scale_y, 1e-4)

        # 1. 3D 鱼眼 IPM 物理光线逆投影解算
        projector = self._get_projector(camera_id, calib_w, calib_h)
        xg, yg, raw_dist = projector.pixel_to_ground(u_calib, v_calib)

        # 2. 视觉防区多边形空间互锁校验 (解决单目盲区与标定带视觉-数值脱节)
        in_zone2 = False
        in_zone1 = False

        if calib is not None:
            pt = (float(u_calib), float(v_calib))
            if calib.zone2_polygon and len(calib.zone2_polygon) >= 3:
                poly2 = np.array(calib.zone2_polygon, dtype=np.int32)
                if cv2.pointPolygonTest(poly2, pt, False) >= 0:
                    in_zone2 = True

            if not in_zone2 and calib.zone1_polygon and len(calib.zone1_polygon) >= 3:
                poly1 = np.array(calib.zone1_polygon, dtype=np.int32)
                if cv2.pointPolygonTest(poly1, pt, False) >= 0:
                    in_zone1 = True

        # 3. 距离约束与分档一致性映射
        if in_zone2:
            # 严格约束至 Zone 2 区间 [0.1, 2.9]m
            dist = float(np.clip(raw_dist, 0.1, 2.9))
        elif in_zone1:
            # 严格约束至 Zone 1 区间 [3.1, 5.9]m
            dist = float(np.clip(raw_dist, 3.1, 5.9))
        else:
            # 位于划线外部: 若靠近车体则保持物理距离，若在防区外侧远端则至少 6.0m
            if calib and (calib.zone1_polygon or calib.zone2_polygon):
                if raw_dist < 3.0:
                    # 车身极近盲区/门板下方
                    dist = float(np.clip(raw_dist, 0.1, 2.9))
                elif raw_dist >= 6.0:
                    dist = float(raw_dist)
                else:
                    dist = float(raw_dist)
            else:
                dist = float(raw_dist)

        return float(np.clip(dist, 0.1, 15.0))


    def update_radial_velocity(self, track_id: int, current_dist: float, current_time: float) -> float:
        """
        计算目标的径向逼近速度 (米/秒)
        负数表示朝车身方向靠近，正数表示远离车身
        """
        if track_id not in self.history_records:
            self.history_records[track_id] = []

        history = self.history_records[track_id]
        history.append((current_time, current_dist))

        # 仅保留最近 1.0 秒内的历史点
        while len(history) > 1 and (current_time - history[0][0]) > 1.0:
            history.pop(0)

        if len(history) < 2:
            return 0.0

        dt = history[-1][0] - history[0][0]
        if dt <= 0.05:
            return 0.0

        # delta_dist: 负数说明距离缩小，即逼近
        delta_dist = history[-1][1] - history[0][1]
        vr = delta_dist / dt
        return float(vr)

    def clean_stale_tracks(self, active_track_ids: set):
        """清理已消失轨迹的测距缓存"""
        stale = [tid for tid in self.history_records if tid not in active_track_ids]
        for tid in stale:
            del self.history_records[tid]

    def reassign_track_id(self, old_id: int, new_id: int):
        """跨遮挡 ReID 轨迹重新关联时同步映射距离时序历史"""
        if old_id in self.history_records:
            self.history_records[new_id] = self.history_records.pop(old_id)


class ZoneHysteresisFilter:
    """
    防区边界迟滞滤波器 (第二章工程约束)
    边界加迟滞 (±0.3 m)，避免目标在边界抖动造成防区和状态频繁切换
    """
    def __init__(self):
        # 记录每个目标的当前防区状态: {track_id: VehicleZone}
        self.current_zones: Dict[int, VehicleZone] = {}

    def reassign_track_id(self, old_id: int, new_id: int):
        """跨遮挡 ReID 轨迹重新关联时同步防区状态"""
        if old_id in self.current_zones:
            self.current_zones[new_id] = self.current_zones.pop(old_id)

    def clean_stale_tracks(self, active_track_ids: set):
        """清理已移除轨迹的防区状态，防止长时间运行内存泄漏"""
        stale = [tid for tid in self.current_zones if tid not in active_track_ids]
        for tid in stale:
            del self.current_zones[tid]

    def update_zone(self, track_id: int, dist: float) -> VehicleZone:
        buf = GuardConfig.HYSTERESIS_BUFFER # 0.3m

        # 首次出现的目标直接根据物理距离初始化防区 (0~3.0m 直达车身统一归入 Zone 2 核心研判区)
        if track_id not in self.current_zones:
            if dist < GuardConfig.ZONE_INNER_DIST: # < 3.0m (0 ~ 3.0m 直达车身)
                init_zone = VehicleZone.ZONE_2_INNER
            elif dist < GuardConfig.ZONE_FAR_DIST: # < 6.0m
                init_zone = VehicleZone.ZONE_1_OUTER
            else:
                init_zone = VehicleZone.ZONE_0_FAR
            self.current_zones[track_id] = init_zone
            return init_zone

        current_zone = self.current_zones[track_id]

        # 具备迟滞区间的连续更新
        if current_zone == VehicleZone.ZONE_0_FAR:
            if dist < (GuardConfig.ZONE_FAR_DIST - buf):
                current_zone = VehicleZone.ZONE_1_OUTER
                if dist < (GuardConfig.ZONE_INNER_DIST - buf):
                    current_zone = VehicleZone.ZONE_2_INNER
        elif current_zone == VehicleZone.ZONE_1_OUTER:
            if dist > (GuardConfig.ZONE_FAR_DIST + buf):
                current_zone = VehicleZone.ZONE_0_FAR
            elif dist < (GuardConfig.ZONE_INNER_DIST - buf):
                current_zone = VehicleZone.ZONE_2_INNER
        elif current_zone == VehicleZone.ZONE_2_INNER:
            if dist > (GuardConfig.ZONE_INNER_DIST + buf):
                current_zone = VehicleZone.ZONE_1_OUTER

        self.current_zones[track_id] = current_zone
        return current_zone
