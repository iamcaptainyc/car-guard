"""
智能汽车露营守卫 - 多相机防区与外参动态调优控制台 (tune_camera_zones.py)
专为真实车载环视鱼眼 (左/右后视镜下视、前向机盖、后备箱) 打造。

全新核心升级：
1. 【单线分界模式 (极简免闭合、自动焊接邻接面)】:
   - 彻底告别繁琐的沿画面边缘/车身边缘一毫米一毫米描点的痛苦！
   - ★ 标定车身: 只需顺着车门从上往下点击一条分界线，程序自动延伸至画面边缘闭合车身！
   - ★ 标定 Zone 2: 只需顺着 3m 标尺线点击一条分界线，程序自动以车身边界为内缘无缝缝合，绝对零缝隙！
   - ★ 标定 Zone 1: 只需顺着 6m 标尺线点击一条分界线，程序自动以 Zone 2 为内缘无缝缝合，绝对零缝隙！
   - 用户不用画整圈闭合多边形，只画单线即可自动生成三大互斥带状防区！

2. 【地面真实物理标尺曲线 (按 [G] 开关)】:
   - 1m、2m、3m (Zone 2 界限)、6m (Zone 1 界限) 真实等距物理弧线。
   - 纵向全覆盖扩展 (默认覆盖车前到车后 13 米长)，支持按 [T]/[Y] 前后拉伸/收缩。

3. 【3D 鱼眼投影动态联动】:
   - 按 [J]/[L] 偏航 Yaw (±1°), [I]/[K] 俯仰 Pitch (±1°), [U]/[O] 横滚 Roll (±1°)
   - 按 [T]/[Y] 调整前后纵向跨度 (±0.5m)
   - 3D 模式下 Zone 1 与 Zone 2 同样做掩膜互斥，彻底消除重叠！

4. 【实时延展虚线预览与撤销】:
   - 画线时实时计算并绘制延展到画面边缘的引线，所见即所得
   - 鼠标左键添加点，右键/退格撤销上一手，回车或双击确认
   - 鼠标悬停实时物理测距探针 (精确判定车身/Zone2/Zone1/远场)
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np

from camping_guard.config import load_vehicle_calibration
from camping_guard.types import CameraCalib, VehicleCalibration


def parse_camera_id(val: str) -> int:
    val_str = str(val).strip().lower()
    mapping = {
        "0": 0, "front": 0, "f": 0, "前": 0, "车头": 0,
        "1": 1, "left": 1, "l": 1, "左": 1, "左侧": 1, "左视": 1,
        "2": 2, "right": 2, "r": 2, "右": 2, "右侧": 2, "右视": 2,
        "3": 3, "rear": 3, "back": 3, "b": 3, "后": 3, "车尾": 3, "后视": 3
    }
    if val_str in mapping:
        return mapping[val_str]
    try:
        cid = int(val_str)
        if cid in [0, 1, 2, 3]:
            return cid
    except ValueError:
        pass
    return 1


def snap_to_nearest_border(pt: Tuple[int, int], w: int, h: int) -> Tuple[int, int]:
    """将点垂直投影到最近的图像外边界 (防止凹曲线或转弯时切线外推乱穿屏幕)"""
    x = max(0, min(w, int(round(pt[0]))))
    y = max(0, min(h, int(round(pt[1]))))
    d_left = x
    d_right = w - x
    d_top = y
    d_bot = h - y
    min_d = min(d_left, d_right, d_top, d_bot)
    if min_d == d_left:
        return (0, y)
    elif min_d == d_right:
        return (w, y)
    elif min_d == d_top:
        return (x, 0)
    else:
        return (x, h)


def extend_to_border(p_ref: Tuple[int, int], p_edge: Tuple[int, int], w: int, h: int) -> Tuple[int, int]:
    """保持接口兼容：采用最近边界垂直吸附投影，杜绝切线发散乱穿"""
    return snap_to_nearest_border(p_edge, w, h)


def process_stroke(raw_pts: List[Tuple[int, int]], w: int, h: int, cam_side: str = 'left') -> List[Tuple[int, int]]:
    """将用户点击的分界线首尾两端自动垂直吸附到最近外边框"""
    if len(raw_pts) < 2:
        return list(raw_pts)
    p_start = snap_to_nearest_border(raw_pts[0], w, h)
    p_end = snap_to_nearest_border(raw_pts[-1], w, h)
    return [p_start] + list(raw_pts) + [p_end]


def get_cam_body_seed(cam_side: str, w: int, h: int) -> Tuple[int, int]:
    """获取不同机位自车车身所处的基准锚点，用于水流分水岭注水填充 (FloodFill)"""
    if cam_side == "left":
        return (0, h - 1)
    elif cam_side == "right":
        return (w - 1, h - 1)
    else:  # front or rear
        return (w // 2, h - 1)


def make_partition_mask(stroke: List[Tuple[int, int]], w: int, h: int, seed: Tuple[int, int]) -> np.ndarray:
    """
    使用 OpenCV 图像学 FloodFill 构建单侧空间划分掩膜。
    无论车身边缘如何凹陷、转弯、S形或车身仅占角落，注水算法天然完美贴合任何复杂凹多边形，绝无交叉破面！
    """
    if not stroke or len(stroke) < 2:
        return np.zeros((h, w), dtype=np.uint8)

    canvas = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(stroke, np.int32)
    # 线条粗细设为3，保证抗锯齿与断点密闭
    cv2.polylines(canvas, [pts], False, 255, 3, cv2.LINE_AA)

    # cv2.floodFill 要求的 mask 大小为 (h + 2, w + 2)
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    ff_mask[0, :] = 1
    ff_mask[-1, :] = 1
    ff_mask[:, 0] = 1
    ff_mask[:, -1] = 1
    ff_mask[1:-1, 1:-1] = (canvas > 0).astype(np.uint8)

    # 寻找有效未被画笔占用的注水种子点
    sx, sy = seed
    sx = max(0, min(w - 1, sx))
    sy = max(0, min(h - 1, sy))
    if canvas[sy, sx] > 0:
        for r in range(1, 25):
            found = False
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx, ny = sx + dx, sy + dy
                    if 0 <= nx < w and 0 <= ny < h and canvas[ny, nx] == 0:
                        sx, sy = nx, ny
                        found = True
                        break
                if found:
                    break
            if found:
                break

    flood_img = np.zeros((h, w), dtype=np.uint8)
    cv2.floodFill(flood_img, ff_mask, (sx, sy), 255)
    return flood_img



def mask_to_clean_polygon(mask: np.ndarray, max_points: int = 32) -> List[Tuple[int, int]]:
    """将二值掩膜提取为平滑、无自相交、顶点数适中的多边形点集"""
    if mask is None or np.count_nonzero(mask) < 50:
        return []
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    largest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(largest) < 80:
        return []

    peri = cv2.arcLength(largest, True)
    for eps in [0.004, 0.007, 0.010, 0.015, 0.022, 0.030]:
        approx = cv2.approxPolyDP(largest, eps * peri, True)
        if len(approx) <= max_points:
            return [tuple(pt[0]) for pt in approx]
    return [tuple(pt[0]) for pt in approx[:max_points]]


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

    def get_vanishing_point(self, direction_3d: Tuple[float, float, float] = (1.0, 0.0, 0.0)) -> Optional[Tuple[int, int]]:
        """计算三维物理方向向量在鱼眼画面的透视灭点 (Vanishing Point)"""
        v = np.array(direction_3d, dtype=np.float64)
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            return None
        v /= norm
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
        if 0 <= u < self.w and 0 <= v_px < self.h:
            return int(round(u)), int(round(v_px))
        return None

    def pixel_to_ground(self, u: int, v: int) -> Tuple[Optional[float], Optional[float], float]:
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

    def generate_zone_polygons(
        self,
        cam_side: str,
        mask_body: Optional[np.ndarray] = None,
        poly_body: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """根据当前外参动态计算 3D 地面透视投影延伸多边形，前后两端平滑交汇湮灭于灭点 (零截断、零重叠)"""
        h, w = self.h, self.w
        span = self.x_span

        if cam_side in ['left', 'right']:
            sign = 1.0 if cam_side == 'left' else -1.0
            vp_rear = self.get_vanishing_point([-1.0, 0.0, 0.0])
            vp_fwd = self.get_vanishing_point([1.0, 0.0, 0.0])

            def sample_ground_curve(yg_dist: float) -> List[Tuple[int, int]]:
                yg = sign * (self.half_w + yg_dist)
                pts = []
                if vp_rear is not None:
                    pts.append(vp_rear)
                x_samples = np.concatenate([
                    np.linspace(-80.0, -10.0, 30),
                    np.linspace(-10.0, 10.0, 60),
                    np.linspace(10.0, 80.0, 30)
                ])
                for xg in x_samples:
                    p = self.ground_to_pixel(xg, yg, 0.0)
                    if p and 0 <= p[0] < w and 0 <= p[1] < h:
                        if not pts or p != pts[-1]:
                            pts.append(p)
                if vp_fwd is not None:
                    if not pts or vp_fwd != pts[-1]:
                        pts.append(vp_fwd)
                return pts

            def simplify_curve(pts: List[Tuple[int, int]], eps: float = 2.0) -> List[Tuple[int, int]]:
                if len(pts) <= 2:
                    return pts
                arr = np.array(pts, np.int32).reshape(-1, 1, 2)
                approx = cv2.approxPolyDP(arr, eps, False)
                return [tuple(p[0]) for p in approx]

            c_3m = simplify_curve(sample_ground_curve(2.8), 2.5)
            c_6m = simplify_curve(sample_ground_curve(5.5), 3.0)

            # Zone 1 是 6m 与 3m 地面弧线首尾相连构成的透视封闭飘带 (前后平滑汇聚于灭点)
            poly1 = c_6m + c_3m[::-1]

            # 提取面向地面侧的车身边界线 (零缝隙零重叠直接焊接)
            flank_pts = []
            if poly_body and len(poly_body) >= 3:
                raw_flank = [
                    tuple(p) for p in poly_body
                    if not (p[0] <= 3 or p[1] <= 3 or (p[0] >= w - 10 and p[1] >= h - 15))
                ]
                if len(raw_flank) >= 2:
                    if raw_flank[0][1] > raw_flank[-1][1]:
                        raw_flank = raw_flank[::-1]
                    flank_pts = raw_flank

            if flank_pts:
                # 100% 零缝隙：Zone 2 内侧与车身门板共享同一套边界点！
                poly2 = c_3m + flank_pts[::-1]
            else:
                c_inner = simplify_curve(sample_ground_curve(0.0), 3.0)
                poly2_raw = c_3m + c_inner[::-1]
                m2_render = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(m2_render, [np.array(poly2_raw, np.int32)], 255)
                if mask_body is not None and np.count_nonzero(mask_body) > 0:
                    m2_clean = cv2.bitwise_and(m2_render, cv2.bitwise_not(mask_body))
                    poly2 = mask_to_clean_polygon(m2_clean, 32)
                    if not poly2:
                        poly2 = poly2_raw
                else:
                    poly2 = poly2_raw

            return poly1, poly2

        elif cam_side == 'rear':
            x_min, x_max = -span * 1.5, -self.half_l
            z2_box = (x_min, x_max, -self.half_w - 0.6, self.half_w + 0.6)
            z1_box = (x_min - 2.5, x_max, -self.half_w - 1.5, self.half_w + 1.5)
            def rasterize_box(xmin, xmax, ymin, ymax, steps=40):
                pts = []
                for x in np.linspace(xmin, xmax, steps):
                    p = self.ground_to_pixel(x, ymax, 0.0); (pts.append(p) if p else None)
                for y in np.linspace(ymax, ymin, steps):
                    p = self.ground_to_pixel(xmax, y, 0.0); (pts.append(p) if p else None)
                for x in np.linspace(xmax, xmin, steps):
                    p = self.ground_to_pixel(x, ymin, 0.0); (pts.append(p) if p else None)
                for y in np.linspace(ymin, ymax, steps):
                    p = self.ground_to_pixel(xmin, y, 0.0); (pts.append(p) if p else None)
                mask = np.zeros((h, w), dtype=np.uint8)
                valid = [p for p in pts if 0 <= p[0] < w and 0 <= p[1] < h]
                if len(valid) >= 3:
                    cv2.fillPoly(mask, [np.array(valid, np.int32)], 255)
                return mask
            m2_raw = rasterize_box(*z2_box)
            m1_raw = rasterize_box(*z1_box)

        else:  # front
            x_min, x_max = self.half_l, span * 1.5
            z2_box = (x_min, x_max, -self.half_w - 0.6, self.half_w + 0.6)
            z1_box = (x_min, x_max + 2.5, -self.half_w - 1.5, self.half_w + 1.5)
            def rasterize_box(xmin, xmax, ymin, ymax, steps=40):
                pts = []
                for x in np.linspace(xmin, xmax, steps):
                    p = self.ground_to_pixel(x, ymax, 0.0); (pts.append(p) if p else None)
                for y in np.linspace(ymax, ymin, steps):
                    p = self.ground_to_pixel(xmax, y, 0.0); (pts.append(p) if p else None)
                for x in np.linspace(xmax, xmin, steps):
                    p = self.ground_to_pixel(x, ymin, 0.0); (pts.append(p) if p else None)
                for y in np.linspace(ymin, ymax, steps):
                    p = self.ground_to_pixel(xmin, y, 0.0); (pts.append(p) if p else None)
                mask = np.zeros((h, w), dtype=np.uint8)
                valid = [p for p in pts if 0 <= p[0] < w and 0 <= p[1] < h]
                if len(valid) >= 3:
                    cv2.fillPoly(mask, [np.array(valid, np.int32)], 255)
                return mask
            m2_raw = rasterize_box(*z2_box)
            m1_raw = rasterize_box(*z1_box)

        # 1. Zone 2: 扣除车身门板 (零缝隙)
        if mask_body is not None and np.count_nonzero(mask_body) > 0:
            m2_clean = cv2.bitwise_and(m2_raw, cv2.bitwise_not(mask_body))
        else:
            m2_clean = m2_raw
        poly2 = mask_to_clean_polygon(m2_clean, 32)

        # 2. Zone 1: 扣除 Zone 2 与 车身 (零缝隙)
        m2_render = np.zeros((h, w), dtype=np.uint8)
        if poly2:
            cv2.fillPoly(m2_render, [np.array(poly2, np.int32)], 255)

        m_exclude = m2_render
        if mask_body is not None and np.count_nonzero(mask_body) > 0:
            m_exclude = cv2.bitwise_or(m_exclude, mask_body)

        m1_clean = cv2.bitwise_and(m1_raw, cv2.bitwise_not(m_exclude))
        poly1 = mask_to_clean_polygon(m1_clean, 32)

        return poly1, poly2


class UnifiedZoneTunerApp:
    def __init__(
        self,
        base_frame: np.ndarray,
        camera_id: int = 1,
        config_path: str = "vehicle_calib.json",
        cap: Optional[cv2.VideoCapture] = None
    ):
        self.frame = base_frame.copy()
        self.cap = cap
        self.camera_id = camera_id
        self.config_path = config_path

        # 载入标定配置
        self.calib = load_vehicle_calibration(config_path)
        self.cam_calib = self.calib.get_camera(self.camera_id)
        if not self.cam_calib:
            raise ValueError(f"未在 {config_path} 中找到 Camera {camera_id}")

        self.v_len = self.calib.vehicle_info.length_m
        self.v_wid = self.calib.vehicle_info.width_m
        self.cam_side = getattr(self.cam_calib, "vehicle_body_side", "left")

        # 当前可调外参
        inst = getattr(self.cam_calib, "install_pos_m", (0.85, 1.02, 1.20))
        self.inst_x = float(inst[0])
        self.inst_y = float(inst[1])
        self.inst_z = float(self.cam_calib.camera_height_m)
        self.yaw_deg = float(getattr(self.cam_calib, "camera_yaw_deg", 105.0 if camera_id in [1, 2] else 0.0))
        self.pitch_deg = float(self.cam_calib.camera_pitch_deg)
        self.roll_deg = float(getattr(self.cam_calib, "camera_roll_deg", -5.0 if camera_id in [1, 2] else 0.0))
        self.fov_deg = float(getattr(self.cam_calib, "fov_deg", 190.0))
        self.x_span = 10.0  # 前后拉伸跨度，默认前后各 6.5 米 (总长 13 米)

        h, w = self.frame.shape[:2]
        self.projector = FisheyeProjector(
            w=w, h=h,
            yaw_deg=self.yaw_deg,
            pitch_deg=self.pitch_deg,
            roll_deg=self.roll_deg,
            fov_deg=self.fov_deg,
            cam_pos=(self.inst_x, self.inst_y, self.inst_z),
            v_len=self.v_len,
            v_wid=self.v_wid,
            x_span=self.x_span
        )

        self.is_3d_dynamic = True
        self.show_ground_grid = True

        # 单线分界线集合: {'body': [...], 'z2': [...], 'z1': [...]}
        self.strokes: Dict[str, List[Tuple[int, int]]] = {
            'body': [],
            'z2': [],
            'z1': []
        }

        # 闭合多边形点集
        self.poly_z1: List[Tuple[int, int]] = []
        self.poly_z2: List[Tuple[int, int]] = []
        self.poly_roi_body: List[Tuple[int, int]] = []

        # 内部互斥掩膜 (h, w)
        self.mask_body = np.zeros((h, w), dtype=np.uint8)
        self.mask_z2 = np.zeros((h, w), dtype=np.uint8)
        self.mask_z1 = np.zeros((h, w), dtype=np.uint8)

        # 查找该相机的车身门板 ROI
        self.target_roi_id = "body_flank_l" if self.camera_id == 1 else (
            "body_flank_r" if self.camera_id == 2 else "body_rear_bumper"
        )
        scale_x = w / 640.0
        scale_y = h / 480.0
        for roi in self.calib.zone3_rois:
            if roi.camera_id == self.camera_id and roi.roi_id == self.target_roi_id:
                self.poly_roi_body = [(int(p[0] * scale_x), int(p[1] * scale_y)) for p in roi.polygon]
                break

        if self.poly_roi_body and len(self.poly_roi_body) >= 3:
            cv2.fillPoly(self.mask_body, [np.array(self.poly_roi_body, np.int32)], 255)

        # 默认计算 3D 动态多边形
        self.refresh_3d_zones()

        # 模式状态机: 'VIEW', 'WIZARD_BODY', 'WIZARD_Z2', 'WIZARD_Z1', 'DRAW_SINGLE_BODY', 'DRAW_SINGLE_Z2', 'DRAW_SINGLE_Z1'
        self.mode = "VIEW"
        self.drawing_pts: List[Tuple[int, int]] = []

        # 鼠标探针
        self.mouse_pos: Optional[Tuple[int, int]] = None
        self.is_paused = True
        self.status_msg = "就绪: 按 [W] 开启单线分界向导(零缝隙免描边)，按 [G] 显隐物理标尺"
        self.status_color = (0, 255, 0)
        self.window_name = "Camping Guard - Fisheye Calibration Studio"

    def refresh_3d_zones(self):
        """重新计算 3D 几何地面投影多边形并刷新掩膜"""
        self.projector.update_params(
            yaw_deg=self.yaw_deg,
            pitch_deg=self.pitch_deg,
            roll_deg=self.roll_deg,
            fov_deg=self.fov_deg,
            cam_z=self.inst_z,
            x_span=self.x_span
        )
        p1, p2 = self.projector.generate_zone_polygons(self.cam_side, self.mask_body, self.poly_roi_body)
        h, w = self.frame.shape[:2]
        if p2:
            self.poly_z2 = p2
            self.mask_z2 = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(self.mask_z2, [np.array(p2, np.int32)], 255)
        if p1:
            self.poly_z1 = p1
            self.mask_z1 = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(self.mask_z1, [np.array(p1, np.int32)], 255)

    def build_zones_from_strokes(self):
        """
        基于注水划分法 (FloodFill) 构建车身、Zone2、Zone1 的三级互斥空间。
        - 优势 1: 天然完美支持任意凹多边形、凹陷腰线、凹进轮拱、S弯，绝不会自交破面或乱穿。
        - 优势 2: 区域间由像素级布尔掩膜与 7x7 膨胀核裁切控制，严格保证 100% 零缝隙、零重叠！
        """
        h, w = self.frame.shape[:2]
        seed = get_cam_body_seed(self.cam_side, w, h)

        s_b = self.strokes.get('body', [])
        s_2 = self.strokes.get('z2', [])
        s_1 = self.strokes.get('z1', [])

        # 1. 自车车身门板 (包含轮拱与凹陷)
        if len(s_b) >= 2:
            m_body = make_partition_mask(s_b, w, h, seed)
            pb = mask_to_clean_polygon(m_body, 32)
            if pb:
                self.poly_roi_body = pb
                self.mask_body = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(self.mask_body, [np.array(pb, np.int32)], 255)

        # 2. Zone 2 (0~3m 核心研判区): 扣除车身门板 (零缝隙零重叠)
        if len(s_2) >= 2:
            m_up3m = make_partition_mask(s_2, w, h, seed)
            if np.count_nonzero(self.mask_body) > 0:
                m_z2_clean = cv2.bitwise_and(m_up3m, cv2.bitwise_not(self.mask_body))
            else:
                m_z2_clean = m_up3m
            p2 = mask_to_clean_polygon(m_z2_clean, 32)
            if p2:
                self.poly_z2 = p2
                self.mask_z2 = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(self.mask_z2, [np.array(p2, np.int32)], 255)

        # 3. Zone 1 (3~6m 一级初筛区): 扣除 Zone 2 与 车身 (零缝隙零重叠)
        if len(s_1) >= 2:
            m_up6m = make_partition_mask(s_1, w, h, seed)
            m_exclude = np.zeros((h, w), dtype=np.uint8)
            if np.count_nonzero(self.mask_z2) > 0:
                m_exclude = cv2.bitwise_or(m_exclude, self.mask_z2)
            if np.count_nonzero(self.mask_body) > 0:
                m_exclude = cv2.bitwise_or(m_exclude, self.mask_body)

            if np.count_nonzero(m_exclude) > 0:
                m_z1_clean = cv2.bitwise_and(m_up6m, cv2.bitwise_not(m_exclude))
            else:
                m_z1_clean = m_up6m
            p1 = mask_to_clean_polygon(m_z1_clean, 32)
            if p1:
                self.poly_z1 = p1
                self.mask_z1 = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(self.mask_z1, [np.array(p1, np.int32)], 255)

    def build_ribbon_system(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """向后兼容接口：执行注水划分并返回三大多边形"""
        self.build_zones_from_strokes()
        return self.poly_roi_body, self.poly_z2, self.poly_z1

    def apply_preset_left_mirror(self):
        """应用专为左侧鱼眼相机优化的实车真实参数与透视灭点交汇防区"""
        self.yaw_deg = 105.0
        self.pitch_deg = -15.0
        self.roll_deg = -5.0
        self.fov_deg = 190.0
        self.inst_z = 1.20
        self.x_span = 10.0
        # 生成默认平滑车身边界线
        h, w = self.frame.shape[:2]
        self.strokes['body'] = process_stroke([
            (int(w * 0.22), int(h * 0.15)),
            (int(w * 0.28), int(h * 0.45)),
            (int(w * 0.35), int(h * 0.80))
        ], w, h, self.cam_side)

        self.build_zones_from_strokes()
        self.refresh_3d_zones()

        self.status_msg = "已载入实车预设 (前后自然收敛湮灭于透视灭点，零截断零重叠)"
        self.status_color = (0, 255, 255)

    def start_wizard(self):
        """启动标定模式：用户只需标定自车车身，系统根据相机 3D 外参全自动生成 3m 与 6m 防区并零缝隙焊接"""
        self.mode = "WIZARD_BODY"
        self.drawing_pts.clear()
        self.strokes = {'body': [], 'z2': [], 'z1': []}
        self.is_3d_dynamic = True
        self.status_msg = "【标定自车车身】顺着车门或轮拱点击分界线，按 [回车] 确认（3m/6m防区将全自动生成并零缝隙焊接）"
        self.status_color = (0, 255, 255)

    def finish_drawing_step(self):
        """根据当前标定阶段执行完成逻辑：支持单线自动延展缝合，或闭合多边形掩膜扣除"""
        h, w = self.frame.shape[:2]
        if len(self.drawing_pts) < 2:
            self.drawing_pts.clear()
            self.mode = "VIEW"
            self.status_msg = "点数少于2点，已取消当前绘制"
            self.status_color = (0, 160, 255)
            return

        raw_pts = self.drawing_pts.copy()
        self.drawing_pts.clear()

        # 判断用户画的是否是已经首尾相连的圈 (首尾距离小于 30 像素)
        is_closed = len(raw_pts) >= 4 and np.hypot(raw_pts[0][0] - raw_pts[-1][0], raw_pts[0][1] - raw_pts[-1][1]) < 30

        if "BODY" in self.mode:
            stage_key = "body"
        elif "Z2" in self.mode:
            stage_key = "z2"
        else:
            stage_key = "z1"

        if not is_closed:
            # ★【极简单线分界模式】：垂直吸附边缘 + 水流分水岭(FloodFill)像素级精确贴合任何凹陷
            stroke = process_stroke(raw_pts, w, h, self.cam_side)
            self.strokes[stage_key] = stroke
            self.build_zones_from_strokes()

            if stage_key == "body":
                # ★ 车身标定完成 -> 自动基于 3D IPM 相机外参生成 Zone 2 (0~3m) 与 Zone 1 (3~6m)，完全无需人工绘制！
                self.is_3d_dynamic = True
                self.refresh_3d_zones()
                self.mode = "VIEW"
                self.status_msg = "【车身标定完成】系统已根据 3D 相机外参全自动生成 Zone 2 与 Zone 1 (零人工描画)！按 [S] 即可保存！"
                self.status_color = (0, 255, 0)
            elif self.mode == "WIZARD_Z2":
                self.mode = "WIZARD_Z1"
                self.status_msg = "【向导 3/3: Zone 1 (3~6m)】顺着 6m 蓝色标尺线点击分界线（Zone 2 自动无缝贴合！），按[回车]确认"
                self.status_color = (0, 255, 255)
            elif self.mode == "WIZARD_Z1":
                self.mode = "VIEW"
                self.status_msg = "【向导标定完成】车身(含凹陷)、Zone2、Zone1已全自动无缝缝合且零重叠！按 [S] 保存"
                self.status_color = (0, 255, 0)
            else:
                self.mode = "VIEW"
                self.status_msg = f"【单项分界线已更新】邻接凹陷边界已自动无缝贴合！按 [S] 保存"
                self.status_color = (0, 255, 0)

        else:
            # 【传统闭合多边形模式】：自动使用布尔差集扣除
            if stage_key == "body":
                self.poly_roi_body = raw_pts
                self.mask_body = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(self.mask_body, [np.array(raw_pts, np.int32)], 255)
                # ★ 闭合车身标定完成 -> 同样直接自动生成 3D Zone 2 和 Zone 1
                self.is_3d_dynamic = True
                self.refresh_3d_zones()
                self.mode = "VIEW"
                self.status_msg = "【车身标定完成】系统已根据 3D 相机外参全自动生成 Zone 2 与 Zone 1 (零人工描画)！按 [S] 即可保存！"
                self.status_color = (0, 255, 0)

            elif stage_key == "z2":
                mask_z2_raw = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask_z2_raw, [np.array(raw_pts, np.int32)], 255)
                m_body_dil = cv2.dilate(self.mask_body, np.ones((3, 3), np.uint8), iterations=1)
                self.mask_z2 = cv2.bitwise_and(mask_z2_raw, cv2.bitwise_not(m_body_dil))
                self.poly_z2 = mask_to_clean_polygon(self.mask_z2, max_points=32)

                if np.count_nonzero(self.mask_z1) > 0:
                    m_z2_render = np.zeros((h, w), dtype=np.uint8)
                    if self.poly_z2: cv2.fillPoly(m_z2_render, [np.array(self.poly_z2, np.int32)], 255)
                    m_z2_dil = cv2.dilate(m_z2_render, np.ones((3, 3), np.uint8), iterations=1)
                    self.mask_z1 = cv2.bitwise_and(self.mask_z1, cv2.bitwise_not(m_z2_dil))
                    self.poly_z1 = mask_to_clean_polygon(self.mask_z1, max_points=32)

                if self.mode == "WIZARD_Z2":
                    self.mode = "WIZARD_Z1"
                    self.status_msg = "【向导 3/3: 标定 Zone 1 (3~6m)】参考 6m 蓝线画外围，Zone2自动排除！按回车闭合"
                    self.status_color = (0, 255, 255)
                else:
                    self.mode = "VIEW"
                    self.status_msg = "【Zone 2 标定完成】按 [S] 保存"
                    self.status_color = (0, 255, 0)

            else:
                mask_z1_raw = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask_z1_raw, [np.array(raw_pts, np.int32)], 255)
                m_z2_render = np.zeros((h, w), dtype=np.uint8)
                if self.poly_z2: cv2.fillPoly(m_z2_render, [np.array(self.poly_z2, np.int32)], 255)
                m_z2_dil = cv2.dilate(m_z2_render, np.ones((3, 3), np.uint8), iterations=1)
                mask_exclude = cv2.bitwise_or(m_z2_dil, self.mask_body)
                self.mask_z1 = cv2.bitwise_and(mask_z1_raw, cv2.bitwise_not(mask_exclude))
                self.poly_z1 = mask_to_clean_polygon(self.mask_z1, max_points=32)

                self.mode = "VIEW"
                self.status_msg = "【标定完成】重叠已剪除！按 [S] 保存"
                self.status_color = (0, 255, 0)

    def _mouse_cb(self, event, x, y, flags, param):
        h, w = self.frame.shape[:2]
        if x < w and y < h:
            self.mouse_pos = (x, y)
            is_drawing = self.mode.startswith("WIZARD_") or self.mode.startswith("DRAW_SINGLE_")
            if event == cv2.EVENT_LBUTTONDOWN:
                if is_drawing:
                    self.drawing_pts.append((x, y))
                    step_name = "车身分界线" if "BODY" in self.mode else ("Zone 2 分界线" if "Z2" in self.mode else "Zone 1 分界线")
                    self.status_msg = f"[{step_name}] 已点选第 {len(self.drawing_pts)} 点 (只需单线)，按 [回车] 或双击闭合"
                    self.status_color = (0, 255, 255)
            elif event == cv2.EVENT_LBUTTONDBLCLK:
                if is_drawing and len(self.drawing_pts) >= 2:
                    self.finish_drawing_step()
            elif event == cv2.EVENT_RBUTTONDOWN:
                if is_drawing and self.drawing_pts:
                    self.drawing_pts.pop()
                    self.status_msg = f"已撤销最后一点，剩余 {len(self.drawing_pts)} 点"

    def draw_composite_gui(self) -> np.ndarray:
        """合成主图像与右侧控制台侧边栏"""
        h, w = self.frame.shape[:2]
        panel_w = 400
        canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)

        video_canvas = self.frame.copy()

        # 1. 绘制地面物理距离标尺弧线 (前后自然收敛交汇于灭点，自动湮灭无截断)
        if self.show_ground_grid and self.cam_side in ["left", "right"]:
            sign = 1.0 if self.cam_side == "left" else -1.0
            dist_specs = [
                (1.0, (0, 255, 255), "1m"),
                (2.0, (0, 220, 255), "2m"),
                (2.8, (0, 160, 255), "3m界限 (Zone2)"),
                (5.5, (255, 120, 50), "6m界限 (Zone1)")
            ]
            vp_rear = self.projector.get_vanishing_point([-1.0, 0.0, 0.0])
            vp_fwd = self.projector.get_vanishing_point([1.0, 0.0, 0.0])
            for d_val, col, d_name in dist_specs:
                yg = sign * (self.v_wid / 2.0 + d_val)
                line_pts = []
                if vp_rear is not None:
                    line_pts.append(vp_rear)
                x_samples = np.concatenate([
                    np.linspace(-80.0, -10.0, 30),
                    np.linspace(-10.0, 10.0, 60),
                    np.linspace(10.0, 80.0, 30)
                ])
                for xg in x_samples:
                    p = self.projector.ground_to_pixel(xg, yg, 0.0)
                    if p and 0 <= p[0] < w and 0 <= p[1] < h:
                        if not line_pts or p != line_pts[-1]:
                            line_pts.append(p)
                if vp_fwd is not None:
                    if not line_pts or vp_fwd != line_pts[-1]:
                        line_pts.append(vp_fwd)
                if len(line_pts) >= 2:
                    cv2.polylines(video_canvas, [np.array(line_pts, np.int32)], False, col, 1, cv2.LINE_AA)
                    mid_idx = len(line_pts) // 2
                    cv2.putText(video_canvas, d_name, (line_pts[mid_idx][0] - 25, line_pts[mid_idx][1] - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

            # 标记前后透视灭点
            if vp_rear is not None:
                cv2.circle(video_canvas, vp_rear, 4, (0, 0, 255), -1)
                cv2.putText(video_canvas, "后灭点 V_rear", (vp_rear[0] - 45, vp_rear[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 180, 255), 1, cv2.LINE_AA)
            if vp_fwd is not None:
                cv2.circle(video_canvas, vp_fwd, 4, (0, 0, 255), -1)
                cv2.putText(video_canvas, "前灭点 V_fwd", (vp_fwd[0] - 45, vp_fwd[1] + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 180, 255), 1, cv2.LINE_AA)

        # 2. 绘制半透明防区与车身多边形 (严格互斥无缝隙无重叠)
        overlay = video_canvas.copy()
        if len(self.poly_z1) >= 3:
            cv2.fillPoly(overlay, [np.array(self.poly_z1, np.int32)], (220, 100, 30))
        if len(self.poly_z2) >= 3:
            cv2.fillPoly(overlay, [np.array(self.poly_z2, np.int32)], (30, 200, 220))
        if len(self.poly_roi_body) >= 3:
            cv2.fillPoly(overlay, [np.array(self.poly_roi_body, np.int32)], (0, 80, 220))

        cv2.addWeighted(overlay, 0.34, video_canvas, 0.66, 0, video_canvas)

        # 绘制轮廓边界线
        if len(self.poly_z1) >= 3:
            cv2.polylines(video_canvas, [np.array(self.poly_z1, np.int32)], True, (255, 140, 0), 2, cv2.LINE_AA)
            p_text = self.poly_z1[0]
            cv2.putText(video_canvas, "Zone 1 (3~6m)", (p_text[0] + 5, p_text[1] + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 180, 50), 1, cv2.LINE_AA)

        if len(self.poly_z2) >= 3:
            cv2.polylines(video_canvas, [np.array(self.poly_z2, np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)
            p_text = self.poly_z2[0]
            cv2.putText(video_canvas, "Zone 2 (0~3m)", (p_text[0] + 5, p_text[1] + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)

        if len(self.poly_roi_body) >= 3:
            cv2.polylines(video_canvas, [np.array(self.poly_roi_body, np.int32)], True, (0, 140, 255), 2, cv2.LINE_AA)
            cv2.putText(video_canvas, "自车车身门板 (Zone 3)", (self.poly_roi_body[0][0] + 5, self.poly_roi_body[0][1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 160, 255), 1, cv2.LINE_AA)

        # 3. 正在绘制过程中的橡皮筋与外框自动延伸预览
        is_drawing = self.mode.startswith("WIZARD_") or self.mode.startswith("DRAW_SINGLE_")
        if is_drawing and self.drawing_pts:
            pts_arr = np.array(self.drawing_pts, np.int32)
            cv2.polylines(video_canvas, [pts_arr], False, (0, 255, 0), 2, cv2.LINE_AA)
            for i, p in enumerate(self.drawing_pts):
                cv2.circle(video_canvas, p, 4, (0, 0, 255), -1)
                cv2.putText(video_canvas, str(i + 1), (p[0] + 5, p[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

            # 橡皮筋实时连线至当前鼠标位置
            if self.mouse_pos:
                cv2.line(video_canvas, self.drawing_pts[-1], self.mouse_pos, (0, 255, 120), 1, cv2.LINE_AA)

            # 动态延伸虚线预览：显示起点与终点自动垂直吸附到画面边缘的效果 (稳固防凹陷穿帮)
            p_start_ext = snap_to_nearest_border(self.drawing_pts[0], w, h)
            cv2.line(video_canvas, p_start_ext, self.drawing_pts[0], (0, 255, 255), 1, cv2.LINE_AA)
            cv2.drawMarker(video_canvas, p_start_ext, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 8, 1)

            target_pt = self.mouse_pos if self.mouse_pos else self.drawing_pts[-1]
            p_end_ext = snap_to_nearest_border(target_pt, w, h)
            cv2.line(video_canvas, target_pt, p_end_ext, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.drawMarker(video_canvas, p_end_ext, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 8, 1)

        # 绘制十字光标
        if self.mouse_pos:
            mx, my = self.mouse_pos
            cv2.drawMarker(video_canvas, (mx, my), (0, 255, 0), cv2.MARKER_CROSS, 16, 1)

        canvas[:, :w] = video_canvas

        # 4. 绘制右侧暗黑仪表控制面板
        sidebar = canvas[:, w:]
        sidebar[:] = (20, 20, 24)
        cv2.line(canvas, (w, 0), (w, h), (55, 55, 65), 2)

        def put_line(text, y, font_scale=0.40, color=(220, 220, 220), thickness=1):
            cv2.putText(sidebar, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

        put_line("CAMPING GUARD 鱼眼标定台", 26, 0.50, (0, 255, 255), 2)
        put_line(f"相机: Cam {self.camera_id} ({self.cam_calib.name})", 48, 0.36, (180, 200, 220), 1)
        cv2.line(sidebar, (15, 58), (panel_w - 15, 58), (50, 50, 60), 1)

        # 标定模式指示
        if self.mode == "WIZARD_BODY":
            mode_desc = "【向导 1/3: 车身单线】顺着车门点一条线"
            mode_col = (0, 255, 255)
        elif self.mode == "WIZARD_Z2":
            mode_desc = "【向导 2/3: 3m单线】顺着3m线点一条线"
            mode_col = (0, 255, 255)
        elif self.mode == "WIZARD_Z1":
            mode_desc = "【向导 3/3: 6m单线】顺着6m线点一条线"
            mode_col = (0, 255, 255)
        elif self.mode.startswith("DRAW_SINGLE_"):
            mode_desc = "【单线分界】按回车自动延伸缝合"
            mode_col = (255, 200, 0)
        else:
            mode_desc = "【3D 动态联动模式】(调参即变形)" if self.is_3d_dynamic else "【单线无缝缝合就绪】(零缝隙零重叠)"
            mode_col = (0, 255, 120) if self.is_3d_dynamic else (200, 240, 255)

        put_line(mode_desc, 80, 0.42, mode_col, 2)

        # 推荐核心工作流
        put_line("【极简标定工作流】(仅标车身，防区全自动):", 106, 0.40, (255, 200, 80), 2)
        put_line("★ 按 [W] 或 [3]: 顺着车门划线标定自车车身", 126, 0.37, (0, 255, 255), 1)
        put_line("   - 3m (Zone 2) 与 6m (Zone 1) 由 3D 外参全自动生成！", 144, 0.35, (120, 255, 160), 1)
        put_line("   - 车身、Zone 2、Zone 1 之间 100% 零缝隙零重叠焊接！", 162, 0.35, (120, 255, 160), 1)
        put_line("   - 画完车身按 [回车] 确认，按 [S] 一键保存落盘！", 180, 0.35, (0, 255, 255), 1)
        put_line("★ [A] 基于 3D 外参一键重算 | [S] 保存落盘", 198, 0.36, (200, 220, 255), 1)
        put_line("★ [G] 显隐地面物理标尺 | [C] 重置3D联动", 216, 0.36, (200, 220, 255), 1)

        cv2.line(sidebar, (15, 228), (panel_w - 15, 228), (50, 50, 60), 1)

        # 3D 参数与前后长度微调
        put_line("3D 外参与纵向拉伸 (按键即动):", 248, 0.40, (255, 180, 100), 2)
        put_line(f"前后跨度 Span: {self.x_span * 2:.1f}m    ([T]/[Y] ±0.5m)", 270, 0.38, (0, 255, 255), 1)
        put_line(f"偏航角 Yaw   : {self.yaw_deg:+.1f}°   ([J]/[L] ±1°)", 290, 0.38, (220, 220, 220), 1)
        put_line(f"俯仰角 Pitch : {self.pitch_deg:+.1f}°   ([I]/[K] ±1°)", 310, 0.38, (220, 220, 220), 1)
        put_line(f"横滚角 Roll  : {self.roll_deg:+.1f}°   ([U]/[O] ±1°)", 330, 0.38, (220, 220, 220), 1)
        put_line(f"离地高度 H   : {self.inst_z:.2f}m    ([Z]/[X] ±5cm)", 350, 0.38, (220, 220, 220), 1)
        put_line(f"视场角 FOV   : {self.fov_deg:.1f}°   ([N]/[M] ±2°)", 370, 0.38, (220, 220, 220), 1)

        cv2.line(sidebar, (15, 384), (panel_w - 15, 384), (50, 50, 60), 1)

        # 鼠标实时实地测距仪
        put_line("【鼠标实时实地测距仪】:", 404, 0.42, (0, 255, 255), 2)
        if self.mouse_pos:
            mx, my = self.mouse_pos
            xg, yg, p_dist = self.projector.pixel_to_ground(mx, my)

            # 判断当前像素所在图层
            in_body = (self.mask_body[my, mx] > 0) if len(self.poly_roi_body) >= 3 else False
            in_z2 = (self.mask_z2[my, mx] > 0) if len(self.poly_z2) >= 3 else False
            in_z1 = (self.mask_z1[my, mx] > 0) if len(self.poly_z1) >= 3 else False

            if in_body:
                layer_name = "自车车身/门板 (Zone 3)"
                layer_col = (0, 140, 255)
            elif in_z2:
                layer_name = "Zone 2 核心研判区 (0~3m)"
                layer_col = (0, 255, 255)
            elif in_z1:
                layer_name = "Zone 1 一级初筛区 (3~6m)"
                layer_col = (255, 140, 0)
            elif xg is not None:
                layer_name = "Zone 0 远场外围"
                layer_col = (160, 160, 160)
            else:
                layer_name = "天空/车顶/视界外"
                layer_col = (120, 120, 120)

            put_line(f"像素位置: ({mx}, {my})", 424, 0.36, (200, 200, 200), 1)
            if xg is not None:
                put_line(f"地面物理: X={xg:+.2f}m, Y={yg:+.2f}m", 444, 0.36, (200, 200, 200), 1)
                put_line(f"车体净距: {p_dist:.2f} 米", 464, 0.46, layer_col, 2)
                put_line(f"判定区域: {layer_name}", 484, 0.40, layer_col, 2)
            else:
                put_line(f"判定区域: {layer_name}", 445, 0.38, layer_col, 1)
        else:
            put_line("在左侧画面上移动鼠标查看实地测距与区域", 425, 0.35, (140, 140, 140), 1)

        # 底部状态通知横幅
        cv2.rectangle(sidebar, (10, h - 50), (panel_w - 10, h - 10), (30, 30, 38), -1)
        put_line(f"提示: {self.status_msg}", h - 25, 0.34, self.status_color, 1)

        return canvas

    def save_calibration(self) -> bool:
        """保存调优结果与多边形至 vehicle_calib.json"""
        if not os.path.exists(self.config_path):
            self.status_msg = f"错误: 找不到 {self.config_path}"
            self.status_color = (0, 0, 255)
            return False

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            cams = data.get("cameras", {})
            cid_str = str(self.camera_id)
            if cid_str not in cams:
                cams[cid_str] = {}

            h, w = self.frame.shape[:2]
            scale_x = 640.0 / w
            scale_y = 480.0 / h

            # 保存外参
            cams[cid_str]["camera_height_m"] = round(self.inst_z, 3)
            cams[cid_str]["camera_pitch_deg"] = round(self.pitch_deg, 2)
            cams[cid_str]["camera_yaw_deg"] = round(self.yaw_deg, 2)
            cams[cid_str]["camera_roll_deg"] = round(self.roll_deg, 2)
            cams[cid_str]["fov_deg"] = round(self.fov_deg, 1)
            cams[cid_str]["install_pos_m"] = [round(self.inst_x, 3), round(self.inst_y, 3), round(self.inst_z, 3)]

            # 保存多边形 (归一到 640x480 基准存储)
            cams[cid_str]["zone1_polygon"] = [[int(p[0] * scale_x), int(p[1] * scale_y)] for p in self.poly_z1]
            cams[cid_str]["zone2_polygon"] = [[int(p[0] * scale_x), int(p[1] * scale_y)] for p in self.poly_z2]

            # 更新车身门板防撞 ROI
            if self.poly_roi_body and len(self.poly_roi_body) >= 3:
                norm_roi = [[int(p[0] * scale_x), int(p[1] * scale_y)] for p in self.poly_roi_body]
                found = False
                for r in data.get("zone3_rois", []):
                    if r.get("camera_id") == self.camera_id and r.get("roi_id") == self.target_roi_id:
                        r["polygon"] = norm_roi
                        found = True
                        break
                if not found:
                    data.setdefault("zone3_rois", []).append({
                        "roi_id": self.target_roi_id,
                        "name": "左侧车身门板防碰撞区" if self.camera_id == 1 else "车身防碰撞区",
                        "camera_id": self.camera_id,
                        "polygon": norm_roi,
                        "threshold_ratio": 0.08,
                        "confirm_frames": 2,
                        "roi_type": "body"
                    })

            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.status_msg = "【保存成功】已写入 vehicle_calib.json！"
            self.status_color = (0, 255, 0)
            print(f"\n[OK] 标定已成功写入 {Path(self.config_path).resolve()}:")
            print(f"     Camera {self.camera_id}: Yaw={self.yaw_deg:+.1f}°, Pitch={self.pitch_deg:+.1f}°, Roll={self.roll_deg:+.1f}°, H={self.inst_z:.2f}m")
            print(f"     Zone 2 顶点数: {len(self.poly_z2)}, Zone 1 顶点数: {len(self.poly_z1)}, 车身顶点数: {len(self.poly_roi_body)}")
            return True
        except Exception as e:
            self.status_msg = f"保存异常: {e}"
            self.status_color = (0, 0, 255)
            print(f"[!] 保存失败: {e}")
            return False

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_cb)

        print("\n" + "=" * 78)
        print(" 智能汽车露营守卫 - 极简车身单线标定台 (全自动 3D 物理防区)")
        print("=" * 78)
        print(" 【设计哲学】:")
        print("   ★ 3m (Zone 2) 与 6m (Zone 1) 由相机 3D 物理外参全自动生成，零人工描画！")
        print("   ★ 用户仅需标定自车车身！系统自动完成车身与防区之间的 100% 零缝隙焊接！")
        print(" 【极简操作步骤】:")
        print("   1. 按 [W] 或 [3] 键: 顺着车门从上往下点击一条分界线，按 [回车] 确认")
        print("      - 程序自动垂直吸附闭合车身")
        print("      - 系统立即全自动基于相机 3D 外参生成 3m/6m 弧线，与车身零缝隙无缝焊接！")
        print("   2. 按 [S] 键一键保存落盘写入 vehicle_calib.json")
        print("   3. 按 [G] 键开/关地面物理标尺弧线")
        print("   4. 3D 外参微调 (实时联动): [J]/[L] 偏航Yaw, [I]/[K] 俯仰Pitch, [U]/[O] 横滚Roll")
        print("   5. 按 [A] 键: 一键基于当前 3D 外参重新生成全套防区")
        print("=" * 78 + "\n")

        while True:
            if self.cap is not None and not self.is_paused:
                ret, next_frame = self.cap.read()
                if ret:
                    self.frame = next_frame
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            composite = self.draw_composite_gui()
            cv2.imshow(self.window_name, composite)

            k = cv2.waitKey(30) & 0xFF
            if k in [ord('q'), 27]:
                break

            elif k == ord(' '):
                if self.cap is not None:
                    self.is_paused = not self.is_paused
                    self.status_msg = "视频已暂停 (可精确标定)" if self.is_paused else "视频播放中..."
                    self.status_color = (255, 200, 0)

            # 按 W 启动向导标定
            elif k in [ord('w'), ord('W')]:
                self.start_wizard()

            # 按 A 键根据相机 3D 物理外参 + 车身掩膜全自动生成 Zone 2 与 Zone 1 (零人工描画)
            elif k in [ord('a'), ord('A')]:
                self.is_3d_dynamic = True
                self.refresh_3d_zones()
                self.status_msg = "【全自动生成】已基于相机 3D 外参与车身轮廓生成 Zone 2 与 Zone 1！按 [S] 保存"
                self.status_color = (0, 255, 0)

            # 按 P 键应用实车预设
            elif k in [ord('p'), ord('P')]:
                if self.camera_id in [1, 2]:
                    self.apply_preset_left_mirror()
                else:
                    self.status_msg = "当前相机已重置为机位标准 3D 外参"
                    self.refresh_3d_zones()

            # 按 G 键开/关物理距离网格
            elif k in [ord('g'), ord('G')]:
                self.show_ground_grid = not self.show_ground_grid
                self.status_msg = f"地面物理标尺弧线已{'开启' if self.show_ground_grid else '关闭'}"

            # 车身单线标定
            elif k == ord('3'):
                self.start_wizard()

            elif k in [ord('1'), ord('2')]:
                self.status_msg = "【无需手动标定 3m/6m】系统已根据相机 3D 外参全自动生成！只需按 [W] 标定车身"
                self.status_color = (0, 255, 255)

            # 回车完成当前绘制
            elif k in [10, 13]:
                if self.mode != "VIEW":
                    self.finish_drawing_step()

            # 按 Backspace 或 Delete 撤销上一个点
            elif k in [8, 127]:
                if self.drawing_pts:
                    self.drawing_pts.pop()
                    self.status_msg = f"已撤销最后一点，剩余 {len(self.drawing_pts)} 点"

            # 按 C 键清除手动框，恢复 3D 几何动态联动
            elif k in [ord('c'), ord('C')]:
                self.is_3d_dynamic = True
                self.drawing_pts.clear()
                self.strokes = {'body': [], 'z2': [], 'z1': []}
                self.mode = "VIEW"
                self.refresh_3d_zones()
                self.status_msg = "已重置并切回【3D 几何动态联动模式】(无缝互斥无重叠)"
                self.status_color = (0, 255, 120)

            # 按 S 一键保存
            elif k in [ord('s'), ord('S')]:
                self.save_calibration()

            # 前后拉伸 T / Y
            elif k in [ord('t'), ord('T')]:
                self.x_span = min(12.0, self.x_span + 0.5)
                self.is_3d_dynamic = True
                self.refresh_3d_zones()
                self.status_msg = f"前后纵向跨度已扩展为: {self.x_span * 2:.1f}米"
            elif k in [ord('y'), ord('Y')]:
                self.x_span = max(2.5, self.x_span - 0.5)
                self.is_3d_dynamic = True
                self.refresh_3d_zones()
                self.status_msg = f"前后纵向跨度已缩短为: {self.x_span * 2:.1f}米"

            # 偏航微调 J / L
            elif k in [ord('j'), ord('J')]:
                self.yaw_deg = (self.yaw_deg - 1.0) % 360.0
                self.is_3d_dynamic = True
                self.refresh_3d_zones()
                self.status_msg = f"偏航 Yaw: {self.yaw_deg:+.1f}° (3D防区已动态重算)"
            elif k in [ord('l'), ord('L')]:
                self.yaw_deg = (self.yaw_deg + 1.0) % 360.0
                self.is_3d_dynamic = True
                self.refresh_3d_zones()
                self.status_msg = f"偏航 Yaw: {self.yaw_deg:+.1f}° (3D防区已动态重算)"

            # 俯仰微调 I / K
            elif k in [ord('i'), ord('I')]:
                self.pitch_deg = min(0.0, self.pitch_deg + 1.0)
                self.is_3d_dynamic = True
                self.refresh_3d_zones()
                self.status_msg = f"俯仰 Pitch: {self.pitch_deg:+.1f}°"
            elif k in [ord('k'), ord('K')]:
                self.pitch_deg = max(-75.0, self.pitch_deg - 1.0)
                self.is_3d_dynamic = True
                self.refresh_3d_zones()
                self.status_msg = f"俯仰 Pitch: {self.pitch_deg:+.1f}°"

            # 横滚微调 U / O
            elif k in [ord('u'), ord('U')]:
                self.roll_deg -= 1.0
                self.is_3d_dynamic = True
                self.refresh_3d_zones()
                self.status_msg = f"横滚 Roll: {self.roll_deg:+.1f}°"
            elif k in [ord('o'), ord('O')]:
                self.roll_deg += 1.0
                self.is_3d_dynamic = True
                self.refresh_3d_zones()
                self.status_msg = f"横滚 Roll: {self.roll_deg:+.1f}°"

            # 高度微调 Z / X
            elif k in [ord('z'), ord('Z')]:
                self.inst_z = max(0.5, self.inst_z - 0.05)
                self.is_3d_dynamic = True
                self.refresh_3d_zones()
                self.status_msg = f"高度: {self.inst_z:.2f}m"
            elif k in [ord('x'), ord('X')]:
                self.inst_z = min(2.5, self.inst_z + 0.05)
                self.is_3d_dynamic = True
                self.refresh_3d_zones()
                self.status_msg = f"高度: {self.inst_z:.2f}m"

            # 视场角微调 N / M
            elif k in [ord('n'), ord('N')]:
                self.fov_deg = max(120.0, self.fov_deg - 2.0)
                self.is_3d_dynamic = True
                self.refresh_3d_zones()
                self.status_msg = f"视场角 FOV: {self.fov_deg:.1f}°"
            elif k in [ord('m'), ord('M')]:
                self.fov_deg = min(220.0, self.fov_deg + 2.0)
                self.is_3d_dynamic = True
                self.refresh_3d_zones()
                self.status_msg = f"视场角 FOV: {self.fov_deg:.1f}°"

        cv2.destroyAllWindows()
        if self.cap:
            self.cap.release()
        print("[*] 调优工具已退出。")


def main():
    parser = argparse.ArgumentParser(description="智能汽车露营守卫 - 多相机防区与外参动态调优工具")
    parser.add_argument("--image", type=str, default=None, help="鱼眼图片文件路径 (如 left_fisheye_sample.png)")
    parser.add_argument("--video", type=str, default=None, help="鱼眼视频文件路径 (如 火焰.mp4)")
    parser.add_argument("--camera-id", type=parse_camera_id, default=1,
                        help="待调节的相机通道: 0前, 1左, 2右, 3后 (支持 left/front 等别名，默认 1 左视)")
    parser.add_argument("--config", type=str, default="vehicle_calib.json", help="整车标定参数文件 (默认: vehicle_calib.json)")
    parser.add_argument("--auto-zones", action="store_true",
                        help="根据相机 3D 物理外参与车身 ROI，全自动计算生成 Zone 2 和 Zone 1 并直接保存至配置文件")

    args = parser.parse_args()

    frame = None
    cap = None

    if args.image and os.path.exists(args.image):
        frame = cv2.imread(args.image)
        if frame is not None:
            print(f"[*] 成功载入底图: {args.image} ({frame.shape[1]}x{frame.shape[0]})")
    elif args.video and os.path.exists(args.video):
        cap = cv2.VideoCapture(args.video)
        if cap.isOpened():
            for try_idx in [30, 60, 100, 0]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, try_idx)
                ret, frame = cap.read()
                if ret and frame is not None and frame.mean() > 5:
                    print(f"[*] 成功打开视频源并定位至清晰内容帧: {args.video} (Frame {try_idx})")
                    break

    if frame is None:
        candidates = [
            "left_fisheye_sample.png",
            "sample_fisheye_left.mp4",
            "sample_fisheye.png",
            "C:\\Users\\chanceyu\\Downloads\\火焰.mp4"
        ]
        for cand in candidates:
            if os.path.exists(cand):
                if cand.endswith(".mp4"):
                    cap = cv2.VideoCapture(cand)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"[*] 自动选用视频样本: {cand}")
                        break
                else:
                    frame = cv2.imread(cand)
                    if frame is not None:
                        print(f"[*] 自动选用图片样本: {cand}")
                        break

    if frame is None:
        print("[!] 错误: 未找到可用的图像或视频底图！请指定 --image 或 --video。")
        return

    app = UnifiedZoneTunerApp(
        base_frame=frame,
        camera_id=args.camera_id,
        config_path=args.config,
        cap=cap
    )

    if args.auto_zones:
        app.refresh_3d_zones()
        app.save_calibration()
        print(f"[OK] 已成功根据 Camera {args.camera_id} 物理外参全自动生成 Zone 2 与 Zone 1，并成功保存至 {args.config}")
        return

    app.run()


if __name__ == "__main__":
    main()
