"""
Zone 3 敏感部件固定 ROI 标定与管理工具 (ROICalibrator)
支持 JSON 序列化与反序列化、ROI 多边形几何合法性校验、以及交互式标定
"""

import json
import os
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from camping_guard.types import Zone3ROI

class ROICalibrator:
    def __init__(self, config_path: str = "roi_config.json"):
        self.config_path = config_path

    def load_rois(self) -> List[Zone3ROI]:
        """从 JSON 文件载入标定结果"""
        if not os.path.exists(self.config_path):
            return []

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            rois = []
            for item in data:
                poly = [tuple(p) for p in item.get("polygon", [])]
                roi = Zone3ROI(
                    roi_id=item["roi_id"],
                    name=item.get("name", item["roi_id"]),
                    camera_id=item.get("camera_id", 1),
                    polygon=poly,
                    threshold_ratio=item.get("threshold_ratio", 0.12),
                    confirm_frames=item.get("confirm_frames", 2),
                    roi_type=item.get("roi_type", "part")
                )
                rois.append(roi)
            return rois
        except Exception as e:
            print(f"[ROICalibrator] 载入标定文件失败: {e}")
            return []

    def save_rois(self, rois: List[Zone3ROI]) -> bool:
        """保存标定结果至 JSON 文件"""
        try:
            data = []
            for r in rois:
                data.append({
                    "roi_id": r.roi_id,
                    "name": r.name,
                    "camera_id": r.camera_id,
                    "polygon": [list(pt) for pt in r.polygon],
                    "threshold_ratio": r.threshold_ratio,
                    "confirm_frames": r.confirm_frames,
                    "roi_type": getattr(r, "roi_type", "part")
                })

            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"[ROICalibrator] 保存标定文件失败: {e}")
            return False

    def sync_to_vehicle_calib(self, rois: List[Zone3ROI], calib_path: str = "vehicle_calib.json") -> bool:
        """同步更新 vehicle_calib.json 中的 zone3_rois 字段"""
        if not os.path.exists(calib_path):
            return False
        try:
            with open(calib_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["zone3_rois"] = [
                {
                    "roi_id": r.roi_id,
                    "name": r.name,
                    "camera_id": r.camera_id,
                    "polygon": [list(pt) for pt in r.polygon],
                    "threshold_ratio": r.threshold_ratio,
                    "confirm_frames": r.confirm_frames,
                    "roi_type": getattr(r, "roi_type", "part")
                }
                for r in rois
            ]
            with open(calib_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"[ROICalibrator] 同步 vehicle_calib.json 失败: {e}")
            return False

    def validate_roi(self, roi: Zone3ROI, frame_shape: Tuple[int, int] = (480, 640)) -> Tuple[bool, str]:
        """校验 ROI 多边形几何合法性"""
        if len(roi.polygon) < 3:
            return False, f"ROI {roi.roi_id} 多边形顶点数不足 (< 3)"

        h, w = frame_shape[:2]
        pts = np.array(roi.polygon, dtype=np.int32)

        # 检查是否越界
        if np.any(pts[:, 0] < 0) or np.any(pts[:, 0] >= w) or np.any(pts[:, 1] < 0) or np.any(pts[:, 1] >= h):
            return False, f"ROI {roi.roi_id} 顶点超出画面范围 [0, {w}) x [0, {h})"

        area = cv2.contourArea(pts)
        if area < 50:
            return False, f"ROI {roi.roi_id} 面积过小 ({area:.1f} px^2 < 50)"

        return True, "合法"

    def interactive_calibrate(self, frame: np.ndarray, roi_id: str, name: str, camera_id: int = 1) -> Optional[Zone3ROI]:
        """
        在指定画面上交互式标定一个多边形 ROI
        左键点击添加点，右键撤销最后一点，'s'键完成并保存，'c'键取消
        """
        pts = []
        window_name = f"ROI Calibrator - {name} ({roi_id})"

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pts.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                if pts:
                    pts.pop()

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_cb)

        print(f"[*] 开始标定 【{name}】({roi_id}):")
        print("    - 鼠标左键点击添加多边形顶点")
        print("    - 鼠标右键点击撤销上一个点")
        print("    - 按键盘 's' 键确认完成")
        print("    - 按键盘 'c' 或 ESC 键取消")

        while True:
            display = frame.copy()
            if len(pts) > 0:
                for p in pts:
                    cv2.circle(display, p, 4, (0, 0, 255), -1)
                if len(pts) >= 2:
                    cv2.polylines(display, [np.array(pts, np.int32)], False, (0, 255, 255), 2)
                if len(pts) >= 3:
                    overlay = display.copy()
                    cv2.fillPoly(overlay, [np.array(pts, np.int32)], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)

            cv2.imshow(window_name, display)
            k = cv2.waitKey(20) & 0xFF
            if k == ord('s'):
                if len(pts) >= 3:
                    roi = Zone3ROI(
                        roi_id=roi_id,
                        name=name,
                        camera_id=camera_id,
                        polygon=pts
                    )
                    cv2.destroyWindow(window_name)
                    return roi
                else:
                    print("多边形至少需要 3 个顶点！")
            elif k in [ord('c'), 27]:
                cv2.destroyWindow(window_name)
                return None
