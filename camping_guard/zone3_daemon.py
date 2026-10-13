"""
车身敏感部件与防碰撞固定 ROI 扰动监视守护进程 (阶段 0.5)
实现门把手、后备箱锁、充电口、车侧防碰撞等车身区域的低功耗常驻监视
包含全局大面积环境扰动抑制 (暴雨/起雾/晃动) 与直跳 S2 机制
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import time
from camping_guard.types import Zone3ROI, BodyROI
from camping_guard.config import GuardConfig

class Zone3ResidentDaemon:
    def __init__(self, rois: Optional[List[BodyROI]] = None, on_disturbance_callback: Optional[Callable[[str, int], None]] = None):
        """
        :param rois: 标定的敏感部件 ROI 列表
        :param on_disturbance_callback: ROI 扰动确认触发回调函数 func(roi_id, camera_id)
        """
        self.rois = rois if rois is not None else GuardConfig.get_default_rois()
        self.on_disturbance_callback = on_disturbance_callback

        # 为每个 ROI 维护专属背景减除器与多帧确认计数器
        self.bg_subtractors: Dict[str, cv2.BackgroundSubtractorMOG2] = {}
        self.trigger_counters: Dict[str, int] = {}
        self.last_disturbance_time: Dict[str, float] = {}

        for roi in self.rois:
            # 采用 MOG2，设置较长历史与高阈值，抑制阴影与环境小抖动
            self.bg_subtractors[roi.roi_id] = cv2.createBackgroundSubtractorMOG2(
                history=150, varThreshold=30, detectShadows=False
            )
            self.trigger_counters[roi.roi_id] = 0
            self.last_disturbance_time[roi.roi_id] = 0.0

        # 全局异常检测状态 (暴雨、大风、镜头起雾)
        self.is_env_anomaly: bool = False
        self.env_anomaly_desc: str = ""

    def process_frames(self, camera_frames: Dict[int, np.ndarray], current_time: float) -> List[Dict]:
        """
        常驻处理各路相机画面
        :param camera_frames: {camera_id: BGR frame}
        :param current_time: 当前时间戳
        :return: 触发确认的 ROI 事件列表
        """
        active_disturbances = []

        # 1. 第一步：仅对标定的局部 ROI 切片执行灰度转换与 MOG2 背景减除 (彻底移除全图无用灰度转换)
        roi_candidates = []

        for roi in self.rois:
            cam_id = roi.camera_id
            if cam_id not in camera_frames or camera_frames[cam_id] is None:
                continue

            frame = camera_frames[cam_id]
            h, w = frame.shape[:2]

            # 提取 ROI 掩码与包围盒
            pts = np.array(roi.polygon, dtype=np.int32)
            rx, ry, rw, rh = cv2.boundingRect(pts)
            rx, ry = max(0, rx), max(0, ry)
            rw = min(w - rx, rw)
            rh = min(h - ry, rh)
            if rw <= 0 or rh <= 0:
                continue

            roi_img = frame[ry:ry+rh, rx:rx+rw]
            # 对大面积车身防碰撞 ROI 降采样至 <= 160x120，兼顾亚毫秒级耗时与高灵敏度前景检测
            if rw > 160 or rh > 120:
                proc_w = min(rw, 160)
                proc_h = min(rh, 120)
                roi_img = cv2.resize(roi_img, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
            else:
                proc_w, proc_h = rw, rh

            roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            roi_blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)

            # MOG2 前景提取
            subtractor = self.bg_subtractors[roi.roi_id]
            fg_mask = subtractor.apply(roi_blurred, learningRate=0.005)
            _, fg_thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

            # 计算前景扰动像素占 ROI 总像素的比值
            roi_area = proc_w * proc_h
            fg_pixels = cv2.countNonZero(fg_thresh)
            disturbance_ratio = fg_pixels / float(roi_area) if roi_area > 0 else 0.0

            if disturbance_ratio >= roi.threshold_ratio:
                roi_candidates.append((roi, disturbance_ratio))
            else:
                self.trigger_counters[roi.roi_id] = max(0, self.trigger_counters[roi.roi_id] - 1)

        # 2. 第二步：全局异常判定 (若多个相机的 ROI 同时剧烈扰动)
        affected_cams = set(r.camera_id for r, _ in roi_candidates)
        if len(affected_cams) >= 2:
            self.is_env_anomaly = True
            self.env_anomaly_desc = "多路相机同时出现剧烈ROI扰动，判定为环境扰动(暴雨/狂风/起雾)，抑制接触误报"
            # 清空候选，不按接触事件处理
            return []
        else:
            self.is_env_anomaly = False
            self.env_anomaly_desc = ""

        # 3. 第三步：连续多帧确认，确认后发布事件
        for roi, ratio in roi_candidates:
            self.trigger_counters[roi.roi_id] += 1
            if self.trigger_counters[roi.roi_id] >= roi.confirm_frames:
                # 冷却检测 (防短时间连续狂刷)
                if current_time - self.last_disturbance_time[roi.roi_id] > 3.0:
                    self.last_disturbance_time[roi.roi_id] = current_time
                    event = {
                        "roi_id": roi.roi_id,
                        "name": roi.name,
                        "camera_id": roi.camera_id,
                        "ratio": ratio,
                        "timestamp": current_time,
                        "roi_type": getattr(roi, "roi_type", "part")
                    }
                    active_disturbances.append(event)
                    if self.on_disturbance_callback:
                        self.on_disturbance_callback(roi.roi_id, roi.camera_id)

        return active_disturbances

BodyROIResidentDaemon = Zone3ResidentDaemon # 兼容别名
