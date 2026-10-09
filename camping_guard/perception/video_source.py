"""
四路环视车载视频流管理与分发服务 (camping_guard/perception/video_source.py)
支持本地视频文件循环回放、车载 RTSP/V4L2 摄像头与离线合成流
支持 S0 (1~2fps) / S1 (10~15fps) / S2 (全帧率) 动态功耗抽帧降频
"""

import cv2
import numpy as np
import time
from typing import Dict, Optional, Union, List

class CameraStream:
    """单路相机流封装"""
    def __init__(self, camera_id: int, source: Union[str, int]):
        self.camera_id = camera_id
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False
        self.is_video_file = isinstance(source, str) and not source.startswith("rtsp")

        self._init_capture()

    def _init_capture(self):
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if self.cap and self.cap.isOpened():
            self.is_opened = True
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            self.is_opened = False

    def read_frame(self) -> Optional[np.ndarray]:
        if not self.is_opened or self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            # 若是视频文件，支持自动循环倒回首帧
            if self.is_video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    return None
            else:
                return None

        if frame is not None and (frame.shape[1] != 640 or frame.shape[0] != 480):
            frame = cv2.resize(frame, (640, 480))
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
            self.is_opened = False


class MultiCameraStreamManager:
    """四路环视相机管理器 (0:前, 1:左, 2:右, 3:后)"""
    def __init__(self, stream_sources: Optional[Dict[int, Union[str, int]]] = None):
        self.streams: Dict[int, CameraStream] = {}
        self.last_fetch_time: Dict[int, float] = {}

        # 默认离线合成流发生器
        self.synthetic_enabled = False
        self.synthetic_bg: Optional[np.ndarray] = None

        if stream_sources:
            for cam_id, src in stream_sources.items():
                self.streams[cam_id] = CameraStream(cam_id, src)
        else:
            # 未提供物理流时，启用合成背景模式
            self.synthetic_enabled = True
            self._init_synthetic_background()

    def _init_synthetic_background(self):
        w, h = 640, 480
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        # 远景与地面
        cv2.rectangle(bg, (0, 0), (w, int(h * 0.35)), (30, 25, 20), -1)
        cv2.rectangle(bg, (0, int(h * 0.35)), (w, h), (40, 45, 40), -1)
        self.synthetic_bg = bg

    def fetch_frames(self, target_fps: float = 15.0) -> Dict[int, np.ndarray]:
        """
        按指定 FPS 抽帧读取四路画面
        :param target_fps: S0 待机通常 1~2 fps, S1 10~15 fps, S2 25~30 fps
        """
        now = time.time()
        min_interval = 1.0 / max(0.1, target_fps)
        frames = {}

        if self.synthetic_enabled:
            # 合成模式返回标准 4 路帧
            for cam_id in [0, 1, 2, 3]:
                last_t = self.last_fetch_time.get(cam_id, 0.0)
                if now - last_t >= min_interval:
                    self.last_fetch_time[cam_id] = now
                    f = self.synthetic_bg.copy()
                    cv2.putText(f, f"CAM {cam_id} [SYNTHETIC FEED]", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
                    frames[cam_id] = f
            return frames

        # 真实物理相机或文件流
        for cam_id, stream in self.streams.items():
            last_t = self.last_fetch_time.get(cam_id, 0.0)
            if now - last_t >= min_interval:
                frame = stream.read_frame()
                if frame is not None:
                    self.last_fetch_time[cam_id] = now
                    frames[cam_id] = frame

        return frames

    def release_all(self):
        for s in self.streams.values():
            s.release()
        self.streams.clear()
