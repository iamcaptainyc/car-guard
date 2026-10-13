"""
Person ReID 模块 (camping_guard/perception/reid.py)
基于 OSNet (Omni-Scale Network) 512维深度外观特征提取
实现多视角样本池 (Exemplar Bank)、防特征污染机制、跨相机/遮挡重识别与累计徘徊判定
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import time

from camping_guard.types import ReIDExemplar, PersonSession, BoundingBox

class OSNetReIDExtractor:
    """
    基于 ONNXRuntime 的 OSNet 车载轻量 ReID 特征提取器
    输入: 人体框裁剪图像 -> 输出: 512 维 L2 归一化特征向量
    """
    def __init__(
        self,
        model_path: str = "models/reid/osnet_x0_25.onnx",
        input_size: Tuple[int, int] = (128, 256), # (width, height)
        use_trt: bool = False
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.session = None
        self.backend = "unknown"
        self.fixed_batch_size = 16

        # ImageNet 均值与标准差 (RGB)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

        is_engine = str(model_path).lower().endswith(".engine")
        if os.path.exists(model_path) or is_engine:
            try:
                from camping_guard.perception.trt_engine import load_inference_backend
                self.session, self.backend = load_inference_backend(model_path, prefer_tensorrt=(use_trt or is_engine))
                if hasattr(self.session, "engine_path"):
                    self.model_path = self.session.engine_path
                elif hasattr(self.session, "model_path"):
                    self.model_path = self.session.model_path
                self.input_name = self.session.get_inputs()[0].name
                inp_shape = self.session.get_inputs()[0].shape
                if len(inp_shape) > 0 and isinstance(inp_shape[0], int) and inp_shape[0] > 0:
                    self.fixed_batch_size = inp_shape[0]
                print(f"[OSNetReIDExtractor] 成功加载 ReID 模型 ({self.backend}): {self.model_path} (batch={self.fixed_batch_size})")
            except Exception as e:
                if os.path.exists(model_path) and not is_engine:
                    try:
                        import onnxruntime as ort
                        sess_opts = ort.SessionOptions()
                        sess_opts.intra_op_num_threads = min(4, os.cpu_count() or 4)
                        sess_opts.inter_op_num_threads = 1
                        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                        self.session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=['CPUExecutionProvider'])
                        self.input_name = self.session.get_inputs()[0].name
                        inp_shape = self.session.get_inputs()[0].shape
                        if isinstance(inp_shape[0], int) and inp_shape[0] > 0:
                            self.fixed_batch_size = inp_shape[0]
                        self.backend = "onnxruntime"
                        print(f"[OSNetReIDExtractor] 成功加载 ReID ONNX 模型: {model_path} (batch={self.fixed_batch_size}, threads={sess_opts.intra_op_num_threads})")
                    except Exception as ex:
                        print(f"[OSNetReIDExtractor] 初始化 ReID ONNX 失败: {ex}")
                else:
                    print(f"[OSNetReIDExtractor] 初始化模型失败: {e}")

    def extract_crop_feature(self, crop_bgr: np.ndarray) -> np.ndarray:
        """从单个人体 BGR 裁剪图像提取 512 维特征"""
        if crop_bgr is None or crop_bgr.size == 0:
            return np.zeros((512,), dtype=np.float32)

        if self.session is None:
            # 合成伪特征 (用于无模型时的测试桩)
            h, w = crop_bgr.shape[:2]
            avg_color = crop_bgr.mean(axis=(0, 1))
            pseudo = np.zeros((512,), dtype=np.float32)
            pseudo[:3] = avg_color / 255.0
            pseudo[3] = h / max(w, 1)
            norm = np.linalg.norm(pseudo)
            return pseudo / (norm + 1e-7)

        # 1. 预处理: BGR -> RGB, Resize, Transpose
        t_pre0 = time.perf_counter()
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(crop_rgb, self.input_size)
        tensor = resized.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1) # (3, 256, 128)
        tensor = np.expand_dims(tensor, axis=0) # (1, 3, 256, 128)
        tensor = (tensor - self.mean) / self.std

        # 2. 补齐到固定 batch 大小 (复用预分配连续缓冲区，消除每次 np.repeat 分配开销)
        if self.fixed_batch_size > 1:
            buf_shape = (self.fixed_batch_size, 3, self.input_size[1], self.input_size[0])
            if getattr(self, "_batch_buffer", None) is None or self._batch_buffer.shape != buf_shape:
                self._batch_buffer = np.zeros(buf_shape, dtype=np.float32)
            self._batch_buffer[0] = tensor[0]
            self.last_preprocess_ms = (time.perf_counter() - t_pre0) * 1000.0
            t_inf0 = time.perf_counter()
            outputs = self.session.run(None, {self.input_name: self._batch_buffer})[0]
            inf_wall_ms = (time.perf_counter() - t_inf0) * 1000.0
            feature = outputs[0]
        else:
            self.last_preprocess_ms = (time.perf_counter() - t_pre0) * 1000.0
            t_inf0 = time.perf_counter()
            feature = self.session.run(None, {self.input_name: tensor})[0][0]
            inf_wall_ms = (time.perf_counter() - t_inf0) * 1000.0

        self.last_gpu_compute_ms = float(getattr(self.session, "last_gpu_compute_ms", 0.0) or inf_wall_ms)

        # 3. L2 归一化
        t_post0 = time.perf_counter()
        norm = np.linalg.norm(feature)
        if norm > 1e-7:
            feature = feature / norm
        res_feat = feature.astype(np.float32)
        self.last_postprocess_ms = (time.perf_counter() - t_post0) * 1000.0
        try:
            self.session.last_preprocess_ms = self.last_preprocess_ms
            self.session.last_postprocess_ms = self.last_postprocess_ms
        except Exception:
            pass
        return res_feat

    def extract_from_frame(self, frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """从全景画面依据目标边界框安全裁剪并提取特征"""
        H, W = frame.shape[:2]
        x1 = max(0, int(bbox.x))
        y1 = max(0, int(bbox.y))
        x2 = min(W, int(bbox.x + bbox.w))
        y2 = min(H, int(bbox.y + bbox.h))

        if x2 <= x1 or y2 <= y1:
            return np.zeros((512,), dtype=np.float32)

        crop = frame[y1:y2, x1:x2]
        return self.extract_crop_feature(crop)


class PersonSessionManager:
    """
    跨机位与长时序全局人员会话管理器 (用于 ReID 徘徊研判与特征库防污染)
    实现:
    1. 多视角样本池 (每人保留 5~8 个代表性清晰特征)
    2. 特征入库防污染机制 (仅在关联可信且带来新视角时写入)
    3. 离开后再入 (Re-entry) 累计停留时长缝合
    """
    def __init__(
        self,
        max_exemplars_per_person: int = 6,
        reid_match_threshold: float = 0.65,
        new_view_diversity_threshold: float = 0.88,
        session_expire_seconds: float = 1800.0 # 30分钟无活动则会话归档
    ):
        self.max_exemplars = max_exemplars_per_person
        self.reid_match_threshold = reid_match_threshold
        self.new_view_thresh = new_view_diversity_threshold
        self.session_expire_sec = session_expire_seconds

        # 活跃人员会话库: {session_id: PersonSession}
        self.sessions: Dict[str, PersonSession] = {}
        # 局部轨迹到全局会话映射: {track_uid: session_id}
        self.track_uid_to_session: Dict[str, str] = {}
        self._next_session_idx = 1

    @staticmethod
    def cosine_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
        """计算两个 L2 归一化特征向量的余弦相似度"""
        if f1 is None or f2 is None or len(f1) == 0 or len(f2) == 0:
            return 0.0
        return float(np.dot(f1, f2))

    def compute_max_similarity(self, feature: np.ndarray, session: PersonSession) -> float:
        """计算候选特征与该人员特征库中所有样本的最大相似度"""
        if not session.exemplars:
            return 0.0
        sims = [self.cosine_similarity(feature, ex.feature) for ex in session.exemplars]
        return max(sims)

    def match_or_create_session(
        self,
        track_uid: str,
        feature: np.ndarray,
        current_time: float,
        quality_score: float = 0.8,
        camera_id: int = 0,
        track_id: Optional[int] = None
    ) -> Tuple[PersonSession, bool]:
        """
        根据当前人体 ReID 特征匹配全局人员会话
        :return: (PersonSession, is_new_session)
        """
        if track_id is None and "trk" in track_uid:
            try:
                track_id = int(track_uid.split("trk")[-1])
            except ValueError:
                track_id = None

        # 1. 如果该 track_uid 已经绑定会话，直接返回
        if track_uid in self.track_uid_to_session:
            sid = self.track_uid_to_session[track_uid]
            session = self.sessions[sid]
            session.last_seen_time = current_time
            if session.root_track_id is None and track_id is not None:
                session.root_track_id = track_id
            # 尝试增量更新特征库 (防污染入库)
            self._try_add_exemplar(session, feature, quality_score, camera_id, current_time)
            return session, False

        # 2. 清理超时已归档会话
        stale_sids = [
            sid for sid, s in self.sessions.items()
            if current_time - s.last_seen_time > self.session_expire_sec
        ]
        for sid in stale_sids:
            # 清理映射
            for tuid in self.sessions[sid].current_track_uids:
                if tuid in self.track_uid_to_session:
                    del self.track_uid_to_session[tuid]
            del self.sessions[sid]

        # 3. 在已有活跃与离线会话中检索最佳匹配
        best_sid: Optional[str] = None
        best_sim: float = -1.0

        for sid, session in self.sessions.items():
            max_sim = self.compute_max_similarity(feature, session)
            if max_sim > best_sim:
                best_sim = max_sim
                best_sid = sid

        # 4. 判定是否为同一人 (Re-entry 重入或跨相机接力)
        if best_sid is not None and best_sim >= self.reid_match_threshold:
            session = self.sessions[best_sid]
            session.last_seen_time = current_time
            session.visit_count += 1
            if session.root_track_id is None and track_id is not None:
                session.root_track_id = track_id
            if track_uid not in session.current_track_uids:
                session.current_track_uids.append(track_uid)
            self.track_uid_to_session[track_uid] = session.session_id
            # 写入代表性新特征
            self._try_add_exemplar(session, feature, quality_score, camera_id, current_time)
            return session, False

        # 5. 未匹配到任何人，建立全新全局会话
        new_sid = f"P{self._next_session_idx:04d}"
        self._next_session_idx += 1

        new_session = PersonSession(
            session_id=new_sid,
            first_seen_time=current_time,
            last_seen_time=current_time,
            cumulative_dwell_time=0.0,
            visit_count=1,
            is_whitelisted=False,
            current_track_uids=[track_uid],
            root_track_id=track_id
        )
        # 初始特征直接入库
        new_session.exemplars.append(ReIDExemplar(
            feature=feature,
            timestamp=current_time,
            quality_score=quality_score,
            camera_id=camera_id,
            view_tag="initial"
        ))
        self.sessions[new_sid] = new_session
        self.track_uid_to_session[track_uid] = new_sid
        return new_session, True

    def _try_add_exemplar(
        self,
        session: PersonSession,
        feature: np.ndarray,
        quality_score: float,
        camera_id: int,
        current_time: float
    ):
        """
        特征库防污染与多视角样本池维护策略:
        1. 只有质量评分合格 (quality_score >= 0.5) 才考虑入库;
        2. 与库内所有已有特征的最大相似度在 [0.55, 0.88] 之间，说明是同一人且带来了有价值的新视角/新光影;
        3. 若与库内某个特征相似度 > 0.90，说明视角高度重复，不浪费样本池槽位;
        4. 样本池超额时，淘汰时间最早且质量最低的样本。
        """
        if feature is None or len(feature) == 0:
            return
        if quality_score < 0.5:
            return

        max_sim = self.compute_max_similarity(feature, session)
        # 太相似 (冗余) 或太离谱 (可能串 ID 污染)
        if max_sim > self.new_view_thresh or max_sim < 0.55:
            return

        new_ex = ReIDExemplar(
            feature=feature,
            timestamp=current_time,
            quality_score=quality_score,
            camera_id=camera_id,
            view_tag=f"cam{camera_id}"
        )

        if len(session.exemplars) < self.max_exemplars:
            session.exemplars.append(new_ex)
        else:
            # 找到质量最低的样本替换
            min_idx = int(np.argmin([e.quality_score for e in session.exemplars]))
            session.exemplars[min_idx] = new_ex

    def update_dwell_time(self, track_uid: str, dt: float):
        """为当前活跃轨迹所属全局会话累加累计滞留时间"""
        if track_uid in self.track_uid_to_session:
            sid = self.track_uid_to_session[track_uid]
            if sid in self.sessions:
                self.sessions[sid].cumulative_dwell_time += dt

    def mark_session_whitelisted(self, track_uid: str, name: str):
        """当某条轨迹通过 FaceID 确认白名单时，反哺并永久豁免该全局人员会话"""
        if track_uid in self.track_uid_to_session:
            sid = self.track_uid_to_session[track_uid]
            if sid in self.sessions:
                session = self.sessions[sid]
                session.is_whitelisted = True
                session.whitelist_name = name
