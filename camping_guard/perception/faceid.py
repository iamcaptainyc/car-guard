"""
FaceID 识别与状态机管理模块 (camping_guard/perception/faceid.py)
基于 InsightFace SCRFD-500M 人脸检测与 MobileFaceNet (ArcFace 512维) 特征识别
实现:
1. FQA (Face Quality Assessment) 人脸质量多维门禁 (尺寸、模糊度、姿态角)
2. 细粒度状态机 (PENDING / UNAVAILABLE / NO_MATCH / MATCH_CANDIDATE / VERIFIED / RECHECK)
3. 机会主义异步调度与限频重试
4. 多帧验证确权与白名单状态后验反哺
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import time

from camping_guard.types import FaceIDState, TrackedEntity, BoundingBox

# ArcFace 标准 5 点对齐基准模板坐标 (112x112)
ARCFACE_REFERENCE_POINTS = np.array([
    [38.2946, 51.6963],  # 左眼
    [73.5318, 51.5014],  # 右眼
    [56.0252, 71.7366],  # 鼻尖
    [41.5493, 92.3655],  # 左嘴角
    [70.7299, 92.2041]   # 右嘴角
], dtype=np.float32)


class FaceQualityAssessor:
    """
    FQA (人脸质量评估) 门禁初筛
    对检测出的人脸做几何与图像质检，不合格者直接拦截，杜绝无效计算与底库污染
    """
    def __init__(
        self,
        min_face_size: int = 40,        # 最小有效人脸边长 (像素)
        min_blur_variance: float = 35.0, # 最小清晰度方差 (拉普拉斯梯度)
        max_yaw_ratio: float = 2.4       # 偏航角对称比上限 (超过代表大角度侧脸 > 35°)
    ):
        self.min_face_size = min_face_size
        self.min_blur_variance = min_blur_variance
        self.max_yaw_ratio = max_yaw_ratio

    def evaluate_quality(
        self,
        frame: np.ndarray,
        face_bbox: Tuple[float, float, float, float], # (x1, y1, x2, y2)
        kps: Optional[np.ndarray] = None              # (5, 2) 关键点
    ) -> Tuple[bool, float, str]:
        """
        评估人脸质量
        :return: (is_qualified: bool, score: float, reason: str)
        """
        x1, y1, x2, y2 = face_bbox
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)

        # 1. 尺寸检验
        if min(w, h) < self.min_face_size:
            return False, 0.2, "FACE_TOO_SMALL"

        # 裁剪人脸 ROI
        H, W = frame.shape[:2]
        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
        ix2, iy2 = min(W, int(x2)), min(H, int(y2))
        if ix2 <= ix1 or iy2 <= iy1:
            return False, 0.0, "INVALID_CROP"

        crop = frame[iy1:iy2, ix1:ix2]

        # 2. 清晰度检验 (Laplacian 方差)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_var < self.min_blur_variance:
            return False, 0.3, "MOTION_BLURRED"

        # 3. 姿态对称性检验 (基于 5 点估计水平偏航角)
        if kps is not None and len(kps) == 5:
            # kps: [left_eye, right_eye, nose, left_mouth, right_mouth]
            le, re, nose = kps[0], kps[1], kps[2]
            d_left = abs(nose[0] - le[0])
            d_right = abs(re[0] - nose[0])
            if min(d_left, d_right) > 1e-4:
                yaw_ratio = max(d_left, d_right) / min(d_left, d_right)
                if yaw_ratio > self.max_yaw_ratio:
                    return False, 0.4, "EXTREME_YAW_PROFILE"

        # 综合评分 (0~1)
        quality_score = min(1.0, 0.5 + (blur_var / 200.0) * 0.3 + (min(w, h) / 100.0) * 0.2)
        return True, float(quality_score), "QUALIFIED"


class SCRFDFaceDetector:
    """
    基于 ONNXRuntime 的 SCRFD-500M 轻量人脸检测器 (带 5 关键点)
    """
    def __init__(
        self,
        model_path: str = "models/insightface/det_500m.onnx",
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        input_size: Tuple[int, int] = (640, 640),
        use_trt: bool = False
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size
        self.session = None
        self.backend = "unknown"
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2

        # 预计算各 FPN 步长的静态锚点中心 (杜绝每帧使用 np.mgrid 带来的 25ms CPU 额外开销)
        self._cached_anchor_centers = {}
        in_h, in_w = self.input_size[1], self.input_size[0]
        for s in self._feat_stride_fpn:
            h = in_h // s
            w = in_w // s
            ac = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32) * s
            ac = np.repeat(ac[:, :, None, :], self._num_anchors, axis=2).reshape(-1, 2)
            self._cached_anchor_centers[s] = ac

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
                print(f"[SCRFDFaceDetector] 成功加载人脸检测模型 ({self.backend}): {self.model_path}")
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
                        self.backend = "onnxruntime"
                        print(f"[SCRFDFaceDetector] 成功加载人脸检测 ONNX 模型: {model_path} (threads={sess_opts.intra_op_num_threads})")
                    except Exception as ex:
                        print(f"[SCRFDFaceDetector] 初始化 SCRFD 失败: {ex}")
                else:
                    print(f"[SCRFDFaceDetector] 初始化模型失败: {e}")

    def detect(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        在图像中检测人脸
        :return: (bboxes: Nx5 [x1, y1, x2, y2, score], kpss: Nx5x2 [5 keypoints])
        """
        if self.session is None or img is None or img.size == 0:
            return np.empty((0, 5)), np.empty((0, 5, 2))

        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(self.input_size[1]) / self.input_size[0]
        if im_ratio > model_ratio:
            new_height = self.input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = self.input_size[0]
            new_height = int(new_width * im_ratio)

        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        input_tensor = (det_img.astype(np.float32) - 127.5) / 128.0
        input_tensor = input_tensor.transpose(2, 0, 1)[None, ...]

        outputs = self.session.run(None, {self.input_name: input_tensor})

        scores_list = []
        bboxes_list = []
        kpss_list = []

        input_height, input_width = self.input_size[1], self.input_size[0]

        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outputs[idx]
            bbox_preds = outputs[idx + self.fmc] * stride
            kps_preds = outputs[idx + self.fmc * 2] * stride

            anchor_centers = self._cached_anchor_centers[stride]

            pos_inds = np.where(scores >= self.conf_thresh)[0]
            if len(pos_inds) > 0:
                pos_scores = scores[pos_inds]
                pos_bbox = bbox_preds[pos_inds]
                pos_kps = kps_preds[pos_inds]
                pos_centers = anchor_centers[pos_inds]

                x1 = (pos_centers[:, 0] - pos_bbox[:, 0]) / det_scale
                y1 = (pos_centers[:, 1] - pos_bbox[:, 1]) / det_scale
                x2 = (pos_centers[:, 0] + pos_bbox[:, 2]) / det_scale
                y2 = (pos_centers[:, 1] + pos_bbox[:, 3]) / det_scale
                bboxes = np.stack([x1, y1, x2, y2], axis=-1)

                kpss = pos_kps.reshape(-1, 5, 2) + pos_centers[:, None, :]
                kpss /= det_scale

                scores_list.append(pos_scores)
                bboxes_list.append(bboxes)
                kpss_list.append(kpss)

        if not scores_list:
            return np.empty((0, 5)), np.empty((0, 5, 2))

        scores = np.vstack(scores_list).flatten()
        bboxes = np.vstack(bboxes_list)
        kpss = np.vstack(kpss_list)

        keep = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), self.conf_thresh, self.nms_thresh)
        if len(keep) > 0:
            keep = keep.flatten()
            return np.hstack([bboxes[keep], scores[keep, None]]), kpss[keep]
        return np.empty((0, 5)), np.empty((0, 5, 2))


class ArcFaceRecognizer:
    """
    基于 ONNXRuntime 的 MobileFaceNet (ArcFace 512维) 人脸特征提取器
    包含 5 点相似变换仿射对齐至 112x112
    """
    def __init__(
        self,
        model_path: str = "models/insightface/w600k_mbf.onnx",
        use_trt: bool = False
    ):
        self.model_path = model_path
        self.session = None
        self.backend = "unknown"

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
                print(f"[ArcFaceRecognizer] 成功加载人脸特征提取模型 ({self.backend}): {self.model_path}")
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
                        self.backend = "onnxruntime"
                        print(f"[ArcFaceRecognizer] 成功加载人脸特征提取 ONNX 模型: {model_path} (threads={sess_opts.intra_op_num_threads})")
                    except Exception as ex:
                        print(f"[ArcFaceRecognizer] 初始化 ArcFace 失败: {ex}")
                else:
                    print(f"[ArcFaceRecognizer] 初始化模型失败: {e}")

    @staticmethod
    def align_face(img: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """根据 5 个关键点仿射对齐到标准 112x112"""
        # 计算相似变换矩阵
        M, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_REFERENCE_POINTS, method=cv2.RANSAC)
        if M is None:
            # 降级裁剪
            return cv2.resize(img, (112, 112))
        aligned = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
        return aligned

    def extract_feature(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        """从 112x112 对齐人脸提取 512 维归一化特征"""
        if self.session is None or aligned_face_bgr is None:
            pseudo = np.random.randn(512).astype(np.float32)
            return pseudo / np.linalg.norm(pseudo)

        face_rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)
        tensor = (face_rgb.astype(np.float32) - 127.5) / 127.5
        tensor = tensor.transpose(2, 0, 1)[None, ...]

        feat = self.session.run(None, {self.input_name: tensor})[0][0]
        norm = np.linalg.norm(feat)
        if norm > 1e-7:
            feat = feat / norm
        return feat.astype(np.float32)


class FaceIDPipeline:
    """
    工业级 FaceID 完整调度流水线
    集成:
    - 预注册车主/营友白名单特征库
    - 细粒度状态流转 (PENDING -> UNAVAILABLE / NO_MATCH / MATCH_CANDIDATE -> VERIFIED)
    - 异步机会主义抓拍与限频重试
    - 多帧确认与异步版本号防竞争绑定
    """
    def __init__(
        self,
        detector: Optional[SCRFDFaceDetector] = None,
        recognizer: Optional[ArcFaceRecognizer] = None,
        fqa: Optional[FaceQualityAssessor] = None,
        match_threshold: float = 0.65,          # ArcFace 余弦相似度白名单阈值
        verify_confirm_frames: int = 2,          # 连续 2 次有效匹配才升至 VERIFIED
        retry_interval_sec: float = 0.8,         # UNAVAILABLE 时重试间隔
        use_trt: bool = False
    ):
        det_path = "models/insightface/det_500m.engine" if (use_trt and os.path.exists("models/insightface/det_500m.engine")) else "models/insightface/det_500m.onnx"
        rec_path = "models/insightface/w600k_mbf.engine" if (use_trt and os.path.exists("models/insightface/w600k_mbf.engine")) else "models/insightface/w600k_mbf.onnx"
        self.detector = detector or SCRFDFaceDetector(model_path=det_path, use_trt=use_trt)
        self.recognizer = recognizer or ArcFaceRecognizer(model_path=rec_path, use_trt=use_trt)
        self.fqa = fqa or FaceQualityAssessor()
        self.match_threshold = match_threshold
        self.verify_confirm_frames = verify_confirm_frames
        self.retry_interval_sec = retry_interval_sec
        self.profiler: Optional[Any] = None

        # 白名单底库: {identity_name: 512-dim embedding}
        self.whitelist_db: Dict[str, np.ndarray] = {}

        # 轨迹运行状态缓存: {track_id: {"last_check_time": float, "candidate_name": str, "candidate_count": int}}
        self.track_face_context: Dict[int, dict] = {}

    def register_whitelist(self, name: str, face_img_bgr: np.ndarray) -> bool:
        """从底库照片注册白名单人脸"""
        bboxes, kpss = self.detector.detect(face_img_bgr)
        if len(bboxes) == 0:
            print(f"[FaceIDPipeline] 注册失败: 未检测到有效人脸 ({name})")
            return False

        # 取置信度最高的人脸
        best_idx = int(np.argmax(bboxes[:, 4]))
        aligned = self.recognizer.align_face(face_img_bgr, kpss[best_idx])
        feature = self.recognizer.extract_feature(aligned)
        self.whitelist_db[name] = feature
        print(f"[FaceIDPipeline] 成功注册白名单: {name}")
        return True

    def register_whitelist_feature(self, name: str, feature: np.ndarray):
        """直接注册已提取好的 512 维特征向量"""
        norm = np.linalg.norm(feature)
        if norm > 1e-7:
            feature = feature / norm
        self.whitelist_db[name] = feature.astype(np.float32)

    def process_target_face(
        self,
        frame: np.ndarray,
        target: TrackedEntity,
        current_time: float
    ) -> FaceIDState:
        """
        对被追踪目标执行机会主义 FaceID 状态机研判
        :param frame: 原始全景帧
        :param target: 目标实体 (包含 bbox 与 face_state)
        :param current_time: 当前时钟
        :return: 更新后的 FaceIDState
        """
        tid = target.track_id

        # 1. 如果已确认白名单，且轨迹平稳，保持 VERIFIED，不再浪费算力
        if target.face_state == FaceIDState.VERIFIED and target.is_whitelisted:
            return FaceIDState.VERIFIED

        # 2. 状态机上下文初始化
        if tid not in self.track_face_context:
            self.track_face_context[tid] = {
                "last_check_time": 0.0,
                "candidate_name": None,
                "candidate_count": 0,
                "last_version": target.association_version
            }

        ctx = self.track_face_context[tid]

        # 检查异步防踩内存关联版本号
        if target.association_version != ctx["last_version"]:
            ctx["candidate_name"] = None
            ctx["candidate_count"] = 0
            ctx["last_version"] = target.association_version
            target.face_state = FaceIDState.RECHECK

        # 3. 机会主义频控检查 (未匹配或不可用时限频重试，避免每帧重复调用 SCRFD 挤占算力)
        if target.face_state in [FaceIDState.PENDING, FaceIDState.UNAVAILABLE, FaceIDState.NO_MATCH]:
            if ctx["last_check_time"] > 0 and (current_time - ctx["last_check_time"] < self.retry_interval_sec):
                return target.face_state

        ctx["last_check_time"] = current_time

        # 4. 从人体框上半身提取人脸区域
        # 人脸通常位于人体边界框上部 1/3 ~ 1/2 区域
        H, W = frame.shape[:2]
        bx, by, bw, bh = int(target.bbox.x), int(target.bbox.y), int(target.bbox.w), int(target.bbox.h)
        head_x1 = max(0, bx)
        head_y1 = max(0, by)
        head_x2 = min(W, bx + bw)
        head_y2 = min(H, by + int(bh * 0.55))

        if head_x2 <= head_x1 or head_y2 <= head_y1:
            target.face_state = FaceIDState.UNAVAILABLE
            return FaceIDState.UNAVAILABLE

        head_crop = frame[head_y1:head_y2, head_x1:head_x2]

        # 5. 人脸检测
        if self.profiler:
            det_s = getattr(self.detector, 'session', None)
            with self.profiler.profile_model("SCRFD-500M", condition=f"ID:{tid} 上半身人脸定位 (4m以内)", session=det_s):
                bboxes, kpss = self.detector.detect(head_crop)
        else:
            bboxes, kpss = self.detector.detect(head_crop)

        if len(bboxes) == 0:
            # 未检出人脸 (背对相机、低头、遮挡)
            target.face_state = FaceIDState.UNAVAILABLE
            return FaceIDState.UNAVAILABLE

        # 选置信度最高的人脸
        best_idx = int(np.argmax(bboxes[:, 4]))
        face_box = bboxes[best_idx, :4]
        face_kps = kpss[best_idx]

        # 6. FQA 质量评估门禁
        is_qual, q_score, reason = self.fqa.evaluate_quality(head_crop, face_box, face_kps)
        if not is_qual:
            target.face_state = FaceIDState.UNAVAILABLE
            return FaceIDState.UNAVAILABLE

        # 7. 人脸对齐与特征提取
        aligned_face = self.recognizer.align_face(head_crop, face_kps)
        if self.profiler:
            rec_s = getattr(self.recognizer, 'session', None)
            with self.profiler.profile_model("ArcFace-MobileFaceNet", condition=f"ID:{tid} FQA通过白名单特征提取", session=rec_s):
                feat = self.recognizer.extract_feature(aligned_face)
        else:
            feat = self.recognizer.extract_feature(aligned_face)

        # 8. 匹配白名单底库
        if not self.whitelist_db:
            target.face_state = FaceIDState.NO_MATCH
            return FaceIDState.NO_MATCH

        best_name = None
        best_sim = -1.0
        for name, db_feat in self.whitelist_db.items():
            sim = float(np.dot(feat, db_feat))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        # 9. 状态流转判定
        if best_sim >= self.match_threshold and best_name is not None:
            if ctx["candidate_name"] == best_name:
                ctx["candidate_count"] += 1
            else:
                ctx["candidate_name"] = best_name
                ctx["candidate_count"] = 1

            if ctx["candidate_count"] >= self.verify_confirm_frames:
                # 多次确认为同一人，升至 VERIFIED 确权
                target.face_state = FaceIDState.VERIFIED
                target.face_id_name = best_name
                target.is_whitelisted = True
            else:
                target.face_state = FaceIDState.MATCH_CANDIDATE
                target.face_id_name = best_name
        else:
            # 清晰可见，但未匹配白名单 (判定为陌生人)
            ctx["candidate_name"] = None
            ctx["candidate_count"] = 0
            target.face_state = FaceIDState.NO_MATCH
            target.face_id_name = None

        return target.face_state
