"""
S1 初筛目标检测引擎抽象与实现 (camping_guard/perception/detector.py)
支持 ONNXRuntime 模型推理 (YOLO-Nano / YOLO11) 与无权重下的合成/回放检测器
输出类别规范: 'person', 'car', 'small_animal', 'large_beast', 'fire', 'smoke'
"""

import abc
import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

class BaseObjectDetector(abc.ABC):
    @abc.abstractmethod
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.35) -> List[Dict]:
        """
        对输入画面进行目标检测
        :return: [{'bbox': (x, y, w, h), 'category': str, 'conf': float}, ...]
        """
        pass


class ONNXObjectDetector(BaseObjectDetector):
    """基于 ONNXRuntime / TensorRT Engine 的车载轻量 YOLO 检测引擎"""
    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.25,
        use_trt: bool = False
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.session = None
        self.backend = "unknown"

        # COCO 类别映射到露营守卫规范类别
        self.coco_to_guard_map = {
            0: 'person',
            2: 'car', 5: 'car', 7: 'car', # 轿车, 公交车, 卡车
            14: 'small_animal', 15: 'small_animal', 16: 'small_animal', # 鸟, 猫, 狗
            17: 'small_animal', 18: 'small_animal', 19: 'small_animal', # 马, 羊, 牛
            21: 'large_beast', # 熊 (大型猛兽)
            80: 'fire', 81: 'smoke' # 扩展类
        }

        # 专有鱼眼数据集 (Fisheye8K) 类别映射: {0: 'bus', 1: 'bike', 2: 'car', 3: 'pedestrian', 4: 'truck'}
        self.fisheye_to_guard_map = {
            3: 'person',  # pedestrian (鱼眼行人)
            2: 'car',     # car
            0: 'car',     # bus
            4: 'car',     # truck
            1: 'car'      # bike
        }

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
                print(f"[ObjectDetector] 成功载入模型 ({self.backend}): {self.model_path} (置信度阈值: {self.conf_threshold:.2f})")
            except Exception as e:
                # 兼容回退原生 ONNXRuntime
                if os.path.exists(model_path) and not is_engine:
                    try:
                        import onnxruntime as ort
                        providers = ['CPUExecutionProvider']
                        sess_opts = ort.SessionOptions()
                        sess_opts.intra_op_num_threads = min(4, os.cpu_count() or 4)
                        sess_opts.inter_op_num_threads = 1
                        self.session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
                        self.input_name = self.session.get_inputs()[0].name
                        self.backend = "onnxruntime"
                        print(f"[ONNXObjectDetector] 成功载入模型: {model_path} (默认置信度阈值: {self.conf_threshold:.2f})")
                    except Exception as ex:
                        print(f"[ONNXObjectDetector] 初始化 ONNX 失败: {ex}")
                else:
                    print(f"[ObjectDetector] 初始化模型失败: {e}")

    def letterbox(self, img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def detect(self, frame: np.ndarray, conf_threshold: Optional[float] = None) -> List[Dict]:
        if self.session is None or frame is None:
            return []

        eff_conf = conf_threshold if conf_threshold is not None else getattr(self, 'conf_threshold', 0.25)

        h0, w0 = frame.shape[:2]
        img, ratio, (dw, dh) = self.letterbox(frame, new_shape=self.input_size)
        img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # 模型前向推理
        outputs = self.session.run(None, {self.input_name: img})
        preds = outputs[0] # 通常为 [1, 84, 8400] 或 [1, num_boxes, 85]

        # 解析输出
        if preds.shape[1] < preds.shape[2]:
            preds = np.transpose(preds, (0, 2, 1))

        preds = preds[0] # [num_candidates, features]
        scores = preds[:, 4:]
        confidences = np.max(scores, axis=1)
        mask = confidences >= eff_conf

        if not np.any(mask):
            return []

        boxes = preds[mask, :4]
        class_ids = np.argmax(scores[mask], axis=1)
        confidences = confidences[mask]

        # 自适应根据输出通道维度判断分类字典 (COCO 80类 vs Fisheye8K 5类)
        is_fisheye_5cls = (scores.shape[1] == 5)
        cls_map = self.fisheye_to_guard_map if is_fisheye_5cls else self.coco_to_guard_map

        # 转换为 (x, y, w, h) 并做坐标还原
        detections = []
        for box, cid, score in zip(boxes, class_ids, confidences):
            cx, cy, bw, bh = box
            # 还原 letterbox
            x = (cx - bw / 2 - dw) / ratio
            y = (cy - bh / 2 - dh) / ratio
            w = bw / ratio
            h = bh / ratio

            x = max(0, min(w0 - 1, int(x)))
            y = max(0, min(h0 - 1, int(y)))
            w = max(5, min(w0 - x, int(w)))
            h = max(5, min(h0 - y, int(h)))

            category = cls_map.get(int(cid), 'unknown')
            if category != 'unknown':
                detections.append({
                    'bbox': (x, y, w, h),
                    'category': category,
                    'conf': float(score)
                })

        return self._apply_nms(detections, iou_threshold=0.45, score_threshold=eff_conf)

    def _apply_nms(self, dets: List[Dict], iou_threshold: float = 0.45, score_threshold: float = 0.2) -> List[Dict]:
        if not dets:
            return []
        boxes = [d['bbox'] for d in dets]
        scores = [d['conf'] for d in dets]
        indices = cv2.dnn.NMSBoxes(
            bboxes=[[x, y, w, h] for x, y, w, h in boxes],
            scores=scores,
            score_threshold=score_threshold,
            nms_threshold=iou_threshold
        )
        if len(indices) == 0:
            return []
        # 兼容不同 OpenCV 版本返回格式 (1D 数组或 2D 数组)
        flat_indices = np.array(indices).flatten().tolist()
        return [dets[int(i)] for i in flat_indices]


class SyntheticDetector(BaseObjectDetector):
    """用于自动化仿真与外部注入测试的合成目标检测器"""
    def __init__(self):
        self.injected_detections: List[Dict] = []

    def set_injected_detections(self, dets: List[Dict]):
        self.injected_detections = list(dets)

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.35) -> List[Dict]:
        return [d for d in self.injected_detections if d.get('conf', 1.0) >= conf_threshold]


class TRTObjectDetector(ONNXObjectDetector):
    """TensorRT 专用 YOLO 目标检测引擎 (适用 Jetson Orin)"""
    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.25
    ):
        super().__init__(model_path=model_path, input_size=input_size, conf_threshold=conf_threshold, use_trt=True)

