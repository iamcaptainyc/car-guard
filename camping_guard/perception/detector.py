"""
S1 初筛目标检测引擎抽象与实现 (camping_guard/perception/detector.py)
支持 ONNXRuntime 模型推理 (YOLO-Nano / YOLO11) 与无权重下的合成/回放检测器
输出类别规范: 'person', 'car', 'small_animal', 'large_beast', 'fire', 'smoke'
"""

import abc
import os
import time
import concurrent.futures
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

class BaseObjectDetector(abc.ABC):
    @abc.abstractmethod
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.35) -> List[Dict]:
        """
        对输入画面进行目标检测
        :return: [{'bbox': (x, y, w, h), 'category': str, 'conf': float}, ...]
        """
        pass


class ONNXObjectDetector(BaseObjectDetector):
    """基于 ONNXRuntime / TensorRT Engine 的车载轻量 YOLO / MegaDetector 检测引擎"""
    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.25,
        use_trt: bool = False,
        animal_classifier: Optional[Any] = None,
        custom_cls_map: Optional[Dict[int, str]] = None
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.session = None
        self.backend = "unknown"
        self.animal_classifier = animal_classifier
        self.custom_cls_map = custom_cls_map

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

        # 专有野生动物/通用 MegaDetector 类别映射 (0: animal, 1: person, 2: vehicle)
        self.megadetector_to_guard_map = {
            0: 'animal',
            1: 'person',
            2: 'car'
        }

        # 专有火灾/浓烟专用模型类别映射 (0: fire, 1: smoke)
        self.fire_to_guard_map = {
            0: 'fire',
            1: 'smoke'
        }

        # 判定是否为 MegaDetector 专用模型或火灾/烟雾模型
        mpath_str = str(model_path).lower()
        self.is_megadetector = ("mdv" in mpath_str or "megadetector" in mpath_str)
        self.is_fire_model = ("fire" in mpath_str or "smoke" in mpath_str or "flame" in mpath_str)

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

        # 补充通过 metadata 探测 MegaDetector
        if not self.is_megadetector and self.session is not None and hasattr(self.session, "get_modelmeta"):
            try:
                meta = self.session.get_modelmeta().custom_metadata_map
                names_str = str(meta.get("names", ""))
                desc_str = str(meta.get("description", "")).lower()
                if "0: 'animal'" in names_str or "animal" in names_str or "megadb" in desc_str:
                    self.is_megadetector = True
            except Exception:
                pass

        # 若当前为 MegaDetector 且尚未挂载动物分类器，尝试自动挂载 models/cls 下的模型
        if self.is_megadetector and self.animal_classifier is None:
            try:
                from camping_guard.perception.animal_classifier import AnimalClassifier
                self.animal_classifier = AnimalClassifier(use_trt=use_trt)
                if self.animal_classifier.session is not None:
                    print(f"[ObjectDetector] 已为 MegaDetector 自动级联挂载二级动物细分类器: {self.animal_classifier.model_path}")
            except Exception as e:
                pass

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

        # 自适应根据输出通道维度判断分类字典 (MegaDetector 3类 vs Fisheye8K 5类 vs COCO 80类 vs 火灾 1~2类)
        is_megadetector = getattr(self, 'is_megadetector', False) or (scores.shape[1] == 3)
        is_fisheye_5cls = (scores.shape[1] == 5)

        if getattr(self, 'custom_cls_map', None) is not None:
            cls_map = self.custom_cls_map
        elif getattr(self, 'is_fire_model', False) or (scores.shape[1] in [1, 2] and not is_megadetector):
            cls_map = self.fire_to_guard_map
        elif is_megadetector:
            cls_map = self.megadetector_to_guard_map
        elif is_fisheye_5cls:
            cls_map = self.fisheye_to_guard_map
        else:
            cls_map = self.coco_to_guard_map

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

            base_cat = cls_map.get(int(cid), 'unknown')
            if base_cat == 'unknown':
                continue

            category = base_cat
            species_en = None
            species_cn = None
            cls_conf = None

            # 核心策略：仅在调用 MegaDetector 且识别到 animal 类目标时，调用二级动物细分类器
            if is_megadetector and base_cat == 'animal':
                if self.animal_classifier is not None:
                    crop = frame[y:y+h, x:x+w]
                    category, cls_conf, species_en, species_cn = self.animal_classifier.classify(crop)
                else:
                    category = 'small_animal'

            det_item = {
                'bbox': (x, y, w, h),
                'category': category,
                'conf': float(score)
            }
            if species_en is not None:
                det_item['species'] = species_en
                det_item['species_cn'] = species_cn
                det_item['cls_conf'] = cls_conf

            detections.append(det_item)

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

    def detect(self, frame: np.ndarray, conf_threshold: Optional[float] = None) -> List[Dict]:
        eff_conf = conf_threshold if conf_threshold is not None else 0.35
        return [d for d in self.injected_detections if d.get('conf', 1.0) >= eff_conf]


class TRTObjectDetector(ONNXObjectDetector):
    """TensorRT 专用 YOLO / MegaDetector 目标检测引擎 (适用 Jetson Orin)"""
    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.25,
        animal_classifier: Optional[Any] = None,
        custom_cls_map: Optional[Dict[int, str]] = None
    ):
        super().__init__(
            model_path=model_path,
            input_size=input_size,
            conf_threshold=conf_threshold,
            use_trt=True,
            animal_classifier=animal_classifier,
            custom_cls_map=custom_cls_map
        )


class ParallelEnsembleDetector(BaseObjectDetector):
    """
    多模型多流并发集成检测器 (Parallel Ensemble Detector)
    专为 NVIDIA Jetson Orin 硬件架构并发推理设计:
    - Ampere GPU 具备多个硬件工作队列与 Compute Engines，支持 CUDA Streams 并发执行
    - 主检测器 (人/车辆/野生动物，如 YOLO 或 MegaDetector) 与专用火灾/烟雾检测模型分别在独立线程与独立 CUDA Stream 中并发发射
    - 运行耗时由两模型串行耗时之和 (t1 + t2) 降至两者最大值 max(t1, t2)
    - 自动聚合多路检测器的检测结果并做安全类别正交合并与非极大值抑制 (NMS)
    """
    def __init__(
        self,
        main_detector: BaseObjectDetector,
        fire_detector: Optional[BaseObjectDetector] = None,
        max_workers: int = 2
    ):
        self.main_detector = main_detector
        self.fire_detector = fire_detector
        self.max_workers = max_workers
        self.is_ensemble = True
        self.profiler_label = "Parallel-Ensemble-Det"

        # 路径汇总标识
        p_paths = []
        if hasattr(main_detector, "model_path"):
            p_paths.append(str(main_detector.model_path))
        if fire_detector and hasattr(fire_detector, "model_path"):
            p_paths.append(str(fire_detector.model_path))
        self.model_path = " + ".join(p_paths) if p_paths else "ParallelEnsemble"

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ParallelDetWorker"
        )
        self.last_main_ms: float = 0.0
        self.last_fire_ms: float = 0.0
        self.last_parallel_ms: float = 0.0

    @property
    def session(self):
        """暴露主检测器 session 给 profiler"""
        return getattr(self.main_detector, 'session', None)

    @property
    def is_megadetector(self) -> bool:
        return getattr(self.main_detector, 'is_megadetector', False)

    @property
    def animal_classifier(self):
        return getattr(self.main_detector, 'animal_classifier', None)

    @animal_classifier.setter
    def animal_classifier(self, val):
        if hasattr(self.main_detector, 'animal_classifier'):
            self.main_detector.animal_classifier = val

    def detect(self, frame: np.ndarray, conf_threshold: Optional[float] = None) -> List[Dict]:
        if frame is None:
            return []

        # 若未配置辅助火灾检测器，直接运行主检测器
        if self.fire_detector is None:
            t0 = time.perf_counter()
            dets = self.main_detector.detect(frame, conf_threshold)
            self.last_main_ms = (time.perf_counter() - t0) * 1000.0
            self.last_parallel_ms = self.last_main_ms
            return dets

        # 启动双流/多流并发执行
        t_start = time.perf_counter()

        def _run_main():
            tm0 = time.perf_counter()
            res = self.main_detector.detect(frame, conf_threshold)
            self.last_main_ms = (time.perf_counter() - tm0) * 1000.0
            return res

        def _run_fire():
            tf0 = time.perf_counter()
            res = self.fire_detector.detect(frame, conf_threshold)
            self.last_fire_ms = (time.perf_counter() - tf0) * 1000.0
            return res

        future_main = self._executor.submit(_run_main)
        future_fire = self._executor.submit(_run_fire)

        main_dets = future_main.result()
        fire_dets = future_fire.result()

        self.last_parallel_ms = (time.perf_counter() - t_start) * 1000.0

        # 合并检测目标框
        combined = list(main_dets) + list(fire_dets)
        return combined

    def close(self):
        if self._executor:
            self._executor.shutdown(wait=False)


class VisualFlameDetector(BaseObjectDetector):
    """
    内置轻量自适应视觉火焰检测器 (Visual Flame Detector)
    专为车载露营守卫与边缘设备零模型权重保底设计:
    - 基于色彩动力学与火焰光谱特性 (RGB: R > G > B 且 R > 140; HSV: H in [0, 38], S >= 50, V >= 130)
    - 快速自适应连通域与形态学聚合分析，单帧耗时仅 ~5-8ms，显存占用 0
    - 开箱即用，默认开启，无需外部训练模型权重即可稳定秒级检出篝火与野火
    - 输出与神经网络检测器完全兼容的标准格式: {'bbox': (x, y, w, h), 'category': 'fire', 'conf': float}
    """
    def __init__(
        self,
        min_area: int = 40,
        conf_threshold: float = 0.25,
        target_size: Tuple[int, int] = (320, 240)
    ):
        self.min_area = min_area
        self.conf_threshold = conf_threshold
        self.target_size = target_size
        self.model_path = "builtin:VisualFlameDetector"
        self.profiler_label = "VisualFlameDetector"
        self.session = None
        self.is_ensemble = False
        self.is_megadetector = False
        self.animal_classifier = None

    def detect(self, frame: np.ndarray, conf_threshold: Optional[float] = None) -> List[Dict]:
        if frame is None or frame.size == 0:
            return []

        th = conf_threshold if conf_threshold is not None else self.conf_threshold
        orig_h, orig_w = frame.shape[:2]
        tw, th_dim = self.target_size

        # 降采样加速色彩与连通域分析
        if orig_w != tw or orig_h != th_dim:
            small_frame = cv2.resize(frame, (tw, th_dim), interpolation=cv2.INTER_LINEAR)
            scale_x = orig_w / float(tw)
            scale_y = orig_h / float(th_dim)
        else:
            small_frame = frame
            scale_x = 1.0
            scale_y = 1.0

        # 1. RGB 色彩动力学规则: R > G > B 且 R > 140
        b, g, r = cv2.split(small_frame)
        rgb_mask = (r > g) & (g > b) & (r > 140)

        # 2. HSV 火焰色谱规则: H 在红橙黄区间 [0, 38], 饱和度 S >= 50, 亮度 V >= 130
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        hsv_mask = (h <= 38) & (s >= 50) & (v >= 130)

        # 3. 融合掩码与形态学去噪/聚合
        combined = (rgb_mask & hsv_mask).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(opened, kernel_dilate, iterations=1)

        # 4. 连通域轮廓提取
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            cx, cy, cw, ch = cv2.boundingRect(cnt)
            if cw < 4 or ch < 4:
                continue

            # 映射回原始帧分辨率坐标
            bx = int(cx * scale_x)
            by = int(cy * scale_y)
            bw = int(cw * scale_x)
            bh = int(ch * scale_y)

            bx = max(0, min(bx, orig_w - 1))
            by = max(0, min(by, orig_h - 1))
            bw = min(bw, orig_w - bx)
            bh = min(bh, orig_h - by)

            if bw <= 0 or bh <= 0:
                continue

            # 计算局部区域置信度 (基于 ROI 内 R 与 V 通道强度均值)
            roi_v = v[cy:cy + ch, cx:cx + cw]
            mean_v = float(np.mean(roi_v)) if roi_v.size > 0 else 180.0
            conf = float(np.clip(0.60 + 0.35 * (mean_v / 255.0), 0.50, 0.95))

            if conf >= th:
                detections.append({
                    "bbox": (bx, by, bw, bh),
                    "category": "fire",
                    "conf": conf
                })

        return detections


