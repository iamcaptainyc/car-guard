"""
二级动物细分类引擎 (camping_guard/perception/animal_classifier.py)
基于 MobileNetV3-Large 深度分类模型 (224x224 输入, ImageNet-1K)
设计模式：与 MegaDetector 严格级联触发——仅在检测器检出 animal 类目标时按需调用，
细化识别具体动物物种 (熊、狼、野猪、犬、猫、鸟、鹿等)，并精准区分为 large_beast (大型猛兽) 或 small_animal (小型无害动物)
"""

import os
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any


# ImageNet-1K 动物类别映射字典: class_id -> (category, species_en, species_cn)
IMAGENET_ANIMAL_MAP: Dict[int, Tuple[str, str, str]] = {
    # 猛兽 1: 熊类 (Bears)
    294: ('large_beast', 'brown_bear', '棕熊'),
    295: ('large_beast', 'american_black_bear', '美洲黑熊'),
    296: ('large_beast', 'polar_bear', '北极熊'),
    297: ('large_beast', 'sloth_bear', '懒熊'),

    # 猛兽 2: 大型猫科 (Big Cats)
    284: ('large_beast', 'cougar', '美洲狮'),
    286: ('large_beast', 'leopard', '花豹'),
    287: ('large_beast', 'snow_leopard', '雪豹'),
    288: ('large_beast', 'jaguar', '美洲豹'),
    289: ('large_beast', 'lion', '狮子'),
    290: ('large_beast', 'tiger', '老虎'),
    291: ('large_beast', 'cheetah', '猎豹'),

    # 猛兽 3: 野生犬科 / 鬣狗 (Wild Canines & Hyenas)
    269: ('large_beast', 'timber_wolf', '灰狼'),
    270: ('large_beast', 'white_wolf', '白狼'),
    271: ('large_beast', 'red_wolf', '红狼'),
    272: ('large_beast', 'coyote', '郊狼'),
    273: ('large_beast', 'dingo', '澳洲野犬'),
    274: ('large_beast', 'dhole', '豺'),
    275: ('large_beast', 'african_hunting_dog', '非洲野犬'),
    276: ('large_beast', 'hyena', '鬣狗'),

    # 猛兽 4: 具攻击性大型野生动物 (Wild Boar & Large Ungulates & Reptiles)
    341: ('large_beast', 'wild_boar', '野猪'),
    342: ('large_beast', 'warthog', '疣猪'),
    343: ('large_beast', 'hippopotamus', '河马'),
    345: ('large_beast', 'ox', '公牛'),
    346: ('large_beast', 'water_buffalo', '水牛'),
    347: ('large_beast', 'bison', '野牛'),
    385: ('large_beast', 'indian_elephant', '亚洲象'),
    386: ('large_beast', 'african_elephant', '非洲象'),
    49:  ('large_beast', 'alligator', '短吻鳄'),
    50:  ('large_beast', 'crocodile', '鳄鱼'),

    # 小型/无害 1: 家犬 (Domestic Dogs - 151 至 268)
    **{i: ('small_animal', 'dog', '犬类') for i in range(151, 269)},

    # 小型/无害 2: 狐狸 (Foxes)
    277: ('small_animal', 'red_fox', '赤狐'),
    278: ('small_animal', 'kit_fox', '敏狐'),
    279: ('small_animal', 'arctic_fox', '北极狐'),
    280: ('small_animal', 'grey_fox', '灰狐'),

    # 小型/无害 3: 猫类 (Domestic & Small Cats)
    281: ('small_animal', 'tabby_cat', '虎斑猫'),
    282: ('small_animal', 'tiger_cat', '虎皮猫'),
    283: ('small_animal', 'persian_cat', '波斯猫'),
    285: ('small_animal', 'lynx', '猞猁'),

    # 小型/无害 4: 兔类与啮齿动物 (Rabbits & Rodents - 330 至 340)
    **{i: ('small_animal', 'rabbit_rodent', '野兔/啮齿类') for i in range(330, 341)},

    # 小型/无害 5: 鹿类与羚羊 (Deer & Gazelle - 348 至 357)
    **{i: ('small_animal', 'deer_antelope', '鹿/羚羊') for i in range(348, 358)},

    # 小型/无害 6: 鸟类 (Birds - 80 至 150)
    **{i: ('small_animal', 'bird', '鸟禽') for i in range(80, 151)},

    # 小型/无害 7: 其它小型哺乳类 (Badgers, Otters, Panda etc.)
    388: ('small_animal', 'giant_panda', '大熊猫'),
    **{i: ('small_animal', 'small_mammal', '小型哺乳动物') for i in range(358, 385)},
}


class AnimalClassifier:
    """
    基于 ONNXRuntime / TensorRT Engine 的二级动物细分类器
    输入: 动物候选区域裁剪图 (BGR numpy array)
    输出: (category, confidence, species_en, species_cn)
         category 规范为 'large_beast' 或 'small_animal'
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = (224, 224),
        use_trt: bool = False
    ):
        self.input_size = input_size
        self.use_trt = use_trt
        self.session = None
        self.backend = "unknown"
        self.model_path = model_path
        self.last_preprocess_ms: float = 0.0
        self.last_gpu_compute_ms: float = 0.0
        self.last_postprocess_ms: float = 0.0

        # ImageNet 经典归一化参数 (RGB 空间)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

        # 1. 自动探测候选模型
        if self.model_path is None or not os.path.exists(self.model_path):
            self.model_path = self._discover_model_path(use_trt=self.use_trt)

        # 2. 载入推理引擎
        if self.model_path and os.path.exists(self.model_path):
            self._load_model(self.model_path, use_trt=self.use_trt)
        else:
            print("[AnimalClassifier] 未找到分类模型文件，将使用智能规则后备分类器")

    def _discover_model_path(self, use_trt: bool = False) -> Optional[str]:
        """按优先级搜索 models/cls/ 目录下的 .engine 或 .onnx 模型"""
        cands = []
        if use_trt:
            cands.extend([
                "models/cls/mobilenetv3-large.engine",
                "models/cls/mobilenetv3.engine",
            ])
            # 扫描 models/cls 目录下所有 .engine
            cls_dir = Path("models/cls")
            if cls_dir.is_dir():
                cands.extend([str(p) for p in cls_dir.glob("*.engine")])

        cands.extend([
            "models/cls/mobilenetv3-large.onnx",
            "models/cls/mobilenetv3.onnx",
        ])
        cls_dir = Path("models/cls")
        if cls_dir.is_dir():
            cands.extend([str(p) for p in cls_dir.glob("*.onnx")])

        for c in cands:
            if os.path.exists(c):
                return c
        return None

    def _load_model(self, model_path: str, use_trt: bool = False):
        """挂载 TensorRT Engine 或 ONNXRuntime 会话"""
        is_engine = str(model_path).lower().endswith(".engine")
        try:
            from camping_guard.perception.trt_engine import load_inference_backend
            self.session, self.backend = load_inference_backend(model_path, prefer_tensorrt=(use_trt or is_engine))
            if hasattr(self.session, "engine_path"):
                self.model_path = self.session.engine_path
            elif hasattr(self.session, "model_path"):
                self.model_path = self.session.model_path
            self.input_name = self.session.get_inputs()[0].name
            print(f"[AnimalClassifier] 成功载入模型 ({self.backend}): {self.model_path}")
        except Exception as e:
            if os.path.exists(model_path) and not is_engine:
                try:
                    import onnxruntime as ort
                    sess_opts = ort.SessionOptions()
                    sess_opts.intra_op_num_threads = min(2, os.cpu_count() or 2)
                    sess_opts.inter_op_num_threads = 1
                    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    self.session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=['CPUExecutionProvider'])
                    self.input_name = self.session.get_inputs()[0].name
                    self.backend = "onnxruntime"
                    print(f"[AnimalClassifier] 成功载入 ONNX 模型: {model_path}")
                except Exception as ex:
                    print(f"[AnimalClassifier] 载入 ONNX 模型异常: {ex}")
            else:
                print(f"[AnimalClassifier] 初始化模型失败: {e}")

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        对裁剪区域执行 MobileNetV3 标准预处理:
        Resize -> BGR转RGB -> 归一化 -> 减均值除方差 -> 转CHW
        """
        h, w = self.input_size
        resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        norm = rgb.astype(np.float32) / 255.0
        chw = norm.transpose(2, 0, 1) # (3, H, W)
        tensor = np.expand_dims(chw, axis=0) # (1, 3, H, W)
        tensor = (tensor - self.mean) / self.std
        return np.ascontiguousarray(tensor, dtype=np.float32)

    def classify(self, crop: np.ndarray) -> Tuple[str, float, str, str]:
        """
        对输入的动物候选 BBox 进行细粒度分类
        :param crop: BBox 裁剪图像 (BGR 格式)
        :return: (category, confidence, species_en, species_cn)
                 例如: ('large_beast', 0.92, 'brown_bear', '棕熊')
                 或:   ('small_animal', 0.88, 'dog', '犬类')
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return ('small_animal', 0.0, 'unknown', '未知动物')

        if self.session is None:
            # 兼容模型缺失时的启发式退避
            return ('small_animal', 0.5, 'unclassified_animal', '通用动物')

        try:
            t_pre0 = time.perf_counter()
            inp = self.preprocess(crop)
            self.last_preprocess_ms = (time.perf_counter() - t_pre0) * 1000.0

            t_inf0 = time.perf_counter()
            outputs = self.session.run(None, {self.input_name: inp})
            inf_wall_ms = (time.perf_counter() - t_inf0) * 1000.0
            self.last_gpu_compute_ms = float(getattr(self.session, "last_gpu_compute_ms", 0.0) or inf_wall_ms)

            t_post0 = time.perf_counter()
            logits = outputs[0] # (1, 1000)

            # Softmax 概率化
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            top1_id = int(np.argmax(probs[0]))
            top1_conf = float(probs[0, top1_id])

            res_tuple = None
            if top1_id in IMAGENET_ANIMAL_MAP:
                cat, sp_en, sp_cn = IMAGENET_ANIMAL_MAP[top1_id]
                res_tuple = (cat, top1_conf, sp_en, sp_cn)

            if res_tuple is None:
                # 若前 5 中存在猛兽候选且概率较高 (>0.25)，优先召回猛兽提高露营安全性
                top5_ids = np.argsort(probs[0])[::-1][:5]
                for cid in top5_ids:
                    if cid in IMAGENET_ANIMAL_MAP:
                        cat, sp_en, sp_cn = IMAGENET_ANIMAL_MAP[cid]
                        if cat == 'large_beast' and float(probs[0, cid]) > 0.25:
                            res_tuple = ('large_beast', float(probs[0, cid]), sp_en, sp_cn)
                            break

                # 若前 5 均为其它类别或小型动物
                if res_tuple is None:
                    for cid in top5_ids:
                        if cid in IMAGENET_ANIMAL_MAP:
                            cat, sp_en, sp_cn = IMAGENET_ANIMAL_MAP[cid]
                            res_tuple = (cat, float(probs[0, cid]), sp_en, sp_cn)
                            break

            if res_tuple is None:
                # 非动物预测回退
                res_tuple = ('small_animal', top1_conf, 'unclassified_animal', '通用动物')

            self.last_postprocess_ms = (time.perf_counter() - t_post0) * 1000.0
            try:
                self.session.last_preprocess_ms = self.last_preprocess_ms
                self.session.last_postprocess_ms = self.last_postprocess_ms
            except Exception:
                pass
            return res_tuple

        except Exception as e:
            print(f"[AnimalClassifier] 推理过程发生异常: {e}")
            return ('small_animal', 0.0, 'error', '分类异常')
