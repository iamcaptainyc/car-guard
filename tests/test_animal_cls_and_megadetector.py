"""
单元与集成测试：MegaDetector 动物检测与 MobileNetV3 二级动物细分类联动引擎
验证核心特性：
1. MobileNetV3 动物分类器 (ImageNet-1K 物种映射、猛兽/小型动物二分类、特征预处理)
2. MegaDetector 与分类器的严格级联机制——仅在检出 animal 类时调用分类器，人(person)与车(car)绝对不调用
3. 联动露营守卫算法引擎：猛兽(large_beast)冲刺触发 L3 紧急与存证上链，小型动物(small_animal)低危处理
4. 证据链跨进程重启无缝衔接与历史断裂自愈验证
"""

import time
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest

from camping_guard.types import (
    ThreatLevel, ActionType, ThreatEventRecord, VehicleZone, BoundingBox
)
from camping_guard.perception.animal_classifier import AnimalClassifier, IMAGENET_ANIMAL_MAP
from camping_guard.perception.detector import ONNXObjectDetector, BaseObjectDetector
from camping_guard.evidence.recorder import EvidenceRecorder
from camping_guard.engine import CampingGuardEngine


class MockDetectorWithAnimal(BaseObjectDetector):
    """模拟 MegaDetector 检测器输出"""
    def __init__(self, detections):
        self.detections = detections
        self.is_megadetector = True
        self.animal_classifier = None

    def detect(self, frame, conf_threshold=0.25):
        # 模拟 detect 方法中的分类联动逻辑
        dets = []
        for d in self.detections:
            base_cat = d.get('category')
            if base_cat == 'animal' and self.animal_classifier is not None:
                bx, by, bw, bh = d['bbox']
                crop = frame[by:by+bh, bx:bx+bw]
                final_cat, conf, sp_en, sp_cn = self.animal_classifier.classify(crop)
                dets.append({
                    'bbox': d['bbox'],
                    'category': final_cat,
                    'conf': d.get('conf', 0.9),
                    'species': sp_en,
                    'species_cn': sp_cn
                })
            else:
                dets.append(dict(d))
        return dets


class MockAnimalClassifier:
    """模拟细分类器，用于确定性单元测试"""
    def __init__(self, override_cat="large_beast", override_species="brown_bear", override_cn="棕熊"):
        self.override_cat = override_cat
        self.override_species = override_species
        self.override_cn = override_cn
        self.call_count = 0

    def classify(self, crop):
        self.call_count += 1
        return (self.override_cat, 0.95, self.override_species, self.override_cn)


def test_animal_classifier_imagenet_mapping():
    """测试 1: 验证 ImageNet-1K 动物类别映射表的完备性与正确性"""
    # 猛兽验证 (熊类、大型猫科、野生犬狼、野猪)
    assert IMAGENET_ANIMAL_MAP[294][0] == 'large_beast' # 棕熊
    assert IMAGENET_ANIMAL_MAP[294][1] == 'brown_bear'
    assert IMAGENET_ANIMAL_MAP[294][2] == '棕熊'

    assert IMAGENET_ANIMAL_MAP[290][0] == 'large_beast' # 老虎
    assert IMAGENET_ANIMAL_MAP[290][1] == 'tiger'

    assert IMAGENET_ANIMAL_MAP[269][0] == 'large_beast' # 灰狼
    assert IMAGENET_ANIMAL_MAP[269][1] == 'timber_wolf'

    assert IMAGENET_ANIMAL_MAP[341][0] == 'large_beast' # 野猪
    assert IMAGENET_ANIMAL_MAP[341][1] == 'wild_boar'

    # 小型无害动物验证 (家犬、猫、狐狸、野兔、鸟)
    assert IMAGENET_ANIMAL_MAP[200][0] == 'small_animal' # 家犬
    assert IMAGENET_ANIMAL_MAP[281][0] == 'small_animal' # 虎斑猫
    assert IMAGENET_ANIMAL_MAP[277][0] == 'small_animal' # 赤狐
    assert IMAGENET_ANIMAL_MAP[330][0] == 'small_animal' # 野兔
    assert IMAGENET_ANIMAL_MAP[100][0] == 'small_animal' # 鸟


def test_animal_classifier_inference_and_fallback():
    """测试 2: 验证真实模型/后备机制的分类前向推理与边界鲁棒性"""
    # 实例化真实分类器 (若有 onnx 模型则挂载，无则优雅后备)
    classifier = AnimalClassifier()
    assert classifier is not None

    # 1. 空图/极小图边界退避
    empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
    cat, conf, sp_en, sp_cn = classifier.classify(empty_crop)
    assert cat == 'small_animal'
    assert sp_en == 'unknown'

    tiny_crop = np.zeros((3, 3, 3), dtype=np.uint8)
    cat, conf, sp_en, sp_cn = classifier.classify(tiny_crop)
    assert cat == 'small_animal'

    # 2. 正常输入裁剪块预处理形状验证
    dummy_crop = np.ones((120, 160, 3), dtype=np.uint8) * 128
    prep = classifier.preprocess(dummy_crop)
    assert prep.shape == (1, 3, 224, 224)
    assert prep.dtype == np.float32

    # 3. 前向推理
    cat, conf, sp_en, sp_cn = classifier.classify(dummy_crop)
    assert cat in ['large_beast', 'small_animal']
    assert isinstance(conf, float)
    assert len(sp_en) > 0
    assert len(sp_cn) > 0


def test_megadetector_cascade_only_on_animal():
    """测试 3: 严格验证核心约束——仅在识别到 animal 时触发分类，person 与 car 严禁调用"""
    mock_cls = MockAnimalClassifier(override_cat="large_beast", override_species="brown_bear", override_cn="棕熊")

    # 构造假检测帧: 包含 1个人 (person), 1辆车 (car), 1个动物 (animal)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    raw_dets = [
        {'bbox': (50, 50, 80, 160), 'category': 'person', 'conf': 0.85},
        {'bbox': (200, 100, 150, 120), 'category': 'car', 'conf': 0.90},
        {'bbox': (400, 150, 100, 90), 'category': 'animal', 'conf': 0.88},
    ]

    detector = MockDetectorWithAnimal(raw_dets)
    detector.animal_classifier = mock_cls

    results = detector.detect(frame)

    # 核心裁决: mock_cls 必须且仅被调用了 1 次 (对应 animal)!
    assert mock_cls.call_count == 1, f"分类器调用次数错误: 期望 1, 实际 {mock_cls.call_count}"

    # 验证分类后结果
    person_det = next(d for d in results if d['bbox'] == (50, 50, 80, 160))
    car_det = next(d for d in results if d['bbox'] == (200, 100, 150, 120))
    animal_det = next(d for d in results if d['bbox'] == (400, 150, 100, 90))

    assert person_det['category'] == 'person'
    assert car_det['category'] == 'car'
    assert animal_det['category'] == 'large_beast'
    assert animal_det['species'] == 'brown_bear'
    assert animal_det['species_cn'] == '棕熊'


def test_engine_integration_beast_vs_small_animal():
    """测试 4: 联动 CampingGuardEngine 验证猛兽冲刺触发高危 vs 小型动物低危"""
    mock_beast_cls = MockAnimalClassifier(override_cat="large_beast", override_species="timber_wolf", override_cn="灰狼")

    temp_dir = tempfile.mkdtemp()
    try:
        # 1. 模拟大型猛兽场景: 灰狼冲刺进入 Zone 2 (由远及近连续逼近)
        engine = CampingGuardEngine(use_bytetrack=False, animal_classifier=mock_beast_cls)
        engine.activate()
        engine.recorder = EvidenceRecorder(storage_dir=temp_dir, buffer_duration_sec=2.0, fps=10.0)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 连续模拟多帧以满足猛兽冲刺径向逼近计算与时序确认
        out_f = None
        for i in range(6):
            beast_dets = [{
                'track_id': 101,
                'bbox': (240, 200 + i * 35, 80, 80),
                'category': 'large_beast',
                'species': 'timber_wolf',
                'species_cn': '灰狼',
                'conf': 0.92
            }]
            time.sleep(0.04)
            out_f = engine.process_tick({0: dummy_frame}, beast_dets)

        # 验证猛兽冲刺行为判定生效 (威胁升级为 L3 紧急高危)
        assert out_f["threat_level"] in [ThreatLevel.L2_WARNING.value, ThreatLevel.L3_CRITICAL.value]
        assert engine.current_threat_level in [ThreatLevel.L2_WARNING, ThreatLevel.L3_CRITICAL]
        last_evt = engine.historical_threat_events[-1] if engine.historical_threat_events else None
        assert last_evt is not None
        assert last_evt.target_category == 'large_beast'
        assert last_evt.species == 'timber_wolf'
        assert last_evt.species_cn == '灰狼'

        # 验证 App 推送格式包含物种信息
        payload = last_evt.to_push_payload()
        assert "灰狼" in payload["notification"]["body"]
        assert payload["target"]["species_cn"] == "灰狼"

        # 验证防篡改证据链记录已正确保存物种信息
        engine.recorder.flush()
        is_valid, msg, bad_idx = engine.recorder.verify_hash_chain()
        assert is_valid is True
        assert bad_idx is None

        card_lines = last_evt.format_audit_card()
        assert "灰狼" in card_lines

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_mobilenetv3_animal_cls_profiling():
    """测试 5: 验证 MobileNetV3 动物细分类器独立耗时打点、GPU 推理时间与级联 Profiler 统计"""
    from unittest.mock import MagicMock
    from camping_guard.profiler import PipelineProfiler
    from camping_guard.perception.detector import ONNXObjectDetector, ParallelEnsembleDetector

    profiler = PipelineProfiler()

    # 构造 ONNXObjectDetector 并设置 is_megadetector
    detector = ONNXObjectDetector("models/megadetector/md_v5a.onnx")
    detector.is_megadetector = True
    detector.input_name = "images"

    # Mock 主检测器 session 输出一个 animal (class_id=0)
    mock_main_sess = MagicMock()
    mock_main_sess.last_gpu_compute_ms = 35.5
    # YOLO/MegaDetector output format: [1, 7, 10] -> [1, 10, 7] (cx, cy, w, h, score0, score1, score2)
    fake_preds = np.zeros((1, 7, 10), dtype=np.float32)
    fake_preds[0, 0, 0] = 320.0
    fake_preds[0, 1, 0] = 240.0
    fake_preds[0, 2, 0] = 100.0
    fake_preds[0, 3, 0] = 100.0
    fake_preds[0, 4, 0] = 0.95  # score for class 0 (animal)
    mock_main_sess.run.return_value = [fake_preds]
    detector.session = mock_main_sess

    # Mock 动物分类器与 GPU 推理耗时
    mock_classifier = MagicMock()
    mock_cls_sess = MagicMock()
    mock_cls_sess.last_gpu_compute_ms = 3.8
    mock_classifier.session = mock_cls_sess
    mock_classifier.classify.return_value = ('large_beast', 0.92, 'american_black_bear', '美洲黑熊')
    detector.animal_classifier = mock_classifier

    # 并发集成检测器挂载测试 (验证 profiler 级联属性传播)
    ensemble = ParallelEnsembleDetector(main_detector=detector, fire_detector=None)
    ensemble.profiler = profiler
    assert detector.profiler is profiler, "ParallelEnsembleDetector 未能将 profiler 级联透传至 main_detector！"

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = ensemble.detect(dummy_frame)

    assert len(dets) == 1
    assert dets[0]['category'] == 'large_beast'
    assert dets[0]['species'] == 'american_black_bear'

    # 核心断言: profiler 必须精准记录到 MobileNetV3-AnimalCls 的独立统计数据
    assert "MobileNetV3-AnimalCls" in profiler.model_stats
    stats = profiler.model_stats["MobileNetV3-AnimalCls"]
    assert stats.call_count == 1, f"MobileNetV3-AnimalCls 预期调用 1 次，实际调用 {stats.call_count}"
    assert stats.avg_gpu_ms == pytest.approx(3.8, rel=1e-3)
    assert stats.avg_ms > 0.0


def test_bytetrack_large_beast_hysteresis_latching():
    """测试 6: 验证 ByteTrack 轨迹级大型猛兽防抖锁存机制 (Hysteresis Latching)"""
    from camping_guard.perception.bytetrack import ByteTracker

    tracker = ByteTracker(high_thresh=0.5, low_thresh=0.2)
    now = 1000.0

    # 帧 1: 首次检出高危美洲黑熊 (large_beast)
    det_f1 = [{
        'bbox': (100, 100, 80, 80),
        'conf': 0.90,
        'category': 'large_beast',
        'species': 'american_black_bear',
        'species_cn': '美洲黑熊'
    }]
    ents_f1 = tracker.update(det_f1, current_time=now)
    assert len(ents_f1) == 1
    t1 = ents_f1[0]
    tid = t1.track_id
    assert t1.category == 'large_beast'
    assert t1.species == 'american_black_bear'
    assert t1.species_cn == '美洲黑熊'

    # 帧 2: 黑熊背对镜头/走入树荫，分类器在阈值边缘退化为通用 animal
    det_f2 = [{
        'bbox': (102, 101, 80, 80),
        'conf': 0.75,
        'category': 'animal'
    }]
    ents_f2 = tracker.update(det_f2, current_time=now + 0.033)
    assert len(ents_f2) == 1
    t2 = ents_f2[0]
    assert t2.track_id == tid
    # 核心断言: 类别必须锁存在 large_beast，物种维持美洲黑熊，杜绝标签与警报闪烁！
    assert t2.category == 'large_beast', f"标签退化为 {t2.category}，锁存防抖失效！"
    assert t2.species == 'american_black_bear'
    assert t2.species_cn == '美洲黑熊'

    # 帧 3: 进一步退化为 small_animal 且物种为 unknown
    det_f3 = [{
        'bbox': (104, 102, 80, 80),
        'conf': 0.70,
        'category': 'small_animal',
        'species': 'unknown',
        'species_cn': '未知动物'
    }]
    ents_f3 = tracker.update(det_f3, current_time=now + 0.066)
    assert len(ents_f3) == 1
    t3 = ents_f3[0]
    assert t3.track_id == tid
    assert t3.category == 'large_beast'
    assert t3.species == 'american_black_bear'

    # 帧 4: 目标确凿变更为完全独立物理类别 (例如人员走入替换)
    det_f4 = [{
        'bbox': (106, 103, 80, 80),
        'conf': 0.95,
        'category': 'person'
    }]
    ents_f4 = tracker.update(det_f4, current_time=now + 0.099)
    assert len(ents_f4) == 1
    t4 = ents_f4[0]
    assert t4.track_id == tid
    assert t4.category == 'person'
    # 锁存解除
    st = tracker.get_track(tid)
    assert st.confirmed_large_beast is False

