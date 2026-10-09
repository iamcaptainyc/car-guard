"""
FaceID 与 ReID 策略集成自动化单元与回归测试套件 (tests/test_face_reid.py)
验证:
1. OSNet 512维 ReID 特征提取、L2归一化与余弦相似度
2. 多视角样本池 (Exemplar Bank) 与特征防污染机制
3. 跨遮挡/跨ID重入 (Re-entry) 徘徊累计滞留时间关联
4. InsightFace SCRFD 人脸检测与 ArcFace 512维特征识别
5. FQA (人脸质量评估) 门禁 (尺寸、模糊度、侧脸姿态)
6. FaceID 细粒度状态机流转 (PENDING -> UNAVAILABLE -> MATCH_CANDIDATE -> VERIFIED)
7. FaceID 确权反哺 ReID 全局会话与软白名单跨时空继承
"""

import pytest
import numpy as np
import cv2
import time

from camping_guard.types import (
    TrackedEntity, BoundingBox, VehicleZone, FaceIDState,
    ReIDExemplar, PersonSession, ThreatLevel
)
from camping_guard.perception.reid import OSNetReIDExtractor, PersonSessionManager
from camping_guard.perception.faceid import (
    FaceQualityAssessor, SCRFDFaceDetector, ArcFaceRecognizer, FaceIDPipeline
)
from camping_guard.perception.tracker import TargetTrackManager
from camping_guard.perception.whitelist import WhitelistManager


class TestReIDModule:
    """ReID 特征提取与全局人员会话管理测试"""

    def test_osnet_feature_extraction(self):
        """测试 1: 验证 OSNet 模型推理、输出维度及 L2 归一化"""
        extractor = OSNetReIDExtractor()
        # 构造模拟人体裁剪图像 (128x256 BGR)
        dummy_person = np.full((256, 128, 3), (80, 100, 150), dtype=np.uint8)
        cv2.circle(dummy_person, (64, 40), 20, (180, 200, 220), -1)
        cv2.rectangle(dummy_person, (30, 80), (98, 200), (30, 40, 90), -1)

        feat = extractor.extract_crop_feature(dummy_person)
        assert feat.shape == (512,), "ReID 特征向量必须为 512 维"
        norm = float(np.linalg.norm(feat))
        assert abs(norm - 1.0) < 1e-4, f"特征向量必须满足 L2 归一化，当前 norm={norm}"

        # 验证相同图像相似度接近 1.0
        sim_self = PersonSessionManager.cosine_similarity(feat, feat)
        assert sim_self > 0.999, "自相似度必须接近 1.0"

        # 验证不同颜色图像相似度明显降低
        diff_person = np.full((256, 128, 3), (200, 30, 30), dtype=np.uint8)
        feat_diff = extractor.extract_crop_feature(diff_person)
        sim_diff = PersonSessionManager.cosine_similarity(feat, feat_diff)
        assert sim_diff < 0.85, f"差异样本相似度应显著降低，当前={sim_diff}"

    def test_exemplar_bank_anti_pollution(self):
        """测试 2: 验证多视角样本池与特征入库防污染机制"""
        session_mgr = PersonSessionManager(max_exemplars_per_person=4, new_view_diversity_threshold=0.88)
        now = time.time()

        # 基准特征
        f1 = np.random.randn(512).astype(np.float32)
        f1 /= np.linalg.norm(f1)

        sess, is_new = session_mgr.match_or_create_session("cam0_trk1", f1, now, quality_score=0.9)
        assert is_new is True
        assert len(sess.exemplars) == 1, "初始特征应成功入库"

        # 尝试灌入高度冗余特征 (相似度 0.95 > 0.88) -> 应被拒绝入库，不浪费样本槽
        f_redundant = f1 * 0.98 + np.random.randn(512) * 0.02
        f_redundant /= np.linalg.norm(f_redundant)
        session_mgr.match_or_create_session("cam0_trk1", f_redundant, now + 1.0, quality_score=0.9)
        assert len(sess.exemplars) == 1, "高度冗余样本应被抑制，防止挤占样本池"

        # 尝试灌入质量极差的特征 (quality_score < 0.5) -> 应被防污染拦截
        f_bad_quality = np.random.randn(512).astype(np.float32)
        f_bad_quality /= np.linalg.norm(f_bad_quality)
        session_mgr.match_or_create_session("cam0_trk1", f_bad_quality, now + 2.0, quality_score=0.3)
        assert len(sess.exemplars) == 1, "低质量特征禁止入库污染档案"

        # 灌入带来有益新视角的高质量特征 (相似度在 [0.55, 0.88] 之间) -> 成功入库
        # 控制余弦相似度在约 0.7 左右
        noise = np.random.randn(512).astype(np.float32)
        noise -= np.dot(f1, noise) * f1 # 正交化
        noise /= np.linalg.norm(noise)
        f_new_view = f1 * 0.70 + noise * np.sqrt(1 - 0.70**2) # 精确余弦相似度为 0.70
        session_mgr.match_or_create_session("cam0_trk1", f_new_view, now + 3.0, quality_score=0.85)
        assert len(sess.exemplars) == 2, "有代表性的新视角特征应成功入库"

    def test_reentry_loitering_accumulation(self):
        """测试 3: 验证离开后重入 (Re-entry) 跨 ID 缝合并累计徘徊时长"""
        extractor = OSNetReIDExtractor()
        session_mgr = PersonSessionManager(reid_match_threshold=0.60)
        track_mgr = TargetTrackManager(reid_extractor=extractor, session_manager=session_mgr)

        # 准备一个人体画面
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        # 在 (100, 100, 80, 180) 画一个人体
        cv2.rectangle(frame, (100, 100), (180, 280), (120, 150, 200), -1)

        t0 = 1000.0
        # 第一次出现: track_id = 10, 在 Zone 1 逗留 15 秒
        det1 = [{
            'track_id': 10,
            'category': 'person',
            'conf': 0.88,
            'bbox': BoundingBox(100, 100, 80, 180),
            'dist': 4.5,
            'radial_speed': 0.0,
            'zone': VehicleZone.ZONE_1_OUTER
        }]
        track_mgr.update_tracks(det1, t0, frame=frame, camera_id=0)
        track_mgr.update_tracks(det1, t0 + 15.0, frame=frame, camera_id=0)

        # 检查会话与累计时间
        sess_id = track_mgr.active_tracks[10].person_session_id
        assert sess_id is not None
        assert abs(track_mgr.active_tracks[10].cumulative_dwell_time - 15.0) < 0.1

        # 目标离开视野 60 秒 (超过常规 10s 追踪重置窗口，常规追踪器会丢弃该 ID)
        t_reentry = t0 + 75.0
        track_mgr.update_tracks([], t_reentry - 1.0)
        assert 10 not in track_mgr.active_tracks

        # 目标重新进入: 上游检测器赋予了全新的 track_id = 99
        det2 = [{
            'track_id': 99,
            'category': 'person',
            'conf': 0.88,
            'bbox': BoundingBox(120, 100, 80, 180), # 位置微移，但外观相同
            'dist': 4.2,
            'radial_speed': 0.0,
            'zone': VehicleZone.ZONE_1_OUTER
        }]
        updated = track_mgr.update_tracks(det2, t_reentry, frame=frame, camera_id=0)
        new_entity = updated[0]

        # 核心断言: ReID 成功将 track_id 99 关联回之前的全局会话，并继承了历史 15 秒滞留时间！
        assert new_entity.person_session_id == sess_id, "ReID 必须成功识别重入目标并恢复全局会话"
        assert abs(new_entity.cumulative_dwell_time - 15.0) < 0.1, "重入瞬间应恢复历史累计滞留时间"

        # 继续逗留 18 秒 -> 累计达到 33 秒 (>30s 突破 Zone 1 徘徊预警阈值)
        track_mgr.update_tracks(det2, t_reentry + 18.0, frame=frame, camera_id=0)
        assert new_entity.cumulative_dwell_time >= 33.0, "累计滞留时间必须无缝跨轨迹持续累加"


class TestFaceIDModule:
    """FaceID 质检、状态机与确权反哺测试"""

    def test_face_quality_assessment(self):
        """测试 4: 验证 FQA 质量评估门禁 (尺寸、模糊度、侧脸角度)"""
        fqa = FaceQualityAssessor(min_face_size=40, min_blur_variance=30.0, max_yaw_ratio=2.4)
        dummy_frame = np.full((300, 300, 3), 128, dtype=np.uint8)

        # 1. 尺寸过小测试
        small_box = (10, 10, 35, 35) # 25x25 像素
        is_q, score, reason = fqa.evaluate_quality(dummy_frame, small_box)
        assert is_q is False
        assert reason == "FACE_TOO_SMALL"

        # 2. 运动模糊测试 (纯色方块无梯度)
        normal_box = (50, 50, 150, 150) # 100x100 像素纯灰度
        is_q, score, reason = fqa.evaluate_quality(dummy_frame, normal_box)
        assert is_q is False
        assert reason == "MOTION_BLURRED"

        # 3. 大角度侧脸姿态测试
        # 构造有纹理人脸
        face_img = dummy_frame.copy()
        cv2.putText(face_img, "FACE", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        # 侧脸关键点: 鼻子距左眼极近，距右眼极远 (偏航角超标)
        extreme_kps = np.array([
            [60, 80],   # 左眼
            [140, 80],  # 右眼
            [65, 100],  # 鼻尖 (d_left=5, d_right=75 -> ratio=15 > 2.4)
            [70, 130],
            [130, 130]
        ], dtype=np.float32)
        is_q, score, reason = fqa.evaluate_quality(face_img, normal_box, kps=extreme_kps)
        assert is_q is False
        assert reason == "EXTREME_YAW_PROFILE"

    def test_faceid_state_machine_flow(self):
        """测试 5: 验证细粒度状态机 (PENDING -> UNAVAILABLE -> MATCH_CANDIDATE -> VERIFIED)"""
        pipeline = FaceIDPipeline(match_threshold=0.65, verify_confirm_frames=2, retry_interval_sec=0.5)

        # 注册车主特征到白名单底库
        owner_feat = np.random.randn(512).astype(np.float32)
        owner_feat /= np.linalg.norm(owner_feat)
        pipeline.register_whitelist_feature("Owner_Alice", owner_feat)

        target = TrackedEntity(
            track_id=1,
            category="person",
            confidence=0.9,
            bbox=BoundingBox(100, 100, 100, 200),
            ground_dist=2.0
        )
        assert target.face_state == FaceIDState.PENDING

        now = time.time()
        # 场景 A: 背对相机 (纯黑色画面检测不出人脸) -> 变更为 UNAVAILABLE，绝不能误判为 NO_MATCH 陌生人
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = pipeline.process_target_face(black_frame, target, now)
        assert state == FaceIDState.UNAVAILABLE
        assert target.is_whitelisted is False, "背对相机不应确权白名单"

        # 场景 B: 模拟抓拍到合格人脸特征并单次匹配车主 -> 变更为 MATCH_CANDIDATE
        # Mock 提取器返回车主相似特征
        pipeline.recognizer.extract_feature = lambda img: owner_feat
        # Mock 检测器返回有效框与居中五点
        pipeline.detector.detect = lambda img: (
            np.array([[20, 20, 80, 80, 0.95]]),
            np.array([[[35, 45], [65, 45], [50, 60], [40, 75], [60, 75]]])
        )
        pipeline.fqa.evaluate_quality = lambda f, b, k: (True, 0.9, "QUALIFIED")

        # 第一次匹配成功
        state1 = pipeline.process_target_face(black_frame, target, now + 0.6)
        assert state1 == FaceIDState.MATCH_CANDIDATE
        assert target.is_whitelisted is False, "单次匹配必须等待多帧确认"

        # 第二次再次置信匹配 -> 升至 VERIFIED 确权
        state2 = pipeline.process_target_face(black_frame, target, now + 0.7)
        assert state2 == FaceIDState.VERIFIED
        assert target.is_whitelisted is True, "多次确认后必须完成白名单确权"
        assert target.face_id_name == "Owner_Alice"

    def test_faceid_to_reid_whitelist_inheritance(self):
        """测试 6: 验证人脸确权后反哺 ReID 全局会话，背身及跨相机自动享受软白名单免扰"""
        pipeline = FaceIDPipeline()
        session_mgr = PersonSessionManager()
        whitelist_mgr = WhitelistManager(faceid_pipeline=pipeline, session_manager=session_mgr)

        # 模拟目标通过 FaceID 确权
        target1 = TrackedEntity(
            track_id=7,
            track_uid="cam1_trk7",
            person_session_id="P0001",
            category="person",
            confidence=0.9,
            bbox=BoundingBox(100, 100, 80, 180),
            ground_dist=1.8,
            face_state=FaceIDState.VERIFIED,
            face_id_name="Owner_Bob",
            is_whitelisted=True
        )

        # 建立会话并打上白名单标记
        session_mgr.sessions["P0001"] = PersonSession(
            session_id="P0001",
            is_whitelisted=True,
            whitelist_name="Owner_Bob"
        )
        session_mgr.track_uid_to_session["cam1_trk7"] = "P0001"

        # 此时车主走到另一路相机 (cam2) 且完全背对相机 (FaceID 无法使用)
        target2 = TrackedEntity(
            track_id=18,
            track_uid="cam2_trk18",
            person_session_id="P0001", # 由 ReID 成功识别并关联到同一人员会话
            category="person",
            confidence=0.9,
            bbox=BoundingBox(200, 150, 80, 180),
            ground_dist=2.5,
            face_state=FaceIDState.UNAVAILABLE,
            is_whitelisted=False
        )

        now = time.time()
        whitelist_mgr.evaluate_whitelist([target2], uwb_key_pos=None, face_id_results=None, current_time=now)

        # 核心断言: target2 凭借 ReID 全局会话成功继承白名单，无需重新刷脸即可免受告警打扰！
        assert target2.is_whitelisted is True, "ReID 关联的目标必须自动继承会话白名单免扰状态"
        assert target2.face_id_name == "Owner_Bob"
        assert whitelist_mgr.whitelisted_tracks[18]["auth_channel"] == "REID_SESSION_INHERIT"
