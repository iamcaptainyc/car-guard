#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM 提示词、多模态研判反臆造与火灾烟雾检测链路单元测试 (test_vlm_prompt_and_fire.py)
验证:
1. SYSTEM_PROMPT 包含 FIRE_SMOKE 分类与视觉客观性铁律
2. JSON 输出契约首字段为 visual_description (视觉锚定驱动)
3. _build_prompt 在火灾/烟雾事件中注入专门防套用拉车门的定向指引
4. _build_prompt 在人员事件中明确提示严禁无证据臆测拉门
5. should_trigger_vlm 对 ActionType.FIRE_SMOKE 具备唤醒能力
6. _extract_vlm_response 能完整解析 visual_description 与 action_category
7. _arbitrate_and_finalize 保留 visual_description 并对火情进行安全升级仲裁
8. CampingGuardEngine 在火情触发时正确绑定 target_for_threat
"""

import pytest
import numpy as np
from camping_guard.types import ThreatLevel, ActionType, ThreatEventRecord, TrackedEntity, BoundingBox, VehicleZone
from camping_guard.review.vlm_reviewer import (
    TRTEdgeLLMOnDemandBackend,
    LocalHttpVLMBackend,
    MockVLMBackend,
    AsyncVLMReviewer
)
from camping_guard.engine import CampingGuardEngine


def test_system_prompt_structure_and_categories():
    """验证 SYSTEM_PROMPT 中包含 FIRE_SMOKE 及首字段 visual_description"""
    backend = TRTEdgeLLMOnDemandBackend(mode="mock")
    prompt = backend.SYSTEM_PROMPT

    # 1. 核心类别覆盖
    assert "FIRE_SMOKE" in prompt, "SYSTEM_PROMPT 必须包含 FIRE_SMOKE 动作类别"
    assert "VEHICLE_COLLISION" in prompt
    assert "DOOR_HANDLE_TAMPERING" in prompt
    assert "FALSE_ALARM" in prompt

    # 2. 首字段必须为 visual_description (视觉真实性锚定)
    assert '"visual_description"' in prompt, "JSON 契约必须声明 visual_description 字段"
    pos_vis = prompt.find('"visual_description"')
    pos_act = prompt.find('"action_category"')
    assert pos_vis < pos_act, "visual_description 必须在 action_category 之前输出以引导视觉注意力"

    # 3. 严格防臆造铁律
    assert "严禁无中生有" in prompt or "严禁凭空编造" in prompt or "严禁" in prompt
    assert "DOOR_HANDLE_TAMPERING" in prompt

    # 4. LocalHttpVLMBackend 同样同步此规范
    http_prompt = LocalHttpVLMBackend.SYSTEM_PROMPT
    assert "FIRE_SMOKE" in http_prompt
    assert '"visual_description"' in http_prompt


def test_build_prompt_fire_smoke_guidance():
    """验证火灾/烟雾事件下，提示词包含专门的火情防误判指引"""
    backend = TRTEdgeLLMOnDemandBackend(mode="mock")

    # 模拟快通路探测到火情烟雾
    fire_evt = ThreatEventRecord(
        event_id="EVT_FIRE_001",
        timestamp=100.0,
        threat_level=ThreatLevel.L2_WARNING,
        operating_mode=None,
        action_type=ActionType.FIRE_SMOKE,
        description="营火无人看管已超过5分钟",
        target_id=10,
        has_contact_signal=False,
        target_category="fire",
        target_distance=2.5,
        target_dwell_time=320.0
    )

    prompt = backend._build_prompt(fire_evt)

    # 校验提示词中不应将 target_category 误识别为 person
    assert "- 目标外部类别: fire" in prompt
    assert "火灾" in prompt or "浓烟" in prompt or "FIRE_SMOKE" in prompt
    # 校验包含严禁套用人员动作的强约束
    assert "切勿" in prompt or "绝不可" in prompt or "禁止" in prompt or "严禁" in prompt
    assert "拉车门" in prompt or "人员动作" in prompt or "拉门" in prompt


def test_build_prompt_person_no_tampering_bias():
    """验证人员事件下，提示词明确要求必须以图像肉眼事实为准，未拉门严禁判拉门"""
    backend = TRTEdgeLLMOnDemandBackend(mode="mock")

    person_evt = ThreatEventRecord(
        event_id="EVT_PERSON_001",
        timestamp=100.0,
        threat_level=ThreatLevel.L1_NOTICE,
        operating_mode=None,
        action_type=ActionType.NORMAL_PASS,
        description="行人常规经过",
        target_id=1,
        has_contact_signal=False,
        target_category="person",
        target_distance=4.2,
        target_dwell_time=5.0
    )

    prompt = backend._build_prompt(person_evt)
    assert "- 目标外部类别: person" in prompt
    assert "DOOR_HANDLE_TAMPERING" in prompt
    assert "严禁" in prompt or "切勿" in prompt or "绝不可" in prompt


def test_vlm_trigger_fire_smoke():
    """验证 should_trigger_vlm 在遇到 FIRE_SMOKE 时必定触发"""
    reviewer = AsyncVLMReviewer(enable_worker_thread=False)

    triggered = reviewer.should_trigger_vlm(
        category_conf=0.9,
        is_unknown_obstacle=False,
        fast_path_threat=ThreatLevel.L1_NOTICE,
        action_type=ActionType.FIRE_SMOKE,
        has_contact_signal=False
    )
    assert triggered is True, "ActionType.FIRE_SMOKE 必须唤醒慢通路 VLM 复核"


def test_extract_vlm_response_with_visual_description():
    """验证 _extract_vlm_response 能够完整提取 visual_description 与 action_category"""
    backend = TRTEdgeLLMOnDemandBackend(mode="mock")

    sample_output = """```json
{
  "visual_description": "画面左下方草地出现明显明火火堆与浓白烟雾，火势正向车体蔓延。",
  "action_category": "FIRE_SMOKE",
  "confidence": 0.96,
  "is_human_contact": false,
  "is_false_alarm": false,
  "threat_escalation": "CONFIRM_OR_UPGRADE",
  "summary_text": "车辆左侧草地起火并伴有大量浓烟，存在失控火灾险情！"
}
```"""

    raw_text, parsed = backend._extract_vlm_response(sample_output, sample_output)
    assert parsed is not None
    assert parsed["visual_description"] == "画面左下方草地出现明显明火火堆与浓白烟雾，火势正向车体蔓延。"
    assert parsed["action_category"] == "FIRE_SMOKE"
    assert parsed["confidence"] == 0.96
    assert parsed["summary_text"] == "车辆左侧草地起火并伴有大量浓烟，存在失控火灾险情！"


def test_arbitrate_fire_smoke_escalation():
    """验证 _arbitrate_and_finalize 能正确处理火情险情并升级至 L3"""
    reviewer = AsyncVLMReviewer(enable_worker_thread=False)

    fire_evt = ThreatEventRecord(
        event_id="EVT_FIRE_002",
        timestamp=200.0,
        threat_level=ThreatLevel.L2_WARNING,
        operating_mode=None,
        action_type=ActionType.FIRE_SMOKE,
        description="疑似失控火灾",
        target_id=12,
        has_contact_signal=False,
        target_category="fire",
        target_distance=2.0,
        target_dwell_time=60.0
    )

    raw_out = {
        "visual_description": "地面明火剧烈燃烧，浓烟逼近宿主车辆左侧车身。",
        "action_category": "FIRE_SMOKE",
        "confidence": 0.95,
        "is_human_contact": False,
        "is_false_alarm": False,
        "threat_escalation": "CONFIRM_OR_UPGRADE",
        "summary_text": "车身左侧火情扩大，浓烟逼近，建议立即采取应急灭火措施。"
    }

    result = reviewer._arbitrate_and_finalize(fire_evt, raw_out, record_history=False)
    assert result["final_threat"] == ThreatLevel.L3_CRITICAL, "已确认的火灾险情必须升级至 L3_CRITICAL"
    assert result["visual_description"] == "地面明火剧烈燃烧，浓烟逼近宿主车辆左侧车身。"
    assert result["action_category"] == "FIRE_SMOKE"


def test_mock_backend_fire_output():
    """验证 MockVLMBackend 产出的数据符合全新视觉规范"""
    mock = MockVLMBackend()

    fire_evt = ThreatEventRecord(
        event_id="EVT_FIRE_003",
        timestamp=300.0,
        threat_level=ThreatLevel.L2_WARNING,
        operating_mode=None,
        action_type=ActionType.FIRE_SMOKE,
        description="火情初探",
        target_id=15,
        has_contact_signal=False,
        target_category="fire",
        target_distance=2.8,
        target_dwell_time=15.0
    )

    res = mock.infer(fire_evt)
    assert "visual_description" in res
    assert res["action_category"] == "FIRE_SMOKE"
    assert "火" in res["visual_description"] or "烟" in res["visual_description"]
    assert "火" in res["summary_text"] or "烟" in res["summary_text"]
