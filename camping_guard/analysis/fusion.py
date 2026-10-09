"""
多源时序联合确认模块 (第三章、步骤 2.1)
实现视觉动作与整车 CAN / IMU 接触信号在 ±2.0 秒时间窗口内的联合仲裁 (升 L3)
"""

from typing import List, Dict, Optional, Tuple
import time
from camping_guard.types import ContactSensorEvent, ActionType, ThreatLevel, TrackedEntity
from camping_guard.config import GuardConfig

class MultiSourceFusionEngine:
    def __init__(self):
        # 缓存最近 10 秒内的整车物理传感器事件
        self.sensor_event_buffer: List[ContactSensorEvent] = []

    def record_sensor_event(self, event: ContactSensorEvent):
        """记录来自 CAN / IMU 的硬件事件 (如震动、门把手被拉动、异常解锁)"""
        self.sensor_event_buffer.append(event)
        # 清理超过 10 秒的历史硬件事件
        cutoff = event.timestamp - 10.0
        self.sensor_event_buffer = [e for e in self.sensor_event_buffer if e.timestamp >= cutoff]

    def arbitrate_threat(
        self,
        target: TrackedEntity,
        visual_action: ActionType,
        visual_threat: ThreatLevel,
        current_time: float
    ) -> Tuple[ThreatLevel, bool, str]:
        """
        联合仲裁威胁等级
        :return: (最终威胁等级, 是否有接触信号确认, 描述)
        """
        # 如果白名单豁免，直接 L0
        if target is not None and target.is_whitelisted:
            return ThreatLevel.L0_SAFE, False, "白名单车主豁免"

        win = GuardConfig.CONTACT_FUSION_WINDOW_SEC # 2.0s
        min_t = current_time - win
        max_t = current_time + win

        # 查找 ±2.0 秒窗口内匹配的接触事件
        matching_events = [
            e for e in self.sensor_event_buffer
            if min_t <= e.timestamp <= max_t
        ]

        has_vibration = any(e.event_type == "IMU_VIBRATION_PEAK" for e in matching_events)
        has_handle_pull = any(e.event_type in ["DOOR_HANDLE_PULL", "UNAUTHORIZED_UNLOCK"] for e in matching_events)
        has_status_tamper = any(e.event_type in ["CHARGING_TAMPER", "DOOR_AJAR"] for e in matching_events)

        # 1. 拉门撬锁联合确认: 视觉手部动作 + 把手被拉动/异常解锁/震动 -> L3
        if visual_action == ActionType.DOOR_HANDLE_PULL:
            if has_handle_pull or has_vibration:
                return ThreatLevel.L3_CRITICAL, True, "拉门撬锁确立：视觉手部动作 + 车端门把手/车锁/震动物理信号联合确认"
            return ThreatLevel.L2_WARNING, False, "疑似拉门撬锁：视觉手部停留在把手区域，暂无车身接触信号"

        # 2. 蓄意踢踹/撞击联合确认: 视觉踢踹 + 车身震动峰值 -> L3
        if visual_action == ActionType.INTENTIONAL_KICK:
            if has_vibration:
                return ThreatLevel.L3_CRITICAL, True, "蓄意踢踹确立：视觉踢踹动作 + 车身 IMU 震动峰值联合确认"
            return ThreatLevel.L2_WARNING, False, "疑似踢踹动作：视觉动作指向车身，未检测到车体明显冲击震动"

        # 3. 装备盗窃联合确认: 视觉手部持续在行李架/充电口 + 车身状态变化 -> L3
        if visual_action == ActionType.EQUIPMENT_THEFT:
            if has_status_tamper or has_vibration:
                return ThreatLevel.L3_CRITICAL, True, "外挂装备盗窃确立：视觉盗窃动作 + 充电口/车体状态变化联合确认"
            return ThreatLevel.L2_WARNING, False, "疑似装备盗窃：手部在车顶行李架/充电口滞留，待 VLM 或接触信号复核"

        # 4. 车身碰撞/外物撞击联合确认: 视觉车身区域运动扰动 (开门碰撞/树木倒塌/砸车) + 车身震动峰值 -> L3
        if visual_action == ActionType.VEHICLE_COLLISION:
            if has_vibration or has_status_tamper:
                return ThreatLevel.L3_CRITICAL, True, "车身碰撞确立：视觉车身区域运动扰动 + 车端 IMU 震动/状态变化联合确认 (如邻车开门撞击/倒木砸车)"
            return ThreatLevel.L2_WARNING, False, "疑似车身碰撞：车身敏感区域检测到显著运动扰动 (如邻车开门贴近/外物触碰)，待 VLM 复核"

        # 5. 猛兽冲刺 (已在视觉侧 3 帧确认直接可定 L3)
        if visual_action == ActionType.BEAST_SPRINT and visual_threat == ThreatLevel.L3_CRITICAL:
            return ThreatLevel.L3_CRITICAL, False, "大型猛兽冲刺：径向高速逼近，连续 3 帧确认"

        # 5. 硬触发检测 (即便是视觉未识别动作，但检测到暴力破坏/倾斜/玻璃破碎)
        violent_hard_triggers = [
            e for e in matching_events
            if e.event_type in ["GLASS_BREAK", "VEHICLE_TILTED"]
        ]
        if violent_hard_triggers:
            trigger_name = violent_hard_triggers[0].event_type
            return ThreatLevel.L3_CRITICAL, True, f"物理硬触发：检测到整车高危传感器信号 ({trigger_name})"

        return visual_threat, False, "按常规防区与行为等级执行"
