"""
露营守卫系统主状态机与退出防护机制 (第一、五章)
管理 S0 待机、S1 初筛、S2 研判流转，中继攻击拦截防御，与相机遮挡故障降级
"""

from typing import Optional, Tuple, List, Dict
import time
from camping_guard.types import GuardState, ThreatLevel, OperatingMode
from camping_guard.config import GuardConfig

class GuardStateMachine:
    def __init__(self):
        self.state: GuardState = GuardState.S0_STANDBY
        self.operating_mode: OperatingMode = OperatingMode.IN_CAR

        # 状态切换与计时器
        self.last_transition_time: float = time.time()
        self.s0_motion_start_time: Optional[float] = None
        self.s1_empty_start_time: Optional[float] = None
        self.s2_empty_start_time: Optional[float] = None
        self.last_high_threat_time: float = 0.0 # 最近一次 L2/L3 事件时间 (60s 冷却)

        # 相机遮挡计时器: {camera_id: occlusion_start_time}
        self.camera_occlusion_starts: Dict[int, float] = {}

        # 退出防御状态
        self.is_active: bool = False
        self.relay_attack_detected: bool = False

    def activate_system(
        self,
        gear: str,
        epb_engaged: bool,
        guard_switch_on: bool,
        all_doors_locked: bool,
        seat_occupied: bool,
        uwb_key_inside: bool = True
    ) -> Tuple[bool, str]:
        """
        激活条件检查 (第一章.1)
        """
        if not (gear == "P" and epb_engaged):
            return False, "激活失败：车辆未处于 P 档或电子手刹未拉起"
        if not guard_switch_on:
            return False, "激活失败：露营守卫开关未开启"
        if not all_doors_locked:
            return False, "激活失败：检测到车门/后备箱未完全锁闭"

        self.is_active = True
        self.state = GuardState.S0_STANDBY
        self.last_transition_time = time.time()

        # 工作模式判定: 座椅占位/UWB 在车内 -> 车内模式；否则车外模式
        if seat_occupied and uwb_key_inside:
            self.operating_mode = OperatingMode.IN_CAR
        else:
            self.operating_mode = OperatingMode.OUT_CAR

        return True, f"露营守卫已成功激活，当前工作模式为【{self.operating_mode.value}】"

    def handle_unlock_request(
        self,
        unlock_source: str,
        uwb_distance_to_car: Optional[float],
        current_time: float
    ) -> Tuple[bool, Optional[ThreatLevel], str]:
        """
        退出机制与中继攻击防御校验 (第一章.2)
        :return: (是否允许正常退出, 威胁等级更新, 状态说明)
        """
        if not self.is_active:
            return True, None, "系统未激活"

        # 校验 UWB 钥匙空间对齐 (是否在车旁 1.5 米内)
        if uwb_distance_to_car is not None and uwb_distance_to_car <= GuardConfig.UWB_UNLOCK_EXIT_DIST:
            self.is_active = False
            self.relay_attack_detected = False
            return True, None, "车主 UWB 钥匙鉴权通过 (<= 1.5m)，守卫正常安全退出"

        # 无法与现场 UWB 匹配 -> 判定为疑似中继攻击/越权解锁！
        self.relay_attack_detected = True
        self.last_high_threat_time = current_time
        # 强制保持守卫并不退出，升至 L3 紧急
        self.transition_to(GuardState.S2_ANALYSIS, current_time)
        return False, ThreatLevel.L3_CRITICAL, f"【警告】检测到非法越权/中继攻击解锁！钥匙不在场或位置异常，守卫强制保持并升级 L3 紧急高危！"

    def transition_to(self, new_state: GuardState, current_time: float):
        if self.state != new_state:
            # print(f"[状态流转] {self.state.value} -> {new_state.value}")
            self.state = new_state
            self.last_transition_time = current_time
            # 状态重置
            self.s0_motion_start_time = None
            self.s1_empty_start_time = None
            self.s2_empty_start_time = None

    def update_camera_occlusion(self, occluded_cam_ids: List[int], current_time: float) -> Tuple[Optional[ThreatLevel], str]:
        """
        相机遮挡故障判定与告警 (第一章故障与降级)
        单路相机 > 5s -> L2; 两路及以上 -> L3
        """
        # 更新遮挡字典
        for cid in occluded_cam_ids:
            if cid not in self.camera_occlusion_starts:
                self.camera_occlusion_starts[cid] = current_time

        # 移除已恢复的相机
        recovered = [cid for cid in self.camera_occlusion_starts if cid not in occluded_cam_ids]
        for cid in recovered:
            del self.camera_occlusion_starts[cid]

        # 统计持续遮挡超 5 秒的相机数量
        long_occluded = [
            cid for cid, t in self.camera_occlusion_starts.items()
            if (current_time - t) >= GuardConfig.CAMERA_OCCLUSION_L2_SEC
        ]

        if len(long_occluded) >= 2:
            return ThreatLevel.L3_CRITICAL, f"故障紧急告警：多路相机 ({long_occluded}) 发生持续严重遮挡！"
        elif len(long_occluded) == 1:
            return ThreatLevel.L2_WARNING, f"故障预警：相机 {long_occluded[0]} 被遮挡超过 5 秒，盲区告警已推送"

        return None, ""

    def step(
        self,
        has_motion_s0: bool,
        has_zone1_targets: bool,
        has_zone2_targets: bool = False,
        max_radial_speed: float = 0.0,
        has_fire_smoke: bool = False,
        has_contact_event: bool = False,
        is_body_roi_disturbed: bool = False,
        current_time: float = 0.0,
        has_zone2_3_targets: Optional[bool] = None,
        is_zone3_disturbed: Optional[bool] = None
    ):
        """
        主状态机单步流转 (第五章)
        """
        if not self.is_active:
            return

        # 兼容旧参数名
        z2_flag = has_zone2_targets if has_zone2_3_targets is None else (has_zone2_targets or has_zone2_3_targets)
        roi_disturbed_flag = is_body_roi_disturbed if is_zone3_disturbed is None else (is_body_roi_disturbed or is_zone3_disturbed)

        # 1. 准硬触发与硬触发 (直接从 S0/S1 直跳 S2)
        if has_contact_event or roi_disturbed_flag or self.relay_attack_detected:
            self.transition_to(GuardState.S2_ANALYSIS, current_time)
            return

        # 2. S0 待机态流转逻辑
        if self.state == GuardState.S0_STANDBY:
            if has_motion_s0:
                if self.s0_motion_start_time is None:
                    self.s0_motion_start_time = current_time
                elif current_time - self.s0_motion_start_time >= GuardConfig.S0_MOTION_CONFIRM_SEC: # >= 0.5s
                    self.transition_to(GuardState.S1_SCREENING, current_time)
            else:
                self.s0_motion_start_time = None

        # 3. S1 初筛态流转逻辑
        elif self.state == GuardState.S1_SCREENING:
            # S1 -> S2 常规升级: 目标进入 Zone 2 (0~3.0m)，或径向速度 > 0.8m/s，或疑似火情
            if z2_flag or max_radial_speed < -GuardConfig.RADIAL_SPEED_WAKE_S2 or has_fire_smoke:
                self.transition_to(GuardState.S2_ANALYSIS, current_time)
                return

            # S1 -> S0 回落: 范围内无目标持续 15 秒，且已过 L2/L3 60s 冷却期
            in_cooldown = (current_time - self.last_high_threat_time < GuardConfig.EVENT_COOLDOWN_SEC)
            if not has_zone1_targets and not in_cooldown:
                if self.s1_empty_start_time is None:
                    self.s1_empty_start_time = current_time
                elif current_time - self.s1_empty_start_time >= GuardConfig.S1_TO_S0_TIMEOUT_SEC:
                    self.transition_to(GuardState.S0_STANDBY, current_time)
            else:
                self.s1_empty_start_time = None

        # 4. S2 核心研判态流转逻辑
        elif self.state == GuardState.S2_ANALYSIS:
            # S2 -> S1 回落: Zone 2 无目标持续 10 秒且无接触事件与车身扰动
            if not z2_flag and not has_contact_event and not roi_disturbed_flag:
                if self.s2_empty_start_time is None:
                    self.s2_empty_start_time = current_time
                elif current_time - self.s2_empty_start_time >= GuardConfig.S2_TO_S1_TIMEOUT_SEC:
                    self.transition_to(GuardState.S1_SCREENING, current_time)
            else:
                self.s2_empty_start_time = None
