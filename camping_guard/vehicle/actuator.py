"""
响应动作矩阵执行器 (阶段 3)
实现车内模式 (静默/座椅震动) 与车外模式 (大灯爆闪/车外扬声器) 的差异化执行
包含升级驱逐逻辑与火情特殊响应
"""

from typing import Dict, List, Optional
import time
from camping_guard.types import ThreatLevel, OperatingMode, ActionType, ThreatEventRecord

class VehicleActuator:
    def __init__(self):
        # 记录当前执行状态
        self.seat_vibration_active: bool = False
        self.headlight_strobe_active: bool = False
        self.exterior_speaker_active: bool = False
        self.puddle_light_active: bool = False
        self.cockpit_screen_view_active: bool = False

        # 驱逐升级计时器
        self.critical_start_time: Optional[float] = None
        self.is_escalated_eviction: bool = False

    def execute_response(
        self,
        record: ThreatEventRecord,
        current_time: float,
        is_fire_event: bool = False
    ) -> List[str]:
        """
        根据威胁等级与模式，执行整车联动动作
        :return: 触发的动作列表说明
        """
        actions = []
        mode = record.operating_mode
        level = record.threat_level

        if level == ThreatLevel.L1_NOTICE:
            actions.append("【L1 提示】静默日志记录，不打扰车主")
            self._reset_all()
            return actions

        if level == ThreatLevel.L2_WARNING:
            self._reset_critical()
            actions.append("【L2 预警】事件录像加密锁定 (事前10s至事后20s)")
            if mode == OperatingMode.IN_CAR:
                actions.append("【车内响应】App 静默通知；座舱屏幕保持暗色")
            else:
                actions.append("【车外响应】App 常规图文推送；车底照地灯低照度补光")
                self.puddle_light_active = True
            return actions

        if level == ThreatLevel.L3_CRITICAL:
            actions.append("【L3 紧急】完整录像加盖防篡改时间戳落盘")
            if is_fire_event:
                # 火情特殊响应: 不使用驱逐动作！
                if mode == OperatingMode.IN_CAR:
                    actions.append("【火情紧急】主驾座椅脉冲震动唤醒车主 + App 强提醒 + 车内撤离提示音")
                    self.seat_vibration_active = True
                else:
                    actions.append("【火情紧急】App 强提醒 + 车外持续声光警示")
                    self.headlight_strobe_active = True
                    self.exterior_speaker_active = True
                return actions

            # 非火情常规破坏入侵 L3 响应
            if mode == OperatingMode.IN_CAR:
                actions.append("【车内响应】App 最高优先级强震动警报")
                actions.append("【车内响应】主驾座椅电机脉冲震动唤醒车主")
                self.seat_vibration_active = True

                # 升级驱逐逻辑: 车主 30 秒内未响应或接触持续，自动启动车外灯光/声音驱逐
                if self.critical_start_time is None:
                    self.critical_start_time = current_time

                if current_time - self.critical_start_time > 30.0:
                    self.is_escalated_eviction = True
                    actions.append("【升级驱逐】车主超过 30 秒未响应，自动启动车外大灯爆闪与声音驱逐")
                    self.headlight_strobe_active = True
                    self.exterior_speaker_active = True
            else:
                # 车外无人模式
                actions.append("【车外响应】大灯/日行灯高频爆闪 3 次！")
                actions.append("【车外响应】车外扬声器播放低频警示音！")
                actions.append("【车外响应】手机 App 立即弹出紧急告警！")
                self.headlight_strobe_active = True
                self.exterior_speaker_active = True

            return actions

        return actions

    def _reset_critical(self):
        self.seat_vibration_active = False
        self.headlight_strobe_active = False
        self.exterior_speaker_active = False
        self.critical_start_time = None
        self.is_escalated_eviction = False

    def _reset_all(self):
        self._reset_critical()
        self.puddle_light_active = False
        self.cockpit_screen_view_active = False
