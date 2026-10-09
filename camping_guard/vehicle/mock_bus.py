"""
整车总线与传感器模拟发生器 (Mock CAN / IMU / UWB)
提供整车状态信号、物理接触模拟与中继攻击测试接口
"""

from typing import Dict, Any, Optional, Tuple, List
import time
from camping_guard.types import ContactSensorEvent

class MockVehicleBus:
    def __init__(self):
        # 整车基础状态 (满足第一章激活条件)
        self.gear: str = "P"            # P 档
        self.epb_engaged: bool = True   # EPB 电子手刹已拉起
        self.all_doors_locked: bool = True # 四门/后备箱上锁
        self.guard_switch_on: bool = True  # 露营守卫开关开启
        self.battery_soc: float = 78.0     # 动力电池 SOC (%)

        # 工作模式相关
        self.seat_occupied: bool = True # 主驾/座舱有人占位 (决定车内/车外模式)

        # UWB 车钥匙空间物理坐标 (车身坐标系米: x横向, y纵向)
        # None 表示钥匙未在车周或离车遥远
        self.uwb_key_pos: Optional[Tuple[float, float]] = (0.5, 0.8) # 默认在车旁 0.94m 内

        # 待消费的物理硬件事件列表
        self.pending_events: List[ContactSensorEvent] = []

    def inject_vibration_peak(self, intensity: float = 8.5):
        """注入车身 IMU 剧烈冲击震动 (踢踹/撞击)"""
        event = ContactSensorEvent(
            timestamp=time.time(),
            event_type="IMU_VIBRATION_PEAK",
            intensity=intensity,
            extra_info={"axis": "lateral", "accel_rms": intensity}
        )
        self.pending_events.append(event)

    def inject_door_handle_pull(self, handle_id: str = "handle_fl"):
        """注入门把手被机械拉动/电容触摸信号"""
        event = ContactSensorEvent(
            timestamp=time.time(),
            event_type="DOOR_HANDLE_PULL",
            intensity=1.0,
            extra_info={"handle_id": handle_id}
        )
        self.pending_events.append(event)

    def inject_charging_tamper(self):
        """注入充电口锁止被强行撬动/拔出信号"""
        event = ContactSensorEvent(
            timestamp=time.time(),
            event_type="CHARGING_TAMPER",
            intensity=1.0
        )
        self.pending_events.append(event)

    def inject_unauthorized_unlock_request(self):
        """注入越权/异常解锁请求 (例如黑客中继攻击模拟)"""
        event = ContactSensorEvent(
            timestamp=time.time(),
            event_type="UNAUTHORIZED_UNLOCK",
            intensity=1.0,
            extra_info={"source": "BLE_RELAY_SIMULATION"}
        )
        self.pending_events.append(event)

    def pull_pending_events(self) -> List[ContactSensorEvent]:
        events = list(self.pending_events)
        self.pending_events.clear()
        return events
