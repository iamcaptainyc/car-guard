"""
露营火情专用监测模块 (步骤 1.4)
区分稳定火源 (营火/烧烤) 与失控火源 (蔓延扩散/浓烟逼近)
火情响应以通知撤离为主，严禁触发车外声光驱逐
"""

from typing import List, Dict, Tuple, Optional
import time
from camping_guard.types import TrackedEntity, ThreatLevel, ActionType, VehicleZone
from camping_guard.config import GuardConfig

class CampfireMonitor:
    def __init__(self):
        # 记录火源状态: {track_id: {"first_seen": float, "last_seen": float, "initial_area": float, "last_area": float, "last_person_seen_time": float}}
        self.fire_records: Dict[int, dict] = {}

    def update_fire_status(
        self,
        fire_targets: List[TrackedEntity],
        person_targets: List[TrackedEntity],
        current_time: float
    ) -> Tuple[ThreatLevel, str]:
        """
        更新火源与在场人员状态，给出火情威胁等级
        """
        if not fire_targets:
            return ThreatLevel.L0_SAFE, "无火情"

        # 检查是否有人在火源附近 (3米内)
        has_person_nearby = False
        for fire in fire_targets:
            for person in person_targets:
                # 距离相近
                if abs(fire.ground_dist - person.ground_dist) < 2.5:
                    has_person_nearby = True
                    break
            if has_person_nearby:
                break

        highest_threat = ThreatLevel.L0_SAFE
        desc = "营地火源正常受控"

        for fire in fire_targets:
            tid = fire.track_id
            fire_area = fire.bbox.w * fire.bbox.h

            if tid not in self.fire_records:
                self.fire_records[tid] = {
                    "first_seen": current_time,
                    "last_seen": current_time,
                    "initial_area": fire_area,
                    "last_area": fire_area,
                    "last_person_seen_time": current_time if has_person_nearby else 0.0
                }

            rec = self.fire_records[tid]
            rec["last_seen"] = current_time
            if has_person_nearby:
                rec["last_person_seen_time"] = current_time

            # 1. 失控火情判定: 火焰面积持续增大且进入 Zone 2 (< 3m)，或浓烟逼近车体 -> L3
            is_expanding = (fire_area > rec["initial_area"] * 1.8)
            is_close = (fire.ground_dist < GuardConfig.CAMPFIRE_SPREAD_L3_DIST) # < 3.0m

            if is_expanding and is_close:
                return ThreatLevel.L3_CRITICAL, f"失控火源蔓延！火焰面积持续增大且已逼近车体 ({fire.ground_dist:.1f}m)"

            if fire.category == 'smoke' and is_close:
                return ThreatLevel.L3_CRITICAL, f"浓烟严重逼近车体 ({fire.ground_dist:.1f}m)，触发紧急撤离预警"

            # 2. 稳定火源判定
            unattended_duration = current_time - rec["last_person_seen_time"]
            if unattended_duration > GuardConfig.CAMPFIRE_UNATTENDED_L2_SEC: # > 5 分钟 (300s)
                if highest_threat.value < ThreatLevel.L2_WARNING.value:
                    highest_threat = ThreatLevel.L2_WARNING
                    desc = f"营火无人看管已超过 {unattended_duration / 60:.1f} 分钟，提醒车主查看"
            else:
                # 有人看管或刚离开不久 -> L0 监控静默记录
                if highest_threat.value < ThreatLevel.L0_SAFE.value:
                    highest_threat = ThreatLevel.L0_SAFE
                    desc = "营地炉具/篝火稳定且有人看管"

        return highest_threat, desc
