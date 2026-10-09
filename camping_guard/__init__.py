"""
智能汽车露营守卫算法系统核心包
"""
from camping_guard.engine import CampingGuardEngine
from camping_guard.config import GuardConfig
from camping_guard.types import (
    GuardState, ThreatLevel, OperatingMode, VehicleZone,
    TrackedEntity, BoundingBox, ContactSensorEvent, ThreatEventRecord, ActionType
)

__all__ = [
    "CampingGuardEngine",
    "GuardConfig",
    "GuardState",
    "ThreatLevel",
    "OperatingMode",
    "VehicleZone",
    "TrackedEntity",
    "BoundingBox",
    "ContactSensorEvent",
    "ThreatEventRecord",
    "ActionType"
]
