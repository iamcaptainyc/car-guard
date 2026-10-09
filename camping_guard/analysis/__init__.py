"""
露营守卫研判与多模态融合层模块
"""
from camping_guard.analysis.pose_action import PoseActionClassifier
from camping_guard.analysis.fusion import MultiSourceFusionEngine
from camping_guard.analysis.fire_monitor import CampfireMonitor

__all__ = ["PoseActionClassifier", "MultiSourceFusionEngine", "CampfireMonitor"]
