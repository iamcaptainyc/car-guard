"""
S2 阶段姿态与短时序动作识别模块 (步骤 2.1)
接入 RTMPose 17 骨骼关键点，实现贴窗脊柱倾角几何计算、手腕-门把手拓扑判定与踢踹冲量分析
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import time

from camping_guard.types import TrackedEntity, ActionType, ThreatLevel, VehicleZone
from camping_guard.config import GuardConfig

class PoseActionClassifier:
    def __init__(self):
        # 记录各目标动作持续起始时间: {track_id: {"action": ActionType, "start_time": float, "frame_count": int}}
        self.action_states: Dict[int, dict] = {}
        # 记录目标历史脚踝位置用于踢踹检测: {track_id: [(time, ankle_x, ankle_y)]}
        self.ankle_history: Dict[int, list] = {}

    def classify_action(
        self,
        target: TrackedEntity,
        current_time: float,
        window_roi_polygons: Optional[List[List[Tuple[int, int]]]] = None,
        handle_roi_polygons: Optional[List[List[Tuple[int, int]]]] = None,
        rack_roi_polygons: Optional[List[List[Tuple[int, int]]]] = None
    ) -> Tuple[ActionType, ThreatLevel, str]:
        """
        输入目标轨迹、骨骼点与场景 ROI，输出动作类型与纯视觉建议威胁等级
        """
        tid = target.track_id
        if tid not in self.action_states:
            self.action_states[tid] = {
                "action": ActionType.NORMAL_PASS,
                "start_time": current_time,
                "frame_count": 0
            }

        state = self.action_states[tid]
        state["frame_count"] += 1
        kpts = target.keypoints or {}

        # ================= 1. 猛兽冲刺判定 =================
        if target.category == 'large_beast' and target.confidence >= 0.6:
            if target.radial_speed < -GuardConfig.RADIAL_SPEED_BEAST_L3 and target.zone == VehicleZone.ZONE_2_INNER:
                if state["action"] == ActionType.BEAST_SPRINT:
                    if state["frame_count"] >= 3:
                        return ActionType.BEAST_SPRINT, ThreatLevel.L3_CRITICAL, "大型猛兽高速径向冲刺进入警戒防区"
                else:
                    state["action"] = ActionType.BEAST_SPRINT
                    state["start_time"] = current_time
                    state["frame_count"] = 1
                return ActionType.BEAST_SPRINT, ThreatLevel.L2_WARNING, "大型猛兽逼近警戒"

        # ================= 2. 骨骼点驱动：蓄意踢踹判定 =================
        if kpts:
            la = kpts.get("left_ankle")
            ra = kpts.get("right_ankle")
            active_ankle = la if (la and la[2] > 0.5) else ra
            if active_ankle:
                if tid not in self.ankle_history:
                    self.ankle_history[tid] = []
                self.ankle_history[tid].append((current_time, active_ankle[0], active_ankle[1]))
                while len(self.ankle_history[tid]) > 5:
                    self.ankle_history[tid].pop(0)

                # 计算瞬时脚踝速度 (像素/秒)
                if len(self.ankle_history[tid]) >= 2:
                    dt = self.ankle_history[tid][-1][0] - self.ankle_history[tid][0][0]
                    if dt > 0.05:
                        dy = abs(self.ankle_history[tid][-1][2] - self.ankle_history[tid][0][2])
                        ankle_speed = dy / dt
                        # 若脚部高速向前踢击且接近车身
                        if ankle_speed > 220.0 and target.ground_dist <= 1.0:
                            state["action"] = ActionType.INTENTIONAL_KICK
                            return ActionType.INTENTIONAL_KICK, ThreatLevel.L2_WARNING, "视觉判定脚部高速踢向车身钣金"

        if state["action"] == ActionType.INTENTIONAL_KICK:
            return ActionType.INTENTIONAL_KICK, ThreatLevel.L2_WARNING, "视觉踢踹动作指向车身"

        # ================= 3. 骨骼点驱动：贴窗窥探几何判定 =================
        is_peeking_pose = False
        if kpts:
            ls = kpts.get("left_shoulder")
            rs = kpts.get("right_shoulder")
            lh = kpts.get("left_hip")
            rh = kpts.get("right_hip")
            nose = kpts.get("nose")

            if ls and rs and lh and rh:
                # 计算脊柱中点与上身前倾夹角
                mid_shoulder = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
                mid_hip = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
                dx = abs(mid_shoulder[0] - mid_hip[0])
                dy = abs(mid_shoulder[1] - mid_hip[1])
                spine_tilt_angle = np.degrees(np.arctan2(dx, max(dy, 1e-3)))

                # 判定条件: 脊柱前倾角 > 25° 且头部贴近车窗 (< 0.5m)
                if spine_tilt_angle > 25.0 and target.ground_dist <= GuardConfig.WINDOW_PEEK_DISTANCE:
                    is_peeking_pose = True

        if is_peeking_pose or state["action"] == ActionType.WINDOW_PEEKING:
            if state["action"] == ActionType.WINDOW_PEEKING:
                duration = current_time - state["start_time"]
                if duration >= GuardConfig.WINDOW_PEEK_DURATION_SEC:
                    return ActionType.WINDOW_PEEKING, ThreatLevel.L2_WARNING, f"贴窗窥探已持续 {duration:.1f} 秒 (上身前倾>25°)"
                return ActionType.WINDOW_PEEKING, ThreatLevel.L1_NOTICE, f"贴窗观察姿态 ({duration:.1f}s)"
            elif is_peeking_pose:
                state["action"] = ActionType.WINDOW_PEEKING
                state["start_time"] = current_time
                state["frame_count"] = 1
                return ActionType.WINDOW_PEEKING, ThreatLevel.L1_NOTICE, "检测到上身前倾贴窗观察姿态"

        # ================= 4. 骨骼点驱动：手部触碰门把手判定 =================
        is_hand_on_handle = False
        if kpts and handle_roi_polygons:
            lw = kpts.get("left_wrist")
            rw = kpts.get("right_wrist")
            wrists = [w for w in [lw, rw] if w and w[2] > 0.4]

            for h_poly in handle_roi_polygons:
                pts = np.array(h_poly, dtype=np.int32)
                for w_pt in wrists:
                    dist_to_handle = cv2.pointPolygonTest(pts, (float(w_pt[0]), float(w_pt[1])), True)
                    # 在把手多边形内 (>= 0) 或距把手边缘 <= 35 像素 (约 15cm)
                    if dist_to_handle >= -35.0:
                        is_hand_on_handle = True
                        break
                if is_hand_on_handle:
                    break

        # 结合目标空间距离回退判定
        if is_hand_on_handle or state["action"] == ActionType.DOOR_HANDLE_PULL:
            if state["action"] == ActionType.DOOR_HANDLE_PULL:
                duration = current_time - state["start_time"]
                if duration >= GuardConfig.DOOR_PULL_DURATION_SEC:
                    return ActionType.DOOR_HANDLE_PULL, ThreatLevel.L2_WARNING, f"视觉手部持续在门把手区域拉动 {duration:.1f} 秒"
                return ActionType.DOOR_HANDLE_PULL, ThreatLevel.L1_NOTICE, f"手部触碰门把手区域 ({duration:.1f}s)"
            elif is_hand_on_handle:
                state["action"] = ActionType.DOOR_HANDLE_PULL
                state["start_time"] = current_time
                state["frame_count"] = 1
                return ActionType.DOOR_HANDLE_PULL, ThreatLevel.L1_NOTICE, "手部触碰门把手区域"

        # ================= 5. 装备盗窃持续判定 =================
        if state["action"] == ActionType.EQUIPMENT_THEFT:
            duration = current_time - state["start_time"]
            if duration >= GuardConfig.THEFT_DURATION_SEC:
                return ActionType.EQUIPMENT_THEFT, ThreatLevel.L2_WARNING, f"视觉手部在外挂装备/充电口滞留 {duration:.1f} 秒"
            return ActionType.EQUIPMENT_THEFT, ThreatLevel.L1_NOTICE, f"手部接近外挂装备区域 ({duration:.1f}s)"

        # ================= 6. 贴车极近距离 (<0.5m) 托底防漏 (Floor Rule) =================
        # 非人员目标 (车辆/障碍物) 极近距离贴车兜底 (绝对不可为 L0 安全)
        if target.category != 'person' and target.ground_dist <= 0.8:
            if target.radial_speed < -GuardConfig.RADIAL_SPEED_WAKE_S2:
                return ActionType.NORMAL_PASS, ThreatLevel.L2_WARNING, f"外部{target.category}快速逼近车体 ({target.ground_dist:.2f}m)"
            return ActionType.NORMAL_PASS, ThreatLevel.L1_NOTICE, f"外部{target.category}极度贴近车身 ({target.ground_dist:.2f}m)"

        # ================= 7. 滞留徘徊判定 =================
        if target.category == 'person' and not target.is_whitelisted:
            if target.zone == VehicleZone.ZONE_2_INNER and target.cumulative_dwell_time > GuardConfig.DWELL_TIME_Z2_L2_SEC:
                return ActionType.LINGERING, ThreatLevel.L2_WARNING, f"陌生人在核心防区滞留超 {target.cumulative_dwell_time:.1f} 秒"
            elif target.zone == VehicleZone.ZONE_1_OUTER and target.cumulative_dwell_time > GuardConfig.DWELL_TIME_Z1_L2_SEC:
                return ActionType.LINGERING, ThreatLevel.L2_WARNING, f"陌生人在外围警戒区滞留超 {target.cumulative_dwell_time:.1f} 秒"

        # ================= 8. 径向快速逼近 =================
        if target.category == 'person' and target.zone == VehicleZone.ZONE_2_INNER:
            if target.radial_speed < -GuardConfig.RADIAL_SPEED_HUMAN_L2:
                return ActionType.NORMAL_PASS, ThreatLevel.L2_WARNING, f"人员快速逼近车体 (径向速度 {abs(target.radial_speed):.2f} m/s)"

        # 默认正常通行与近场兜底 (0.5m 范围内托底铁律：绝对不可降级为 L0_SAFE)
        if target.ground_dist <= GuardConfig.ZONE_INNER_DIST or target.ground_dist <= 0.5:
            return ActionType.NORMAL_PASS, ThreatLevel.L1_NOTICE, f"目标处于贴车极近距离 ({target.ground_dist:.2f}m)"
        if target.ground_dist <= 0.8 or target.zone in [VehicleZone.ZONE_1_OUTER, VehicleZone.ZONE_2_INNER]:
            return ActionType.NORMAL_PASS, ThreatLevel.L1_NOTICE, "常规活动经过"

        return ActionType.NORMAL_PASS, ThreatLevel.L0_SAFE, "安全"
