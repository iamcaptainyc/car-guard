"""
智能汽车露营守卫系统 - 核心数据类型与枚举定义
基于 camping_guard_algorithm_flow_revised_1.md
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import time

class GuardState(Enum):
    """系统运行状态 (第五章)"""
    S0_STANDBY = "S0 待机"      # 硬件级低功耗巡检
    S1_SCREENING = "S1 初筛"    # Zone 1 目标检测、追踪与白名单初筛
    S2_ANALYSIS = "S2 研判"     # Zone 2/3 姿态、时序动作与多模态联合研判

class ThreatLevel(Enum):
    """四级威胁分级 (第四章)"""
    L0_SAFE = "L0 安全"         # 无威胁 / 白名单豁免 / 飞虫风噪 / 稳定有人看管篝火
    L1_NOTICE = "L1 提示"       # 陌生实体常规经过 (Zone1 <= 30s, Zone2 <= 10s)
    L2_WARNING = "L2 预警"      # 可疑徘徊、窥探、无人看管火源、ROI扰动、单路相机遮挡
    L3_CRITICAL = "L3 紧急"     # 撬锁/拉门、踢踹撞击、外挂装备盗窃、猛兽冲刺、失控火灾、双路遮挡、中继攻击

class OperatingMode(Enum):
    """工作模式 (第一章)"""
    IN_CAR = "车内模式"         # 车主在车内就寝，侧重静默唤醒车主与内部报警
    OUT_CAR = "车外模式"        # 车内无人，侧重车外声光驱逐与手机远程告警

class VehicleZone(Enum):
    """同心防区 (第二章: Zone 0 远场 >6m, Zone 1 一级区 3~6m, Zone 2 研判区 0~3m)"""
    ZONE_0_FAR = "Zone 0 远场"     # > 6.0m (忽略)
    ZONE_1_OUTER = "Zone 1 一级区" # 3.0m ~ 6.0m
    ZONE_2_INNER = "Zone 2 研判区" # 0 ~ 3.0m (直达车身外轮廓)
    ZONE_3_CONTACT = "Zone 2 研判区" # 兼容别名：Zone 3 取消，统一归入 Zone 2 直达车身

class ActionType(Enum):
    """行为分类"""
    NORMAL_PASS = "常规经过"
    LINGERING = "徘徊滞留"
    WINDOW_PEEKING = "贴窗窥探"
    DOOR_HANDLE_PULL = "拉门撬锁"
    INTENTIONAL_KICK = "蓄意踢踹"
    EQUIPMENT_THEFT = "装备盗窃"
    BEAST_SPRINT = "猛兽冲刺"
    FIRE_SMOKE = "火情烟雾"
    VEHICLE_COLLISION = "车身碰撞"   # 邻车开门碰撞、树木倒塌砸车、外物撞击等车身区域运动扰动

@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    def __getitem__(self, idx: int) -> int:
        return (self.x, self.y, self.w, self.h)[idx]

    def __iter__(self):
        return iter((self.x, self.y, self.w, self.h))

    def __len__(self) -> int:
        return 4

    @property
    def bottom_center(self) -> Tuple[int, int]:
        """接地脚点估算"""
        return (int(self.x + self.w / 2), int(self.y + self.h))

    @property
    def center(self) -> Tuple[int, int]:
        return (int(self.x + self.w / 2), int(self.y + self.h / 2))

class FaceIDState(Enum):
    """FaceID 状态机细粒度状态 (区分无条件 vs 未匹配)"""
    PENDING = "PENDING"                   # 刚建立轨迹，等待人脸观测
    UNAVAILABLE = "UNAVAILABLE"           # 背对相机、脸太小、模糊、遮挡或光照不足
    NO_MATCH = "NO_MATCH"                 # 获得清晰人脸，但未匹配白名单 (陌生人)
    MATCH_CANDIDATE = "MATCH_CANDIDATE"   # 单次命中白名单候选，等待多帧确认
    VERIFIED = "VERIFIED"                 # 多帧置信命中白名单 (车主/合法营友)
    RECHECK = "RECHECK"                   # 遮挡重现、多人交叉或轨迹跳变，需重新验证

@dataclass
class ReIDExemplar:
    """人员外观特征样本 (多视角样本池)"""
    feature: Any                          # 512-dim L2-normalized numpy array
    timestamp: float
    quality_score: float                  # 人体框检测置信度与清晰度评分
    camera_id: int = 0                    # 采集该特征的机位 (0:前, 1:左, 2:右, 3:后)
    view_tag: str = "unknown"             # "front", "back", "side", "unknown"

@dataclass
class PersonSession:
    """跨机位/跨遮挡全局人员会话实体 (用于徘徊研判与身份保持)"""
    session_id: str                       # 全局会话标识，如 "person_session_001"
    exemplars: List[ReIDExemplar] = field(default_factory=list)
    first_seen_time: float = field(default_factory=time.time)
    last_seen_time: float = field(default_factory=time.time)
    cumulative_dwell_time: float = 0.0    # 跨轨迹累计可见滞留时长 (秒)
    visit_count: int = 1                  # 重访进入次数
    is_whitelisted: bool = False          # 是否确认为白名单
    whitelist_name: Optional[str] = None  # 白名单姓名/身份备注
    current_track_uids: List[str] = field(default_factory=list) # 当前绑定的活跃轨迹

@dataclass
class TrackedEntity:
    """被追踪目标元数据"""
    track_id: int
    category: str                   # 'person', 'car', 'small_animal', 'large_beast', 'fire', 'smoke'
    confidence: float
    bbox: BoundingBox
    ground_dist: float              # 接地点到车身水平距离 (米)
    radial_speed: float = 0.0       # 径向逼近速度 (米/秒, 负数朝向车身)
    zone: VehicleZone = VehicleZone.ZONE_0_FAR
    is_whitelisted: bool = False
    first_seen_time: float = field(default_factory=time.time)
    last_seen_time: float = field(default_factory=time.time)
    cumulative_dwell_time: float = 0.0 # 累计滞留时长 (秒)
    action_type: ActionType = ActionType.NORMAL_PASS
    action_start_time: Optional[float] = None
    keypoints: Optional[Dict[str, Tuple[float, float]]] = None # 骨骼关键点
    # FaceID 与 ReID 扩展字段
    track_uid: str = ""                                         # 唯一轨迹标识符 (如 "cam1_trk12")
    person_session_id: Optional[str] = None                     # 绑定的全局人员会话 ID
    face_state: FaceIDState = FaceIDState.PENDING               # 细粒度 FaceID 状态机
    face_id_name: Optional[str] = None                          # 命中的白名单名称
    reid_feature: Optional[Any] = None                          # 当前提取的最优 ReID 特征向量
    association_version: int = 1                                # 异步任务防踩内存关联版本号
    velocity_pixel: Tuple[float, float] = (0.0, 0.0)            # 图像坐标系中心运动速度 (vx, vy, 像素/帧)

@dataclass
class BodyROI:
    """车身敏感部件与防碰撞固定 ROI 规格 (阶段 0.5)"""
    roi_id: str                     # "handle_fl", "handle_fr", "charge_port", "trunk_lock", "roof_rack", "body_flank_left"
    name: str                       # 中文名称
    camera_id: int                  # 0:前, 1:左, 2:右, 3:后
    polygon: List[Tuple[int, int]]  # 像素坐标多边形
    threshold_ratio: float = 0.12   # 前景扰动面积触发阈值比例
    confirm_frames: int = 2         # 连续确认帧数
    roi_type: str = "part"          # "part" (敏感部件: 门把手/充电口/车锁) 或 "body" (车身碰撞监视区: 门板/车侧/车顶)

Zone3ROI = BodyROI # 兼容别名

@dataclass
class ContactSensorEvent:
    """整车接触类传感器硬件事件 (CAN / IMU)"""
    timestamp: float
    event_type: str                 # "DOOR_HANDLE_PULL", "IMU_VIBRATION_PEAK", "DOOR_AJAR", "CHARGING_TAMPER", "UNAUTHORIZED_UNLOCK"
    intensity: float = 1.0          # 震动强度或信号幅值
    extra_info: Dict[str, Any] = field(default_factory=dict)

class ClockSource(Enum):
    """授时时钟源 (阶段 4 防篡改存证)"""
    GNSS_PPS_SYNC = "GNSS_PPS_SYNC"           # 卫星原子授时 (最高信任度)
    CELLULAR_NTP = "CELLULAR_NTP"             # 4G/5G 基站网络授时
    LOCAL_RTC_UNSYNCED = "LOCAL_RTC_UNSYNCED" # 本地硬件 RTC 离线时钟

@dataclass
class ThreatEventRecord:
    """威胁事件审计记录"""
    event_id: str
    timestamp: float
    threat_level: ThreatLevel
    operating_mode: OperatingMode
    action_type: ActionType
    description: str
    target_id: Optional[int] = None
    location_desc: str = "车周"
    has_contact_signal: bool = False
    vlm_summary: Optional[str] = None
    video_clip_path: Optional[str] = None
    hash_signature: Optional[str] = None
    final_threat_level: Optional[ThreatLevel] = None
    clock_source: ClockSource = ClockSource.GNSS_PPS_SYNC

@dataclass
class CameraCalib:
    """相机标定外参及规格 (车规级 4 路环视 AVM 标准)"""
    camera_id: int
    name: str
    camera_height_m: float
    camera_pitch_deg: float
    fov_deg: float = 190.0
    resolution: Tuple[int, int] = (640, 480)
    install_pos_m: Tuple[float, float, float] = (0.0, 0.0, 1.0) # [x, y, z] 车身物理安装坐标 (m)
    camera_yaw_deg: float = 0.0                                 # 偏航角 (deg, 0:前, 90:左, -90:右, 180:后)
    camera_roll_deg: float = 0.0                                # 翻滚角 (deg)
    vehicle_body_side: str = "front"                            # "front", "left", "right", "rear"
    zone1_polygon: Optional[List[Tuple[int, int]]] = None       # Zone 1 (3~6m) 像素参考多边形
    zone2_polygon: Optional[List[Tuple[int, int]]] = None       # Zone 2 (0~3m) 像素参考多边形

@dataclass
class VehicleInfo:
    """车辆基础参数"""
    model_name: str
    vin: str
    wheelbase_m: float = 2.95
    width_m: float = 1.98
    length_m: float = 4.98

@dataclass
class VehicleCalibration:
    """全车统一标定对象 (从 vehicle_calib.json 导入)"""
    vehicle_info: VehicleInfo
    cameras: Dict[int, CameraCalib]
    zone3_rois: List[Zone3ROI]
    thresholds: Dict[str, float] = field(default_factory=dict)

    def get_camera(self, camera_id: int) -> Optional[CameraCalib]:
        return self.cameras.get(camera_id)

    def get_rois_for_camera(self, camera_id: int) -> List[Zone3ROI]:
        return [r for r in self.zone3_rois if r.camera_id == camera_id]

