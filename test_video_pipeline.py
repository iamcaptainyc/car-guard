"""
单路鱼眼相机 MP4 视频实测与算法回放评估工具 (test_video_pipeline.py)
用于导入实车/测试台架拍摄的单路鱼眼视频，完整验证 S0/S1/S2 状态流转、接地测距、车身固定 ROI 微动、RTMPose 姿态、多模态联合确认与 VLM 复核
"""

import argparse
import copy
import os
import sys
import queue
import threading
import time
from typing import Optional, List, Dict, Tuple
from pathlib import Path

if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import cv2
import numpy as np

from camping_guard.engine import CampingGuardEngine
from camping_guard.types import GuardState, ThreatLevel, VehicleZone, ActionType
from camping_guard.config import load_vehicle_calibration
from camping_guard.perception.detector import ONNXObjectDetector, ParallelEnsembleDetector, VisualFlameDetector
from camping_guard.perception.pose_estimator import RTMPoseEstimator
from camping_guard.perception.distance import GroundDistanceEstimator
from run_simulation import draw_skeleton, draw_vehicle_and_zones
from camping_guard.profiler import PipelineProfiler


_SIDEBAR_FONT_CACHE = {}


def get_sidebar_font(size: int = 14):
    """自适应查找并缓存高质量 TrueType 中文字体，避免 cv2.putText 乱码"""
    if size in _SIDEBAR_FONT_CACHE:
        return _SIDEBAR_FONT_CACHE[size]
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",                             # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",                           # 黑体
        "C:/Windows/Fonts/simsun.ttc",                           # 宋体
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",        # Ubuntu/Jetson 文泉驿微米黑
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",# Noto Sans CJK
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                from PIL import ImageFont
                f = ImageFont.truetype(c, size)
                _SIDEBAR_FONT_CACHE[size] = f
                return f
            except Exception:
                pass
    try:
        from PIL import ImageFont
        f = ImageFont.load_default()
    except Exception:
        f = None
    _SIDEBAR_FONT_CACHE[size] = f
    return f


def draw_text_utf8(image: np.ndarray, text: str, org: Tuple[int, int], font_size: int = 14, color: Tuple[int, int, int] = (255, 255, 255)):
    """支持 UTF-8 中文与英文的高质感抗锯齿无乱码文字渲染 (含中文自动走 PIL TrueType，纯英文走 cv2.putText)"""
    if not text:
        return image
    has_unicode = any(ord(c) > 127 for c in text)
    if not has_unicode:
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_size / 28.0, color, 1, cv2.LINE_AA)
        return image

    try:
        from PIL import Image, ImageDraw
        font = get_sidebar_font(font_size)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img_pil)
        rgb_color = (color[2], color[1], color[0]) if len(color) == 3 else color
        draw_y = max(0, org[1] - font_size)
        draw.text((org[0], draw_y), text, font=font, fill=rgb_color)
        image[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_size / 28.0, color, 1, cv2.LINE_AA)
    return image


def draw_telemetry_sidebar(
    video_frame: np.ndarray,
    engine: CampingGuardEngine,
    tick_result: dict,
    real_fps: float,
    frame_idx: int,
    total_frames: int,
    is_paused: bool,
    sim_status: str,
    camera_id: int,
    detector_name: str = "",
    panel_w: int = 380,
    targets_override: list = None,
    roi_disturbances_override: list = None,
    ai_fps: float = None,
    ai_e2e_ms: float = None,
    is_realtime_mode: bool = False
) -> np.ndarray:
    """
    非遮挡式独立侧边栏监控仪表盘:
    基于 PIL TrueType 硬件渲染，彻底根治 OpenCV 中文乱码 (???)，呈现 100% 原始视频原画
    """
    h, w = video_frame.shape[:2]
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = video_frame  # 视频原画无遮挡放置于左侧

    sidebar = canvas[:, w:]
    sidebar[:] = (20, 24, 30)  # 暗黑高质感仪表板背景

    # 左右分割线与区块分割线
    cv2.line(canvas, (w, 0), (w, h), (55, 65, 80), 2)
    cv2.line(sidebar, (15, 54), (panel_w - 15, 54), (40, 48, 60), 1)
    cv2.line(sidebar, (15, 148), (panel_w - 15, 148), (40, 48, 60), 1)
    cv2.line(sidebar, (15, 230), (panel_w - 15, 230), (40, 48, 60), 1)
    cv2.line(sidebar, (15, 280), (panel_w - 15, 280), (40, 48, 60), 1)
    cv2.line(sidebar, (15, 380), (panel_w - 15, 380), (40, 48, 60), 1)
    cv2.line(sidebar, (15, 428), (panel_w - 15, 428), (40, 48, 60), 1)

    if sim_status:
        cv2.rectangle(sidebar, (10, h - 38), (panel_w - 10, h - 6), (25, 45, 60), -1)
        cv2.rectangle(sidebar, (10, h - 38), (panel_w - 10, h - 6), (50, 180, 255), 1)

    # 转换为 PIL 一次性批量绘制抗锯齿中英文字符，耗时 < 1ms
    from PIL import Image, ImageDraw
    sidebar_rgb = cv2.cvtColor(sidebar, cv2.COLOR_BGR2RGB)
    sidebar_pil = Image.fromarray(sidebar_rgb)
    draw = ImageDraw.Draw(sidebar_pil)

    def put_sb(text: str, y: int, font_size: int = 13, color=(220, 220, 220)):
        font = get_sidebar_font(font_size)
        rgb_color = (color[2], color[1], color[0]) if len(color) == 3 else color
        top_y = max(2, y - font_size - 1)
        draw.text((15, top_y), str(text), font=font, fill=rgb_color)

    # 1. 顶部监控标题
    cam_info = engine.calib.get_camera(camera_id) if hasattr(engine, 'calib') and engine.calib else None
    cam_name = cam_info.name if cam_info else f"Cam_{camera_id}"
    mode_tag = " [实时流速]" if is_realtime_mode else ""
    put_sb("CAMPING GUARD 露营守卫监控", 24, font_size=16, color=(0, 255, 255))
    put_sb(f"Camera {camera_id} ({cam_name}) | {w}x{h} @ {real_fps:.1f}FPS{mode_tag}", 44, font_size=12, color=(160, 180, 200))

    # 2. 威胁等级与状态
    threat_str = tick_result.get("threat_level", "L0 正常安全")
    if "L3" in threat_str or "紧急" in threat_str:
        threat_color = (0, 0, 255)
    elif "L2" in threat_str or "预警" in threat_str:
        threat_color = (0, 140, 255)
    elif "L1" in threat_str or "提示" in threat_str:
        threat_color = (255, 200, 0)
    else:
        threat_color = (0, 255, 120)

    state_str = tick_result.get("state", "S0 待机")
    mode_str = tick_result.get("mode", "车内模式")

    put_sb(f"系统威胁: [ {threat_str} ]", 76, font_size=14, color=threat_color)
    put_sb(f"守卫状态: {state_str}", 98, font_size=13, color=(210, 220, 240))
    put_sb(f"工作模式: {mode_str}", 118, font_size=13, color=(210, 220, 240))

    uwb_online = (engine.vehicle_bus.uwb_key_pos is not None)
    uwb_text = "UWB 钥匙: 在场 <=1.5m (白名单)" if uwb_online else "UWB 钥匙: 远离现场 / 不在线"
    uwb_color = (0, 255, 120) if uwb_online else (0, 160, 255)
    put_sb(uwb_text, 138, font_size=12, color=uwb_color)

    # 3. 目标感知追踪
    active_targets = targets_override if targets_override is not None else engine.active_targets
    num_targets = len(active_targets)
    put_sb(f"目标感知追踪 (活跃目标: {num_targets})", 170, font_size=13, color=(255, 200, 80))
    if num_targets == 0:
        put_sb("当前无动态目标，防区安全", 194, font_size=12, color=(140, 140, 150))
    else:
        for idx, t in enumerate(active_targets[:2]):
            y_base = 192 + idx * 28
            z_short = t.zone.value.split()[0] if hasattr(t.zone, 'value') else str(t.zone)
            pose_status = "17点骨骼活跃" if t.keypoints else "休眠"
            cat_lower = str(t.category).lower()
            if "person" in cat_lower or "人" in str(t.category):
                cat_cn = "行人"
            elif "large_beast" in cat_lower or "beast" in cat_lower:
                sp_cn = getattr(t, 'species_cn', None)
                cat_cn = f"猛兽({sp_cn})" if sp_cn and sp_cn not in ['未知动物', '通用动物'] else "猛兽"
            elif "small_animal" in cat_lower or "animal" in cat_lower:
                sp_cn = getattr(t, 'species_cn', None)
                cat_cn = f"动物({sp_cn})" if sp_cn and sp_cn not in ['未知动物', '通用动物'] else "动物"
            elif "fire" in cat_lower:
                cat_cn = "火灾"
            elif "smoke" in cat_lower:
                cat_cn = "浓烟"
            elif "car" in cat_lower or "车" in str(t.category):
                cat_cn = "车辆"
            else:
                cat_cn = str(t.category)
            sess_txt = f"({t.person_session_id})" if t.person_session_id else ""
            put_sb(f"ID:{t.track_id}{sess_txt} [{cat_cn}] {t.ground_dist:.1f}m ({z_short}) vr:{t.radial_speed:+.1f}m/s", y_base, font_size=12, color=(220, 220, 220))
            put_sb(f"姿态: {pose_status} | 驻留: {t.cumulative_dwell_time:.1f}s", y_base + 14, font_size=11, color=(120, 255, 180))

    # 4. 车身敏感部位 (Zone 3 Fixed ROIs)
    put_sb("车身敏感部件监视 (Zone 3):", 250, font_size=13, color=(180, 200, 240))
    active_disturbances = roi_disturbances_override if roi_disturbances_override is not None else engine.active_roi_disturbances
    if active_disturbances:
        d_names = [d.get("roi_id", "") for d in active_disturbances]
        put_sb(f"【警报】部件扰动: {', '.join(d_names)}", 270, font_size=12, color=(0, 0, 255))
    else:
        put_sb("门把手/充电口/防撞区: 正常 (无触碰)", 270, font_size=12, color=(0, 255, 120))

    # 5. E2E 性能与模型开销
    frame_metrics = tick_result.get("metrics")
    e2e_val = ai_e2e_ms if ai_e2e_ms is not None else tick_result.get("e2e_ms", 0.0)
    det_label = f" ({detector_name})" if detector_name else ""
    put_sb(f"E2E 端到端耗时与模型负载{det_label}:", 302, font_size=13, color=(80, 240, 255))
    if is_realtime_mode and ai_fps is not None:
        put_sb(f"视频显示: {real_fps:4.1f} FPS (实时平滑流速)", 322, font_size=12, color=(0, 255, 200))
        put_sb(f"AI耗时: {e2e_val:5.1f} ms | AI吞吐: {ai_fps:4.1f} FPS", 340, font_size=12, color=(220, 220, 220))
        put_sb("渲染模式: 帧框同源合一 (100%零框延迟)", 358, font_size=11, color=(120, 255, 180))
    else:
        put_sb(f"E2E 延时: {e2e_val:5.1f} ms ({1000.0/max(e2e_val, 1e-3):4.1f} FPS)", 324, font_size=12, color=(220, 220, 220))

    if frame_metrics:
        model_items = []
        for m_name, m_info in frame_metrics.get_model_summary().items():
            short = m_name.replace("ArcFace-MobileFaceNet", "ArcFace").replace("YOLO11n-Det", "YOLO").replace("SCRFD-500M", "SCRFD").replace("RTMPose-s", "RTMPose").replace("OSNet-ReID", "ReID").replace("MobileNetV3-AnimalCls", "AnimalCls")
            model_items.append(f"{short}:{m_info['total_ms']:.1f}ms")
        if model_items:
            put_sb(" | ".join(model_items[:3]), 370, font_size=11, color=(160, 200, 220))

    # 6. VLM 复核与防篡改审计
    vlm_busy = tick_result.get("vlm_is_inferencing", False)
    vlm_dur = tick_result.get("vlm_last_duration_ms", 0.0)
    vlm_summary = tick_result.get("vlm_last_summary") or tick_result.get("vlm_summary") or "慢通路待机"
    if vlm_busy:
        vlm_display = "VLM 复核: [推理中] 正在深度语义研判..."
        vlm_color = (0, 200, 255)
    elif vlm_dur > 0:
        vlm_display = f"VLM 复核 ({vlm_dur:.0f}ms): {vlm_summary}"
        vlm_color = (120, 255, 180)
    else:
        vlm_display = f"VLM 复核: {vlm_summary}"
        vlm_color = (180, 230, 180)

    latest_hash = engine.recorder.prev_hash[:10] + ".." if engine.recorder.prev_hash != engine.recorder.GENESIS_HASH else "GENESIS"
    put_sb(vlm_display[:32], 400, font_size=12, color=vlm_color)
    put_sb(f"防篡改审计: {latest_hash} (区块: {len(engine.recorder.manifest)})", 418, font_size=11, color=(160, 180, 210))

    # 7. 播放进度与快捷键提示
    play_status = "[PAUSED]" if is_paused else "[PLAYING]"
    put_sb(f"播放进度: Frame {frame_idx}/{total_frames} {play_status}", 444, font_size=12, color=(0, 255, 255))
    put_sb("按键: [Space]暂停 [d]步进 [v]切换视图", 462, font_size=11, color=(180, 180, 190))
    put_sb("注入: [h]门把手 [k]震动 [u]钥匙 [q]退出", 478, font_size=11, color=(180, 180, 190))

    # 8. 突发事件消息通知条
    if sim_status:
        put_sb(f"[EVENT] {sim_status}", h - 18, font_size=12, color=(50, 255, 255))

    # 写回 canvas
    sidebar[:] = cv2.cvtColor(np.array(sidebar_pil), cv2.COLOR_RGB2BGR)
    return canvas


def disable_windows_quick_edit():
    """在 Windows 终端下禁用 QuickEdit 模式，防止用户鼠标误点击控制台导致 Python 进程被操作系统内核强行挂起暂停"""
    if os.name != 'nt':
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        h_stdin = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE = -10
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(h_stdin, ctypes.byref(mode)):
            ENABLE_QUICK_EDIT_MODE = 0x0040
            ENABLE_EXTENDED_FLAGS = 0x0080
            new_mode = (mode.value & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS
            kernel32.SetConsoleMode(h_stdin, new_mode)
    except Exception:
        pass


def parse_camera_id(val: str) -> int:
    """支持 0,1,2,3 数字或 front/left/right/rear 中英文别名"""
    val_str = str(val).strip().lower()
    mapping = {
        "0": 0, "front": 0, "f": 0, "前": 0, "车头": 0,
        "1": 1, "left": 1, "l": 1, "左": 1, "左侧": 1, "左视": 1,
        "2": 2, "right": 2, "r": 2, "右": 2, "右侧": 2, "右视": 2,
        "3": 3, "rear": 3, "back": 3, "b": 3, "后": 3, "车尾": 3, "后视": 3
    }
    if val_str in mapping:
        return mapping[val_str]
    try:
        cid = int(val_str)
        if cid in [0, 1, 2, 3]:
            return cid
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"无法识别相机通道: '{val}' (支持 0/1/2/3 或 front/left/right/rear)")


class ConsoleKeyListener:
    """
    跨平台控制台非阻塞按键监听器 (后台守护线程)
    无需依赖 OpenCV 窗口聚焦，支持直接在终端控制台中捕获单个按键 (q, ESC, 空格, d, r 等)
    - Windows: 基于 msvcrt.kbhit() / getch()
    - Linux / Jetson: 基于 termios cbreak 模式 / select
    """
    def __init__(self):
        self.queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True, name="ConsoleKeyListener")
        self.thread.start()

    def _run(self):
        if sys.platform.startswith("win"):
            try:
                import msvcrt
                while self.running:
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if ch in [b'\x00', b'\xe0']:
                            if msvcrt.kbhit():
                                msvcrt.getch()
                            continue
                        if ch in [b'\x1b', b'\x03']:  # ESC 或 Ctrl+C
                            self.queue.put(27)
                        else:
                            try:
                                self.queue.put(ord(ch.decode('utf-8', errors='ignore').lower()))
                            except Exception:
                                self.queue.put(ord(ch))
                    time.sleep(0.02)
                return
            except Exception:
                pass

        # Linux / Jetson / macOS
        try:
            import termios
            import tty
            import select
            if sys.stdin.isatty():
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setcbreak(fd)
                    while self.running:
                        r, _, _ = select.select([sys.stdin], [], [], 0.05)
                        if r:
                            ch = sys.stdin.read(1)
                            if ch:
                                if ch in ['\x1b', '\x03']:
                                    self.queue.put(27)
                                else:
                                    self.queue.put(ord(ch.lower()))
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                return
        except Exception:
            pass

        # Fallback for piped or non-tty environments
        while self.running:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                for c in line.strip():
                    self.queue.put(ord(c.lower()))
            except Exception:
                break

    def get_key(self) -> Optional[int]:
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False


def run_video_pipeline(
    video_path: str,
    config_path: str = "vehicle_calib.json",
    camera_id: int = 1,
    camera_height: float = None,
    camera_pitch: float = None,
    camera_yaw: float = None,
    detector_onnx_path: str = None,
    pose_onnx_path: str = None,
    loop_video: Optional[bool] = None,
    output_path: str = None,
    stats_json_path: str = None,
    verbose: bool = False,
    headless: bool = False,
    max_frames: int = None,
    realtime: bool = False,
    conf_threshold: float = 0.25,
    show_sidebar: bool = False,
    detect_interval: int = 1,
    use_trt: bool = False,
    enable_web: bool = False,
    web_port: int = 8080,
    vlm_model_dir: Optional[str] = None,
    vlm_engine_path: Optional[str] = None,
    vlm_visual_engine_path: Optional[str] = None,
    edgellm_dir: Optional[str] = None,
    build_dir: Optional[str] = None,
    vlm_timeout: float = 35.0,
    vlm_max_tokens: int = 128,
    vlm_mode: str = "auto",
    vlm_engine_so: Optional[str] = None,
    vlm_server_port: int = 8090,
    vlm_idle_timeout: int = 60,
    vlm_server_bin: Optional[str] = None,
    cls_onnx_path: Optional[str] = None,
    fire_detector_path: Optional[str] = None,
    enable_fire: bool = True,
    detector_size: Optional[int] = None,
    fire_size: Optional[int] = None,
    vlm_throttle_enabled: bool = True,
    vlm_active_fps: float = 5.0,
    dla_core: Optional[int] = None
):
    from camping_guard.perception.trt_engine import is_jetson
    use_trt = use_trt or is_jetson()
    disable_windows_quick_edit()

    is_cam_or_pipe = isinstance(video_path, int) or (
        isinstance(video_path, str) and (
            video_path.isdigit() or
            video_path.startswith("nvargus") or
            video_path.startswith("rtsp://") or
            video_path.startswith("http://") or
            video_path.startswith("/dev/video")
        )
    )
    if not is_cam_or_pipe:
        video_file = Path(video_path)
        if not video_file.exists():
            print(f"[!] 错误: 找不到视频文件 '{video_path}'")
            return
        cap_arg = str(video_file)
        display_video_name = video_file.name
    else:
        cap_arg = int(video_path) if (isinstance(video_path, int) or (isinstance(video_path, str) and video_path.isdigit())) else video_path
        display_video_name = f"Camera/LiveStream ({video_path})"

    cap = cv2.VideoCapture(cap_arg)
    if not cap.isOpened():
        print(f"[!] 错误: 无法使用 OpenCV 打开视频源 '{video_path}'")
        return

    # 智能设置循环回放策略:
    # 若指定了 output_path 录制输出视频，默认单次处理完毕后自动封口保存并退出 (loop_video=False)
    # 若未指定 output_path (纯实时预览)，默认循环播放 (loop_video=True)
    if loop_video is None:
        loop_video = False if output_path else True

    # 0. 载入统一整车标定配置
    calib = load_vehicle_calibration(config_path)
    cam_info = calib.get_camera(camera_id)
    actual_height = camera_height if camera_height is not None else (cam_info.camera_height_m if cam_info else 1.15)
    actual_pitch = camera_pitch if camera_pitch is not None else (cam_info.camera_pitch_deg if cam_info else -38.0)
    actual_yaw = camera_yaw if camera_yaw is not None else (getattr(cam_info, "camera_yaw_deg", 90.0) if cam_info else 90.0)

    # 动态写入覆盖外参
    if cam_info:
        cam_info.camera_height_m = actual_height
        cam_info.camera_pitch_deg = actual_pitch
        cam_info.camera_yaw_deg = actual_yaw

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    side_desc = getattr(cam_info, "vehicle_body_side", "side") if cam_info else "side"
    cam_name = cam_info.name if cam_info else f"camera_{camera_id}"

    print("=" * 70)
    print(" 智能汽车露营守卫系统 - 单路鱼眼视频评估工具已启动")
    print(f" 车辆型号: {calib.vehicle_info.model_name} (VIN: {calib.vehicle_info.vin})")
    print(f" 标定配置: {config_path}")
    print(f" 视频文件: {display_video_name} ({video_w}x{video_h} @ {fps:.1f} fps, 共 {total_frames} 帧)")
    print(f" 挂载相机通道: Camera {camera_id} ({cam_name}, 车身朝向: {side_desc})")
    print(f" 相机外参: 安装高度={actual_height:.2f}m, 偏航角={actual_yaw:+.1f}°, 俯仰角={actual_pitch:+.1f}°")
    print(f" 循环模式: {'循环重播' if loop_video else '单次处理 (处理完自动保存并退出)'}")
    print(" 快捷键指南 (支持终端控制台直接输入或可视化窗口输入):")
    print("   [SPACE] 暂停 / 继续播放")
    print("   [d]     单帧步进 (暂停时有效)")
    print("   [v]     切换视图模式 (右侧独立数据看板 vs 纯画面极简模式)")
    print("   [h]     注入 CAN 门把手信号 (与视频中拉门动作进行联合确认升 L3)")
    print("   [k]     注入 IMU 震动冲击信号 (与视频中踢踹动作进行联合确认升 L3)")
    print("   [l]     注入中继攻击解锁测试")
    print("   [u]     切换车主 UWB 钥匙在线/远离 (测试白名单免扰)")
    print("   [8]     触发 VLM 假阳性复核测试")
    print("   [9]     审计防篡改哈希链完整性")
    print("   [r]     视频重头播放")
    print("   [q/ESC] 退出程序并安全保存输出视频")
    print("=" * 70)

    # 1. 目标检测器初始化 (支持自动发现/手动加载 YOLO 与 MegaDetector 模型)
    detector = None
    chosen_detector_path = detector_onnx_path

    # 支持用户传入 megadetector 别名或仅传入模型文件名 (自动在 models/megadetector/, models/yolo/ 或当前目录检索)
    if chosen_detector_path:
        cp_lower = str(chosen_detector_path).lower()
        if cp_lower in ["megadetector", "mdv", "mdv6"]:
            for md_cand in [
                "models/megadetector/MDV6-yolov10-c.engine",
                "models/megadetector/MDV6-yolov10-c.onnx",
                "models/yolo/MDV6-yolov10-c.engine",
                "models/yolo/MDV6-yolov10-c.onnx"
            ]:
                if os.path.exists(md_cand):
                    chosen_detector_path = md_cand
                    break
        elif not os.path.exists(chosen_detector_path):
            for search_dir in ["models/megadetector", "models/yolo", "models"]:
                sub_cand = os.path.join(search_dir, chosen_detector_path)
                if os.path.exists(sub_cand):
                    chosen_detector_path = sub_cand
                    break

    if not chosen_detector_path:
        # 优先使用在车周视角人体与车辆检测能力最强的高召回通用模型
        candidates = []
        if use_trt:
            candidates.extend([
                "models/yolo/yolo11s.engine",
                "models/yolo/yolo11n.engine",
                "models/yolo/yolo11n-fisheye.engine",
                "models/yolo/yolo26n.engine",
                "models/megadetector/MDV6-yolov10-c.engine",
                "models/yolo/MDV6-yolov10-c.engine",
                "yolo11s.engine",
                "yolo11n.engine"
            ])
        candidates.extend([
            "models/yolo/yolo11s.onnx",          # ★ 高精度与高召回通用 YOLO11s (实测车周人体检出率最佳)
            "models/yolo/yolo11n.onnx",          # ★ 极速轻量通用 YOLO11n
            "models/yolo/yolov8m.onnx",          # 中容量通用 YOLOv8m
            "models/yolo/yolov8s.onnx",          # 均衡通用 YOLOv8s
            "models/yolo/yolov8n.onnx",          # 轻量通用 YOLOv8n
            "models/yolo/yolo11n-fisheye.onnx", # 俯视交通鱼眼模型 (Fisheye8K)
            "models/yolo/yolov8n-fisheye.onnx",
            "models/yolo/yolo11x-fisheye.onnx",
            "models/megadetector/MDV6-yolov10-c.onnx",
            "models/yolo/MDV6-yolov10-c.onnx",
            "yolo11s.onnx",
            "yolo11n.onnx"
        ])
        for cand in candidates:
            if os.path.exists(cand):
                chosen_detector_path = cand
                break

    if chosen_detector_path:
        if use_trt:
            eng_cand = Path(chosen_detector_path).with_suffix(".engine")
            if eng_cand.is_file():
                chosen_detector_path = str(eng_cand)

    # 1. 目标检测器选择与主检测模式判定
    is_md = False
    if chosen_detector_path:
        det_str = str(chosen_detector_path).lower()
        is_md = ("mdv" in det_str or "megadetector" in det_str)

    # 1.1 二级动物细分类器初始化 (统一规范：无论 YOLO 检出 10 类动物还是 MegaDetector 检出 animal，均支持二级细分类)
    animal_classifier = None
    chosen_cls_path = cls_onnx_path
    if chosen_cls_path and not os.path.exists(chosen_cls_path):
        for search_dir in ["models/cls", "models"]:
            sub_cand = os.path.join(search_dir, chosen_cls_path)
            if os.path.exists(sub_cand):
                chosen_cls_path = sub_cand
                break
    if not chosen_cls_path:
        for cand in [
            "models/cls/mobilenetv3-large.engine" if use_trt else None,
            "models/cls/mobilenetv3.engine" if use_trt else None,
            "models/cls/mobilenetv3-large.onnx",
            "models/cls/mobilenetv3.onnx"
        ]:
            if cand and os.path.exists(cand):
                chosen_cls_path = cand
                break
    if chosen_cls_path and os.path.exists(chosen_cls_path):
        try:
            from camping_guard.perception.animal_classifier import AnimalClassifier
            animal_classifier = AnimalClassifier(model_path=chosen_cls_path, use_trt=use_trt)
        except Exception as e:
            pass

    if chosen_detector_path and os.path.exists(chosen_detector_path):
        tag = "[TensorRT Engine]" if str(chosen_detector_path).endswith(".engine") else "[ONNX]"
        model_type_str = "MegaDetector (野生动物监测模式)" if is_md else "YOLO (人车/野生动物统一检测模式)"
        print(f"[*] 正在挂载 {model_type_str} 目标检测模型 {tag}: {chosen_detector_path} (置信度阈值: {conf_threshold:.2f})")
        if animal_classifier is not None and getattr(animal_classifier, 'session', None) is not None:
            cls_tag = "[TensorRT Engine]" if str(animal_classifier.model_path).endswith(".engine") else "[ONNX]"
            print(f"[*] 已级联挂载二级动物物种细分类器 {cls_tag}: {animal_classifier.model_path}")
        det_input_size = (detector_size, detector_size) if detector_size else (640, 640)
        detector = ONNXObjectDetector(
            model_path=chosen_detector_path,
            input_size=det_input_size,
            conf_threshold=conf_threshold,
            use_trt=use_trt,
            animal_classifier=animal_classifier,
            dla_core=dla_core
        )
    else:
        model_type_str = "未挂载"
        print("[i] 提示: 未指定且未找到 YOLO/MegaDetector 模型，系统将启用【车身固定 ROI MOG2 局部微动守护】模式")
        print("    (若需全图检测，可传入 --detector yolo11s.onnx / MDV6-yolov10-c.onnx)")

    # 1.2 辅助火灾/烟雾检测模型初始化 (多模型多流并发检测，默认开启)
    if enable_fire:
        chosen_fire_path = fire_detector_path
        if chosen_fire_path and not os.path.exists(chosen_fire_path):
            for search_dir in ["models/fire", "models/yolo", "models"]:
                sub_cand = os.path.join(search_dir, chosen_fire_path)
                if os.path.exists(sub_cand):
                    chosen_fire_path = sub_cand
                    break
        if not chosen_fire_path:
            fire_cands = []
            if use_trt:
                fire_cands.extend([
                    "models/fire/yolo11n_fire.engine",
                    "models/yolo/yolo11n_fire.engine",
                    "models/fire/fire.engine",
                    "models/fire/yolov8n-fire.engine",
                    "models/fire/fire_detector.engine"
                ])
            fire_cands.extend([
                "models/fire/yolo11n_fire.onnx",
                "models/yolo/yolo11n_fire.onnx",
                "models/fire/fire.onnx",
                "models/fire/yolov8n-fire.onnx",
                "models/fire/fire_detector.onnx"
            ])
            for fc in fire_cands:
                if os.path.exists(fc):
                    chosen_fire_path = fc
                    break

        fire_det_obj = None
        if chosen_fire_path and os.path.exists(chosen_fire_path):
            if use_trt:
                eng_cand = Path(chosen_fire_path).with_suffix(".engine")
                if eng_cand.is_file():
                    chosen_fire_path = str(eng_cand)
            ftag = "[TensorRT Engine]" if str(chosen_fire_path).endswith(".engine") else "[ONNX]"
            print(f"[*] 正在挂载火灾/浓烟检测模型 {ftag}: {chosen_fire_path}")
            fire_input_size = (fire_size, fire_size) if fire_size else (640, 640)
            fire_det_obj = ONNXObjectDetector(
                model_path=chosen_fire_path,
                input_size=fire_input_size,
                conf_threshold=conf_threshold,
                use_trt=use_trt
            )
        else:
            # 未提供外部模型权重时，默认启用内置自适应视觉火焰检测引擎 (零权重开箱即用保底)
            print("[*] 正在挂载内置自适应视觉火焰检测器 [VisualFlameDetector] (色彩动力学与光谱特征保底引擎，默认开启)")
            fire_det_obj = VisualFlameDetector(conf_threshold=conf_threshold)

        if fire_det_obj is not None:
            if detector is not None:
                detector = ParallelEnsembleDetector(
                    main_detector=detector,
                    fire_detector=fire_det_obj
                )
                fire_name = Path(chosen_fire_path).name if chosen_fire_path else "VisualFlameDetector"
                print(f"[ParallelEnsemble] ✅ 已构建 Orin 双流并发集成引擎: 主检测器({model_type_str}) + 火焰检测器({fire_name}) 并发执行")
            else:
                detector = fire_det_obj
    else:
        print("[*] 火焰检测已显式关闭 (--no-fire)")

    # 2. 姿态估计器初始化
    pose_estimator = None
    chosen_pose_path = pose_onnx_path
    if chosen_pose_path and not os.path.exists(chosen_pose_path):
        sub_cand_pose = os.path.join("models", "pose", chosen_pose_path)
        if os.path.exists(sub_cand_pose):
            chosen_pose_path = sub_cand_pose
    if not chosen_pose_path:
        pose_cands = []
        if use_trt:
            pose_cands.extend(["models/pose/rtmpose-s.engine", "rtmpose-s.engine"])
        pose_cands.extend(["models/pose/rtmpose-s.onnx", "rtmpose-s.onnx"])
        for default_pose in pose_cands:
            if os.path.exists(default_pose):
                chosen_pose_path = default_pose
                break

    if chosen_pose_path:
        if use_trt:
            eng_cand = Path(chosen_pose_path).with_suffix(".engine")
            if eng_cand.is_file():
                chosen_pose_path = str(eng_cand)

    if chosen_pose_path and os.path.exists(chosen_pose_path):
        tag = "[TensorRT Engine]" if str(chosen_pose_path).endswith(".engine") else "[ONNX]"
        print(f"[*] 正在挂载 RTMPose 姿态估计模型 {tag}: {chosen_pose_path}")
        pose_estimator = RTMPoseEstimator(model_path=chosen_pose_path, use_trt=use_trt)

    # 3. 初始化主算法引擎与性能分析器 (自动绑定标定配置与多相机外参)
    profiler = PipelineProfiler()
    dist_estimator = GroundDistanceEstimator(
        camera_height=actual_height,
        camera_pitch=actual_pitch,
        cameras_calib=calib.cameras,
        vehicle_length_m=calib.vehicle_info.length_m,
        vehicle_width_m=calib.vehicle_info.width_m
    )
    engine = CampingGuardEngine(
        calib=calib,
        detector=detector,
        pose_estimator=pose_estimator,
        distance_estimator=dist_estimator,
        use_bytetrack=True,
        profiler=profiler,
        use_tensorrt=use_trt,
        animal_classifier=animal_classifier,
        vlm_throttle_enabled=vlm_throttle_enabled,
        vlm_active_fps=vlm_active_fps
    )
    success, msg = engine.activate()
    print(f"[*] 系统激活状态: {msg}")
    engine.vehicle_bus.uwb_key_pos = None # 默认钥匙不在现场

    # 挂载端侧按需低功耗 Qwen-VLM 后端 (支持 vlm_server 持久化服务与 llm_inference 自动降级)
    from camping_guard.review.vlm_reviewer import TRTEdgeLLMOnDemandBackend
    vlm_backend = TRTEdgeLLMOnDemandBackend(
        model_dir=vlm_model_dir,
        engine_path=vlm_engine_path,
        visual_engine_path=vlm_visual_engine_path,
        build_dir=build_dir or edgellm_dir,
        timeout=vlm_timeout,
        max_tokens=vlm_max_tokens,
        mode=vlm_mode,
        engine_so_path=vlm_engine_so,
        server_bin=vlm_server_bin,
        server_port=vlm_server_port,
        server_idle_timeout=vlm_idle_timeout
    )
    engine.vlm_reviewer.backend = vlm_backend
    engine.vlm_reviewer.profiler = profiler
    vlm_backend._probe_paths()
    vlm_desc = str(vlm_backend._resolved_engine_dir or vlm_model_dir or "Qwen-VLM")
    engine.profiler.register_model_metadata("Qwen-VLM", vlm_desc, "慢通路异步端侧多模态复核与中文战报生成", is_async=True)

    # 4. 规范化引擎证据录像器：统一设置为标准 30 FPS，环形缓冲容量确保满足事前 10s (450 帧)
    import collections
    if hasattr(engine, 'recorder') and engine.recorder is not None:
        target_rec_fps = 30.0
        engine.recorder.fps = target_rec_fps
        engine.recorder.buffer_size = int(15.0 * target_rec_fps)
        engine.recorder.ring_buffer = collections.deque(maxlen=engine.recorder.buffer_size)

    # 看板与视图配置 (默认启用右侧看板，0% 遮挡原始视频区域)
    panel_w = 380
    view_mode = "SIDEBAR" if show_sidebar else "MINIMAL"

    # 可选的输出录制器 (保持与原始视频完全相同的物理帧率)
    writer = None
    if output_path:
        out_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_w = (video_w + panel_w) if show_sidebar else video_w
        writer = cv2.VideoWriter(output_path, out_fourcc, fps, (out_w, video_h))
        print(f"[*] 评估结果将同步录制保存至: {output_path} (尺寸: {out_w}x{video_h}, 目标帧率: {fps:.1f} FPS)")

    window_name = "Camping Guard - Single Video Test Console"
    if not headless and sys.platform.startswith("linux") and "DISPLAY" not in os.environ:
        headless = True
        print("[!] 检测到 Linux 无图形界面环境 (DISPLAY 环境变量为空)，已自动切换为 --headless 无头模式")

    if not headless:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        except cv2.error:
            headless = True
            print("[!] 检测到无显示环境，已自动切换为 --headless 无头静默模式")

    # 可选启动 Web 实时推流服务 (无显示器/Jetson/远程服务器专属)
    web_streamer = None
    if enable_web:
        try:
            from camping_guard.tools.web_streamer import WebLiveStreamer
            web_streamer = WebLiveStreamer(port=web_port)
        except Exception as e:
            print(f"[!] 启动 Web 实时推流服务器失败: {e}")

    is_paused = False
    step_once = False
    frame_idx = 0
    sim_event_msg = ""
    sim_event_time = 0.0

    base_sim_time = 1774880000.0  # 视频仿真时钟基准原点 (秒)
    video_timestamp = base_sim_time
    target_frame_interval = 1.0 / fps
    start_wall_time = time.perf_counter()
    t_prev = time.perf_counter()
    real_fps = fps
    skipped_frames_count = 0
    output_written_frames_count = 0

    # 5. 若启用 --realtime 实时流速模式，启动处理后实时帧流步调控制器
    if realtime:
        print("[*] 正在启动同步处理实时帧流引擎 (--realtime 已激活): 帧框同源合一渲染，按物理帧率节拍播放与录制，零框延迟、零骨骼漂移")

    def handle_interactive_key(k: int, now_time: float):
        """处理快捷键交互，返回 (event_msg, should_quit, req_reset)"""
        if k in [ord('q'), 27]:  # q or ESC
            return None, True, False
        if k == ord('r'):
            return "已重置视频回到起点", False, True
        if k == ord('h'):
            engine.vehicle_bus.inject_door_handle_pull("handle_fl")
            return "【CAN 注入】门把手被机械拉动 -> 联合确认升 L3 紧急高危！", False, False
        if k == ord('k'):
            engine.vehicle_bus.inject_vibration_peak(8.5)
            return "【IMU 注入】车身剧烈冲击震动 -> 联合确认升 L3 紧急！", False, False
        if k == ord('u'):
            if engine.vehicle_bus.uwb_key_pos is None:
                engine.vehicle_bus.uwb_key_pos = (0.5, 0.8)
                return "【UWB 钥匙】车主在场 (激活白名单豁免)", False, False
            else:
                engine.vehicle_bus.uwb_key_pos = None
                return "【UWB 钥匙】钥匙远离现场", False, False
        if k == ord('l'):
            uwb_d = 6.0 if engine.vehicle_bus.uwb_key_pos is None else 0.8
            can_exit, _, _ = engine.state_machine.handle_unlock_request(
                unlock_source="TEST_BLE_RELAY", uwb_distance_to_car=uwb_d, current_time=now_time
            )
            msg = "【中继攻击拦截】钥匙不在场，拒绝退出并升 L3" if not can_exit else "合法退出"
            return msg, False, False
        if k == ord('v'):
            return "TOGGLE_VIEW", False, False
        if k == ord('8'):
            if hasattr(engine.vlm_reviewer.backend, 'force_false_alarm'):
                engine.vlm_reviewer.backend.force_false_alarm = True
            return "【VLM 测试】强制假阳性复核 -> 允许降级至 L1 提示", False, False
        if k == ord('9'):
            is_valid, report_msg, _ = engine.recorder.verify_hash_chain()
            return f"【防篡改审计】{report_msg}", False, False
        return None, False, False

    term_listener = ConsoleKeyListener()
    stop_requested = False

    def sig_handler(sig, frame):
        nonlocal stop_requested
        print("\n[!] 收到终端强制中断信号 (Ctrl+C)，正在安全退出并妥善封口输出视频...")
        stop_requested = True

    try:
        import signal
        prev_sigint = signal.signal(signal.SIGINT, sig_handler)
    except Exception:
        prev_sigint = None

    cleaned_up = False
    def cleanup():
        nonlocal cleaned_up, writer
        if cleaned_up:
            return
        cleaned_up = True
        if term_listener is not None:
            term_listener.stop()
        if web_streamer:
            try:
                web_streamer.stop()
            except Exception:
                pass
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if writer is not None:
            try:
                writer.release()
                writer = None
                if output_path and os.path.exists(output_path):
                    f_size = os.path.getsize(output_path)
                    print(f"\n[OK] 导出视频已安全封口保存: {output_path} ({f_size / 1024 / 1024:.2f} MB, 共写入 {output_written_frames_count} 帧)")
            except Exception as e:
                print(f"[!] 释放 VideoWriter 异常: {e}")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if hasattr(engine, 'recorder') and hasattr(engine.recorder, 'flush'):
            try:
                engine.recorder.flush(timeout=2.0)
            except Exception:
                pass
        if hasattr(engine, 'vlm_reviewer') and hasattr(engine.vlm_reviewer, 'wait_for_completion'):
            try:
                engine.vlm_reviewer.wait_for_completion(timeout=1.0)
            except Exception:
                pass
        print("[*] 评估测试结束。")

    import atexit
    atexit.register(cleanup)

    while True:
        if stop_requested:
            break

        # 1. 窗口关闭检测 (防止用户点击右上角 [X] 后进程变成 100% CPU 孤儿死循环)
        if not headless:
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("\n[*] 检测到可视化窗口被用户关闭，安全退出并封口保存...")
                    break
            except Exception:
                pass

        # 2. 暂停模式逻辑 (核心：暂停时严禁循环跑模型推理，只低负载等待按键，CPU 占用降为 0%)
        if is_paused and not step_once:
            web_act = web_streamer.pop_action() if web_streamer else None
            if web_act:
                if web_act in ["quit", "exit"]:
                    print("\n[*] 收到网页端退出指令，安全结束测试...")
                    break
                elif web_act in ["pause", "pause_resume"]:
                    is_paused = False
                    start_wall_time = time.perf_counter() - (frame_idx * target_frame_interval)
                    sim_event_msg = "网页端指令: 继续播放"
                    sim_event_time = video_timestamp
                    continue
                elif web_act == "step":
                    step_once = True
                elif web_act == "toggle_view":
                    view_mode = "MINIMAL" if view_mode == "SIDEBAR" else "SIDEBAR"
                    sim_event_msg = f"网页端指令: 切换视图 -> {view_mode}"
                    sim_event_time = video_timestamp
                elif web_act == "reset":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    video_timestamp = base_sim_time
                    start_wall_time = time.perf_counter()
                    sim_event_msg = "网页端指令: 重置视频起点"
                    sim_event_time = video_timestamp
                elif web_act == "inject_handle":
                    engine.vehicle_bus.inject_door_handle_pull("handle_fl")
                    sim_event_msg = "【CAN 注入】门把手被机械拉动 -> 联合确认升 L3 紧急！"
                    sim_event_time = video_timestamp
                elif web_act == "inject_vibe":
                    engine.vehicle_bus.inject_vibration_peak(8.5)
                    sim_event_msg = "【IMU 注入】车身剧烈冲击震动 -> 联合确认升 L3 紧急！"
                    sim_event_time = video_timestamp
                elif web_act == "toggle_key":
                    engine.vehicle_bus.uwb_key_pos = (1.5, 0.0) if engine.vehicle_bus.uwb_key_pos is None else None
                    key_s = "钥匙靠近(免扰)" if engine.vehicle_bus.uwb_key_pos else "钥匙远离(设防)"
                    sim_event_msg = f"【钥匙切换】{key_s}"
                    sim_event_time = video_timestamp
                elif web_act == "test_relay":
                    uwb_d = 50.0 if engine.vehicle_bus.uwb_key_pos is None else 1.0
                    can_exit = engine.try_exit_guard(
                        unlock_source="TEST_BLE_RELAY", uwb_distance_to_car=uwb_d, current_time=video_timestamp
                    )
                    sim_event_msg = "【中继攻击拦截】钥匙不在场，拒绝退出并升 L3" if not can_exit else "合法退出"
                    sim_event_time = video_timestamp

            term_key = term_listener.get_key() if term_listener else None
            if not headless:
                gui_key = cv2.waitKey(30) & 0xFF
            else:
                time.sleep(0.03)
                gui_key = 0xFF
            key = term_key if term_key is not None else gui_key

            if key != 0xFF:
                if key in [ord('q'), ord('Q'), 27]:
                    print("\n[*] 收到退出按键 (q/ESC)，安全结束测试...")
                    break
                elif key == ord(' '):
                    is_paused = False
                    # 恢复播放时重新锚定物理时钟起点，防止时间跳跃
                    start_wall_time = time.perf_counter() - (frame_idx * target_frame_interval)
                    continue
                elif key == ord('d'):
                    step_once = True
                    # 允许继续往下走单帧读取与单帧研判
                else:
                    msg, should_quit, req_reset = handle_interactive_key(key, video_timestamp)
                    if should_quit:
                        break
                    if req_reset:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_idx = 0
                        video_timestamp = base_sim_time
                        start_wall_time = time.perf_counter()
                        sim_event_msg = msg
                        sim_event_time = video_timestamp
                    elif msg == "TOGGLE_VIEW":
                        view_mode = "MINIMAL" if view_mode == "SIDEBAR" else "SIDEBAR"
                        sim_event_msg = f"已切换视图: {view_mode}"
                        sim_event_time = video_timestamp
                    elif msg:
                        sim_event_msg = msg
                        sim_event_time = video_timestamp
            continue

        step_once = False  # 重置单帧标志

        # 3. 读取视频下一帧 (平滑顺序解码，杜绝 cap.set 强行随机 seek 导致的解码卡顿)
        ret, frame = cap.read()
        if not ret:
            if loop_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    # Windows MSMF 下 cap.set(0) 有时失效，彻底释放并重新打开
                    cap.release()
                    cap = cv2.VideoCapture(str(video_file))
                    ret, frame = cap.read()
                if not ret:
                    print("[!] 错误: 视频回放无法重置，退出。")
                    break
                frame_idx = 1
                start_wall_time = time.perf_counter()
                print("[*] 视频播放完毕，循环重播中...")
            else:
                print(f"\n[*] 视频全部帧已处理完毕 (共 {frame_idx} 帧)。")
                break
        else:
            frame_idx += 1

        if max_frames and frame_idx > max_frames:
            print(f"[*] 已达到设定最大测试帧数 ({max_frames})，正常结束。")
            break

        # 视频虚拟时间戳推进 (严格与视频源物理帧率对齐，彻底解决 CPU 处理慢导致速度与滞留时间被扭曲的缺陷)
        video_timestamp = base_sim_time + (frame_idx / fps)

        # 4. 算法研判分发 (铁律 1：100% 帧框同源合一，当前帧同步推理并直接在该帧上绘制 OpenCV 检测框与骨骼点)
        camera_dict = {camera_id: frame}
        run_detection = (frame_idx % detect_interval == 0) if detect_interval > 1 else True
        try:
            tick_res = engine.process_tick(
                camera_frames=camera_dict,
                raw_detections=None if run_detection else [],
                current_time=video_timestamp
            )
        except Exception as e:
            print(f"[!] [Pipeline] process_tick 发生异常 (帧 #{frame_idx}): {e}")
            tick_res = {"threat_level": "L0", "e2e_ms": 0.0}

        active_targets = engine.active_targets
        active_rois = engine.active_roi_disturbances
        ai_e2e_ms = tick_res.get("e2e_ms", 0.0)
        ai_fps = 1000.0 / max(ai_e2e_ms, 1e-3)

        # 5. 可视化增强绘制与输出推流 (纳入 Render_and_Output 阶段性能统计)
        with engine.profiler.profile_stage("Render_and_Output"):
            render_frame = frame.copy()

            # 绘制车周防区参考线与固定 ROI (根据当前机位 camera_id 独立匹配与渲染)
            draw_vehicle_and_zones(render_frame, engine, camera_id=camera_id)

            # 绘制检测与追踪目标 (框上所有提示字符 100% 采用 ASCII 英文，杜绝 OpenCV putText 乱码)
            for t in active_targets:
                bx, by, bw, bh = t.bbox.x, t.bbox.y, t.bbox.w, t.bbox.h
                z_val = t.zone.value if hasattr(t.zone, 'value') else str(t.zone)
                z_short = "Zone 2" if ("Zone 2" in z_val or "2" in z_val) else ("Zone 1" if ("Zone 1" in z_val or "1" in z_val) else "Zone 0")
                cat_str = str(t.category).lower()
                if "person" in cat_str or "人" in str(t.category):
                    cat_short = "Person"
                elif "car" in cat_str or "veh" in cat_str or "车" in str(t.category):
                    cat_short = "Vehicle"
                elif "fire" in cat_str or "火" in str(t.category):
                    cat_short = "Fire"
                elif "smoke" in cat_str or "烟" in str(t.category):
                    cat_short = "Smoke"
                elif "large_beast" in cat_str or "beast" in cat_str:
                    sp = getattr(t, 'species', None)
                    cat_short = f"Beast({sp})" if sp and sp not in ['unknown', 'unclassified_animal'] else "Beast"
                elif "small_animal" in cat_str or "animal" in cat_str or "兽" in str(t.category) or "物" in str(t.category):
                    sp = getattr(t, 'species', None)
                    cat_short = f"Animal({sp})" if sp and sp not in ['unknown', 'unclassified_animal'] else "Animal"
                else:
                    cat_short = str(t.category).split(".")[-1].capitalize()

                color = (0, 255, 120) if t.is_whitelisted else ((0, 160, 255) if "Zone 2" in z_short else (255, 180, 0))
                cv2.rectangle(render_frame, (bx, by), (bx + bw, by + bh), color, 2)

                sess_suffix = f"({t.person_session_id})" if t.person_session_id else ""
                dwell_s = max(0.0, float(getattr(t, "cumulative_dwell_time", 0.0)))
                dist_val = float(getattr(t, "ground_dist", 0.0))
                vr_val = float(getattr(t, "radial_speed", 0.0))
                info_text = f"ID:{t.track_id}{sess_suffix} {cat_short} | {dist_val:.1f}m ({z_short}) | Dwell:{dwell_s:.1f}s | vr:{vr_val:+.1f}m/s"
                cv2.putText(render_frame, info_text, (bx, max(15, by - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

                # 若有骨骼点则渲染 (传入当前 bbox，严格限制连线在人体合理生理轮廓内，防止骨骼划出框外)
                if t.keypoints:
                    draw_skeleton(render_frame, t.keypoints, bbox=t.bbox)

            # 画面组合输出: 默认直通 1080P 原始画面 (0% CPU 绘图损耗，所有遥测指标实时推流至 Web 仪表盘)
            if show_sidebar and view_mode == "SIDEBAR":
                display_frame = draw_telemetry_sidebar(
                    video_frame=render_frame,
                    engine=engine,
                    tick_result=tick_res,
                    real_fps=real_fps,
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                    is_paused=is_paused,
                    sim_status=sim_event_msg,
                    camera_id=camera_id,
                    detector_name=Path(chosen_detector_path).stem if chosen_detector_path else "MOG2",
                    panel_w=panel_w,
                    targets_override=active_targets,
                    roi_disturbances_override=active_rois,
                    ai_fps=ai_fps,
                    ai_e2e_ms=ai_e2e_ms,
                    is_realtime_mode=realtime
                )
            else:
                display_frame = render_frame

            # 显示与保存
            if not headless:
                try:
                    cv2.imshow(window_name, display_frame)
                except cv2.error:
                    headless = True
                    print("[!] 检测到无显示环境，已自动切换为 --headless 无头静默模式")

            if writer is not None:
                writer.write(display_frame)
                output_written_frames_count += 1

            # 网页端实时推流与全维度遥测指标同步 (视频 100% 直通 1080P 帧框合一画面，指标交由 Web DOM 渲染)
            if web_streamer:
                web_streamer.push_frame(render_frame)
                web_streamer.update_telemetry(
                    active_targets=active_targets,
                    active_rois=active_rois,
                    latest_event=engine.historical_threat_events[-1] if getattr(engine, "historical_threat_events", None) else None,
                    threat_level=f"{tick_res.get('threat_level', 'L0')}",
                    state=f"{tick_res.get('state', 'S0')}",
                    mode=f"{tick_res.get('mode', '车内模式')}",
                    fps=real_fps,
                    ai_fps=ai_fps,
                    e2e_ms=ai_e2e_ms or 0.0,
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                    uwb_online=(engine.vehicle_bus.uwb_key_pos is not None)
                )

        # 6. 交互按键监听与动态帧率步调控制
        now_render = time.perf_counter()
        if realtime:
            # 严格锁相物理时钟控制: 计算到达目标帧所需等待的毫秒数
            target_wall_time = start_wall_time + (frame_idx * target_frame_interval)
            remaining_sec = target_wall_time - now_render
            wait_ms = max(1, int(remaining_sec * 1000.0)) if remaining_sec > 0 else 1
            # 追赶物理时钟与保证 1.0x 平滑录像:
            # 当偶发重负载帧导致单帧耗时超出物理帧间隔时，跳过积压帧以追赶物理时钟；
            # 若开启了 --output 录像，按实际跨越的时间槽位补齐当前已渲染的同源帧，防止导出视频因缺帧产生快进跳变卡顿
            if remaining_sec < -max(0.04, target_frame_interval):
                lag_frames = min(int((-remaining_sec) / target_frame_interval), 3)
                for _ in range(lag_frames):
                    if not is_cam_or_pipe:
                        if cap.grab():
                            frame_idx += 1
                            skipped_frames_count += 1
                            if writer is not None:
                                writer.write(display_frame)
                                output_written_frames_count += 1
                    else:
                        cap.grab()
                        skipped_frames_count += 1
        else:
            # 同步模式: 减去当帧算法处理耗时
            e2e_ms = tick_res.get("e2e_ms", 0.0)
            target_delay = (1000.0 / fps) - e2e_ms
            wait_ms = max(1, int(target_delay)) if not is_paused else 30
            remaining_sec = 0.0

        term_key = term_listener.get_key() if term_listener else None
        if not headless:
            gui_key = cv2.waitKey(wait_ms) & 0xFF
        else:
            if realtime and remaining_sec > 0:
                time.sleep(remaining_sec)
            gui_key = 0xFF

        if frame_idx % 25 == 0 or verbose:
            vlm_busy = tick_res.get("vlm_is_inferencing", False)
            vlm_dur = tick_res.get("vlm_last_duration_ms", 0.0)
            vlm_cnt = tick_res.get("vlm_total_reviews", 0)
            if vlm_busy:
                vlm_tag = " | VLM: [推理中]"
            elif vlm_cnt > 0:
                vlm_tag = f" | VLM: [完成: {vlm_dur:.0f}ms](x{vlm_cnt})"
            else:
                vlm_tag = " | VLM: [待机]"

            if realtime:
                print(f"[*] [Frame {frame_idx:03d}/{total_frames}] 视频流速: {real_fps:4.1f} FPS (实时平滑) | AI耗时: {ai_e2e_ms:5.1f}ms ({ai_fps:4.1f} FPS) | 目标: {len(active_targets)} | 状态: {tick_res.get('state')} | 威胁: {tick_res.get('threat_level')}{vlm_tag}")
            else:
                fps_e2e = 1000.0 / max(ai_e2e_ms, 1e-3)
                print(f"[*] [Frame {frame_idx:03d}/{total_frames}] E2E: {ai_e2e_ms:5.1f}ms ({fps_e2e:4.1f} FPS) | 目标: {len(active_targets)} | 状态: {tick_res.get('state')} | 威胁: {tick_res.get('threat_level')}{vlm_tag}")

        key = term_key if term_key is not None else gui_key

        now_loop = time.perf_counter()
        dt = now_loop - t_prev
        t_prev = now_loop
        if dt > 0:
            real_fps = 0.9 * real_fps + 0.1 * (1.0 / dt)

        # 网页交互动作检测
        web_act = web_streamer.pop_action() if web_streamer else None
        if web_act:
            if web_act in ["quit", "exit"]:
                print("\n[*] 收到网页端退出指令，安全结束测试...")
                break
            elif web_act in ["pause", "pause_resume"]:
                is_paused = not is_paused
                if not is_paused:
                    start_wall_time = time.perf_counter() - (frame_idx * target_frame_interval)
                sim_event_msg = f"网页端指令: {'已暂停' if is_paused else '继续播放'}"
                sim_event_time = video_timestamp
            elif web_act == "step":
                step_once = True
            elif web_act == "toggle_view":
                view_mode = "MINIMAL" if view_mode == "SIDEBAR" else "SIDEBAR"
                sim_event_msg = f"网页端指令: 切换视图模式 -> {view_mode}"
                sim_event_time = video_timestamp
            elif web_act == "reset":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                video_timestamp = base_sim_time
                start_wall_time = time.perf_counter()
                sim_event_msg = "网页端指令: 重置视频起点"
                sim_event_time = video_timestamp
            elif web_act == "inject_handle":
                engine.vehicle_bus.inject_door_handle_pull("handle_fl")
                sim_event_msg = "【CAN 注入】门把手被机械拉动 -> 联合确认升 L3 紧急！"
                sim_event_time = video_timestamp
            elif web_act == "inject_vibe":
                engine.vehicle_bus.inject_vibration_peak(8.5)
                sim_event_msg = "【IMU 注入】车身剧烈冲击震动 -> 联合确认升 L3 紧急！"
                sim_event_time = video_timestamp
            elif web_act == "toggle_key":
                engine.vehicle_bus.uwb_key_pos = (1.5, 0.0) if engine.vehicle_bus.uwb_key_pos is None else None
                key_s = "钥匙靠近(免扰)" if engine.vehicle_bus.uwb_key_pos else "钥匙远离(设防)"
                sim_event_msg = f"【钥匙切换】{key_s}"
                sim_event_time = video_timestamp
            elif web_act == "test_relay":
                uwb_d = 50.0 if engine.vehicle_bus.uwb_key_pos is None else 1.0
                can_exit = engine.try_exit_guard(
                    unlock_source="TEST_BLE_RELAY", uwb_distance_to_car=uwb_d, current_time=video_timestamp
                )
                sim_event_msg = "【中继攻击拦截】钥匙不在场，拒绝退出并升 L3" if not can_exit else "合法退出"
                sim_event_time = video_timestamp

        if key != 0xFF:
            if key in [ord('q'), ord('Q'), 27]:  # q or ESC
                print("\n[*] 收到退出按键 (q/ESC)，安全结束测试...")
                break
            elif key == ord(' '):  # 空格暂停/播放
                is_paused = not is_paused
                if not is_paused:
                    start_wall_time = time.perf_counter() - (frame_idx * target_frame_interval)
            elif key == ord('d'):  # 单步步进
                step_once = True
            else:
                msg, should_quit, req_reset = handle_interactive_key(key, video_timestamp)
                if should_quit:
                    break
                if req_reset:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    video_timestamp = base_sim_time
                    start_wall_time = time.perf_counter()
                    sim_event_msg = msg
                    sim_event_time = video_timestamp
                elif msg == "TOGGLE_VIEW":
                    view_mode = "MINIMAL" if view_mode == "SIDEBAR" else "SIDEBAR"
                    sim_event_msg = f"已切换视图模式: {view_mode}"
                    sim_event_time = video_timestamp
                elif msg:
                    sim_event_msg = msg
                    sim_event_time = video_timestamp

    cleanup()
    try:
        atexit.unregister(cleanup)
    except Exception:
        pass
    if prev_sigint is not None:
        try:
            signal.signal(signal.SIGINT, prev_sigint)
        except Exception:
            pass

    # 汇总运行期上下文供全维度 JSON 性能诊断报告使用
    processed_cnt = engine.profiler.total_frames
    total_seen = processed_cnt + skipped_frames_count
    vlm_audit_summary = (
        engine.vlm_reviewer.get_audit_summary()
        if hasattr(engine, "vlm_reviewer") and hasattr(engine.vlm_reviewer, "get_audit_summary")
        else {}
    )
    engine.profiler.set_runtime_context(
        video_path=str(video_path),
        video_name=display_video_name,
        resolution=f"{video_w}x{video_h}",
        source_fps=round(float(fps), 2),
        total_source_frames=int(total_frames),
        processed_frames=int(processed_cnt),
        last_frame_idx=int(frame_idx),
        skipped_frames=int(skipped_frames_count),
        skip_ratio_pct=round(skipped_frames_count * 100.0 / max(1, total_seen), 2),
        output_path=str(output_path) if output_path else None,
        output_written_frames=int(output_written_frames_count),
        real_fps_final=round(float(real_fps), 2),
        realtime_mode=bool(realtime),
        use_trt=bool(use_trt),
        camera_id=int(camera_id),
        detector_model=str(chosen_detector_path) if chosen_detector_path else "MOG2",
        fire_enabled=bool(enable_fire),
        vlm_audit=vlm_audit_summary,
        threat_events_count=len(getattr(engine, "historical_threat_events", []))
    )

    # 打印视频全流程端到端与各模型综合性能统计报告
    print("\n" + engine.profiler.format_summary_table())

    # 打印全维度守卫事件审计报告与 App 标准推送 JSON
    print_guard_event_summary(engine, str(video_path), frame_idx, fps)

    # 默认自动导出全维度 JSON 性能诊断报告至 logs/ 目录，若用户指定了 --stats-json 则同步导出
    try:
        video_stem = Path(str(video_path)).stem if not is_cam_or_pipe else f"cam{camera_id}"
        safe_stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in video_stem)[:32] or "video"
        ts_tag = time.strftime("%Y%m%d_%H%M%S")
        default_latest_json = Path("logs") / "latest_perf_report.json"
        default_archive_json = Path("logs") / f"perf_report_{safe_stem}_{ts_tag}.json"
        latest_saved = engine.profiler.export_json_report(default_latest_json)
        archive_saved = engine.profiler.export_json_report(default_archive_json)
        print(f"[OK] 全维度性能诊断 JSON 报告已自动导出:")
        print(f"     - 最新报告 (固定路径): {latest_saved}")
        print(f"     - 归档报告 (带时间戳): {archive_saved}")
        if stats_json_path:
            custom_saved = engine.profiler.export_json_report(stats_json_path)
            print(f"     - 指定路径 (--stats-json): {custom_saved}")
    except Exception as e:
        print(f"[!] 导出性能诊断 JSON 报告异常: {e}")


def print_guard_event_summary(engine: CampingGuardEngine, video_source: str, total_frames: int, fps: float):
    """
    打印视频全流程守卫事件审计全貌报告与 App 标准推送 JSON
    涵盖：事件ID、威胁等级、触发目标类别/ID、距离、滞留时间(驻留秒数)、联动执行动作、VLM 复核战报、证据视频与哈希签名
    """
    import json
    events = getattr(engine, "historical_threat_events", [])
    duration_sec = total_frames / max(fps, 1.0)

    print("\n" + "=" * 78)
    print(" [CAMPING GUARD] 露营守卫 - 事件审计与推送全貌报告 (AUDIT REPORT)")
    print("=" * 78)
    print(f" 视频源文件 : {video_source}")
    print(f" 评估时长   : {total_frames} 帧 (约 {duration_sec:.1f} 秒 @ {fps:.1f} FPS)")
    print(f" 捕获事件数 : 共记录 {len(events)} 次威胁事件")
    print("-" * 78)

    if not events:
        print("  [OK] 监控期间未触发任何 L1/L2/L3 威胁事件，车身周围安全无虞。")
    else:
        for idx, evt in enumerate(events, 1):
            print(f"\n【事件 #{idx}】")
            print(evt.format_audit_card())

        # 打印最高威胁等级事件的移动端/云端标准推送 JSON 格式
        highest_evt = max(events, key=lambda e: (
            3 if "L3" in str(e.threat_level) else (
                2 if "L2" in str(e.threat_level) else (
                    1 if "L1" in str(e.threat_level) else 0
                )
            )
        ))
        push_json = highest_evt.to_push_payload()

        print("\n" + "-" * 78)
        print(" [APP PUSH NOTIFICATION] 【标准移动端 / 云端平台推送通知格式 (JSON)】")
        print("    (可直接对接车机后台 MQTT、车主手机端推送 FCM/APNs 或座舱大屏卡片)")
        print("-" * 78)
        print(json.dumps(push_json, indent=2, ensure_ascii=False))

    # 防篡改哈希链审计
    if hasattr(engine, "recorder") and engine.recorder is not None:
        try:
            is_valid, msg, tampered_idx = engine.recorder.verify_hash_chain()
            block_count = len(getattr(engine.recorder, "manifest", []))
            status_tag = "[PASS] 完整合规" if is_valid else f"[FAIL] 区块 #{tampered_idx} 受损"
            print("-" * 78)
            print(f" [EVIDENCE CHAIN] 证据链防篡改审计: [{status_tag}] {msg} (共 {block_count} 个区块签名)")
        except Exception as e:
            print(f" [!] 证据链校验异常: {e}")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="智能汽车露营守卫 - 单路鱼眼 MP4 视频测试工具")
    parser.add_argument("--video", type=str, required=True, help="待测试的鱼眼 MP4 视频文件路径")
    parser.add_argument("--config", type=str, default="vehicle_calib.json",
                        help="全车统一标定配置文件 (包含相机外参、几何尺寸、车身固定 ROI 与阈值，默认: vehicle_calib.json)")
    parser.add_argument("--camera-id", type=parse_camera_id, default=1,
                        help="挂载的相机通道编号: 0前(front), 1左(left), 2右(right), 3后(rear) (默认 1 左侧环视)")
    parser.add_argument("--camera-height", type=float, default=None,
                        help="手动覆盖相机离地安装高度 (米，默认从 --config 读取)")
    parser.add_argument("--camera-pitch", type=float, default=None,
                        help="手动覆盖相机俯仰角 (度，向下为负，默认从 --config 读取)")
    parser.add_argument("--camera-yaw", type=float, default=None,
                        help="手动覆盖相机偏航角 (度，0前, 90左, -90右, 180后，默认从 --config 读取)")
    parser.add_argument("--detector", "--megadetector", type=str, default=None,
                        help="目标检测模型路径或名称 (支持自由传入 YOLO 或 MegaDetector，如 yolo11s.onnx / yolo11n.engine / MDV6-yolov10-c.onnx，默认自动发现最佳模型)")
    parser.add_argument("--conf", "--conf-threshold", dest="conf_threshold", type=float, default=0.25,
                        help="目标检测置信度阈值 (默认 0.25，鱼眼边缘大畸变视角可调低至 0.15~0.20)")
    parser.add_argument("--sidebar", action="store_true", default=False,
                        help="保留旧版 OpenCV 绘制的 CPU 侧边栏 (默认已禁用以降低延迟，推荐使用 Web 仪表盘以获得零开销体验)")
    parser.add_argument("--no-sidebar", action="store_true",
                        help="显式关闭右侧独立遥测数据面板 (默认即为关闭)")
    parser.add_argument("--pose", type=str, default=None,
                        help="RTMPose 姿态估计 ONNX 模型文件路径 (如 rtmpose-s.onnx)")
    parser.add_argument("--reid", type=str, default="models/reid/osnet_x0_25.onnx",
                        help="OSNet 人员重识别模型路径 (默认: models/reid/osnet_x0_25.onnx)")
    parser.add_argument("--loop", dest="loop_video", action="store_true", default=None,
                        help="强制循环播放视频 (即使指定了 --output 也循环)")
    parser.add_argument("--no-loop", dest="loop_video", action="store_false",
                        help="单次播放后自动退出，不循环回放 (当指定 --output 时默认开启此项)")
    parser.add_argument("--output", type=str, default=None, help="导出渲染分析视频的保存路径 (如 output.mp4)")
    parser.add_argument("--stats-json", type=str, default=None, help="导出全流程综合性能统计指标 JSON 文件路径")
    parser.add_argument("--verbose", action="store_true", help="输出每帧的详细耗时指标日志")
    parser.add_argument("--headless", action="store_true", help="无头模式运行 (不弹窗，适合后台/无显示器环境)")
    parser.add_argument("--max-frames", type=int, default=None, help="最大测试评估帧数 (达到后自动结束并输出统计报告)")
    parser.add_argument("--realtime", action="store_true",
                        help="以真实时间速度播放视频 (锁定 30 FPS 物理帧率节拍，画面 100%% 展示实时处理帧，零框延迟)")
    parser.add_argument("--detect-interval", type=int, default=1,
                        help="目标检测推理采样间隔 (默认 1，即每帧推理；设置 >1 可跳帧推理仅运行轻量跟踪)")
    parser.add_argument("--use-trt", "--trt", action="store_true",
                        help="在 Jetson Orin 上启用 TensorRT .engine 模型加速推理 (支持自动搜索并加载同名 .engine)")
    parser.add_argument("--web", action="store_true",
                        help="启用内置 Web 实时流仪表盘 (无头设备/远程服务器专属，浏览器访问即可实时查看视频与遥测)")
    parser.add_argument("--web-port", type=int, default=8080,
                        help="Web 实时流端口号 (默认 8080)")
    parser.add_argument("--model-dir", "--vlm-model-dir", dest="vlm_model_dir", type=str, default=None,
                        help="显式指定端侧 Qwen-VLM 模型总目录 (内部须包含 engines/llm 与 engines/visual，如 /home/nvidia/models/Qwen3.5-0.8B)")
    parser.add_argument("--build-dir", "--edgellm-build-dir", "--edgellm-dir", dest="edgellm_dir", type=str, default=None,
                        help="显式指定 TensorRT-Edge-LLM build 目录 (包含 examples/llm/llm_inference，如 /home/nvidia/qwen35-orin/TensorRT-Edge-LLM/build)")
    parser.add_argument("--vlm-engine", type=str, default=None,
                        help="显式指定端侧 Qwen-VLM LLM 语言模型引擎目录或路径 (如 /home/nvidia/models/Qwen3.5-0.8B/engines/llm)")
    parser.add_argument("--vlm-visual-engine", type=str, default=None,
                        help="显式指定端侧 Qwen-VLM Visual 视觉编码器引擎目录或路径 (如 /home/nvidia/models/Qwen3.5-0.8B/engines/visual)")
    parser.add_argument("--vlm-timeout", type=float, default=35.0,
                        help="端侧 Qwen-VLM 推理超时时间 (秒，默认 35.0，为 Orin 引擎反序列化加载与生成留足裕量)")
    parser.add_argument("--vlm-max-tokens", type=int, default=128,
                        help="端侧 Qwen-VLM 单次复核最大生成 Token 长度 (默认 128)")
    parser.add_argument("--vlm-mode", type=str, default="auto", choices=["auto", "server", "subprocess"],
                        help="端侧 VLM 推理模式: auto (优先常驻持久化服务 vlm_server 零冷启动，自动降级 llm_inference), server (强制常驻服务模式), subprocess (传统一次性 llm_inference 进程模式)")
    parser.add_argument("--vlm-server-port", type=int, default=8090,
                        help="vlm_server 监听的 HTTP 端口 (默认 8090)")
    parser.add_argument("--vlm-idle-timeout", type=int, default=60,
                        help="vlm_server 空闲自动退出卸载显存超时 (秒，默认 60)")
    parser.add_argument("--vlm-server-bin", type=str, default=None,
                        help="显式指定 vlm_server 可执行程序路径 (默认自动在 build 目录检索)")
    parser.add_argument("--cls", "--animal-cls", dest="cls_model", type=str, default=None,
                        help="二级动物细分类模型文件或路径 (支持自动在 models/cls/ 查找，如 mobilenetv3-large.onnx / mobilenetv3-large.engine)")
    parser.add_argument("--fire-detector", "--fire", dest="fire_detector", type=str, default=None,
                        help="火灾/浓烟专用目标检测模型文件或路径 (默认已启用火焰检测并发)")
    parser.add_argument("--no-fire", action="store_true",
                        help="显式关闭火焰检测 (默认开启自适应视觉火焰检测或专用模型并发)")
    parser.add_argument("--detector-size", type=int, default=None,
                        help="目标检测模型输入分辨率 (默认自动适配模型，如 640 或 1280)")
    parser.add_argument("--fire-size", type=int, default=None,
                        help="火灾检测模型输入分辨率 (默认自动适配模型，如 640 或 1280)")

    args = parser.parse_args()
    loop_val = args.loop_video
    if loop_val is None:
        loop_val = False if args.output else True

    show_sb = args.sidebar and not args.no_sidebar

    run_video_pipeline(
        video_path=args.video,
        config_path=args.config,
        camera_id=args.camera_id,
        camera_height=args.camera_height,
        camera_pitch=args.camera_pitch,
        camera_yaw=args.camera_yaw,
        detector_onnx_path=args.detector,
        pose_onnx_path=args.pose,
        loop_video=loop_val,
        output_path=args.output,
        stats_json_path=args.stats_json,
        verbose=args.verbose,
        headless=args.headless,
        max_frames=args.max_frames,
        realtime=args.realtime,
        conf_threshold=args.conf_threshold,
        show_sidebar=show_sb,
        detect_interval=args.detect_interval,
        use_trt=args.use_trt,
        enable_web=args.web,
        web_port=args.web_port,
        vlm_model_dir=args.vlm_model_dir,
        vlm_engine_path=args.vlm_engine,
        vlm_visual_engine_path=args.vlm_visual_engine,
        edgellm_dir=args.edgellm_dir,
        build_dir=args.edgellm_dir,
        vlm_timeout=args.vlm_timeout,
        vlm_max_tokens=args.vlm_max_tokens,
        vlm_mode=args.vlm_mode,
        vlm_server_port=args.vlm_server_port,
        vlm_idle_timeout=args.vlm_idle_timeout,
        vlm_server_bin=args.vlm_server_bin,
        cls_onnx_path=args.cls_model,
        fire_detector_path=args.fire_detector,
        enable_fire=not args.no_fire,
        detector_size=args.detector_size,
        fire_size=args.fire_size
    )



if __name__ == "__main__":
    main()
