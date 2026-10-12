"""
环形视频缓冲、事件录像锁存与防篡改存证模块 (阶段 4)
实现事前 10s 至事后 20s 视频截取、GNSS 授时固化、SHA-256 哈希链防篡改与完整性校验
"""

import collections
import hashlib
import time
import json
import os
import queue
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import cv2

from camping_guard.types import ClockSource


class PendingClip:
    """正在持续收集事后录像的事件切片"""
    def __init__(
        self,
        event_id: str,
        threat_level_str: str,
        trigger_time: float,
        post_end_time: float,
        pre_frames: List[Tuple[float, np.ndarray]],
        clock_source: ClockSource,
        pre_duration_sec: float = 10.0,
        post_duration_sec: float = 20.0,
        event_metadata: Optional[Dict[str, Any]] = None
    ):
        self.event_id = event_id
        self.threat_level_str = threat_level_str
        self.trigger_time = trigger_time
        self.post_end_time = post_end_time
        self.pre_frames = pre_frames
        self.post_frames: List[Tuple[float, np.ndarray]] = []
        self.clock_source = clock_source
        self.pre_duration_sec = pre_duration_sec
        self.post_duration_sec = post_duration_sec
        self.is_finalized = False
        self.event_metadata: Dict[str, Any] = event_metadata or {}


def normalize_clip_frames_to_target_fps(
    all_frames: List[Tuple[float, np.ndarray]],
    target_fps: float = 30.0,
    expected_start: Optional[float] = None,
    expected_end: Optional[float] = None
) -> List[np.ndarray]:
    """
    根据时间戳严格等间距重采样/对齐视频帧至固定目标帧率 (默认 30.0 FPS)，
    彻底解决抽帧、变帧率或推理抖动导致的播放倍速快放或慢放问题。
    确保无论输入源是 25fps、30fps 还是存在丢帧，输出视频均严格以 1.0x 真实物理流速播放。
    """
    if not all_frames:
        return []
    if len(all_frames) == 1:
        return [all_frames[0][1]]

    # 提取时间戳并排序
    sorted_pairs = sorted(all_frames, key=lambda x: x[0])
    timestamps = np.array([p[0] for p in sorted_pairs], dtype=np.float64)
    raw_frames = [p[1] for p in sorted_pairs]

    t_start = expected_start if (expected_start is not None and expected_start >= timestamps[0]) else timestamps[0]
    t_end = expected_end if (expected_end is not None and expected_end <= timestamps[-1]) else timestamps[-1]
    if t_end <= t_start:
        t_start = timestamps[0]
        t_end = timestamps[-1]

    duration = t_end - t_start
    if duration <= 0:
        return raw_frames

    # 计算目标总帧数 (例如 30s * 30fps = 900 帧)
    num_target_frames = max(1, int(round(duration * target_fps)))
    target_timestamps = np.linspace(t_start, t_end, num_target_frames, endpoint=False)

    # 对每个目标时间点定位最近邻帧
    indices = np.searchsorted(timestamps, target_timestamps)
    indices = np.clip(indices, 0, len(timestamps) - 1)

    resampled_frames = []
    for idx, t_target in zip(indices, target_timestamps):
        best_idx = idx
        if idx > 0 and abs(timestamps[idx - 1] - t_target) < abs(timestamps[idx] - t_target):
            best_idx = idx - 1
        resampled_frames.append(raw_frames[best_idx])

    return resampled_frames


class EvidenceRecorder:
    """
    证据链管理与防篡改存证引擎
    维护高可靠滚动内存环形缓冲，并在触发威胁时构建可审计的不可篡改证据链
    """
    GENESIS_HASH = "GENESIS_BLOCK_HASH_CAMPING_GUARD_2026"

    def __init__(
        self,
        buffer_duration_sec: float = 15.0,
        fps: float = 30.0,
        storage_dir: Optional[str] = None,
        clock_source: ClockSource = ClockSource.GNSS_PPS_SYNC,
        async_encoding: bool = False
    ):
        """
        :param buffer_duration_sec: 内存环形缓冲长度 (秒)
        :param fps: 录像帧率 (默认 30.0 FPS，满足人眼查看的标准流畅视频要求)
        :param storage_dir: 加密/安全存储目录 (模拟车端 eMMC secure partition)
        :param clock_source: 当前硬件授时源
        :param async_encoding: 是否开启后台独立工作线程异步编码落盘 (避免阻塞主视觉流水线)
        """
        self.buffer_size = int(buffer_duration_sec * fps)
        self.fps = fps
        self.clock_source = clock_source
        self.async_encoding = async_encoding

        # 环形视频帧队列: [(timestamp, frame)]
        self.ring_buffer: collections.deque = collections.deque(maxlen=self.buffer_size)

        # 正在追踪中的事后录像
        self.pending_clips: Dict[str, PendingClip] = {}

        # 已锁存并落盘的事件存证清单
        self.locked_evidence: List[Dict[str, Any]] = []

        # 防篡改哈希链
        self.prev_hash = self.GENESIS_HASH
        self.manifest: List[Dict[str, Any]] = []

        # 线程安全锁与异步工作队列
        self._lock = threading.Lock()
        self._encode_queue: Optional[queue.Queue] = None
        self._worker_thread: Optional[threading.Thread] = None

        if self.async_encoding:
            self._encode_queue = queue.Queue()
            self._worker_thread = threading.Thread(
                target=self._async_encode_worker,
                daemon=True,
                name="EvidenceRecorder_Worker"
            )
            self._worker_thread.start()

        # 存储路径
        if storage_dir is None:
            self.storage_dir = Path("./secure_storage/events")
        else:
            self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.storage_dir / "evidence_manifest.jsonl"

        # 加载既有存证清单与同步哈希链状态
        self._load_and_sync_manifest()

    def _load_and_sync_manifest(self):
        """
        从存储中加载既有存证清单，恢复防篡改哈希链末端状态，
        并对历史版本因跨进程重启引发的链条断裂进行自愈修复
        """
        if not self.manifest_file.exists():
            return

        entries = []
        try:
            with open(self.manifest_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except Exception as e:
            print(f"[EvidenceRecorder] 读取存证清单文件异常: {e}")
            return

        if not entries:
            return

        # 检查是否存在旧版本重启缺陷 (多个 block_index=0 且 prev_hash=GENESIS_HASH)
        needs_heal = False
        for idx, entry in enumerate(entries):
            payload = entry.get("payload", {})
            if idx > 0 and payload.get("prev_hash") == self.GENESIS_HASH and payload.get("block_index") == 0:
                needs_heal = True
                break

        if needs_heal:
            print("[EvidenceRecorder] 发现历史版本跨进程重启导致的断裂区块，正在执行哈希链自愈修复...")
            try:
                import shutil
                bak_path = self.manifest_file.with_suffix(".jsonl.bak")
                if not bak_path.exists():
                    shutil.copy(self.manifest_file, bak_path)

                cur_prev = self.GENESIS_HASH
                healed_entries = []
                for idx, entry in enumerate(entries):
                    payload = entry.get("payload", {})
                    payload["block_index"] = idx
                    payload["prev_hash"] = cur_prev
                    new_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
                    entry["block_hash"] = new_hash
                    cur_prev = new_hash
                    healed_entries.append(entry)

                with open(self.manifest_file, "w", encoding="utf-8") as f:
                    for he in healed_entries:
                        f.write(json.dumps(he, ensure_ascii=False) + "\n")
                entries = healed_entries
                print(f"[EvidenceRecorder] 哈希链自愈修复完成！共修复并重签 {len(entries)} 个区块。")
            except Exception as ex:
                print(f"[EvidenceRecorder] 哈希链自愈修复异常: {ex}")

        self.manifest = entries
        if self.manifest:
            self.prev_hash = self.manifest[-1].get("block_hash", self.GENESIS_HASH)
            for item in self.manifest:
                payload = item.get("payload", {})
                ev_id = payload.get("event_id")
                if not any(ev.get("event_id") == ev_id for ev in self.locked_evidence):
                    vid_name = payload.get("video_file_name", "")
                    self.locked_evidence.append({
                        "event_id": ev_id,
                        "threat_level": payload.get("threat_level", ""),
                        "trigger_time": payload.get("trigger_time", 0.0),
                        "video_path": str(self.storage_dir / vid_name) if vid_name else "",
                        "video_file_sha256": payload.get("video_file_sha256", ""),
                        "block_hash": item.get("block_hash", ""),
                        "clock_source": payload.get("clock_source", ""),
                        "total_frames": payload.get("total_frame_count", 0),
                        "is_locked": True,
                        "target": payload.get("target", {}),
                        "behavior": payload.get("behavior", {}),
                        "vlm_summary": payload.get("vlm_summary", ""),
                        "executed_actions": payload.get("executed_actions", []),
                        "payload": payload
                    })

    def _async_encode_worker(self):
        """后台独立工作线程：从队列消费待持久化的切片，执行视频编码与哈希计算"""
        while True:
            try:
                event_id = self._encode_queue.get()
                if event_id is None:
                    break
                self._do_finalize_clip(event_id)
                self._encode_queue.task_done()
            except Exception as e:
                print(f"[EvidenceRecorder] 后台视频编码落盘异常: {e}")

    def flush(self, timeout: float = 5.0):
        """等待所有后台录像切片完成视频编码与哈希上链"""
        if self.async_encoding and self._encode_queue:
            self._encode_queue.join()

    def push_frame(self, frame: np.ndarray, current_time: float):
        """
        持续将低码率/高清画面写入环形缓冲，并喂给所有正在录制事后视频的切片
        """
        frame_copy = frame.copy()
        self.ring_buffer.append((current_time, frame_copy))

        # 推进所有待完成的事后录像切片
        completed_ids = []
        with self._lock:
            for event_id, pending in self.pending_clips.items():
                if not pending.is_finalized:
                    pending.post_frames.append((current_time, frame_copy))
                    if current_time >= pending.post_end_time:
                        completed_ids.append(event_id)

        for event_id in completed_ids:
            if self.async_encoding and self._encode_queue:
                pending = self.pending_clips.get(event_id)
                if pending and not pending.is_finalized:
                    pending.is_finalized = True
                    self._encode_queue.put(event_id)
            else:
                self.finalize_clip(event_id)

    def lock_event_clip(
        self,
        event_id: str,
        threat_level_str: str,
        trigger_time: float,
        pre_duration_sec: float = 10.0,
        post_duration_sec: float = 20.0,
        clock_source: Optional[ClockSource] = None,
        event_record: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        触发事件录像锁存：
        提取触发前 10 秒缓冲区数据，并注册事后持续录制 20 秒的任务
        返回当前切片的初步安全签名
        """
        actual_clock = clock_source or self.clock_source
        pre_start_time = trigger_time - pre_duration_sec
        post_end_time = trigger_time + post_duration_sec

        # 提取事前 10 秒内已缓存的所有视频帧
        pre_frames = [
            (t, f) for t, f in self.ring_buffer
            if t >= pre_start_time and t <= trigger_time
        ]

        # 提取事件完整感知上下文 (目标类别、ID、测距、关键滞留时长等)
        meta = {}
        if event_record is not None:
            meta = {
                "target_category": getattr(event_record, "target_category", "unknown"),
                "target_species": getattr(event_record, "species", None),
                "target_species_cn": getattr(event_record, "species_cn", None),
                "target_id": getattr(event_record, "target_id", None),
                "person_session_id": getattr(event_record, "person_session_id", None),
                "target_distance_m": getattr(event_record, "target_distance", 0.0),
                "target_dwell_time_sec": getattr(event_record, "target_dwell_time", 0.0),
                "target_radial_speed_mps": getattr(event_record, "target_radial_speed", 0.0),
                "target_zone": getattr(event_record, "target_zone", ""),
                "is_whitelisted": getattr(event_record, "is_whitelisted", False),
                "action_type": getattr(getattr(event_record, "action_type", None), "value", str(getattr(event_record, "action_type", ""))),
                "description": getattr(event_record, "description", ""),
                "has_contact_signal": getattr(event_record, "has_contact_signal", False),
                "disturbed_rois": list(getattr(event_record, "disturbed_rois", [])),
                "vlm_summary": getattr(event_record, "vlm_summary", ""),
                "executed_actions": list(getattr(event_record, "executed_actions", [])),
                "operating_mode": getattr(getattr(event_record, "operating_mode", None), "value", str(getattr(event_record, "operating_mode", "")))
            }
        if metadata:
            meta.update(metadata)

        pending = PendingClip(
            event_id=event_id,
            threat_level_str=threat_level_str,
            trigger_time=trigger_time,
            post_end_time=post_end_time,
            pre_frames=pre_frames,
            clock_source=actual_clock,
            pre_duration_sec=pre_duration_sec,
            post_duration_sec=post_duration_sec,
            event_metadata=meta
        )
        self.pending_clips[event_id] = pending

        # 计算初步签名 (锁存预哈希)
        pre_hash = hashlib.sha256(
            f"{event_id}|{threat_level_str}|{trigger_time:.3f}|{len(pre_frames)}|{self.prev_hash}".encode('utf-8')
        ).hexdigest()
        return pre_hash

    def update_clip_metadata(self, event_id: str, updates: Dict[str, Any]):
        """在事后录制窗口期内动态增量更新事件元数据 (例如 VLM 复核结果或执行动作回调)"""
        with self._lock:
            pending = self.pending_clips.get(event_id)
            if pending and hasattr(pending, 'event_metadata'):
                pending.event_metadata.update(updates)

    def finalize_clip(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        完成并封装事件录像 (事前 10s + 事后 20s 拼装)
        执行实际视频编码落盘与 SHA-256 哈希链上链签署
        """
        return self._do_finalize_clip(event_id)

    def _do_finalize_clip(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            pending = self.pending_clips.get(event_id)
            if pending is None:
                return None
            # 若已在 locked_evidence 中落盘过则直接返回
            for ev in self.locked_evidence:
                if ev["event_id"] == event_id:
                    return ev
            pending.is_finalized = True

        all_frames = pending.pre_frames + pending.post_frames

        # 0. 规范化重采样为恒定目标帧率 (默认 30 FPS) 正常流速视频帧序列
        exp_start = pending.trigger_time - getattr(pending, 'pre_duration_sec', 10.0)
        exp_end = pending.post_end_time
        resampled_frames = normalize_clip_frames_to_target_fps(
            all_frames,
            target_fps=self.fps,
            expected_start=exp_start,
            expected_end=exp_end
        )
        if not resampled_frames:
            resampled_frames = [f for _, f in all_frames]

        # 1. 编码生成视频文件
        video_filename = f"{event_id}_{int(pending.trigger_time)}.mp4"
        video_path = self.storage_dir / video_filename

        frame_h, frame_w = 480, 640
        if len(resampled_frames) > 0 and resampled_frames[0] is not None:
            frame_h, frame_w = resampled_frames[0].shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (frame_w, frame_h))
        if not out_writer.isOpened():
            # 降级尝试 MJPG 编码
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_filename = f"{event_id}_{int(pending.trigger_time)}.avi"
            video_path = self.storage_dir / video_filename
            out_writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (frame_w, frame_h))

        if out_writer.isOpened():
            for frm in resampled_frames:
                out_writer.write(frm)
            out_writer.release()
        else:
            # 兼容无编码器环境：写入元数据占位文件
            video_path.write_bytes(f"MOCK_VIDEO_DATA_{event_id}_{len(resampled_frames)}".encode('utf-8'))

        # 2. 计算视频文件的完整 SHA-256 校验和
        video_bytes = video_path.read_bytes()
        video_file_hash = hashlib.sha256(video_bytes).hexdigest()

        # 3. 构造防篡改哈希区块 (Hash Chain Block) - 包含完整目标画像与语义上下文
        meta = getattr(pending, 'event_metadata', {}) or {}
        dwell_sec = round(float(meta.get("target_dwell_time_sec", 0.0)), 1)
        dist_m = round(float(meta.get("target_distance_m", 0.0)), 2)
        vr_mps = round(float(meta.get("target_radial_speed_mps", 0.0)), 2)

        with self._lock:
            block_index = len(self.manifest)
            formatted_iso_time = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(pending.trigger_time))
            formatted_iso_time += f".{int((pending.trigger_time % 1) * 1000):03d}Z"

            block_payload = {
                "block_index": block_index,
                "event_id": pending.event_id,
                "threat_level": pending.threat_level_str,
                "trigger_time": round(pending.trigger_time, 3),
                "trigger_time_iso": formatted_iso_time,
                "clock_source": pending.clock_source.value,
                "pre_frame_count": len(pending.pre_frames),
                "post_frame_count": len(pending.post_frames),
                "total_frame_count": len(resampled_frames),
                "video_file_name": video_filename,
                "video_file_sha256": video_file_hash,
                "prev_hash": self.prev_hash,
                # 语义审计核心字段
                "target": {
                    "category": meta.get("target_category", "unknown"),
                    "species": meta.get("target_species"),
                    "species_cn": meta.get("target_species_cn"),
                    "track_id": meta.get("target_id"),
                    "session_id": meta.get("person_session_id"),
                    "is_whitelisted": meta.get("is_whitelisted", False),
                    "distance_m": dist_m,
                    "dwell_time_sec": dwell_sec,      # 关键指标：在防区内持续累积徘徊/滞留时间 (秒)
                    "radial_speed_mps": vr_mps,
                    "zone": meta.get("target_zone", "")
                },
                "behavior": {
                    "action_type": meta.get("action_type", ""),
                    "description": meta.get("description", ""),
                    "has_contact_signal": meta.get("has_contact_signal", False),
                    "disturbed_rois": list(meta.get("disturbed_rois", [])),
                    "operating_mode": meta.get("operating_mode", "")
                },
                "vlm_summary": meta.get("vlm_summary", ""),
                "executed_actions": list(meta.get("executed_actions", []))
            }

            # 严格按字段排序序列化计算区块哈希
            payload_serialized = json.dumps(block_payload, sort_keys=True)
            block_hash = hashlib.sha256(payload_serialized.encode('utf-8')).hexdigest()

            # 更新哈希链头指针
            self.prev_hash = block_hash

            manifest_entry = {
                "block_hash": block_hash,
                "payload": block_payload
            }
            self.manifest.append(manifest_entry)

            # 追加写入持久化清单文件
            with open(self.manifest_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")

            # 写入单事件结构化 JSON 卡片
            card_file = self.storage_dir / f"{pending.event_id}_event_card.json"
            try:
                with open(card_file, "w", encoding="utf-8") as cf:
                    cf.write(json.dumps(block_payload, indent=2, ensure_ascii=False))
            except Exception:
                pass

            clip_duration_sec = len(resampled_frames) / max(self.fps, 1.0)
            log_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(pending.trigger_time))
            tgt_display = meta.get('target_category', 'unknown')
            if meta.get('target_species_cn'):
                tgt_display += f" ({meta.get('target_species_cn')})"
            elif meta.get('target_species'):
                tgt_display += f" ({meta.get('target_species')})"
            human_log = (
                f"\n================================================================================\n"
                f" 🛡️ 车载露营守卫 - 事件审计卡片 (Block #{block_index})\n"
                f"================================================================================\n"
                f" 事件编号       : {pending.event_id}\n"
                f" 触发时间       : {log_time_str} ({formatted_iso_time})\n"
                f" 威胁等级       : {pending.threat_level_str} | 运行模式: {meta.get('operating_mode', '未知')}\n"
                f" 目标主体       : {tgt_display} (Track ID: {meta.get('target_id', '-')}, 会话: {meta.get('person_session_id') or '无'}) | 白名单: {'是' if meta.get('is_whitelisted') else '否'}\n"
                f" 空间与测距     : 距离: {dist_m:.1f}m ({meta.get('target_zone', '防区内')}) | 相对速度: {vr_mps:+.1f} m/s\n"
                f" 徘徊/滞留时间  : {dwell_sec:.1f} 秒 (防区内持续累积)\n"
                f" 行为与接触     : {meta.get('action_type', '')} ({meta.get('description', '')}) | 物理接触: {'是 (已联合确认)' if meta.get('has_contact_signal') else '否'}\n"
                f" 车身扰动部位   : {', '.join(meta.get('disturbed_rois', [])) if meta.get('disturbed_rois') else '无'}\n"
                f" VLM 语义复核   : \"{meta.get('vlm_summary', '')}\"\n"
                f" 整车联动动作   : {', '.join(meta.get('executed_actions', [])) if meta.get('executed_actions') else '无'}\n"
                f" 存证录像文件   : {video_path} ({self.fps:.1f} FPS, {clip_duration_sec:.1f}s, 共 {len(resampled_frames)} 帧)\n"
                f" 视频 SHA-256   : {video_file_hash}\n"
                f" 防篡改区块哈希 : {block_hash}\n"
                f"================================================================================\n"
            )
            try:
                with open(self.storage_dir / "guard_events.log", "a", encoding="utf-8") as lf:
                    lf.write(human_log)
            except Exception:
                pass

            evidence_record = {
                "event_id": pending.event_id,
                "threat_level": pending.threat_level_str,
                "trigger_time": pending.trigger_time,
                "video_path": str(video_path),
                "video_file_sha256": video_file_hash,
                "block_hash": block_hash,
                "clock_source": pending.clock_source.value,
                "total_frames": len(resampled_frames),
                "is_locked": True,
                "target": block_payload["target"],
                "behavior": block_payload["behavior"],
                "vlm_summary": block_payload["vlm_summary"],
                "executed_actions": block_payload["executed_actions"],
                "payload": block_payload
            }
            self.locked_evidence.append(evidence_record)
            return evidence_record

    def verify_hash_chain(self, manifest_path: Optional[str] = None) -> Tuple[bool, str, Optional[int]]:
        """
        审计与校验防篡改哈希链完整性 (Tamper-Evidence Verification)
        验证每一区块的 prev_hash 是否连续匹配，以及 payload 哈希签名是否一致
        :return: (is_valid, report_message, tampered_block_index)
        """
        target_path = Path(manifest_path) if manifest_path else self.manifest_file
        if not target_path.exists():
            if len(self.manifest) == 0:
                return (True, "清单为空，无记录需要验证", None)
            entries = self.manifest
        else:
            entries = []
            with open(target_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))

        if not entries:
            return (True, "存证清单为空", None)

        expected_prev_hash = self.GENESIS_HASH

        for idx, entry in enumerate(entries):
            block_hash = entry.get("block_hash")
            payload = entry.get("payload", {})

            # 1. 验证前序哈希链接
            if payload.get("prev_hash") != expected_prev_hash:
                return (
                    False,
                    f"哈希链断裂！区块 [{idx}] 的 prev_hash ({payload.get('prev_hash')}) 与预期上一区块哈希 ({expected_prev_hash}) 不符！存在非法篡改或条目被删除！",
                    idx
                )

            # 2. 重新计算当前区块哈希并核对签名
            recomputed_hash = hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode('utf-8')
            ).hexdigest()

            if recomputed_hash != block_hash:
                return (
                    False,
                    f"数据签名被篡改！区块 [{idx}] (Event: {payload.get('event_id')}) 重新计算的 SHA-256 ({recomputed_hash}) 与记录的区块签名 ({block_hash}) 不匹配！",
                    idx
                )

            expected_prev_hash = block_hash

        return (True, f"存证哈希链完整性验证通过！共审计 {len(entries)} 个区块，链条未受任何篡改。", None)
