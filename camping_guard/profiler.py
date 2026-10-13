"""
端到端流水线与模型耗时性能剖析器 (camping_guard/profiler.py)
用于高精度统计:
1. 每一帧的端到端耗时 (E2E Latency, ms) 与等效 FPS
2. 各深度学习模型 (YOLO检测, RTMPose姿态, OSNet ReID, SCRFD人脸检测, ArcFace人脸识别等) 的调用情况:
   - 触发时机/前置条件
   - 每一帧与全流程累计调用次数
   - 每次调用的耗时 (最新耗时、最小、最大、平均值、P95分位数)
   - 各模型耗时占 E2E 的百分比
3. 提供终端表格渲染、HUD 仪表盘标注与 JSON 指标导出能力
"""

import time
import os
import json
import platform
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np


@dataclass
class ModelCallRecord:
    """单次模型调用事件记录"""
    model_name: str
    duration_ms: float
    condition: str = ""
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    input_shape: Optional[Tuple] = None
    details: str = ""
    gpu_ms: float = 0.0
    preprocess_ms: float = 0.0
    postprocess_ms: float = 0.0


@dataclass
class ModelStats:
    """各模型调用聚合统计指标 (支持预处理 / 纯GPU推理 / 后处理三阶段拆解)"""
    model_name: str
    weight_path: str = ""
    trigger_condition: str = ""
    call_count: int = 0
    durations_ms: List[float] = field(default_factory=list)
    gpu_durations_ms: List[float] = field(default_factory=list)
    preprocess_durations_ms: List[float] = field(default_factory=list)
    postprocess_durations_ms: List[float] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    is_async: bool = False
    is_substream: bool = False

    def record_call(
        self,
        duration_ms: float,
        condition: str = "",
        gpu_ms: Optional[float] = None,
        preprocess_ms: Optional[float] = None,
        postprocess_ms: Optional[float] = None
    ):
        self.call_count += 1
        self.durations_ms.append(float(duration_ms))
        if gpu_ms is not None and isinstance(gpu_ms, (int, float)) and gpu_ms > 0:
            self.gpu_durations_ms.append(float(gpu_ms))
        if preprocess_ms is not None and isinstance(preprocess_ms, (int, float)) and preprocess_ms >= 0:
            self.preprocess_durations_ms.append(float(preprocess_ms))
        if postprocess_ms is not None and isinstance(postprocess_ms, (int, float)) and postprocess_ms >= 0:
            self.postprocess_durations_ms.append(float(postprocess_ms))
        if condition and condition not in self.conditions:
            self.conditions.append(condition)

    @property
    def avg_gpu_ms(self) -> float:
        return (sum(self.gpu_durations_ms) / len(self.gpu_durations_ms)) if self.gpu_durations_ms else 0.0

    @property
    def avg_preprocess_ms(self) -> float:
        return (sum(self.preprocess_durations_ms) / len(self.preprocess_durations_ms)) if self.preprocess_durations_ms else 0.0

    @property
    def avg_postprocess_ms(self) -> float:
        if self.postprocess_durations_ms:
            return sum(self.postprocess_durations_ms) / len(self.postprocess_durations_ms)
        if self.avg_gpu_ms > 0:
            return max(0.0, self.avg_ms - self.avg_gpu_ms - self.avg_preprocess_ms)
        return 0.0

    @property
    def total_ms(self) -> float:
        return sum(self.durations_ms) if self.durations_ms else 0.0

    @property
    def avg_ms(self) -> float:
        return (self.total_ms / len(self.durations_ms)) if self.durations_ms else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.durations_ms) if self.durations_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.durations_ms) if self.durations_ms else 0.0

    @property
    def std_ms(self) -> float:
        return float(np.std(self.durations_ms)) if self.durations_ms else 0.0

    @property
    def p50_ms(self) -> float:
        return float(np.percentile(self.durations_ms, 50)) if self.durations_ms else 0.0

    @property
    def p90_ms(self) -> float:
        return float(np.percentile(self.durations_ms, 90)) if self.durations_ms else 0.0

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.durations_ms, 95)) if self.durations_ms else 0.0

    @property
    def p99_ms(self) -> float:
        return float(np.percentile(self.durations_ms, 99)) if self.durations_ms else 0.0

    @property
    def last_ms(self) -> float:
        return self.durations_ms[-1] if self.durations_ms else 0.0


@dataclass
class FrameMetrics:
    """单帧性能评估指标快照"""
    frame_idx: int
    start_time: float
    end_time: float
    e2e_ms: float
    model_calls: List[ModelCallRecord] = field(default_factory=list)
    stage_durations_ms: Dict[str, float] = field(default_factory=dict)
    threat_level: str = "L0_SAFE"
    target_count: int = 0
    vlm_is_inferencing: bool = False
    rss_mb: float = 0.0

    @property
    def fps(self) -> float:
        return (1000.0 / self.e2e_ms) if self.e2e_ms > 0 else 0.0

    def get_model_summary(self) -> Dict[str, Dict[str, Any]]:
        """按模型聚合当前帧的调用次数与耗时"""
        summary = {}
        for call in self.model_calls:
            name = call.model_name
            if name not in summary:
                summary[name] = {"count": 0, "total_ms": 0.0, "calls": []}
            summary[name]["count"] += 1
            summary[name]["total_ms"] += call.duration_ms
            summary[name]["calls"].append(call.duration_ms)
        return summary


class ModelProfileContext:
    """模型调用测量上下文管理器"""
    def __init__(
        self,
        profiler: "PipelineProfiler",
        model_name: str,
        condition: str = "",
        input_shape: Optional[Tuple] = None,
        session: Optional[Any] = None
    ):
        self.profiler = profiler
        self.model_name = model_name
        self.condition = condition
        self.input_shape = input_shape
        self.session = session
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.t0) * 1000.0
        success = (exc_type is None)
        gpu_ms = getattr(self.session, "last_gpu_compute_ms", None) if self.session else None
        preprocess_ms = getattr(self.session, "last_preprocess_ms", None) if self.session else None
        postprocess_ms = getattr(self.session, "last_postprocess_ms", None) if self.session else None
        self.profiler.record_model_call(
            model_name=self.model_name,
            duration_ms=duration_ms,
            condition=self.condition,
            success=success,
            input_shape=self.input_shape,
            gpu_ms=gpu_ms,
            preprocess_ms=preprocess_ms,
            postprocess_ms=postprocess_ms
        )


class StageProfileContext:
    """流水线业务阶段耗时上下文管理器"""
    def __init__(self, profiler: "PipelineProfiler", stage_name: str):
        self.profiler = profiler
        self.stage_name = stage_name
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.t0) * 1000.0
        self.profiler.record_stage_duration(self.stage_name, duration_ms)


class PipelineProfiler:
    """端到端流水线与模型调用性能分析器"""

    def __init__(self):
        # 模型级全局统计字典: {model_name: ModelStats}
        self.model_stats: Dict[str, ModelStats] = {}
        # 历史每帧记录
        self.frame_history: List[FrameMetrics] = []
        # 业务阶段全局统计: {stage_name: List[float]}
        self.stage_history: Dict[str, List[float]] = {}

        # GPU 显存与硬件资源统计
        self.peak_vram_allocated_mb: float = 0.0
        self.peak_vram_reserved_mb: float = 0.0
        self.peak_global_vram_used_mb: float = 0.0
        self.initial_rss_mb: float = self._read_process_rss_mb()
        self.peak_rss_mb: float = self.initial_rss_mb
        self._last_sampled_rss_mb: float = self.initial_rss_mb

        # 运行期外部注入的上下文 (视频参数、跳帧/录制帧数、CLI参数、VLM审计等)
        self.runtime_context: Dict[str, Any] = {}

        # 当前帧临时变量
        self._current_frame_idx: int = 0
        self._current_frame_start: float = 0.0
        self._current_frame_calls: List[ModelCallRecord] = []
        self._current_stage_durations: Dict[str, float] = {}
        self._is_profiling_frame: bool = False

        # 预设模型元数据 (便于展示与归类)
        self.model_metadata: Dict[str, Dict[str, Any]] = {
            "YOLO11n-Det": {
                "default_weight": "yolo11n.onnx",
                "condition": "S1全图目标检测初筛 (640x480)",
                "category": "Detection"
            },
            "RTMPose-s": {
                "default_weight": "rtmpose-s.onnx",
                "condition": "Zone2/3人员姿态与17骨骼点提取",
                "category": "Pose"
            },
            "OSNet-ReID": {
                "default_weight": "models/reid/osnet_x0_25.onnx",
                "condition": "人员跨时空重识别512维特征建档",
                "category": "ReID"
            },
            "SCRFD-500M": {
                "default_weight": "models/insightface/det_500m.onnx",
                "condition": "4m以内人员上半身人脸与5关键点初筛",
                "category": "FaceDetection"
            },
            "ArcFace-MobileFaceNet": {
                "default_weight": "models/insightface/w600k_mbf.onnx",
                "condition": "FQA质检通过后5点对齐白名单比对",
                "category": "FaceRecognition"
            },
            "MegaDetector": {
                "default_weight": "models/megadetector/md_v5a.onnx",
                "condition": "野生动物与夜间生物检测初筛",
                "category": "Detection"
            },
            "MobileNetV3-AnimalCls": {
                "default_weight": "models/cls/mobilenetv3-large.onnx",
                "condition": "MegaDetector二级动物细分类 (224x224)",
                "category": "Classification"
            },
            "Qwen-VLM": {
                "default_weight": "Qwen2.5-VL / Qwen3.5-VLM",
                "condition": "慢通路异步端侧多模态复核与中文战报生成",
                "category": "MultimodalReview",
                "is_async": True
            }
        }

    @staticmethod
    def _read_process_rss_mb() -> float:
        try:
            import psutil
            return float(psutil.Process().memory_info().rss / (1024.0 * 1024.0))
        except Exception:
            return 0.0

    def set_runtime_context(self, **kwargs):
        """更新运行期环境、视频流与录像统计上下文"""
        self.runtime_context.update(kwargs)

    def register_model_metadata(
        self,
        model_name: str,
        weight_path: str = "",
        condition: str = "",
        is_async: bool = False,
        is_substream: bool = False
    ):
        """注册或更新模型元信息"""
        if model_name not in self.model_stats:
            self.model_stats[model_name] = ModelStats(
                model_name=model_name,
                weight_path=weight_path,
                trigger_condition=condition,
                is_async=is_async,
                is_substream=is_substream
            )
        else:
            if weight_path:
                self.model_stats[model_name].weight_path = weight_path
            if condition:
                self.model_stats[model_name].trigger_condition = condition
            if is_async:
                self.model_stats[model_name].is_async = is_async
            if is_substream:
                self.model_stats[model_name].is_substream = is_substream

    def start_frame(self, frame_idx: int):
        """开启一帧的 E2E 计时"""
        self._current_frame_idx = frame_idx
        self._current_frame_start = time.perf_counter()
        self._current_frame_calls = []
        self._current_stage_durations = {}
        self._is_profiling_frame = True

    def end_frame(
        self,
        threat_level: str = "L0_SAFE",
        target_count: int = 0,
        vlm_is_inferencing: bool = False
    ) -> FrameMetrics:
        """结束一帧计时并返回当前帧指标快照"""
        t_end = time.perf_counter()
        e2e_ms = (t_end - self._current_frame_start) * 1000.0 if self._current_frame_start > 0 else 0.0

        # 实时监控 GPU 显存峰值与周期性采样内存 (每 25 帧采样一次 RSS/全局显存以保持 0 开销)
        try:
            import torch
            if torch.cuda.is_available():
                max_alloc = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                if max_alloc > self.peak_vram_allocated_mb:
                    self.peak_vram_allocated_mb = max_alloc
                max_res = torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)
                if max_res > self.peak_vram_reserved_mb:
                    self.peak_vram_reserved_mb = max_res
                if self._current_frame_idx % 25 == 1:
                    try:
                        free_b, total_b = torch.cuda.mem_get_info()
                        used_mb = (total_b - free_b) / (1024.0 * 1024.0)
                        if used_mb > self.peak_global_vram_used_mb:
                            self.peak_global_vram_used_mb = used_mb
                    except Exception:
                        pass
        except Exception:
            pass

        if self._current_frame_idx % 25 == 1:
            rss = self._read_process_rss_mb()
            if rss > 0:
                self._last_sampled_rss_mb = rss
                if self.initial_rss_mb <= 0:
                    self.initial_rss_mb = rss
                if rss > self.peak_rss_mb:
                    self.peak_rss_mb = rss

        for stg_name, stg_dur in self._current_stage_durations.items():
            if stg_name not in self.stage_history:
                self.stage_history[stg_name] = []
            self.stage_history[stg_name].append(float(stg_dur))

        metrics = FrameMetrics(
            frame_idx=self._current_frame_idx,
            start_time=self._current_frame_start,
            end_time=t_end,
            e2e_ms=e2e_ms,
            model_calls=list(self._current_frame_calls),
            stage_durations_ms=dict(self._current_stage_durations),
            threat_level=threat_level,
            target_count=target_count,
            vlm_is_inferencing=bool(vlm_is_inferencing),
            rss_mb=self._last_sampled_rss_mb
        )
        self.frame_history.append(metrics)
        self._is_profiling_frame = False
        return metrics

    def profile_model(
        self,
        model_name: str,
        condition: str = "",
        input_shape: Optional[Tuple] = None,
        session: Optional[Any] = None
    ) -> ModelProfileContext:
        """模型调用测量上下文"""
        return ModelProfileContext(self, model_name=model_name, condition=condition, input_shape=input_shape, session=session)

    def profile_stage(self, stage_name: str) -> StageProfileContext:
        """阶段耗时测量上下文"""
        return StageProfileContext(self, stage_name=stage_name)

    def record_model_call(
        self,
        model_name: str,
        duration_ms: float,
        condition: str = "",
        success: bool = True,
        input_shape: Optional[Tuple] = None,
        gpu_ms: Optional[float] = None,
        is_async: Optional[bool] = None,
        is_substream: Optional[bool] = None,
        preprocess_ms: Optional[float] = None,
        postprocess_ms: Optional[float] = None
    ):
        """记录单次模型调用"""
        if model_name not in self.model_stats:
            meta = self.model_metadata.get(model_name, {})
            model_is_async = meta.get("is_async", False) or (meta.get("category") == "MultimodalReview") or ("vlm" in model_name.lower())
            if is_async is not None:
                model_is_async = is_async
            model_is_substream = meta.get("is_substream", False)
            if is_substream is not None:
                model_is_substream = is_substream
            self.model_stats[model_name] = ModelStats(
                model_name=model_name,
                weight_path=meta.get("default_weight", ""),
                trigger_condition=condition or meta.get("condition", ""),
                is_async=model_is_async,
                is_substream=model_is_substream
            )
        else:
            if is_async is not None:
                self.model_stats[model_name].is_async = is_async
            if is_substream is not None:
                self.model_stats[model_name].is_substream = is_substream

        self.model_stats[model_name].record_call(
            duration_ms,
            condition=condition,
            gpu_ms=gpu_ms,
            preprocess_ms=preprocess_ms,
            postprocess_ms=postprocess_ms
        )

        rec = ModelCallRecord(
            model_name=model_name,
            duration_ms=duration_ms,
            condition=condition,
            success=success,
            input_shape=input_shape,
            gpu_ms=float(gpu_ms or 0.0),
            preprocess_ms=float(preprocess_ms or 0.0),
            postprocess_ms=float(postprocess_ms or 0.0)
        )
        # 仅主同步模型调用才计入当前帧快照，异步与并发子流不污染单帧同步 E2E 列表
        if self._is_profiling_frame and not self.model_stats[model_name].is_async and not self.model_stats[model_name].is_substream:
            self._current_frame_calls.append(rec)

    def record_stage_duration(self, stage_name: str, duration_ms: float):
        """记录某业务阶段的耗时；若在 end_frame 之后记录 (如 Render_and_Output)，自动同步至最新帧与全局阶段统计"""
        if self._is_profiling_frame:
            self._current_stage_durations[stage_name] = (
                self._current_stage_durations.get(stage_name, 0.0) + duration_ms
            )
        else:
            if self.frame_history:
                last_f = self.frame_history[-1]
                last_f.stage_durations_ms[stage_name] = last_f.stage_durations_ms.get(stage_name, 0.0) + duration_ms
            if stage_name not in self.stage_history:
                self.stage_history[stage_name] = []
            self.stage_history[stage_name].append(float(duration_ms))

    @property
    def total_frames(self) -> int:
        return len(self.frame_history)

    @staticmethod
    def _compute_series_stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {
                "count": 0,
                "avg_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "std_ms": 0.0,
                "p50_ms": 0.0,
                "p90_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "fps": 0.0
            }
        arr = np.asarray(values, dtype=np.float64)
        avg = float(np.mean(arr))
        return {
            "count": int(len(values)),
            "avg_ms": round(avg, 3),
            "min_ms": round(float(np.min(arr)), 3),
            "max_ms": round(float(np.max(arr)), 3),
            "std_ms": round(float(np.std(arr)), 3),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p90_ms": round(float(np.percentile(arr, 90)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "fps": round(1000.0 / avg, 2) if avg > 0 else 0.0
        }

    def get_summary(self) -> Dict[str, Any]:
        """计算全周期的聚合统计指标"""
        total_frames = len(self.frame_history)
        if total_frames == 0:
            return {
                "total_frames": 0,
                "e2e": {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "p95_ms": 0.0, "fps": 0.0},
                "models": {},
                "stages": {}
            }

        e2e_list = [f.e2e_ms for f in self.frame_history]
        total_e2e_ms = sum(e2e_list)
        overall_stats = self._compute_series_stats(e2e_list)

        idle_e2e = [f.e2e_ms for f in self.frame_history if not f.vlm_is_inferencing]
        busy_e2e = [f.e2e_ms for f in self.frame_history if f.vlm_is_inferencing]
        idle_stats = self._compute_series_stats(idle_e2e)
        busy_stats = self._compute_series_stats(busy_e2e)

        model_summaries = {}
        for name, stats in self.model_stats.items():
            is_trt = str(stats.weight_path).lower().endswith(".engine") or "engine" in str(stats.weight_path).lower()
            pre_ms = stats.avg_preprocess_ms
            gpu_ms = stats.avg_gpu_ms
            post_ms = stats.avg_postprocess_ms
            pre_post_combined = max(0.0, stats.avg_ms - gpu_ms) if gpu_ms > 0 else stats.avg_ms
            model_summaries[name] = {
                "model_name": name,
                "weight_path": stats.weight_path,
                "backend": "TensorRT" if is_trt else "ONNXRuntime",
                "trigger_condition": stats.trigger_condition,
                "is_async": stats.is_async,
                "is_substream": stats.is_substream,
                "total_calls": stats.call_count,
                "calls_per_frame": round((stats.call_count / total_frames), 4) if total_frames > 0 else 0.0,
                "total_time_ms": round(stats.total_ms, 2),
                "pct_of_e2e": round(((stats.total_ms / total_e2e_ms) * 100.0), 2) if (total_e2e_ms > 0 and not stats.is_async and not stats.is_substream) else 0.0,
                "avg_preprocess_ms": round(pre_ms, 3),
                "avg_gpu_compute_ms": round(gpu_ms, 3),
                "avg_postprocess_ms": round(post_ms, 3),
                "avg_pre_post_combined_ms": round(pre_post_combined, 3),
                "avg_duration_ms": round(stats.avg_ms, 3),
                "min_duration_ms": round(stats.min_ms, 3),
                "max_duration_ms": round(stats.max_ms, 3),
                "std_duration_ms": round(stats.std_ms, 3),
                "p50_duration_ms": round(stats.p50_ms, 3),
                "p90_duration_ms": round(stats.p90_ms, 3),
                "p95_duration_ms": round(stats.p95_ms, 3),
                "p99_duration_ms": round(stats.p99_ms, 3),
                "last_duration_ms": round(stats.last_ms, 3),
                "observed_conditions": list(stats.conditions[:10])
            }

        stage_summaries = {}
        for stg_name, durs in self.stage_history.items():
            stg_s = self._compute_series_stats(durs)
            tot_stg = float(sum(durs))
            stage_summaries[stg_name] = {
                "stage_name": stg_name,
                "calls": stg_s["count"],
                "total_time_ms": round(tot_stg, 2),
                "avg_ms": stg_s["avg_ms"],
                "p50_ms": stg_s["p50_ms"],
                "p95_ms": stg_s["p95_ms"],
                "max_ms": stg_s["max_ms"],
                "pct_of_e2e": round((tot_stg / total_e2e_ms) * 100.0, 2) if total_e2e_ms > 0 else 0.0
            }

        slowdown_ms = (busy_stats["avg_ms"] - idle_stats["avg_ms"]) if (busy_stats["count"] > 0 and idle_stats["count"] > 0) else 0.0
        slowdown_pct = ((slowdown_ms / idle_stats["avg_ms"]) * 100.0) if (idle_stats["avg_ms"] > 0 and busy_stats["count"] > 0) else 0.0

        summary_res = {
            "total_frames": total_frames,
            "total_time_sec": round(total_e2e_ms / 1000.0, 3),
            "e2e": {
                "avg_ms": overall_stats["avg_ms"],
                "min_ms": overall_stats["min_ms"],
                "max_ms": overall_stats["max_ms"],
                "std_ms": overall_stats["std_ms"],
                "p50_ms": overall_stats["p50_ms"],
                "p90_ms": overall_stats["p90_ms"],
                "p95_ms": overall_stats["p95_ms"],
                "p99_ms": overall_stats["p99_ms"],
                "fps": overall_stats["fps"],
                "vlm_idle_e2e": idle_stats,
                "vlm_busy_e2e": busy_stats,
                "vlm_contention_slowdown_ms": round(slowdown_ms, 3),
                "vlm_contention_slowdown_pct": round(slowdown_pct, 2)
            },
            "models": model_summaries,
            "stages": stage_summaries,
            "memory": self.get_memory_stats()
        }
        return summary_res

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取当前与历史峰值显存 (VRAM) 及系统物理内存使用指标"""
        final_rss = self._read_process_rss_mb() or self._last_sampled_rss_mb
        peak_rss = max(self.peak_rss_mb, final_rss)
        init_rss = self.initial_rss_mb if self.initial_rss_mb > 0 else final_rss

        mem_info = {
            "has_cuda": False,
            "peak_vram_allocated_mb": round(self.peak_vram_allocated_mb, 2),
            "peak_vram_reserved_mb": round(self.peak_vram_reserved_mb, 2),
            "current_vram_allocated_mb": 0.0,
            "current_vram_reserved_mb": 0.0,
            "global_vram_used_mb": 0.0,
            "peak_global_vram_used_mb": round(self.peak_global_vram_used_mb, 2),
            "global_vram_total_mb": 0.0,
            "global_vram_pct": 0.0,
            "initial_rss_mb": round(init_rss, 2),
            "peak_rss_mb": round(peak_rss, 2),
            "process_rss_mb": round(final_rss, 2),
            "rss_growth_mb": round(final_rss - init_rss, 2)
        }
        try:
            import torch
            if torch.cuda.is_available():
                mem_info["has_cuda"] = True
                cur_alloc = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
                cur_res = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
                max_alloc = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                max_res = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
                try:
                    max_res = torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)
                except Exception:
                    pass

                mem_info["current_vram_allocated_mb"] = round(cur_alloc, 2)
                mem_info["current_vram_reserved_mb"] = round(cur_res, 2)
                mem_info["peak_vram_allocated_mb"] = round(max(self.peak_vram_allocated_mb, max_alloc), 2)
                mem_info["peak_vram_reserved_mb"] = round(max(self.peak_vram_reserved_mb, max_res), 2)

                try:
                    free_b, total_b = torch.cuda.mem_get_info()
                    used_b = total_b - free_b
                    used_mb = used_b / (1024.0 * 1024.0)
                    total_mb = total_b / (1024.0 * 1024.0)
                    mem_info["global_vram_used_mb"] = round(used_mb, 2)
                    mem_info["peak_global_vram_used_mb"] = round(max(self.peak_global_vram_used_mb, used_mb), 2)
                    mem_info["global_vram_total_mb"] = round(total_mb, 2)
                    mem_info["global_vram_pct"] = round((used_b / total_b * 100.0), 2) if total_b > 0 else 0.0
                except Exception:
                    pass
        except Exception:
            pass

        return mem_info

    def build_diagnostic_json(
        self,
        extra_context: Optional[Dict[str, Any]] = None,
        timeseries_stride: int = 25
    ) -> Dict[str, Any]:
        """
        构建包含 8 大维度的全量结构化 JSON 诊断报告，供边缘端 Orin 性能回传与瓶颈分析
        """
        ctx = dict(self.runtime_context)
        if extra_context:
            ctx.update(extra_context)

        summary = self.get_summary()
        total_inferred = summary["total_frames"]

        # 1. 环境与硬件/软件配置信息
        env_info: Dict[str, Any] = {
            "report_generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": time.time(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count() or 1,
            "is_jetson": False,
            "cuda_device_name": "CPU",
            "torch_version": None,
            "tensorrt_version": None,
            "onnxruntime_version": None,
        }
        try:
            from camping_guard.perception.trt_engine import is_jetson
            env_info["is_jetson"] = bool(is_jetson())
        except Exception:
            pass
        try:
            import torch
            env_info["torch_version"] = torch.__version__
            if torch.cuda.is_available():
                env_info["cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
        try:
            import tensorrt as trt
            env_info["tensorrt_version"] = trt.__version__
        except Exception:
            pass
        try:
            import onnxruntime as ort
            env_info["onnxruntime_version"] = ort.__version__
        except Exception:
            pass

        if "cli_args" in ctx:
            env_info["cli_args"] = ctx["cli_args"]
        if "video_info" in ctx:
            env_info["video_info"] = ctx["video_info"]

        # 2. 视频流流速与录像平滑度健康指标
        vid_total = int(ctx.get("video_total_frames", ctx.get("total_source_frames", total_inferred)))
        skipped_frames = int(ctx.get("clock_catchup_skipped_frames", ctx.get("skipped_frames", 0)))
        written_frames = int(ctx.get("output_video_written_frames", ctx.get("output_written_frames", 0)))
        wall_elapsed = float(ctx.get("wall_clock_elapsed_sec", summary.get("total_time_sec", 0.0)))
        src_fps = float(ctx.get("source_fps", 30.0) or 30.0)
        processed_span_frames = total_inferred + skipped_frames
        expected_phys_sec = (processed_span_frames / src_fps) if src_fps > 0 else 0.0
        output_play_sec = (written_frames / src_fps) if (written_frames > 0 and src_fps > 0) else 0.0

        video_health = {
            "source_video_total_frames": vid_total,
            "source_fps": round(src_fps, 2),
            "ai_inferred_frames": total_inferred,
            "clock_catchup_skipped_frames": skipped_frames,
            "skip_ratio_pct": round((skipped_frames / max(1, processed_span_frames)) * 100.0, 2),
            "output_video_written_frames": written_frames,
            "wall_clock_elapsed_sec": round(wall_elapsed, 3),
            "expected_physical_duration_sec": round(expected_phys_sec, 3),
            "output_playback_duration_sec": round(output_play_sec, 3),
            "actual_display_fps": round(processed_span_frames / wall_elapsed, 2) if wall_elapsed > 0 else 0.0,
            "ai_inference_fps": round(total_inferred / wall_elapsed, 2) if wall_elapsed > 0 else 0.0,
            "frame_box_sync_mode": "STRICT_SINGLE_STREAM_SAME_FRAME_SYNC"
        }

        # 8. 帧级时序抽样曲线与 Top-10 最慢卡顿帧定位
        timeseries = []
        stride = max(1, int(timeseries_stride))
        prev_vlm_busy = False
        for i, fm in enumerate(self.frame_history):
            is_transition = (fm.vlm_is_inferencing != prev_vlm_busy)
            prev_vlm_busy = fm.vlm_is_inferencing
            if (i % stride == 0) or is_transition or (i == len(self.frame_history) - 1):
                timeseries.append({
                    "frame_idx": fm.frame_idx,
                    "e2e_ms": round(fm.e2e_ms, 2),
                    "fps": round(fm.fps, 1),
                    "target_count": fm.target_count,
                    "threat_level": fm.threat_level,
                    "vlm_is_inferencing": fm.vlm_is_inferencing,
                    "rss_mb": round(fm.rss_mb, 1),
                    "stages_ms": {k: round(v, 2) for k, v in fm.stage_durations_ms.items()}
                })

        sorted_slowest = sorted(self.frame_history, key=lambda x: x.e2e_ms, reverse=True)[:10]
        slowest_top10 = []
        for fm in sorted_slowest:
            slowest_top10.append({
                "frame_idx": fm.frame_idx,
                "e2e_ms": round(fm.e2e_ms, 2),
                "vlm_is_inferencing": fm.vlm_is_inferencing,
                "target_count": fm.target_count,
                "threat_level": fm.threat_level,
                "stage_durations_ms": {k: round(v, 2) for k, v in fm.stage_durations_ms.items()},
                "model_calls": [
                    {
                        "model": c.model_name,
                        "duration_ms": round(c.duration_ms, 2),
                        "gpu_ms": round(c.gpu_ms, 2),
                        "preprocess_ms": round(c.preprocess_ms, 2),
                        "postprocess_ms": round(c.postprocess_ms, 2),
                        "condition": c.condition
                    }
                    for c in fm.model_calls
                ]
            })

        vlm_audit_data = ctx.get("vlm_audit_summary") or ctx.get("vlm_audit") or {}
        report = {
            "schema_version": "2.0",
            "runtime_context": dict(ctx),
            "environment_and_config": env_info,
            "video_and_recording_health": video_health,
            "e2e_latency_breakdown": summary.get("e2e", {}),
            "e2e_summary": summary.get("e2e", {}),
            "models_detailed": summary.get("models", {}),
            "models": summary.get("models", {}),
            "pipeline_stages": summary.get("stages", {}),
            "vlm_audit_summary": vlm_audit_data,
            "memory_and_hardware": summary.get("memory", {}),
            "slowest_frames_top10": slowest_top10,
            "frame_timeseries": timeseries,
        }
        return report

    def export_json_report(
        self,
        output_paths: Optional[Any] = None,
        extra_context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """将全量结构化性能诊断报告写入 JSON 文件并返回落盘路径(或路径列表)"""
        report = self.build_diagnostic_json(extra_context=extra_context)
        single_input = isinstance(output_paths, (str, Path))
        if output_paths is None:
            paths = ["logs/latest_perf_report.json"]
        elif single_input:
            paths = [output_paths]
        else:
            paths = list(output_paths)

        saved_paths: List[str] = []
        for p in paths:
            if not p:
                continue
            try:
                out_file = Path(p).resolve()
                out_file.parent.mkdir(parents=True, exist_ok=True)
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                saved_paths.append(str(out_file))
            except Exception as e:
                print(f"[!] 导出性能诊断 JSON 失败 ({p}): {e}")
        if single_input:
            return saved_paths[0] if saved_paths else str(output_paths)
        return saved_paths

    def format_single_frame_report(self, frame_metrics: Optional[FrameMetrics] = None) -> str:
        """生成单张图像/单帧的指标统计报表"""
        fm = frame_metrics or (self.frame_history[-1] if self.frame_history else None)
        if not fm:
            return "[!] 无可用单帧指标数据"

        lines = []
        lines.append("=" * 82)
        lines.append(f"【端到端性能与模型调用指标诊断报表 (Frame {fm.frame_idx})】")
        lines.append("=" * 82)
        lines.append(f"  > 单帧端到端 E2E 耗时: {fm.e2e_ms:.2f} ms  (理论峰值帧率: {fm.fps:.1f} FPS)")
        lines.append(f"  > 防护威胁等级: {fm.threat_level} | 场景目标数: {fm.target_count}")
        lines.append("-" * 82)

        # 阶段耗时统计
        if fm.stage_durations_ms:
            lines.append("【业务阶段耗时分布】")
            stage_strs = [f"{stg}: {dur:.2f}ms" for stg, dur in fm.stage_durations_ms.items()]
            lines.append("  | " + " | ".join(stage_strs) + " |")
            lines.append("-" * 82)

        # 模型调用列表
        lines.append("【各模型当帧调用明细与耗时】")
        header = f"{'模型名称':<22} | {'调用次数':<8} | {'每次调用耗时 (ms)':<22} | {'触发条件与上下文':<24}"
        lines.append(header)
        lines.append("-" * 82)

        grouped = fm.get_model_summary()
        if not grouped:
            lines.append("  (当帧未触发任何深度学习模型推理)")
        else:
            for name, info in grouped.items():
                call_ms_str = ", ".join([f"{d:.1f}" for d in info["calls"]])
                if len(call_ms_str) > 20:
                    call_ms_str = f"avg={info['total_ms']/info['count']:.1f} ({info['count']}次)"
                conds = [c.condition for c in fm.model_calls if c.model_name == name and c.condition]
                cond_str = conds[0] if conds else "常规按需触发"
                if len(cond_str) > 24:
                    cond_str = cond_str[:21] + "..."
                lines.append(f"{name:<22} | {info['count']:<8} | {call_ms_str:<22} | {cond_str:<24}")

        lines.append("=" * 82)
        return "\n".join(lines)

    def format_summary_table(self) -> str:
        """格式化打印全流程多帧综合汇总报表 (适合视频回放结束后的总评)"""
        summary = self.get_summary()
        total_frames = summary["total_frames"]
        if total_frames == 0:
            return "[!] 尚无帧数据可供生成汇总统计"

        e2e = summary["e2e"]
        models = summary["models"]

        lines = []
        lines.append("=" * 96)
        lines.append("          露营守卫算法端到端 (E2E) 与各模型调用综合性能评估报告")
        lines.append("=" * 96)
        lines.append(f" 评估样本总数 : {total_frames} 帧  (总耗时: {summary['total_time_sec']:.2f} 秒)")
        lines.append(f" 端到端 (E2E) : 平均 {e2e['avg_ms']:.2f} ms | Min: {e2e['min_ms']:.2f} ms | Max: {e2e['max_ms']:.2f} ms")
        lines.append(f" 分位数统计   : P50 (中位数): {e2e['p50_ms']:.2f} ms | P90: {e2e['p90_ms']:.2f} ms | P95: {e2e['p95_ms']:.2f} ms | P99: {e2e['p99_ms']:.2f} ms")
        lines.append(f" 理论等效帧率 : {e2e['fps']:.1f} FPS")

        idle_s = e2e.get("vlm_idle_e2e", {})
        busy_s = e2e.get("vlm_busy_e2e", {})
        if busy_s.get("count", 0) > 0:
            lines.append(
                f" VLM 并发对比 : 空闲期 ({idle_s.get('count', 0)}帧) 平均 {idle_s.get('avg_ms', 0.0):.2f} ms ({idle_s.get('fps', 0.0):.1f} FPS) | "
                f"VLM推理期 ({busy_s.get('count', 0)}帧) 平均 {busy_s.get('avg_ms', 0.0):.2f} ms ({busy_s.get('fps', 0.0):.1f} FPS)"
            )
        lines.append("-" * 96)
        lines.append("【各模型调用频次与每次调用耗时统计 (Per-Model Profiling Breakdown)】")

        hdr = f"{'模型名称':<26} | {'总调用':<6} | {'纯GPU推理':<10} | {'前后处理':<10} | {'总计(ms)':<8} | {'P95(ms)':<8} | {'E2E占比':<8}"
        lines.append(hdr)
        lines.append("-" * 96)

        for name, m in models.items():
            stats_obj = self.model_stats.get(name)
            is_async = (stats_obj.is_async if stats_obj else False) or ("vlm" in name.lower())
            is_substream = (stats_obj.is_substream if stats_obj else False)
            if is_async:
                pct_str = "异步并发"
            elif is_substream:
                pct_str = "并发子流"
            else:
                pct_str = f"{m['pct_of_e2e']:.1f}%"

            is_trt = str(m['weight_path']).lower().endswith(".engine") or "engine" in str(m['weight_path']).lower()
            display_name = f"{name} (TRT)" if is_trt else name
            gpu_ms = stats_obj.avg_gpu_ms if stats_obj else 0.0
            if gpu_ms > 0:
                gpu_str = f"{gpu_ms:6.2f} ms"
                post_ms = max(0.0, m['avg_duration_ms'] - gpu_ms)
                post_str = f"{post_ms:6.2f} ms"
            else:
                gpu_str = "    -     "
                post_str = f"{m['avg_duration_ms']:6.2f} ms"

            lines.append(
                f"{display_name:<26} | {m['total_calls']:<6} | {gpu_str:<10} | {post_str:<10} | "
                f"{m['avg_duration_ms']:<8.2f} | {m['p95_duration_ms']:<8.2f} | {pct_str:<8}"
            )

        # 显存与硬件资源统计
        mem = self.get_memory_stats()
        if mem["has_cuda"]:
            lines.append("-" * 96)
            lines.append("【检测期间显存与硬件资源开销 (VRAM & Memory Breakdown)】")
            lines.append(f"  * 框架活跃显存峰值 (Peak VRAM Allocated)  : {mem['peak_vram_allocated_mb']:>8.2f} MB (PyTorch/TRT活跃张量与I/O缓冲区)")
            lines.append(f"  * 显存保留缓存峰值 (Peak VRAM Reserved) : {mem['peak_vram_reserved_mb']:>8.2f} MB (CUDA Caching Allocator保留池)")
            if mem["global_vram_total_mb"] > 0:
                lines.append(f"  * Orin 设备总物理显存占用 (Unified VRAM) : {mem['global_vram_used_mb']:>8.2f} MB / {mem['global_vram_total_mb']:>8.2f} MB ({mem['global_vram_pct']:.1f}%, 峰值: {mem['peak_global_vram_used_mb']:.1f} MB)")
            if mem["process_rss_mb"] > 0:
                lines.append(f"  * 守卫主进程物理内存 (Host Process RSS): {mem['process_rss_mb']:>8.2f} MB (初始: {mem['initial_rss_mb']:.1f} MB, 峰值: {mem['peak_rss_mb']:.1f} MB)")

        lines.append("-" * 96)
        lines.append("【模型触发机制与运行说明】")

        def _format_weight_name(p_str: str) -> str:
            if not p_str:
                return "内置引擎"
            if " + " in p_str:
                parts = [p.strip() for p in p_str.split(" + ")]
                return " + ".join(os.path.basename(p) if os.path.basename(p) else p for p in parts)
            return os.path.basename(p_str) if os.path.basename(p_str) else p_str

        for name, m in models.items():
            cond = m['trigger_condition'] or "随场景目标动态触发"
            w_path = m['weight_path'] or "内置引擎"
            stats_obj = self.model_stats.get(name)
            gpu_ms = stats_obj.avg_gpu_ms if stats_obj else 0.0
            pre_ms = stats_obj.avg_preprocess_ms if stats_obj else 0.0
            post_ms = stats_obj.avg_postprocess_ms if stats_obj else 0.0
            is_async = (stats_obj.is_async if stats_obj else False) or ("vlm" in name.lower())
            async_tag = " [后台异步慢通路]" if is_async else ""
            if str(w_path).lower().endswith(".engine") or "engine" in str(w_path).lower():
                stage_detail = f" [预处理: {pre_ms:.2f}ms | 纯GPU: {gpu_ms:.2f}ms | 后处理: {post_ms:.2f}ms]" if gpu_ms > 0 else ""
                backend_desc = f"TensorRT GPU Engine{stage_detail}{async_tag}"
            else:
                backend_desc = f"ONNXRuntime{async_tag}"
            w_disp = _format_weight_name(w_path)
            lines.append(f"  * {name:<22}: {cond} (权重: {w_disp}, 后端: {backend_desc})")
        lines.append("=" * 96)
        return "\n".join(lines)


# 全局单例 (便于跨模块轻量访问)
_GLOBAL_PROFILER = PipelineProfiler()

def get_global_profiler() -> PipelineProfiler:
    return _GLOBAL_PROFILER
