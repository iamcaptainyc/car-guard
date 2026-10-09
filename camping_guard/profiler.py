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


@dataclass
class ModelStats:
    """各模型调用聚合统计指标"""
    model_name: str
    weight_path: str = ""
    trigger_condition: str = ""
    call_count: int = 0
    durations_ms: List[float] = field(default_factory=list)
    gpu_durations_ms: List[float] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)

    def record_call(self, duration_ms: float, condition: str = "", gpu_ms: Optional[float] = None):
        self.call_count += 1
        self.durations_ms.append(duration_ms)
        if gpu_ms is not None and gpu_ms > 0:
            self.gpu_durations_ms.append(gpu_ms)
        if condition and condition not in self.conditions:
            self.conditions.append(condition)

    @property
    def avg_gpu_ms(self) -> float:
        return (sum(self.gpu_durations_ms) / len(self.gpu_durations_ms)) if self.gpu_durations_ms else 0.0

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
    def p95_ms(self) -> float:
        if not self.durations_ms:
            return 0.0
        return float(np.percentile(self.durations_ms, 95))

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
        self.profiler.record_model_call(
            model_name=self.model_name,
            duration_ms=duration_ms,
            condition=self.condition,
            success=success,
            input_shape=self.input_shape,
            gpu_ms=gpu_ms
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

        # 当前帧临时变量
        self._current_frame_idx: int = 0
        self._current_frame_start: float = 0.0
        self._current_frame_calls: List[ModelCallRecord] = []
        self._current_stage_durations: Dict[str, float] = {}
        self._is_profiling_frame: bool = False

        # 预设模型元数据 (便于展示与归类)
        self.model_metadata: Dict[str, Dict[str, str]] = {
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
            }
        }

    def register_model_metadata(self, model_name: str, weight_path: str = "", condition: str = ""):
        """注册或更新模型元信息"""
        if model_name not in self.model_stats:
            self.model_stats[model_name] = ModelStats(
                model_name=model_name,
                weight_path=weight_path,
                trigger_condition=condition
            )
        else:
            if weight_path:
                self.model_stats[model_name].weight_path = weight_path
            if condition:
                self.model_stats[model_name].trigger_condition = condition

    def start_frame(self, frame_idx: int):
        """开启一帧的 E2E 计时"""
        self._current_frame_idx = frame_idx
        self._current_frame_start = time.perf_counter()
        self._current_frame_calls = []
        self._current_stage_durations = {}
        self._is_profiling_frame = True

    def end_frame(self, threat_level: str = "L0_SAFE", target_count: int = 0) -> FrameMetrics:
        """结束一帧计时并返回当前帧指标快照"""
        t_end = time.perf_counter()
        e2e_ms = (t_end - self._current_frame_start) * 1000.0 if self._current_frame_start > 0 else 0.0

        metrics = FrameMetrics(
            frame_idx=self._current_frame_idx,
            start_time=self._current_frame_start,
            end_time=t_end,
            e2e_ms=e2e_ms,
            model_calls=list(self._current_frame_calls),
            stage_durations_ms=dict(self._current_stage_durations),
            threat_level=threat_level,
            target_count=target_count
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
        gpu_ms: Optional[float] = None
    ):
        """记录单次模型调用"""
        if model_name not in self.model_stats:
            meta = self.model_metadata.get(model_name, {})
            self.model_stats[model_name] = ModelStats(
                model_name=model_name,
                weight_path=meta.get("default_weight", ""),
                trigger_condition=condition or meta.get("condition", "")
            )

        self.model_stats[model_name].record_call(duration_ms, condition=condition, gpu_ms=gpu_ms)

        rec = ModelCallRecord(
            model_name=model_name,
            duration_ms=duration_ms,
            condition=condition,
            success=success,
            input_shape=input_shape
        )
        if self._is_profiling_frame:
            self._current_frame_calls.append(rec)

    def record_stage_duration(self, stage_name: str, duration_ms: float):
        """记录某业务阶段的耗时"""
        if self._is_profiling_frame:
            self._current_stage_durations[stage_name] = (
                self._current_stage_durations.get(stage_name, 0.0) + duration_ms
            )

    @property
    def total_frames(self) -> int:
        return len(self.frame_history)

    def get_summary(self) -> Dict[str, Any]:
        """计算全周期的聚合统计指标"""
        total_frames = len(self.frame_history)
        if total_frames == 0:
            return {
                "total_frames": 0,
                "e2e": {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "p95_ms": 0.0, "fps": 0.0},
                "models": {}
            }

        e2e_list = [f.e2e_ms for f in self.frame_history]
        total_e2e_ms = sum(e2e_list)
        avg_e2e = total_e2e_ms / total_frames
        min_e2e = min(e2e_list)
        max_e2e = max(e2e_list)
        p50_e2e = float(np.percentile(e2e_list, 50))
        p95_e2e = float(np.percentile(e2e_list, 95))
        avg_fps = (1000.0 / avg_e2e) if avg_e2e > 0 else 0.0

        model_summaries = {}
        for name, stats in self.model_stats.items():
            model_summaries[name] = {
                "model_name": name,
                "weight_path": stats.weight_path,
                "trigger_condition": stats.trigger_condition,
                "total_calls": stats.call_count,
                "calls_per_frame": (stats.call_count / total_frames) if total_frames > 0 else 0.0,
                "total_time_ms": stats.total_ms,
                "pct_of_e2e": ((stats.total_ms / total_e2e_ms) * 100.0) if total_e2e_ms > 0 else 0.0,
                "avg_duration_ms": stats.avg_ms,
                "min_duration_ms": stats.min_ms,
                "max_duration_ms": stats.max_ms,
                "p95_duration_ms": stats.p95_ms,
                "last_duration_ms": stats.last_ms
            }

        return {
            "total_frames": total_frames,
            "total_time_sec": total_e2e_ms / 1000.0,
            "e2e": {
                "avg_ms": avg_e2e,
                "min_ms": min_e2e,
                "max_ms": max_e2e,
                "p50_ms": p50_e2e,
                "p95_ms": p95_e2e,
                "fps": avg_fps
            },
            "models": model_summaries
        }

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
        lines.append(f" 分位数统计   : P50 (中位数): {e2e['p50_ms']:.2f} ms | P95: {e2e['p95_ms']:.2f} ms")
        lines.append(f" 理论等效帧率 : {e2e['fps']:.1f} FPS")
        lines.append("-" * 96)
        lines.append("【各模型调用频次与每次调用耗时统计 (Per-Model Profiling Breakdown)】")

        hdr = f"{'模型名称':<20} | {'总调用':<6} | {'纯GPU推理':<10} | {'前后处理':<10} | {'总计(ms)':<8} | {'P95(ms)':<8} | {'E2E占比':<8}"
        lines.append(hdr)
        lines.append("-" * 96)

        for name, m in models.items():
            rate_str = f"{m['calls_per_frame']:.2f}/f"
            pct_str = f"{m['pct_of_e2e']:.1f}%"
            is_trt = str(m['weight_path']).lower().endswith(".engine")
            display_name = f"{name} (TRT)" if is_trt else name
            stats_obj = self.model_stats.get(name)
            gpu_ms = stats_obj.avg_gpu_ms if stats_obj else 0.0
            if gpu_ms > 0:
                gpu_str = f"{gpu_ms:6.2f} ms"
                post_ms = max(0.0, m['avg_duration_ms'] - gpu_ms)
                post_str = f"{post_ms:6.2f} ms"
            else:
                gpu_str = "    -     "
                post_str = f"{m['avg_duration_ms']:6.2f} ms"

            lines.append(
                f"{display_name:<20} | {m['total_calls']:<6} | {gpu_str:<10} | {post_str:<10} | "
                f"{m['avg_duration_ms']:<8.2f} | {m['p95_duration_ms']:<8.2f} | {pct_str:<8}"
            )

        lines.append("-" * 96)
        lines.append("【模型触发机制与运行说明】")
        for name, m in models.items():
            cond = m['trigger_condition'] or "随场景目标动态触发"
            w_path = m['weight_path'] or "内置引擎"
            stats_obj = self.model_stats.get(name)
            gpu_ms = stats_obj.avg_gpu_ms if stats_obj else 0.0
            if str(w_path).lower().endswith(".engine"):
                gpu_detail = f" [纯GPU推理: {gpu_ms:.2f}ms]" if gpu_ms > 0 else ""
                backend_desc = f"TensorRT GPU Engine{gpu_detail}"
            else:
                backend_desc = "ONNXRuntime"
            lines.append(f"  * {name:<20}: {cond} (权重: {os.path.basename(w_path)}, 后端: {backend_desc})")
        lines.append("=" * 96)
        return "\n".join(lines)


# 全局单例 (便于跨模块轻量访问)
_GLOBAL_PROFILER = PipelineProfiler()

def get_global_profiler() -> PipelineProfiler:
    return _GLOBAL_PROFILER
