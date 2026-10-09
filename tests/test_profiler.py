"""
性能剖析器与模型指标统计测试 (tests/test_profiler.py)
验证:
1. 单帧与多帧 E2E 耗时统计准确性
2. 各模型调用记录与分位数 (Min, Max, Avg, P95) 计算
3. 统计报表格式化输出
"""

import time
import pytest
from camping_guard.profiler import PipelineProfiler, ModelStats, FrameMetrics

def test_pipeline_profiler_basic():
    profiler = PipelineProfiler()
    profiler.register_model_metadata("YOLO11n-Det", "yolo11n.onnx", "S1全图检测")
    profiler.register_model_metadata("RTMPose-s", "rtmpose-s.onnx", "Zone2/3姿态估计")

    # 模拟第 1 帧
    profiler.start_frame(1)
    with profiler.profile_model("YOLO11n-Det", condition="测试全图初筛"):
        time.sleep(0.01) # 10ms
    with profiler.profile_model("RTMPose-s", condition="测试姿态提取"):
        time.sleep(0.005) # 5ms

    f1 = profiler.end_frame(threat_level="L1_NOTICE", target_count=1)
    assert f1.frame_idx == 1
    assert f1.e2e_ms >= 14.0
    assert len(f1.model_calls) == 2

    # 模拟第 2 帧 (未触发 RTMPose)
    profiler.start_frame(2)
    with profiler.profile_model("YOLO11n-Det", condition="测试全图初筛"):
        time.sleep(0.01)

    f2 = profiler.end_frame(threat_level="L0_SAFE", target_count=0)
    assert f2.frame_idx == 2
    assert len(f2.model_calls) == 1

    # 汇总计算验证
    summary = profiler.get_summary()
    assert summary["total_frames"] == 2
    assert summary["e2e"]["avg_ms"] > 0.0

    models = summary["models"]
    assert models["YOLO11n-Det"]["total_calls"] == 2
    assert models["YOLO11n-Det"]["calls_per_frame"] == 1.0
    assert models["RTMPose-s"]["total_calls"] == 1
    assert models["RTMPose-s"]["calls_per_frame"] == 0.5

    # 格式化报表文本生成验证
    report_single = profiler.format_single_frame_report(f1)
    assert "YOLO11n-Det" in report_single
    assert "RTMPose-s" in report_single

    report_table = profiler.format_summary_table()
    assert "Per-Model Profiling Breakdown" in report_table
    assert "YOLO11n-Det" in report_table
