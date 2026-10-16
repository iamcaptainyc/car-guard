#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_cli_args_integrity.py
=============================================================================
全面自动化测试所有 CLI 入口脚本的参数完整性与解析正确性:
1. AST 静态全量扫描: 保证每个脚本中凡是被 args.<attr> 或 getattr(args, '<attr>')
   引用的属性，必须 100% 在 ArgumentParser.add_argument 中显式定义，严防 AttributeError。
2. 真实 CLI 组合调用测试: 模拟用户各种实战传入的参数组合（如用户报错触发的命令组合），
   确保参数解析、默认值、类型转换均无异常。
=============================================================================
"""

import ast
import glob
from pathlib import Path
import pytest
import sys
import textwrap

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_cli_script_paths():
    """获取代码库中所有具有 CLI 参数解析能力的 Python 文件"""
    candidates = (
        list(REPO_ROOT.glob("*.py")) +
        list((REPO_ROOT / "scripts").glob("*.py"))
    )
    cli_files = []
    for p in sorted(candidates):
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()
        if "add_argument" in content and "parse_args" in content:
            cli_files.append(p)
    return cli_files


@pytest.mark.parametrize("script_path", get_cli_script_paths(), ids=lambda p: p.name)
def test_ast_args_integrity(script_path):
    """
    AST 级别审计: 脚本内所有引用的 args 属性必须在 add_argument 中声明
    """
    with open(script_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(script_path))

    added_args = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                dest = None
                flags = []
                for kw in node.keywords:
                    if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                        dest = kw.value.value
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        flags.append(arg.value)
                if dest:
                    added_args.add(dest)
                else:
                    for flg in flags:
                        if flg.startswith("--"):
                            added_args.add(flg[2:].replace("-", "_"))
                        elif not flg.startswith("-"):
                            # positional argument
                            added_args.add(flg.replace("-", "_"))

    accessed_args = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "args":
            accessed_args.add(node.attr)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and node.args[0].id == "args":
                if isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                    accessed_args.add(node.args[1].value)

    missing = accessed_args - added_args
    assert not missing, (
        f"在脚本 {script_path.name} 中发现未在 ArgumentParser 中声明却被引用的参数属性: {missing}！"
        f"请在 parser.add_argument 中补齐该参数定义。"
    )


def extract_parser(script_path: Path, extra_globals=None):
    """从脚本中稳健提取其 ArgumentParser 实例"""
    import argparse
    with open(script_path, "r", encoding="utf-8") as f:
        src = f.read()

    p_start = src.find("parser = argparse.ArgumentParser(")
    p_end = src.find("args = parser.parse_args()")
    assert p_start != -1 and p_end != -1, f"未能定位 {script_path.name} 中的 parser 定义"

    line_start = src.rfind("\n", 0, p_start)
    if line_start == -1:
        line_start = 0
    parser_code = src[line_start:p_end]

    local_ns = {"argparse": argparse}
    if extra_globals:
        local_ns.update(extra_globals)
    exec(textwrap.dedent(parser_code), local_ns)
    return local_ns["parser"]


def test_run_jetson_guard_user_cli_invocation():
    """
    精确重现并测试用户报错的命令行参数组合:
    python run_jetson_guard.py --video 开门碰撞1.mp4 --model-dir /models --camera-id 0 --realtime --detector models/yolo/yolo11s_dla0.engine
    附加: --max-frames, --no-fire, --vlm-throttle, --dla-core
    """
    parser = extract_parser(
        REPO_ROOT / "run_jetson_guard.py",
        extra_globals={"get_jetson_info": lambda: {}}
    )

    test_cli_args = [
        "--video", "开门碰撞1.mp4",
        "--model-dir", "/home/nvidia/qwen35-orin/models/Qwen3.5-0.8B",
        "--camera-id", "0",
        "--realtime",
        "--detector", "models/yolo/yolo11s_dla0.engine",
        "--dla-core", "0",
        "--max-frames", "10",
        "--no-fire",
        "--vlm-throttle",
        "--vlm-throttle-fps", "4.0",
        "--stats-json", "logs/test_perf.json"
    ]

    parsed = parser.parse_args(test_cli_args)

    # 断言关键参数字段全部健在且解析正确
    assert parsed.video == "开门碰撞1.mp4"
    assert parsed.camera_id == 0
    assert parsed.realtime is True
    assert parsed.detector == "models/yolo/yolo11s_dla0.engine"
    assert parsed.dla_core == 0
    assert parsed.max_frames == 10
    assert parsed.no_fire is True
    assert parsed.vlm_throttle is True
    assert parsed.vlm_throttle_fps == 4.0
    assert parsed.stats_json == "logs/test_perf.json"


def test_test_video_pipeline_cli_invocation():
    """
    测试 test_video_pipeline.py 的参数完整性与解析
    """
    parser = extract_parser(
        REPO_ROOT / "test_video_pipeline.py",
        extra_globals={"parse_camera_id": lambda x: int(x)}
    )

    test_args = [
        "--video", "test.mp4",
        "--dla-core", "1",
        "--vlm-throttle",
        "--vlm-throttle-fps", "6.0",
        "--max-frames", "50",
        "--no-fire"
    ]
    parsed = parser.parse_args(test_args)
    assert parsed.video == "test.mp4"
    assert parsed.dla_core == 1
    assert parsed.vlm_throttle is True
    assert parsed.vlm_throttle_fps == 6.0
    assert parsed.max_frames == 50
    assert parsed.no_fire is True


def test_batch_convert_trt_cli_invocation():
    """
    测试 batch_convert_trt.py 的参数完整性与解析 (包含 DLA 硬件卸载参数)
    """
    parser = extract_parser(REPO_ROOT / "batch_convert_trt.py")

    test_args = [
        "models/yolo",
        "--precision", "fp16",
        "--dla-core", "0",
        "--no-gpu-fallback",
        "--builder-level", "3"
    ]
    parsed = parser.parse_args(test_args)
    assert parsed.models_dir == "models/yolo"
    assert parsed.precision == "fp16"
    assert parsed.dla_core == 0
    assert parsed.no_gpu_fallback is True
    assert parsed.builder_level == 3


def test_test_contention_schemes_cli_invocation():
    """
    测试 scripts/test_contention_schemes.py 的参数解析
    """
    parser = extract_parser(REPO_ROOT / "scripts" / "test_contention_schemes.py")

    test_args = [
        "--model-dir", "/home/nvidia/models/Qwen3.5-0.8B",
        "--full-fps", "25.0",
        "--yield-fps", "4.0",
        "--max-tokens", "64",
        "--so-path", "/usr/local/lib/libvlm_engine.so"
    ]
    parsed = parser.parse_args(test_args)
    assert parsed.full_fps == 25.0
    assert parsed.yield_fps == 4.0
    assert parsed.max_tokens == 64
    assert parsed.so_path == "/usr/local/lib/libvlm_engine.so"
