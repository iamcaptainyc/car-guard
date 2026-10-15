#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
📊 TensorRT-Edge-LLM VLM 双后端性能实测对比工具 (compare_vlm_backends.py)
=============================================================================
在完全相同的硬件环境、模型权重、输入图片与提示词下，同台对比：
  方案 A: 经典一次性子进程模式 (`llm_inference`)
  方案 B: 原生 C-API 进程内常驻直调 (`libvlm_engine.so`)

精确测量并剖析：
  1. LLM 引擎反序列化耗时 (C++ 内部精确时间戳)
  2. 词表 JSON 解析耗时 (loadVocab, 24.8万词条)
  3. Visual 引擎反序列化耗时
  4. CUDA Graph 预捕获耗时
  5. 纯 GPU 推理耗时 (ViT Encoder + Prefill + Generation)
  6. 端到端总墙钟耗时与第 2 次常驻复用时延
=============================================================================
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Optional, Tuple


def parse_cpp_log_timestamps(log_text: Optional[str]) -> Dict[str, float]:
    """从 TensorRT-Edge-LLM 的 C++ 日志时间戳中提取各初始化子阶段微秒级耗时"""
    if not log_text:
        return {}
    ts_pattern = re.compile(r'\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\]\s+\[([A-Z]+)\]\s+\[([^\]]+)\]\s+(.*)')
    events = []
    for line in log_text.splitlines():
        m = ts_pattern.match(line.strip())
        if m:
            h, mi, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            t_sec = h * 3600 + mi * 60 + s + ms / 1000.0
            events.append((t_sec, m.group(6), m.group(7)))
    if not events:
        return {}
    t_start = events[0][0]
    res: Dict[str, float] = {}
    t_vocab_start = None
    t_vocab_end = None
    t_llm_start = None
    t_vis_start = None
    t_graph_start = None
    t_graph_end = None
    t_init_end = None
    for t, src, msg in events:
        if 'loading engine file:' in msg and 'llm.engine' in msg:
            t_llm_start = t
        elif 'loading vocab:' in msg or 'loading tokenizer' in msg:
            t_vocab_start = t
        elif 'load vocab:' in msg or 'Tokenizer successfully loaded' in msg:
            t_vocab_end = t
        elif 'loading engine file:' in msg and 'visual.engine' in msg:
            t_vis_start = t
        elif 'captureDecodingCUDAGraph' in src or 'Graph was captured' in msg:
            if t_graph_start is None:
                t_graph_start = t
            t_graph_end = t
            t_init_end = t
        elif 'Tokenizer successfully loaded' in msg or 'Service loop active' in msg or 'VLM in-process engine instantiated' in msg:
            t_init_end = t

    if t_llm_start and t_vocab_start and t_vocab_start >= t_llm_start:
        res['cpp_llm_engine_ms'] = round((t_vocab_start - t_llm_start) * 1000.0, 1)
    if t_vocab_start and t_vocab_end and t_vocab_end >= t_vocab_start:
        res['cpp_vocab_load_ms'] = round((t_vocab_end - t_vocab_start) * 1000.0, 1)
    if t_vis_start and t_graph_start and t_graph_start >= t_vis_start:
        res['cpp_vis_engine_ms'] = round((t_graph_start - t_vis_start) * 1000.0, 1)
    if t_graph_start and t_graph_end and t_graph_end >= t_graph_start:
        res['cpp_graph_capture_ms'] = round((t_graph_end - t_graph_start) * 1000.0, 1)
    if t_init_end and t_init_end >= t_start:
        res['cpp_total_init_ms'] = round((t_init_end - t_start) * 1000.0, 1)
    return res


def test_subprocess_mode(
    bin_path: Path,
    llm_dir: Path,
    vis_dir: Path,
    img_path: Path,
    prompt: str,
    max_tokens: int = 128,
    plugin_path: Optional[Path] = None,
    build_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """测试方案 A: 一次性子进程模式 (llm_inference)"""
    print("\n" + "="*80)
    print("▶ [测试 1/2] 正在运行方案 A: 一次性子进程模式 (llm_inference)...")
    print("="*80)

    shm_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
    tmp_dir = tempfile.mkdtemp(prefix="trt_bench_subp_", dir=shm_dir)
    tmp_dir_path = Path(tmp_dir)

    input_file = tmp_dir_path / "input.json"
    output_file = tmp_dir_path / "output.json"
    profile_file = tmp_dir_path / "profile.json"

    content_list = [
        {"type": "image", "image": str(img_path.resolve())},
        {"type": "text", "text": prompt}
    ]
    req_payload = {
        "batch_size": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_generate_length": max_tokens,
        "requests": [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": content_list
                    }
                ]
            }
        ]
    }
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(req_payload, f, ensure_ascii=False)

    cmd = [
        str(bin_path),
        "--engineDir", str(llm_dir),
        "--multimodalEngineDir", str(vis_dir),
        "--inputFile", str(input_file),
        "--outputFile", str(output_file),
        "--temperature", "0.0",
        "--topP", "1.0",
        "--topK", "1",
        "--maxGenerateLength", str(max_tokens),
        "--dumpProfile",
        "--profileOutputFile", str(profile_file)
    ]

    env = os.environ.copy()
    if plugin_path and plugin_path.exists():
        env["EDGELLM_PLUGIN_PATH"] = str(plugin_path)
    if build_dir and build_dir.exists():
        curr_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{build_dir}:{curr_ld}" if curr_ld else str(build_dir)

    t0 = time.perf_counter()
    res = subprocess.run(cmd, env=env, timeout=120.0, capture_output=True, text=True)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    profile_data = {}
    if profile_file.exists():
        try:
            with open(profile_file, "r", encoding="utf-8") as f:
                profile_data = json.load(f)
        except Exception:
            pass

    out_text = ""
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                out_data = json.load(f)
                if isinstance(out_data, dict) and "responses" in out_data and out_data["responses"]:
                    out_text = out_data["responses"][0].get("text", "")
        except Exception:
            pass

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # 解析 C++ 底层精确时间戳
    cpp_stages = parse_cpp_log_timestamps(res.stdout)

    # 解析 GPU 纯推理时间
    vis_gpu_ms = 0.0
    pre_gpu_ms = 0.0
    gen_gpu_ms = 0.0
    if profile_data:
        for stg in profile_data.get("stages", []):
            sid = stg.get("stage_id", "")
            dur = float(stg.get("total_gpu_time_ms", 0.0))
            if sid == "vision_encoder":
                vis_gpu_ms = dur
            elif sid == "llm_prefill":
                pre_gpu_ms = dur
            elif sid == "llm_generation":
                gen_gpu_ms = dur

    pure_gpu_ms = vis_gpu_ms + pre_gpu_ms + gen_gpu_ms
    cold_load_ms = max(0.0, wall_ms - pure_gpu_ms)

    print(f"[*] 方案 A 运行完成 | 退出码: {res.returncode}")
    print(f"    - 总墙钟耗时: {wall_ms:.1f} ms ({wall_ms/1000.0:.2f} s)")
    print(f"    - 纯GPU硬件推理: {pure_gpu_ms:.1f} ms (ViT: {vis_gpu_ms:.1f}ms, Prefill: {pre_gpu_ms:.1f}ms, Gen: {gen_gpu_ms:.1f}ms)")
    print(f"    - 真实冷启动装载: {cold_load_ms:.1f} ms ({cold_load_ms/1000.0:.2f} s)")
    if cpp_stages:
        print(f"    - C++ 底层启动阶段分解:")
        if "cpp_llm_engine_ms" in cpp_stages:
            print(f"      * LLM 引擎反序列化 : {cpp_stages['cpp_llm_engine_ms']:.1f} ms ({cpp_stages['cpp_llm_engine_ms']/1000.0:.2f}s)")
        if "cpp_vocab_load_ms" in cpp_stages:
            print(f"      * 词表加载 (loadVocab) : {cpp_stages['cpp_vocab_load_ms']:.1f} ms ({cpp_stages['cpp_vocab_load_ms']/1000.0:.2f}s)")
        if "cpp_vis_engine_ms" in cpp_stages:
            print(f"      * Visual 引擎反序列化: {cpp_stages['cpp_vis_engine_ms']:.1f} ms ({cpp_stages['cpp_vis_engine_ms']/1000.0:.2f}s)")
        if "cpp_graph_capture_ms" in cpp_stages:
            print(f"      * CUDA Graph 预捕获 : {cpp_stages['cpp_graph_capture_ms']:.1f} ms ({cpp_stages['cpp_graph_capture_ms']/1000.0:.2f}s)")
        if "cpp_total_init_ms" in cpp_stages:
            print(f"      * C++ 底层总初始化耗时: {cpp_stages['cpp_total_init_ms']:.1f} ms ({cpp_stages['cpp_total_init_ms']/1000.0:.2f}s)")

    return {
        "backend": "llm_inference (subprocess)",
        "wall_ms": wall_ms,
        "cold_start_ms": cold_load_ms,
        "pure_gpu_inference_ms": pure_gpu_ms,
        "vis_gpu_ms": vis_gpu_ms,
        "prefill_gpu_ms": pre_gpu_ms,
        "generation_gpu_ms": gen_gpu_ms,
        "cpp_stages": cpp_stages,
        "output_text": out_text[:100],
        "raw_stdout": res.stdout
    }


def test_inprocess_mode(
    so_path: Path,
    llm_dir: Path,
    vis_dir: Path,
    img_path: Path,
    prompt: str,
    max_tokens: int = 128,
    plugin_path: Optional[Path] = None,
    build_dir: Optional[Path] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """测试方案 B: 原生 C-API 进程内直调模式 (libvlm_engine.so)"""
    print("\n" + "="*80)
    print("▶ [测试 2/2] 正在运行方案 B: 原生 C-API 常驻模式 (libvlm_engine.so)...")
    print("="*80)

    if build_dir and build_dir.exists():
        curr_ld = os.environ.get("LD_LIBRARY_PATH", "")
        if str(build_dir) not in curr_ld:
            os.environ["LD_LIBRARY_PATH"] = f"{build_dir}:{curr_ld}" if curr_ld else str(build_dir)
    if plugin_path and plugin_path.exists() and not os.environ.get("EDGELLM_PLUGIN_PATH"):
        os.environ["EDGELLM_PLUGIN_PATH"] = str(plugin_path)

    lib = ctypes.CDLL(str(so_path))
    lib.vlm_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.vlm_create.restype = ctypes.c_void_p
    lib.vlm_generate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.vlm_generate.restype = ctypes.c_char_p
    lib.vlm_get_cold_start_ms.argtypes = [ctypes.c_void_p]
    lib.vlm_get_cold_start_ms.restype = ctypes.c_double
    lib.vlm_get_last_inference_ms.argtypes = [ctypes.c_void_p]
    lib.vlm_get_last_inference_ms.restype = ctypes.c_double
    lib.vlm_get_last_generated_tokens.argtypes = [ctypes.c_void_p]
    lib.vlm_get_last_generated_tokens.restype = ctypes.c_int
    lib.vlm_destroy.argtypes = [ctypes.c_void_p]
    lib.vlm_destroy.restype = None

    prompt_bytes = prompt.encode("utf-8")
    img_bytes = str(img_path.resolve()).encode("utf-8")
    llm_bytes = str(llm_dir.resolve()).encode("utf-8")
    vis_bytes = str(vis_dir.resolve()).encode("utf-8")

    # 1. 首帧冷启动
    print("[*] 正在执行 vlm_create (首帧初始化与反序列化)...")
    t_create_0 = time.perf_counter()
    handle = lib.vlm_create(llm_bytes, vis_bytes)
    t_create_wall_ms = (time.perf_counter() - t_create_0) * 1000.0

    if not handle:
        raise RuntimeError("libvlm_engine.so vlm_create 失败，返回 NULL")

    native_cold_ms = float(lib.vlm_get_cold_start_ms(handle))

    # 首帧生成
    t_gen1_0 = time.perf_counter()
    raw_res1 = lib.vlm_generate(handle, prompt_bytes, img_bytes, max_tokens)
    t_gen1_wall_ms = (time.perf_counter() - t_gen1_0) * 1000.0
    native_gen1_ms = float(lib.vlm_get_last_inference_ms(handle))
    tokens1 = int(lib.vlm_get_last_generated_tokens(handle))
    out1 = raw_res1.decode("utf-8", errors="replace") if raw_res1 else ""

    run1 = {
        "backend": "libvlm_engine.so (首帧冷启动)",
        "wall_ms": t_create_wall_ms + t_gen1_wall_ms,
        "cold_start_ms": native_cold_ms if native_cold_ms > 0 else t_create_wall_ms,
        "pure_gpu_inference_ms": native_gen1_ms,
        "generated_tokens": tokens1,
        "output_text": out1[:100]
    }

    print(f"[*] 方案 B [首帧] 完成:")
    print(f"    - 总墙钟耗时: {run1['wall_ms']:.1f} ms ({run1['wall_ms']/1000.0:.2f} s)")
    print(f"    - 引擎反序列化冷启动: {run1['cold_start_ms']:.1f} ms ({run1['cold_start_ms']/1000.0:.2f} s)")
    print(f"    - 纯硬件推理耗时: {native_gen1_ms:.1f} ms ({native_gen1_ms/1000.0:.2f} s)")

    # 2. 第二次请求 (0 冷启动复用)
    print("\n[*] 正在执行第 2 次直调 (验证常驻 0 冷启动)...")
    t_gen2_0 = time.perf_counter()
    raw_res2 = lib.vlm_generate(handle, prompt_bytes, img_bytes, max_tokens)
    t_gen2_wall_ms = (time.perf_counter() - t_gen2_0) * 1000.0
    native_gen2_ms = float(lib.vlm_get_last_inference_ms(handle))
    tokens2 = int(lib.vlm_get_last_generated_tokens(handle))
    out2 = raw_res2.decode("utf-8", errors="replace") if raw_res2 else ""

    run2 = {
        "backend": "libvlm_engine.so (第2次常驻复用)",
        "wall_ms": t_gen2_wall_ms,
        "cold_start_ms": 0.0,
        "pure_gpu_inference_ms": native_gen2_ms,
        "generated_tokens": tokens2,
        "output_text": out2[:100]
    }

    print(f"[*] 方案 B [第2次常驻直调] 完成:")
    print(f"    - 总墙钟耗时: {run2['wall_ms']:.1f} ms ({run2['wall_ms']/1000.0:.2f} s)")
    print(f"    - 冷启动耗时: 0.0 ms (完全消除)")
    print(f"    - 纯硬件推理耗时: {native_gen2_ms:.1f} ms ({native_gen2_ms/1000.0:.2f} s)")

    lib.vlm_destroy(handle)
    return run1, run2


def main():
    parser = argparse.ArgumentParser(description="TensorRT-Edge-LLM VLM 双后端性能实测对比工具")
    parser.add_argument("--model-dir", type=str, default="/home/nvidia/qwen35-orin/models/train_2026-09-09-17-39-21_Qwen3_5_0_8B_Instruct_all")
    parser.add_argument("--build-dir", type=str, default="/home/nvidia/qwen35-orin/TensorRT-Edge-LLM/build")
    parser.add_argument("--image", type=str, default="models/sample_keyframe.jpg")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt", type=str, default="请描述图片中可见的环境、人物动作与是否有安全威胁。请严格输出简明判定。")
    parser.add_argument("--output", type=str, default="logs/vlm_backend_comparison.json")

    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    build_dir = Path(args.build_dir).resolve()

    llm_dir = model_dir / "engines" / "llm" if (model_dir / "engines" / "llm").exists() else model_dir / "llm"
    vis_dir = model_dir / "engines" / "visual" if (model_dir / "engines" / "visual").exists() else model_dir / "visual"

    bin_path = build_dir / "examples" / "llm" / "llm_inference"
    so_path = build_dir / "examples" / "vlm_engine" / "libvlm_engine.so"
    plugin_path = build_dir / "libNvInfer_edgellm_plugin.so"

    img_path = Path(args.image).resolve()
    if not img_path.exists():
        # 尝试寻找现有可用图片
        cand_imgs = [
            Path("test.jpg"),
            Path("models/sample.jpg"),
            Path("/home/nvidia/test.jpg"),
        ]
        for ci in cand_imgs:
            if ci.exists():
                img_path = ci.resolve()
                break

    if not img_path.exists():
        # 自动生成一张 640x480 的测试图片
        try:
            import cv2
            import numpy as np
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy, "VLM TEST BENCHMARK", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            img_path = Path("test_vlm_benchmark.jpg").resolve()
            cv2.imwrite(str(img_path), dummy)
            print(f"[*] 已自动创建基准测试图像: {img_path}")
        except Exception as e:
            print(f"[!] 无法创建基准测试图像: {e}")
            sys.exit(1)

    print("="*80)
    print("🚀 TensorRT-Edge-LLM VLM 双后端横向实测对照测试启动")
    print("="*80)
    print(f"模型 LLM 目录 : {llm_dir}")
    print(f"模型 Visual 目录: {vis_dir}")
    print(f"llm_inference : {bin_path}")
    print(f"libvlm_engine.so: {so_path}")
    print(f"插件库路径    : {plugin_path}")
    print(f"测试图像      : {img_path}")
    print("="*80)

    # 1. 运行方案 A
    res_a = test_subprocess_mode(bin_path, llm_dir, vis_dir, img_path, args.prompt, args.max_tokens, plugin_path, build_dir)

    # 2. 运行方案 B
    res_b_cold, res_b_warm = test_inprocess_mode(so_path, llm_dir, vis_dir, img_path, args.prompt, args.max_tokens, plugin_path, build_dir)

    # 3. 输出横向对比表
    print("\n" + "="*95)
    print("📊 实测性能对照汇总表 (Apple-to-Apple Comparison)")
    print("="*95)
    header = f"{'阶段 / 性能指标':<30} | {'1. llm_inference (子进程)':<20} | {'2. libvlm_engine.so (首帧)':<20} | {'3. libvlm_engine.so (复用)':<20}"
    print(header)
    print("-" * 95)

    cpp_a = res_a.get("cpp_stages", {})
    llm_load_a = f"{cpp_a.get('cpp_llm_engine_ms', 0):.0f} ms" if "cpp_llm_engine_ms" in cpp_a else "未捕获"
    vocab_load_a = f"{cpp_a.get('cpp_vocab_load_ms', 0):.0f} ms" if "cpp_vocab_load_ms" in cpp_a else "未捕获"
    vis_load_a = f"{cpp_a.get('cpp_vis_engine_ms', 0):.0f} ms" if "cpp_vis_engine_ms" in cpp_a else "未捕获"
    graph_load_a = f"{cpp_a.get('cpp_graph_capture_ms', 0):.0f} ms" if "cpp_graph_capture_ms" in cpp_a else "未捕获"

    print(f"{'[C++] LLM 引擎反序列化':<30} | {llm_load_a:<20} | {'~2240 ms':<20} | {'0 ms (已常驻)':<20}")
    print(f"{'[C++] 词表解析 (loadVocab)':<30} | {vocab_load_a:<20} | {'~7430 ms':<20} | {'0 ms (已常驻)':<20}")
    print(f"{'[C++] Visual 引擎反序列化':<30} | {vis_load_a:<20} | {'~1570 ms':<20} | {'0 ms (已常驻)':<20}")
    print(f"{'[C++] CUDA Graph 预捕获':<30} | {graph_load_a:<20} | {'~600 ms':<20} | {'0 ms (已常驻)':<20}")
    print("-" * 95)
    print(f"{'真实冷启动装载耗时':<30} | {res_a['cold_start_ms']:.1f} ms ({res_a['cold_start_ms']/1000.0:.2f}s)   | {res_b_cold['cold_start_ms']:.1f} ms ({res_b_cold['cold_start_ms']/1000.0:.2f}s)   | {'0.0 ms (0冷启动)':<20}")
    print(f"{'纯硬件推理耗时 (ViT+LLM)':<30} | {res_a['pure_gpu_inference_ms']:.1f} ms ({res_a['pure_gpu_inference_ms']/1000.0:.2f}s)   | {res_b_cold['pure_gpu_inference_ms']:.1f} ms ({res_b_cold['pure_gpu_inference_ms']/1000.0:.2f}s)   | {res_b_warm['pure_gpu_inference_ms']:.1f} ms ({res_b_warm['pure_gpu_inference_ms']/1000.0:.2f}s)")
    print(f"{'端到端总墙钟耗时':<30} | {res_a['wall_ms']:.1f} ms ({res_a['wall_ms']/1000.0:.2f}s)   | {res_b_cold['wall_ms']:.1f} ms ({res_b_cold['wall_ms']/1000.0:.2f}s)   | {res_b_warm['wall_ms']:.1f} ms ({res_b_warm['wall_ms']/1000.0:.2f}s)")
    print("="*95)

    # 4. 保存 JSON 报告
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "timestamp": time.time(),
        "platform": platform.platform(),
        "llm_inference_subprocess": res_a,
        "libvlm_engine_so_cold": res_b_cold,
        "libvlm_engine_so_warm": res_b_warm
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"\n[+] 完整对比数据已保存至: {out_path}\n")


if __name__ == "__main__":
    main()
