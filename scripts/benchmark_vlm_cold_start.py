"""Measure the existing VLM C API on Jetson, without the guard pipeline.

Each process creates one engine and sends several identical requests. New
processes measure process-cold initialization; later requests measure reuse.
OS file caches are deliberately left alone. Only the Python standard library
is required. Native logs are kept beside the resulting JSON.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--so", required=True, type=Path)
    parser.add_argument("--engine-dir", required=True, type=Path)
    parser.add_argument("--visual-dir", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path,
                        help="Use a prepared keyframe (guard uses max dimension 640).")
    parser.add_argument("--prompt-file", type=Path,
                        help="UTF-8 prompt; use the original prompt to compare inference.")
    parser.add_argument("--max-tokens", type=positive_int, default=128)
    parser.add_argument("--process-runs", type=positive_int, default=5)
    parser.add_argument("--requests-per-process", type=positive_int, default=6)
    parser.add_argument("--timeout-sec", type=float, default=300,
                        help="Timeout for an entire child process, including all requests.")
    parser.add_argument("--output", type=Path,
                        default=Path("logs/vlm_cold_start_benchmark.json"))
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timeout_sec <= 0:
        parser.error("--timeout-sec must be positive")
    for name in ("so", "image", "prompt_file"):
        path = getattr(args, name)
        if path is not None:
            path = path.resolve()
            if not path.is_file():
                parser.error(f"--{name.replace('_', '-')}: file does not exist: {path}")
            setattr(args, name, path)
    for name in ("engine_dir", "visual_dir"):
        path = getattr(args, name).resolve()
        if not path.is_dir() or not (path / "config.json").is_file():
            parser.error(f"--{name.replace('_', '-')}: needs engine directory with config.json")
        setattr(args, name, path)
    return args


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def worker(args):
    prompt = (args.prompt_file.read_text(encoding="utf-8-sig") if args.prompt_file
              else "请描述图片中可见的动物和人物，并说明是否有明显危险。")
    prompt_bytes = prompt.encode("utf-8")
    image_bytes = str(args.image).encode("utf-8")
    engine_bytes = str(args.engine_dir).encode("utf-8")
    visual_bytes = str(args.visual_dir).encode("utf-8")
    start = time.perf_counter()
    lib = ctypes.CDLL(str(args.so))
    dlopen_ms = (time.perf_counter() - start) * 1000
    lib.vlm_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.vlm_create.restype = ctypes.c_void_p
    lib.vlm_generate.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                ctypes.c_char_p, ctypes.c_int]
    lib.vlm_generate.restype = ctypes.c_char_p
    for name in ("vlm_get_cold_start_ms", "vlm_get_last_inference_ms"):
        func = getattr(lib, name)
        func.argtypes = [ctypes.c_void_p]
        func.restype = ctypes.c_double
    lib.vlm_get_last_generated_tokens.argtypes = [ctypes.c_void_p]
    lib.vlm_get_last_generated_tokens.restype = ctypes.c_int
    lib.vlm_destroy.argtypes = [ctypes.c_void_p]
    lib.vlm_destroy.restype = None

    create_start = time.perf_counter()
    handle = lib.vlm_create(engine_bytes, visual_bytes)
    create_ms = (time.perf_counter() - create_start) * 1000
    if not handle:
        raise RuntimeError("vlm_create returned NULL; inspect the native log")
    ready_ms = (time.perf_counter() - start) * 1000
    try:
        native_load_ms = float(lib.vlm_get_cold_start_ms(handle))
        result = {
            "pid": os.getpid(),
            "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "dlopen_wall_ms": round(dlopen_ms, 3),
            "create_wall_ms": round(create_ms, 3),
            "native_reported_load_ms": round(native_load_ms, 3),
            "create_outside_native_load_ms": round(create_ms - native_load_ms, 3),
            "library_load_to_ready_wall_ms": round(ready_ms, 3),
            "requests": [],
        }
        for index in range(args.requests_per_process):
            request_start = time.perf_counter()
            raw = lib.vlm_generate(handle, prompt_bytes, image_bytes, args.max_tokens)
            generate_ms = (time.perf_counter() - request_start) * 1000
            native_ms = float(lib.vlm_get_last_inference_ms(handle))
            tokens = int(lib.vlm_get_last_generated_tokens(handle))
            text = raw.decode("utf-8", errors="replace") if raw else ""
            if not text or tokens <= 0:
                raise RuntimeError(f"request {index + 1} failed or returned no tokens")
            result["requests"].append({
                "request_number": index + 1,
                "cold_start_charged": index == 0,
                "generate_wall_ms": round(generate_ms, 3),
                "native_request_wall_ms": round(native_ms, 3),
                "generated_tokens": tokens,
                "tokens_per_native_request_second": (
                    round(tokens * 1000 / native_ms, 3) if native_ms > 0 else None),
                "output_text": text,
            })
            if index == 0:
                result["library_load_to_first_result_wall_ms"] = round(
                    (time.perf_counter() - start) * 1000, 3)
        write_json(args.worker_output, result)
    finally:
        lib.vlm_destroy(handle)


def measure(args):
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "platform": platform.platform(), "python": sys.version,
        "cpu_count": os.cpu_count(),
        "config": {name: (str(getattr(args, name)) if getattr(args, name) is not None else None) for name in
                   ("so", "engine_dir", "visual_dir", "image", "prompt_file")},
        "max_tokens": args.max_tokens,
        "environment": {name: os.environ.get(name) for name in
                        ("EDGELLM_PLUGIN_PATH", "LD_LIBRARY_PATH")},
        "notes": [
            "New process per run; OS page cache is not cleared.",
            "No guard models, image resize, JPEG encode or event queue in this benchmark.",
            "Native load timer follows the deployed library's implementation.",
            "In the current vlm_engine.cpp it covers the runtime constructor only.",
            "create_outside_native_load_ms includes plugin/CUDA setup and graph capture;"
            " it is not a graph-only measurement.",
            "native_request_wall_ms is handleRequest wall time, not generation GPU time.",
            "library_load_to_first_result_wall_ms excludes Python interpreter startup.",
        ],
        "runs": [], "failures": [],
    }
    with tempfile.TemporaryDirectory(prefix="vlm_bench_", dir=output.parent) as tmp:
        for index in range(args.process_runs):
            child_output = Path(tmp) / f"run_{index + 1}.json"
            log_path = output.with_name(f"{output.stem}.run_{index + 1}.log")
            cmd = [sys.executable, str(Path(__file__).resolve())]
            for name in ("so", "engine_dir", "visual_dir", "image", "prompt_file"):
                value = getattr(args, name)
                if value is not None:
                    cmd.extend([f"--{name.replace('_', '-')}", str(value)])
            cmd.extend(["--max-tokens", str(args.max_tokens),
                        "--requests-per-process", str(args.requests_per_process),
                        "--worker-output", str(child_output)])
            print(f"Process {index + 1}/{args.process_runs}: log={log_path}", flush=True)
            start = time.perf_counter()
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=args.timeout_sec)
                wall_ms = (time.perf_counter() - start) * 1000
                result = json.loads(child_output.read_text(encoding="utf-8"))
                result["native_log"] = str(log_path)
                result["process_wall_including_shutdown_ms"] = round(wall_ms, 3)
                report["runs"].append(result)
                print(f"  create={result['create_wall_ms'] / 1000:.3f}s; "
                      f"native_load={result['native_reported_load_ms'] / 1000:.3f}s", flush=True)
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                report["failures"].append({"run": index + 1, "error": str(exc),
                                           "native_log": str(log_path)})
                print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
            write_json(output, report)
    if report["runs"]:
        report["summary"] = {}
        for key in ("create_wall_ms", "native_reported_load_ms",
                    "library_load_to_first_result_wall_ms"):
            values = [run[key] for run in report["runs"]]
            report["summary"][key] = {"min": min(values), "max": max(values),
                                       "median": round(statistics.median(values), 3)}
        write_json(output, report)
    print(f"Saved: {output}")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker_output:
        worker(parsed)
    else:
        sys.exit(measure(parsed))
