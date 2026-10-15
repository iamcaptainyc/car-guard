# VLM 冷启动判断与复现（2026-10-14）

## 已确认的事实

来自 `perf_report_Dingqing_wildlife_20261014_182051.json`：

| 项目 | 实测 |
|---|---:|
| 首次 cold_start_load | 12.06550 s |
| 首次 pure_inference | 3.47124 s |
| 首次整个 backend.infer 调用 | 16.18884 s |
| 首次 Python 图像预处理 / 输出解析 | 9.41 / 0.04 ms |
| 首次未归入上述字段的时间 | 642.65 ms |
| 后续 5 次完整复核 | 3.43216、3.42261、2.80889、2.79659、2.74684 s |
| 后续 5 次 cold_start_load | 全为 0 |

这支持“首次初始化较慢，同一进程后续复用引擎”。不支持“所有环境必然 11–12 秒”或“后续永远 2.7 秒”。首次完整复核为 16.19 秒；报告的平均 cold_start_load=2.011 秒是 12.066/6，不能作为一次冷启动的延迟。

对照 `perf_report_Dingqing_wildlife_20261013_182543.json`：同样是 realtime=true、YOLO11s、fire_enabled=true、1920×1080、Orin、8 CPU、TRT 10.3。19 次真实 subprocess VLM 调用的 cold_start_load 为 4.892–5.639 秒，均值 5.225 秒；另外 1 条为保底结果，应排除。

因此，旧报告并非“没有快通路竞争的独占单测”。12 秒作为本次初始化观测值可信，但比旧报告明显增加的原因尚未证实，需要对照测量。

## 计时边界及限制

- 当前 `vlm_engine.cpp`：加载插件、创建 CUDA stream 在 t0 前；`LLMInferenceRuntime` 构造在 t0–t1 内；`captureDecodingCUDAGraph` 在 t1 后。`vlm_get_cold_start_ms` 返回构造耗时，不包含完整 create 的所有工作。
- 当前 Python 优先采用 C++ getter；只有 getter<=0 才采用完整 `vlm_create` 的外层计时。另一个 agent 引用的 getter 日志 11475.2 ms 和报告 12065.5 ms，不能直接视为同一构建、同一轮的同一指标，需要核对部署版本。
- 旧 subprocess 的 cold_start_load 是“子进程墙钟时间 − Vision/Prefill/Generation GPU 时间之和”，还含其他 CPU 工作、文件读写、启动退出等。它不是独立测量的构造时间，不能直接与新 getter 作严格对比。
- 新 pure_inference 是整个 `handleRequest` 的 C++ 墙钟时间。当前代码把它也写进 llm_generation_gpu_ms，Vision/Prefill 写 0；这些不是 GPU 各阶段的测量结果。
- “load vocab”日志之间相隔 7.43 秒，只能指向这个日志区间。仅凭相邻日志无法认定全部用于 vocab.json 读取和 JSON 解析；应在部署的 tokenizer 构造中细分文件读、JSON 解析、词表/merge/正则初始化等实际步骤。
- NVIDIA 当前公开的 `examples/llm/llm_inference.cpp` 也调用 captureDecodingCUDAGraph。实际部署版本可能不同，应检查本机同一 commit 的源码；没有证据支持 CLI 天然少捕获图。

上游参考：https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/examples/llm/llm_inference.cpp

## 1. 保留原工况，比较两种后端

以下在 Jetson 的项目目录运行。BUILD、LLM、VISUAL、SO、VIDEO 必须改为本次实际使用的路径；JSON 没有记录模型身份，不能从报告认定模型是 0.8B 或 2B。

```bash
BUILD=/实际路径/TensorRT-Edge-LLM/build
LLM=/实际模型路径/engines/llm
VISUAL=/实际模型路径/engines/visual
SO=/实际路径/libvlm_engine.so
VIDEO=/实际视频路径/Dingqing_wildlife.mp4
export EDGELLM_PLUGIN_PATH="$BUILD/libNvInfer_edgellm_plugin.so"
export LD_LIBRARY_PATH="$BUILD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p logs

common=(
  --video "$VIDEO" --camera-id 0 --realtime --no-loop
  --detector models/yolo/yolo11s.engine
  --build-dir "$BUILD" --vlm-engine "$LLM" --vlm-visual-engine "$VISUAL"
  --vlm-max-tokens 128
)

# 两轮分别运行；先让前一轮完整退出。
env -u VLM_MODE python3 -u run_jetson_guard.py "${common[@]}" \
  --vlm-mode inprocess --vlm-engine-so "$SO" \
  --stats-json logs/repro_inprocess.json 2>&1 | tee logs/repro_inprocess.log

env -u VLM_MODE python3 -u run_jetson_guard.py "${common[@]}" \
  --vlm-mode subprocess \
  --stats-json logs/repro_subprocess.json 2>&1 | tee logs/repro_subprocess.log
```

保留原来的 pose、cls、fire、Web、显示及校准参数，若原运行使用了显式参数，应加入 common。示例让视频播完以保证取得首条结果；实时丢帧和事件排队会让事件数不同，不要求恰好复现 6 条。

进程内首次应有冷启动，后续同一句柄为 0。再次启动整个 Python 进程，又会产生初始化耗时。0 只是“本次没有重新初始化”，不是初始化操作本身变成 0 耗时。

至少运行 5 轮（每轮用不同报告文件名），交替两种模式，比较中位数和范围。必须查看 is_real_vlm=true / timing_source，排除规则保底。旧 subprocess 每次事件都会创建新进程。

## 2. 独立 C API 基准，分清 create 与构造计时

先停止上面的守卫进程。准备与原任务一致的关键帧 JPEG（守卫会将最长边缩到 640），以及 UTF-8 prompt.txt。未提供 prompt-file 时脚本使用简短通用描述提示词，其推理耗时不宜直接与原报告比较。

将本次新增的 scripts/benchmark_vlm_cold_start.py 同步到 Jetson 后：

```bash
python3 -u scripts/benchmark_vlm_cold_start.py \
  --so "$SO" --engine-dir "$LLM" --visual-dir "$VISUAL" \
  --image keyframe.jpg --prompt-file prompt.txt \
  --process-runs 5 --requests-per-process 6 --max-tokens 128 \
  --output logs/repro_native.json
```

脚本只依赖 Python 标准库，每轮新建一个子进程，每个子进程只 create 一次、连续 generate 6 次，最后 destroy。JSON 和每轮原生日志保存在 logs。

重点读取：

| 字段 | 含义 |
|---|---|
| dlopen_wall_ms | ctypes 加载动态库 |
| create_wall_ms | 完整 vlm_create 外层墙钟时间 |
| native_reported_load_ms | 实际加载的 .so 返回的计时；当前源码为 runtime 构造 |
| create_outside_native_load_ms | create 减原生计时，还含插件、CUDA 初始化及图捕获等，不能全部认作图捕获 |
| library_load_to_first_result_wall_ms | 加载库至第一条输出，不含 Python 解释器启动 |
| requests[].generate_wall_ms | 一次完整 C API generate 调用，含原生图像读取 |
| requests[].native_request_wall_ms | handleRequest 墙钟时间，不是单独的生成 GPU 时间 |
| process_wall_including_shutdown_ms | 整个子进程，包括所有请求和释放，不能当冷启动 |

这是独立基准，用于隔离加载问题；它没有主程序的 Torch/OpenCV/CUDA 上下文和多模型负载。要测“同一个进程的快通路竞争”，用第 1 步的完整管线及加载阶段打点，不能仅以另一进程人为加压作为等价工况。

## 3. 固定并记录环境

不要先改功耗、锁频或重编译，否则不能复现原来状态。运行前读取模式，运行中用 tegrastats 记录 CPU/GPU/EMC 频率、温度、内存和 swap：

```bash
sudo nvpmodel -q
sudo jetson_clocks --show
# 单独一个终端运行，测量结束后 Ctrl+C。
tegrastats --interval 1000 | tee logs/repro_tegrastats.log
```

记录同一份 llm.engine、visual.engine、tokenizer 文件、库与二进制、TensorRT-Edge-LLM commit。模型内容校验放在测量后，避免提前读取大文件改变缓存条件。

在 Edge-LLM 源码根目录检查构建及实际链接库：

```bash
git rev-parse HEAD
rg 'CMAKE_BUILD_TYPE|CMAKE_CXX_FLAGS' "$BUILD/CMakeCache.txt"
rg -- '-O[0-3s]|-g' "$BUILD/cpp/CMakeFiles/edgellmCore.dir/flags.make"
ldd "$SO"
ldd "$BUILD/examples/llm/llm_inference"
sha256sum "$SO" "$BUILD/examples/llm/llm_inference" \
  "$LLM/llm.engine" "$VISUAL/visual.engine"
```

若使用 Ninja，没有 flags.make，查看 compile_commands.json 或 ninja 的编译命令。重点检查 tokenizer 所在核心库的优化选项，不能只给 vlm_engine 包装文件增加 -O3 就认定依赖库已优化。Debug/-O0 是待排查原因，并非目前已证实的原因。

重启进程是“进程冷启动”，不会清空 Linux 文件页缓存。启动后第一轮与随后几轮分别记录；若目标是设备重启后的首次启动，需要在设备重启后再测。不要把反复重启进程当作每次磁盘都冷；也不要为了常规复现清空全机缓存。

若独立 C API 仍约 12 秒：优先检查部署库、编译选项、tokenizer 细分耗时、存储/缓存状态。若独立只有约 5 秒、完整管线约 12 秒：并发负载或进程环境才得到支持；还需加载阶段 CPU/GPU/EMC 数据定位。

## 本地验证范围

Windows 上可检查脚本语法和 --help；实际性能测试必须在带原模型与 libvlm_engine.so 的 Orin 上运行。当前未取得 Orin 执行结果，以上复现命令没有宣称测出新性能数据。
