# Jetson Orin 原生 C++ 串行多模型基准测速程序 (`test_serial_trt`)

专为 **NVIDIA Jetson Orin** 打造的纯 C++ 端到端多模型串行基准测试工具，零 Python 胶水层开销、零 GIL 锁。

---

## 一、 核心特性

1. **100% 原生 C++ TensorRT API** (`NvInfer.h` + `cuda_runtime_api.h`)；
2. **显存池预分配 (Zero-Allocation Loop)**：初始化时一次性将输入/输出 Device Buffer 预先锁存，推理循环中杜绝任何运行期 `cudaMalloc` 显存分配毛刺；
3. **测量口径严格对齐 `trtexec`**：
   $$\text{TRT 原生平均延迟} = \text{H2D (显存上传)} + \text{纯 GPU 核心运算 (CUDA Event)} + \text{D2H (显存回传)}$$
4. **全串行链条**：`YOLO Det` $\rightarrow$ `Face Det` $\rightarrow$ `Face Rec` $\rightarrow$ `OSNet ReID` $\rightarrow$ `RTMPose`；
5. **支持实时视频导出**：叠加检测框、人脸关键点、17 骨骼连线与半透明性能 HUD 仪表盘，安全输出 MP4。

---

## 二、 编译方式 (在 Jetson Orin 上)

```bash
cd benchmark_cpp
chmod +x build.sh
./build.sh
```

编译生成的可执行二进制文件位于：`benchmark_cpp/build/test_serial_trt`。

---

## 三、 运行命令示例

### 1. 默认无头模式运行 (SSH 远程服务器推荐)
```bash
./benchmark_cpp/build/test_serial_trt --video sample_fisheye_left.mp4 --headless
```

### 2. 显式指定各模型引擎与输出路径
```bash
./benchmark_cpp/build/test_serial_trt \
    --video your_test.mp4 \
    --det models/yolo/yolo11n.engine \
    --face-det models/insightface/det_500m.engine \
    --face-rec models/insightface/w600k_mbf.engine \
    --reid models/reid/osnet_x0_25.engine \
    --pose models/pose/rtmpose-s.engine \
    --output tmp/serial_cpp_output.mp4 \
    --headless
```

### 3. 可选参数说明
| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--video <路径>` | 测试视频文件路径 | 自动检索样本 |
| `--output <路径>` | 导出带检测标注的 MP4 路径 | `tmp/serial_cpp_output.mp4` |
| `--headless` | 无窗口后台运行（适合远程 SSH 终端） | 关闭 |
| `--no-force-all` | 真实按需触发模式（仅画面有人时触发下游模型） | 默认每帧全开压测 |
| `--max-frames <N>` | 限制测试总帧数 | 全视频处理 |
