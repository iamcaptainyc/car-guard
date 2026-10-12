"""
TensorRT Engine 推理引擎抽象与实现 (camping_guard/perception/trt_engine.py)
专为 NVIDIA Jetson Orin (Orin Nano / Orin NX / AGX Orin) 车载边缘计算平台打造。
支持：
1. 完整加载并执行 TensorRT 8.x 与 TensorRT 10.x 序列化 .engine 模型文件
2. 与 ONNXRuntime InferenceSession 保持完全一致的接口规范 (get_inputs, get_outputs, run)
3. 优先基于 PyTorch CUDA / CUDA-Python / PyCUDA 进行高吞吐、低开销 GPU 显存流式调度
4. 动态维度 (Dynamic Shapes) 自适应更新绑定
5. 无硬件环境下的优雅降级与模拟测试桩 (Mock Session)
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np


def is_jetson() -> bool:
    """检测当前运行环境是否为 NVIDIA Jetson 嵌入式边缘计算平台"""
    if os.path.exists("/etc/nv_tegra_release"):
        return True
    dt_model = "/sys/firmware/devicetree/base/model"
    if os.path.exists(dt_model):
        try:
            with open(dt_model, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().lower()
                if "jetson" in content or "tegra" in content:
                    return True
        except Exception:
            pass
    return False


def get_jetson_info() -> Dict[str, Any]:
    """获取当前 Jetson Orin 硬件规格与系统状态"""
    info = {
        "is_jetson": is_jetson(),
        "model": "Unknown",
        "jetpack_ver": "Unknown",
        "l4t_ver": "Unknown",
        "cuda_ver": "Unknown",
        "tensorrt_ver": "Unknown",
    }
    # 读取 Tegra 发行版信息
    tegra_rel = "/etc/nv_tegra_release"
    if os.path.exists(tegra_rel):
        try:
            with open(tegra_rel, "r") as f:
                first_line = f.readline().strip()
                info["l4t_ver"] = first_line
        except Exception:
            pass

    # 读取板卡型号
    dt_model = "/sys/firmware/devicetree/base/model"
    if os.path.exists(dt_model):
        try:
            with open(dt_model, "r", encoding="utf-8", errors="ignore") as f:
                info["model"] = f.read().replace("\x00", "").strip()
        except Exception:
            pass

    # 检查 TensorRT 版本
    try:
        import tensorrt as trt
        info["tensorrt_ver"] = trt.__version__
    except ImportError:
        pass

    # 检查 PyTorch 与 CUDA
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            info["cuda_ver"] = torch.version.cuda
    except ImportError:
        pass

    return info


class TensorMeta:
    """输入输出张量元数据，与 onnxruntime.NodeArg 接口对齐"""
    def __init__(self, name: str, shape: Tuple[Union[int, str], ...], dtype: np.dtype):
        self.name = str(name)
        self.shape = list(shape)
        self.dtype = dtype

    def __repr__(self):
        return f"TensorMeta(name='{self.name}', shape={self.shape}, dtype={self.dtype})"


class TRTInferenceSession:
    """
    基于 TensorRT 的高性能模型推理会话 (兼容 JetPack 5 / 6 与 TensorRT 8.x / 10.x)
    接口对齐 onnxruntime.InferenceSession:
        session.get_inputs() -> [TensorMeta, ...]
        session.get_outputs() -> [TensorMeta, ...]
        session.run(output_names, input_feed={name: np_arr}) -> [np_arr, ...]
    """
    def __init__(self, engine_path: str, device: str = "cuda:0"):
        self.engine_path = str(engine_path)
        self.model_path = str(engine_path)
        self.device = device
        self.engine = None
        self.context = None
        self.inputs: List[TensorMeta] = []
        self.outputs: List[TensorMeta] = []
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self.is_trt10 = False
        self._backend = "none" # "torch", "cuda_python", "pycuda", "mock"
        self._stream = None
        self._gpu_buffers: Dict[str, Any] = {}
        self.last_gpu_compute_ms: float = 0.0

        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"[TRTInferenceSession] 未找到 TensorRT engine 文件: {self.engine_path}")

        self._init_engine()

    def _init_engine(self):
        try:
            import tensorrt as trt
        except ImportError as e:
            raise ImportError(
                f"[TRTInferenceSession] 未检测到 tensorrt Python 包！\n"
                f"在 Jetson Orin 上请确认已安装 JetPack 并配置环境: python3 -c 'import tensorrt; print(tensorrt.__version__)'\n"
                f"原始错误: {e}"
            )

        trt_major = int(trt.__version__.split(".")[0])
        self.is_trt10 = (trt_major >= 10)

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"[TRTInferenceSession] 反序列化 TensorRT Engine 失败: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"[TRTInferenceSession] 创建 ExecutionContext 失败: {self.engine_path}")

        # 解析模型 I/O 绑定元数据
        if self.is_trt10 or hasattr(self.engine, "num_io_tensors"):
            num_tensors = self.engine.num_io_tensors
            for i in range(num_tensors):
                t_name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(t_name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(t_name))
                shape = tuple(self.engine.get_tensor_shape(t_name))
                meta = TensorMeta(t_name, shape, dtype)
                if mode == trt.TensorIOMode.INPUT:
                    self.inputs.append(meta)
                    self._input_names.append(t_name)
                else:
                    self.outputs.append(meta)
                    self._output_names.append(t_name)
        else:
            num_bindings = self.engine.num_bindings
            for i in range(num_bindings):
                b_name = self.engine.get_binding_name(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                shape = tuple(self.engine.get_binding_shape(i))
                meta = TensorMeta(b_name, shape, dtype)
                if self.engine.binding_is_input(i):
                    self.inputs.append(meta)
                    self._input_names.append(b_name)
                else:
                    self.outputs.append(meta)
                    self._output_names.append(b_name)

        # 确定显存调度后端 (首选 PyTorch CUDA，备选 cuda-python / pycuda)
        self._setup_backend()
        print(f"[TRTInferenceSession] 成功载入 TensorRT Engine: {Path(self.engine_path).name} "
              f"(Inputs: {len(self.inputs)}, Outputs: {len(self.outputs)}, GPU显存管理: {self._backend})")

    def _setup_backend(self):
        """配置显存搬运与 CUDA 流后端"""
        try:
            import torch
            if torch.cuda.is_available():
                self._backend = "torch"
                self._stream = torch.cuda.Stream()
                return
        except ImportError:
            pass

        try:
            from cuda import cudart
            self._backend = "cuda_python"
            err, self._stream = cudart.cudaStreamCreate()
            return
        except ImportError:
            pass

        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            self._backend = "pycuda"
            self._stream = cuda.Stream()
            return
        except ImportError:
            pass

        raise RuntimeError(
            "[TRTInferenceSession] 未找到可用的 CUDA 显存操作库！"
            "请在 Jetson Orin 上安装 PyTorch (带 CUDA 支持) 或 cuda-python / pycuda。"
        )

    def get_inputs(self) -> List[TensorMeta]:
        """返回输入张量列表 (完全兼容 ONNXRuntime session.get_inputs())"""
        return self.inputs

    def get_outputs(self) -> List[TensorMeta]:
        """返回输出张量列表 (完全兼容 ONNXRuntime session.get_outputs())"""
        return self.outputs

    def run(
        self,
        output_names: Optional[List[str]],
        input_feed: Dict[str, np.ndarray]
    ) -> List[np.ndarray]:
        """
        执行 TensorRT 前向推理
        :param output_names: 目标输出名列表 (若为 None 则返回所有 outputs)
        :param input_feed: {input_name: numpy_array}
        :return: [np_array, ...] 输出结果列表
        """
        if self._backend == "torch":
            return self._run_torch(input_feed, output_names)
        elif self._backend == "cuda_python":
            return self._run_cuda_python(input_feed, output_names)
        elif self._backend == "pycuda":
            return self._run_pycuda(input_feed, output_names)
        else:
            raise RuntimeError(f"未知的后端: {self._backend}")

    def _run_torch(self, input_feed: Dict[str, np.ndarray], output_names: Optional[List[str]]) -> List[np.ndarray]:
        import torch

        stream = self._stream
        with torch.cuda.stream(stream):
            tensor_ptrs = {}
            bindings = [] if not self.is_trt10 else None

            # 1. 设置输入并上传至 GPU
            for meta in self.inputs:
                name = meta.name
                if name not in input_feed:
                    # 容错：允许通过位置匹配
                    if len(input_feed) == 1 and len(self.inputs) == 1:
                        data = list(input_feed.values())[0]
                    else:
                        raise ValueError(f"缺少输入张量: {name}, 实际提供: {list(input_feed.keys())}")
                else:
                    data = input_feed[name]

                # 映射并严格保障输入类型对齐模型要求
                inp_torch_dtype = torch.float32
                if meta.dtype == np.float16:
                    inp_torch_dtype = torch.float16
                elif meta.dtype == np.int32:
                    inp_torch_dtype = torch.int32
                elif meta.dtype == np.int64:
                    inp_torch_dtype = torch.int64

                if not isinstance(data, torch.Tensor):
                    tensor = torch.from_numpy(np.ascontiguousarray(data)).to(device="cuda", dtype=inp_torch_dtype)
                else:
                    tensor = data.to(device="cuda", dtype=inp_torch_dtype)

                # 动态输入尺寸设置 (仅针对含动态 -1 维度的模型更新尺寸，若为静态且不匹配则抛出明确提示)
                if self.is_trt10:
                    engine_shape = tuple(self.engine.get_tensor_shape(name))
                    if any(d < 0 for d in engine_shape):
                        self.context.set_input_shape(name, tuple(tensor.shape))
                    elif engine_shape != tuple(tensor.shape):
                        raise RuntimeError(
                            f"TensorRT Engine 静态输入尺寸不匹配: 模型 '{Path(self.engine_path).name}' 输入 '{name}' 期望 {engine_shape}，"
                            f"但实际传入了 {tuple(tensor.shape)}。请重新以正确的动态维度 (--minShapes/--optShapes/--maxShapes) 编译该 Engine！"
                        )
                    tensor_ptrs[name] = tensor.data_ptr()
                    self.context.set_tensor_address(name, tensor.data_ptr())
                else:
                    idx = self.engine.get_binding_index(name)
                    binding_shape = tuple(self.engine.get_binding_shape(idx))
                    if any(d < 0 for d in binding_shape):
                        self.context.set_binding_shape(idx, tuple(tensor.shape))
                    elif binding_shape != tuple(tensor.shape):
                        raise RuntimeError(
                            f"TensorRT Engine 静态输入尺寸不匹配: 模型 '{Path(self.engine_path).name}' 期望 {binding_shape}，"
                            f"但实际传入了 {tuple(tensor.shape)}。请重新以正确的动态维度 (--minShapes/--optShapes/--maxShapes) 编译该 Engine！"
                        )
                    tensor_ptrs[name] = tensor.data_ptr()

            # 2. 为输出分配显存
            out_buffers_dict = {}

            for meta in self.outputs:
                name = meta.name
                if self.is_trt10:
                    out_shape = tuple(self.context.get_tensor_shape(name))
                else:
                    idx = self.engine.get_binding_index(name)
                    out_shape = tuple(self.context.get_binding_shape(idx))

                # 映射 numpy dtype 到 torch dtype
                torch_dtype = torch.float32
                if meta.dtype == np.float16:
                    torch_dtype = torch.float16
                elif meta.dtype == np.int32:
                    torch_dtype = torch.int32
                elif meta.dtype == np.int64:
                    torch_dtype = torch.int64

                if name in self._gpu_buffers and self._gpu_buffers[name].shape == out_shape and self._gpu_buffers[name].dtype == torch_dtype:
                    out_buf = self._gpu_buffers[name]
                else:
                    out_buf = torch.empty(out_shape, dtype=torch_dtype, device="cuda")
                    self._gpu_buffers[name] = out_buf
                tensor_ptrs[name] = out_buf.data_ptr()

                if self.is_trt10:
                    self.context.set_tensor_address(name, out_buf.data_ptr())

                out_buffers_dict[name] = out_buf

            # 3. 组织 TRT 8 顺序 bindings
            if not self.is_trt10:
                bindings = [tensor_ptrs[self.engine.get_binding_name(i)] for i in range(self.engine.num_bindings)]

            # 4. 执行推理 (以 CUDA 硬件 Event 毫秒级高精度测速)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)

            if self.is_trt10:
                self.context.execute_async_v3(stream.cuda_stream)
            else:
                self.context.execute_async_v2(bindings, stream.cuda_stream)

            end_event.record(stream)

        end_event.synchronize()
        try:
            self.last_gpu_compute_ms = float(start_event.elapsed_time(end_event))
        except Exception:
            self.last_gpu_compute_ms = 0.0

        # 严格按照 output_names 请求顺序返回结果，完全对齐 ONNXRuntime 规范
        if output_names:
            out_tensors = [out_buffers_dict[name] for name in output_names if name in out_buffers_dict]
        else:
            out_tensors = [out_buffers_dict[meta.name] for meta in self.outputs]

        return [t.cpu().numpy() for t in out_tensors]

    def _run_cuda_python(self, input_feed: Dict[str, np.ndarray], output_names: Optional[List[str]]) -> List[np.ndarray]:
        from cuda import cudart
        # 通用 cudart 分配与搬运逻辑
        raise NotImplementedError("cuda_python 底层流式执行正在接入，推荐直接配置 PyTorch 环境")

    def _run_pycuda(self, input_feed: Dict[str, np.ndarray], output_names: Optional[List[str]]) -> List[np.ndarray]:
        import pycuda.driver as cuda
        raise NotImplementedError("pycuda 底层流式执行正在接入，推荐直接配置 PyTorch 环境")


class MockTRTInferenceSession:
    """
    轻量 Mock 会话：用于在无 NVIDIA 显卡或测试用例环境下验证逻辑与接口兼容
    """
    def __init__(self, engine_path: str, fallback_onnx_path: Optional[str] = None):
        self.engine_path = engine_path
        self.fallback_onnx_path = fallback_onnx_path
        self.onnx_session = None

        if fallback_onnx_path and os.path.exists(fallback_onnx_path):
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(fallback_onnx_path, providers=['CPUExecutionProvider'])
            except Exception:
                pass

        # 默认虚拟输入输出
        self.inputs = [TensorMeta("images", (1, 3, 640, 640), np.float32)]
        self.outputs = [TensorMeta("output0", (1, 84, 8400), np.float32)]

    def get_inputs(self) -> List[TensorMeta]:
        if self.onnx_session:
            return [TensorMeta(i.name, i.shape, np.float32) for i in self.onnx_session.get_inputs()]
        return self.inputs

    def get_outputs(self) -> List[TensorMeta]:
        if self.onnx_session:
            return [TensorMeta(o.name, o.shape, np.float32) for o in self.onnx_session.get_outputs()]
        return self.outputs

    def run(self, output_names: Optional[List[str]], input_feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
        if self.onnx_session:
            return self.onnx_session.run(output_names, input_feed)
        # 合成空预测输出
        return [np.zeros((1, 84, 8400), dtype=np.float32)]


def load_inference_backend(
    model_path: str,
    prefer_tensorrt: bool = False,
    providers: Optional[List[str]] = None
) -> Tuple[Any, str]:
    """
    智能选择并加载推理会话：
    - 若 model_path 为 .engine 文件，或 prefer_tensorrt 为 True 且同名 .engine 存在：优先载入 TensorRT Engine
    - 否则回退至 ONNXRuntime
    :return: (session, backend_type: "tensorrt" | "onnxruntime")
    """
    p = Path(model_path)
    engine_candidate = p.with_suffix(".engine")

    # 1. 尝试 TensorRT
    should_try_trt = p.suffix.lower() == ".engine" or (prefer_tensorrt and engine_candidate.is_file())
    target_engine = str(p if p.suffix.lower() == ".engine" else engine_candidate)

    if should_try_trt and os.path.exists(target_engine):
        try:
            session = TRTInferenceSession(target_engine)
            session.model_path = target_engine
            return session, "tensorrt"
        except Exception as e:
            print(f"[load_inference_backend] 加载 TensorRT Engine 失败 ({e})，正在平稳回退至 ONNXRuntime...")

    # 2. 回退 ONNXRuntime
    target_onnx = str(p if p.suffix.lower() == ".onnx" else p.with_suffix(".onnx"))
    if not os.path.exists(target_onnx) and os.path.exists(model_path):
        target_onnx = model_path

    if os.path.exists(target_onnx):
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = min(4, os.cpu_count() or 4)
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        prov = providers or ["CPUExecutionProvider"]
        session = ort.InferenceSession(target_onnx, sess_options=sess_opts, providers=prov)
        try:
            session.model_path = target_onnx
        except Exception:
            pass
        return session, "onnxruntime"

    # 若两者均无
    raise FileNotFoundError(f"[load_inference_backend] 找不到可用的模型权重文件 (尝试了: {target_engine} 与 {target_onnx})")
