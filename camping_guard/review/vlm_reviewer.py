"""
异步 VLM 语义复核模块 (慢通路，阶段 4)
实现端侧大模型门控唤醒、多线程异步非阻塞执行、结构化判定、<=50字中文战报生成与权限沙箱 (只升不降铁律)
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import os
import sys
import time
import json
import base64
import queue
import shutil
import tempfile
import threading
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
import numpy as np

from camping_guard.types import ThreatLevel, ActionType, ThreatEventRecord


class VLMBackendBase:
    """端侧 VLM 推理后端抽象基类"""
    def infer(
        self,
        event_record: ThreatEventRecord,
        keyframes: Optional[List[np.ndarray]] = None
    ) -> Dict[str, Any]:
        raise NotImplementedError


class MockVLMBackend(VLMBackendBase):
    """
    高仿真端侧本地 VLM 模拟后端
    模拟 Qwen2-VL-2B / MiniCPM-V INT4 在高通 8155 上的语义复核行为
    """
    def __init__(self, simulated_delay_sec: float = 0.01):
        self.simulated_delay_sec = simulated_delay_sec
        self.force_false_alarm: bool = False
        self.override_response: Optional[Dict[str, Any]] = None

    def infer(
        self,
        event_record: ThreatEventRecord,
        keyframes: Optional[List[np.ndarray]] = None
    ) -> Dict[str, Any]:
        if self.simulated_delay_sec > 0:
            time.sleep(self.simulated_delay_sec)

        if self.override_response is not None:
            return dict(self.override_response)

        action = event_record.action_type
        has_contact = event_record.has_contact_signal

        # 默认高置信度研判
        confidence = 0.92
        is_human_contact = has_contact
        is_false_alarm = self.force_false_alarm
        threat_escalation = "MAINTAIN"
        summary_text = ""

        if self.force_false_alarm:
            is_false_alarm = True
            confidence = 0.95
            is_human_contact = False
            action_category = "ENVIRONMENT_DISTURBANCE"
            threat_escalation = "DOWNGRADE_TO_L1"
            summary_text = "复核确认为树叶飘落或风吹光影晃动，未发现实质安全风险。"
        elif action == ActionType.DOOR_HANDLE_PULL:
            action_category = "DOOR_HANDLE_TAMPERING"
            is_human_contact = True
            if has_contact:
                threat_escalation = "CONFIRM_OR_UPGRADE"
                summary_text = "发现人员正用力反复拉扯门把手，伴有锁具冲击，已阻断。"
            else:
                threat_escalation = "MAINTAIN"
                summary_text = "监测到人员在门把手附近有抓握动作，正在持续监视。"
        elif action == ActionType.INTENTIONAL_KICK:
            action_category = "VEHICLE_BODY_KICK"
            is_human_contact = True
            if has_contact:
                threat_escalation = "CONFIRM_OR_UPGRADE"
                summary_text = "监测到外侧人员蓄意踢踹车门下边梁，冲击已被锁定存证。"
            else:
                summary_text = "人员向车身做出疑似踢踹动作，暂无机械接触震动。"
        elif action == ActionType.WINDOW_PEEKING:
            action_category = "CABIN_PEEKING"
            is_human_contact = False
            summary_text = "发现人员正贴近车窗窥探座舱内部，建议车内人员保持警惕。"
        elif action == ActionType.EQUIPMENT_THEFT:
            action_category = "ROOF_OR_EXTERNAL_THEFT"
            is_human_contact = True
            # 无接触传感器的疑似装备盗窃，VLM 予以确认为盗窃并建议升级 L3
            threat_escalation = "CONFIRM_OR_UPGRADE"
            summary_text = "发现人员正尝试拆卸车顶行李架或外挂装备，建议立即驱逐！"
        elif action == ActionType.BEAST_SPRINT:
            action_category = "WILD_ANIMAL_ATTACK"
            is_human_contact = False
            threat_escalation = "CONFIRM_OR_UPGRADE"
            summary_text = "警报！大型猛兽正高速逼近车辆，请务必紧闭车窗切勿外出！"
        elif action == ActionType.FIRE_SMOKE:
            action_category = "FIRE_OR_SMOKE_HAZARD"
            is_human_contact = False
            summary_text = "监测到露营火情烟雾扩散，请注意防风并准备应急灭火。"
        elif action == ActionType.VEHICLE_COLLISION:
            action_category = "VEHICLE_COLLISION"
            is_human_contact = False
            threat_escalation = "CONFIRM_OR_UPGRADE" if has_contact else "MAINTAIN"
            if has_contact:
                summary_text = "监测到外物剧烈撞击车身(如邻车开门/倒木砸车)，伴随震动冲击，已存证锁定！"
            else:
                summary_text = "监测到车身区域出现异常运动碰撞(如邻车开门贴近/外物触碰)，已锁定录像并复核。"
        else:
            action_category = "BENIGN_ACTIVITY"
            summary_text = f"车身外围发现常规活动目标，等级为 {event_record.threat_level.value}。"

        return {
            "action_category": action_category,
            "confidence": confidence,
            "is_human_contact": is_human_contact,
            "is_false_alarm": is_false_alarm,
            "threat_escalation": threat_escalation,
            "summary_text": summary_text[:50]
        }


class LocalHttpVLMBackend(VLMBackendBase):
    """
    通过 OpenAI 兼容 API 调用本地部署的轻量端侧 VLM (如 Qwen2-VL-2B / MiniCPM-V-2.6)
    支持部署在本地端口 (例如 http://127.0.0.1:8000/v1) 的 llama.cpp / vLLM / Ollama
    """
    SYSTEM_PROMPT = (
        "你是一个部署在智能汽车座舱域控制器（高通8155）上的安全哨兵多模态复核大模型。"
        "你的职责是对守卫系统快通路提交的可疑事件关键帧进行深度语义研判，"
        "排除光影、树叶飘落、鸟禽等环境假阳性，并为车主生成不超过50字的人性化简明情况通报。\n"
        "请严格输出 JSON 格式：\n"
        "{\n"
        '  "action_category": "DOOR_HANDLE_TAMPERING" | "CABIN_PEEKING" | "VEHICLE_BODY_KICK" | "ROOF_THEFT" | "VEHICLE_COLLISION" | "FALSE_ALARM" | "OTHER",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "is_human_contact": true/false,\n'
        '  "is_false_alarm": true/false,\n'
        '  "threat_escalation": "CONFIRM_OR_UPGRADE" | "DOWNGRADE_TO_L1" | "MAINTAIN",\n'
        '  "summary_text": "中文情况通报 (50字以内)"\n'
        "}"
    )

    def __init__(self, api_url: str = "http://127.0.0.1:8000/v1/chat/completions", model_name: str = "qwen2-vl-2b", timeout: float = 3.0):
        self.api_url = api_url
        self.model_name = model_name
        self.timeout = timeout
        self.fallback_mock = MockVLMBackend()

    def infer(
        self,
        event_record: ThreatEventRecord,
        keyframes: Optional[List[np.ndarray]] = None
    ) -> Dict[str, Any]:
        """构造多图 Vision 请求并调用本地端侧推理"""
        user_content = [
            {"type": "text", "text": f"快通路事件动作: {event_record.action_type.value}, 初始威胁等级: {event_record.threat_level.value}, 是否有物理接触信号: {event_record.has_contact_signal}。请对以下事件关键帧进行语义复核："}
        ]

        # 编码多帧关键帧为 Base64 JPEG
        if keyframes:
            import cv2
            for idx, img in enumerate(keyframes[:3]):
                _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                b64_str = base64.b64encode(buffer).decode('utf-8')
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}
                })

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.1,
            "max_tokens": 150
        }

        try:
            req_data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(self.api_url, data=req_data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                resp_json = json.loads(response.read().decode('utf-8'))
                raw_text = resp_json['choices'][0]['message']['content'].strip()
                # 尝试解析 JSON 代码块
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:]
                parsed = json.loads(raw_text.strip())
                if "summary_text" in parsed:
                    parsed["summary_text"] = parsed["summary_text"][:50]
                return parsed
        except Exception:
            # 端侧离线或服务异常时降级至离线安全保底规则
            return self.fallback_mock.infer(event_record, keyframes)


class TRTEdgeLLMOnDemandBackend(VLMBackendBase):
    """
    基于 NVIDIA TensorRT-Edge-LLM 的车载端侧按需低功耗 VLM 推理后端
    专为端侧部署的 qwen3.5-2b (Qwen2.5-VL-2B / Qwen3.5-2B) 多模态模型打造。

    端侧最高效执行架构设计 (按需门控唤醒 + C++ 原生运行 + 即时显存释放)：
    1. 零常驻服务守护进程：
       - 彻底杜绝 vLLM / Ollama / Triton 等后台常驻服务 24 小时占用 4~8GB 显存并维持高频率怠速 (持续消耗 15~25W 导致电瓶亏电)。
       - 平时 (99.9% 监控工况) 保持 0% GPU 显存占用、0% CPU 大模型负载，仅运行微秒级轻量检测器。
    2. 门控唤醒 (Gate-Triggered On-Demand)：
       - 仅在快通路触发 L2/L3 高威胁或车身核心接触报警时，由门控机制在慢通路异步按需拉起。
    3. C++ 原生执行 (Pure C++ Zero-Python-Overhead Runtime)：
       - 优先调用 TensorRT-Edge-LLM 官方编译的原生二进制 `llm_inference`，跳过 Python GIL 与 PyTorch 庞大运行时。
       - 通过轻量 IPC JSON 传递关键帧图片与上下文，直接利用 Orin TensorRT 与专用 SM87 插件加速。
    4. 执行完毕即时回落 (Instant Memory Reclaim)：
       - 产出 <=50 字结构化判定与人性化通报后，进程立即退出，OS 内核瞬间回收全部 CUDA 显存，GPU 回归待机休眠。
    5. 全自动路径推导与智能降级：
       - 自动检索 EDGELLM_BUILD_DIR、系统环境变量及 Orin 默认安装路径 (/home/nvidia/qwen35-orin/...)。
       - 在开发主机或无硬件环境时，无缝衔接高保真端侧规则兜底引擎，保障单元测试与全流程绝对稳定。
    """
    SYSTEM_PROMPT = (
        "你是一个部署在智能汽车座舱域控制器上的安全哨兵多模态复核大模型。"
        "对快通路提交的可疑事件关键帧进行语义复核，排除风吹草动假阳性，并生成<=50字简明情况通报。\n"
        "请严格输出 JSON 格式：\n"
        "{\n"
        '  "action_category": "DOOR_HANDLE_TAMPERING" | "CABIN_PEEKING" | "VEHICLE_BODY_KICK" | "ROOF_THEFT" | "VEHICLE_COLLISION" | "FALSE_ALARM" | "OTHER",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "is_human_contact": true/false,\n'
        '  "is_false_alarm": true/false,\n'
        '  "threat_escalation": "CONFIRM_OR_UPGRADE" | "DOWNGRADE_TO_L1" | "MAINTAIN",\n'
        '  "summary_text": "中文情况通报 (50字以内)"\n'
        "}"
    )

    def __init__(
        self,
        model_dir: Optional[str] = None,
        engine_path: Optional[str] = None,
        visual_engine_path: Optional[str] = None,
        build_dir: Optional[str] = None,
        edgellm_dir: Optional[str] = None,
        model_name: str = "qwen3.5-2b",
        timeout: float = 8.0,
        max_tokens: int = 128
    ):
        self.model_dir_arg = model_dir
        self.engine_path_arg = engine_path
        self.visual_engine_path_arg = visual_engine_path
        self.build_dir_arg = build_dir or edgellm_dir
        self.edgellm_dir_arg = edgellm_dir
        self.model_name = model_name
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.fallback_mock = MockVLMBackend()

        self._resolved_engine_dir: Optional[Path] = None
        self._resolved_visual_dir: Optional[Path] = None
        self._resolved_edgellm_bin: Optional[Path] = None
        self._resolved_build_dir: Optional[Path] = None
        self._resolved_plugin_path: Optional[Path] = None
        self._paths_probed = False

    def _probe_paths(self):
        """严格按显式设置校验路径，不进行任何猜测和候选目录推断"""
        if self._paths_probed:
            return
        self._paths_probed = True

        # 1. 显式构建目录校验 (build_dir)
        raw_build = self.build_dir_arg or os.environ.get("EDGELLM_BUILD_DIR")
        if raw_build:
            b_path = Path(raw_build).resolve()
            if b_path.is_file() and os.access(b_path, os.X_OK):
                self._resolved_edgellm_bin = b_path
                self._resolved_build_dir = b_path.parent
            elif b_path.is_dir():
                cand_bin = b_path / "examples" / "llm" / "llm_inference"
                if cand_bin.is_file() and os.access(cand_bin, os.X_OK):
                    self._resolved_edgellm_bin = cand_bin
                    self._resolved_build_dir = b_path
                elif (b_path / "llm_inference").is_file() and os.access(b_path / "llm_inference", os.X_OK):
                    self._resolved_edgellm_bin = b_path / "llm_inference"
                    self._resolved_build_dir = b_path
                else:
                    print(f"[!] [TRT-Edge-LLM][显式路径错误] 指定的 build 目录中找不到可执行程序 llm_inference: {cand_bin}")

            if self._resolved_build_dir:
                plugin = self._resolved_build_dir / "libNvInfer_edgellm_plugin.so"
                if plugin.is_file():
                    self._resolved_plugin_path = plugin
                else:
                    print(f"[!] [TRT-Edge-LLM][警告] 未在 build 目录找到专用插件库: {plugin}")

        # 2. 显式模型引擎路径校验
        raw_model_dir = self.model_dir_arg or os.environ.get("MODEL_DIR")
        raw_llm = self.engine_path_arg or os.environ.get("VLM_ENGINE_DIR")
        raw_vis = self.visual_engine_path_arg or os.environ.get("VLM_VISUAL_ENGINE_DIR")

        # 优先方式 A: 用户显式传入模型总目录 (--model-dir / MODEL_DIR)
        # 按照 TensorRT-Edge-LLM 官方标准目录规范要求：
        # <model_dir>/engines/llm/llm.engine 与 config.json
        # <model_dir>/engines/visual/visual.engine 与 config.json
        if raw_model_dir:
            m_path = Path(raw_model_dir).resolve()
            if not m_path.exists():
                print(f"[!] [TRT-Edge-LLM][显式路径错误] 指定的 --model-dir 不存在: {m_path}")
            else:
                llm_cand = m_path / "engines" / "llm"
                vis_cand = m_path / "engines" / "visual"
                if not llm_cand.is_dir():
                    print(f"[!] [TRT-Edge-LLM][配置错误] 缺少 LLM 引擎目录: {llm_cand}")
                elif not (llm_cand / "config.json").is_file():
                    print(f"[!] [TRT-Edge-LLM][配置错误] 缺少配置文件: {llm_cand / 'config.json'}")
                elif not ((llm_cand / "llm.engine").is_file() or any(llm_cand.glob("*.engine"))):
                    print(f"[!] [TRT-Edge-LLM][配置错误] 缺少引擎文件: {llm_cand / 'llm.engine'}")
                else:
                    self._resolved_engine_dir = llm_cand

                if not vis_cand.is_dir():
                    print(f"[i] [TRT-Edge-LLM] 模型目录下未找到独立视觉编码器目录: {vis_cand}")
                elif not (vis_cand / "config.json").is_file():
                    print(f"[!] [TRT-Edge-LLM][配置错误] 缺少视觉配置文件: {vis_cand / 'config.json'}")
                elif not ((vis_cand / "visual.engine").is_file() or any(vis_cand.glob("*.engine"))):
                    print(f"[!] [TRT-Edge-LLM][配置错误] 缺少视觉引擎文件: {vis_cand / 'visual.engine'}")
                else:
                    self._resolved_visual_dir = vis_cand

        # 若用户显式指定了独立的视觉引擎路径，最高优先级挂载
        if raw_vis and not self._resolved_visual_dir:
            v_path = Path(raw_vis).resolve()
            if not v_path.exists():
                print(f"[!] [TRT-Edge-LLM][显式路径错误] 指定的 Visual 引擎路径不存在: {v_path}")
            elif not self._is_valid_engine_dir(v_path):
                print(f"[!] [TRT-Edge-LLM][配置错误] Visual 引擎路径 {v_path} 不包含 config.json 或 .engine 文件")
            else:
                self._resolved_visual_dir = v_path if v_path.is_dir() else v_path.parent

        # 方式 B: 用户通过 --vlm-engine 显式指定 LLM 引擎路径
        if raw_llm and not self._resolved_engine_dir:
            l_path = Path(raw_llm).resolve()
            if not l_path.exists():
                print(f"[!] [TRT-Edge-LLM][显式路径错误] 指定的 LLM 引擎路径不存在: {l_path}")
            else:
                # 若用户传入的是包含 engines/llm 的目录
                if (l_path / "engines" / "llm").is_dir() and self._is_valid_engine_dir(l_path / "engines" / "llm"):
                    self._resolved_engine_dir = l_path / "engines" / "llm"
                    if not self._resolved_visual_dir and (l_path / "engines" / "visual").is_dir() and self._is_valid_engine_dir(l_path / "engines" / "visual"):
                        self._resolved_visual_dir = l_path / "engines" / "visual"
                elif self._is_valid_engine_dir(l_path):
                    self._resolved_engine_dir = l_path if l_path.is_dir() else l_path.parent
                    sibling_vis = self._resolved_engine_dir.parent / "visual"
                    if not self._resolved_visual_dir and sibling_vis.is_dir() and self._is_valid_engine_dir(sibling_vis):
                        self._resolved_visual_dir = sibling_vis
                else:
                    print(f"[!] [TRT-Edge-LLM][配置错误] {l_path} 既不包含 config.json 也不包含 .engine 文件，拒绝挂载")

        # 汇总状态提示
        if self._resolved_edgellm_bin and self._resolved_engine_dir:
            print(f"[*] [TRT-Edge-LLM] 显式配置挂载就绪: LLM={self._resolved_engine_dir}, Visual={self._resolved_visual_dir}, bin={self._resolved_edgellm_bin}")
        else:
            print("[i] [TRT-Edge-LLM] 未显式配置完整引擎路径 (--model-dir 或 --vlm-engine + --build-dir)，VLM 保持休眠（不启动 C++ 推理，使用端侧语义保底）")

    @staticmethod
    def _is_valid_engine_dir(d: Optional[Path]) -> bool:
        """检查路径是否为合法的 TensorRT 引擎目录或文件 (必须包含 config.json 或 *.engine)"""
        if not d:
            return False
        if d.is_file() and d.suffix == ".engine":
            return True
        if not d.is_dir():
            return False
        has_config = (d / "config.json").is_file()
        has_engine = (d / "llm.engine").is_file() or (d / "visual.engine").is_file() or any(d.glob("*.engine"))
        return has_config or has_engine

    def _infer_native_cpp(
        self,
        event_record: ThreatEventRecord,
        keyframes: Optional[List[np.ndarray]] = None
    ) -> Optional[Dict[str, Any]]:
        """最高效调用方案：通过 TensorRT-Edge-LLM 官方 C++ 原生二进制无守护进程推理"""
        if not self._resolved_edgellm_bin or not self._resolved_engine_dir:
            return None
        # 双重防御：确保解析出的目录确实为合法引擎目录，杜绝无效路径引发 C++ runtime 崩溃
        if not self._is_valid_engine_dir(self._resolved_engine_dir):
            return None

        dwell_s = getattr(event_record, "target_dwell_time", 0.0)
        dist_m = getattr(event_record, "target_distance", 0.0)
        target_cat = getattr(event_record, "target_category", "person")

        prompt_text = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"当前事件动作: {event_record.action_type.value}, 目标类别: {target_cat}, "
            f"滞留时间: {dwell_s:.1f}秒, 空间距离: {dist_m:.1f}米, "
            f"是否有物理接触信号: {event_record.has_contact_signal}。\n"
            "请输出判定 JSON："
        )

        # 优先使用 Linux /dev/shm 内存盘加速 (0ms 磁盘读写延迟，0 闪存损耗)
        shm_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
        tmp_dir = tempfile.mkdtemp(prefix="trt_vlm_", dir=shm_dir)
        tmp_dir_path = Path(tmp_dir)
        try:
            input_file = tmp_dir_path / "input.json"
            output_file = tmp_dir_path / "output.json"

            content_list: List[Dict[str, Any]] = []

            # 关键帧处理：优化分辨率至不超过 640px 以确保 ViT 编码器百毫秒内完成
            if keyframes and len(keyframes) > 0:
                import cv2
                img = keyframes[0]
                h, w = img.shape[:2]
                max_dim = max(h, w)
                if max_dim > 640:
                    scale = 640.0 / max_dim
                    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

                img_path = tmp_dir_path / "keyframe.jpg"
                cv2.imwrite(str(img_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                content_list.append({"type": "image", "image": str(img_path.resolve())})

            content_list.append({"type": "text", "text": prompt_text})

            # 严格遵循 TensorRT-Edge-LLM 官方多模态请求契约
            req_payload = {
                "batch_size": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "max_generate_length": self.max_tokens,
                "apply_chat_template": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
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
                json.dump(req_payload, f, ensure_ascii=False, indent=2)

            cmd = [
                str(self._resolved_edgellm_bin),
                "--engineDir", str(self._resolved_engine_dir),
                "--inputFile", str(input_file),
                "--outputFile", str(output_file),
                "--maxGenerateLength", str(self.max_tokens),
                "--dumpOutput"
            ]
            if self._resolved_visual_dir:
                cmd.extend(["--multimodalEngineDir", str(self._resolved_visual_dir)])

            # 注入专用 SM87 插件与运行时动态库环境变量
            env = os.environ.copy()
            if self._resolved_plugin_path and self._resolved_plugin_path.exists():
                env["EDGELLM_PLUGIN_PATH"] = str(self._resolved_plugin_path)
            if self._resolved_build_dir and self._resolved_build_dir.exists():
                curr_ld = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = f"{self._resolved_build_dir}:{curr_ld}" if curr_ld else str(self._resolved_build_dir)

            t0 = time.time()
            res = subprocess.run(cmd, env=env, timeout=self.timeout, capture_output=True, text=True)
            elapsed_ms = (time.time() - t0) * 1000.0

            if res.returncode == 0 and output_file.exists():
                with open(output_file, "r", encoding="utf-8") as f:
                    out_data = json.load(f)

                output_str = ""
                if isinstance(out_data, dict):
                    if "responses" in out_data and len(out_data["responses"]) > 0:
                        resp0 = out_data["responses"][0]
                        output_str = resp0.get("output_text", "") or resp0.get("text", "") or resp0.get("response", "")
                    elif "choices" in out_data and len(out_data["choices"]) > 0:
                        output_str = out_data["choices"][0].get("text", "") or out_data["choices"][0].get("message", {}).get("content", "")
                    elif "text" in out_data:
                        output_str = out_data["text"]
                    elif "choices" in out_data and len(out_data["choices"]) > 0:
                        output_str = out_data["choices"][0].get("text", "") or out_data["choices"][0].get("message", {}).get("content", "")
                    elif "text" in out_data:
                        output_str = out_data["text"]
                elif isinstance(out_data, list) and len(out_data) > 0:
                    first = out_data[0]
                    if isinstance(first, dict):
                        output_str = first.get("text", "") or first.get("response", "")

                if output_str:
                    raw_text = output_str.strip()
                    if "```" in raw_text:
                        raw_text = raw_text.split("```")[1]
                        if raw_text.startswith("json"):
                            raw_text = raw_text[4:]
                    if "{" in raw_text and "}" in raw_text:
                        s = raw_text.find("{")
                        e = raw_text.rfind("}") + 1
                        parsed = json.loads(raw_text[s:e])
                        if "summary_text" in parsed:
                            parsed["summary_text"] = parsed["summary_text"][:50]
                        print(f"[*] [TRT-Edge-LLM] C++ 原生按需推理完成 ({elapsed_ms:.1f}ms): {parsed.get('summary_text')}")
                        return parsed
            else:
                print(f"[!] [TRT-Edge-LLM] 二进制推理非零退出 (code {res.returncode}): {res.stderr[:200]}")
        except Exception as e:
            print(f"[!] [TRT-Edge-LLM] 原生推理过程捕获异常: {e}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return None

    def _infer_python_bindings(
        self,
        event_record: ThreatEventRecord,
        keyframes: Optional[List[np.ndarray]] = None
    ) -> Optional[Dict[str, Any]]:
        """备选方案：通过已安装的 tensorrt_edgellm Python 绑定进行进程内直调"""
        try:
            import tensorrt_edgellm as trt_edge
            if not hasattr(self, "_py_runner") or self._py_runner is None:
                if self._resolved_engine_dir:
                    self._py_runner = trt_edge.ModelRunner(str(self._resolved_engine_dir))
            if getattr(self, "_py_runner", None):
                dwell_s = getattr(event_record, "target_dwell_time", 0.0)
                dist_m = getattr(event_record, "target_distance", 0.0)
                prompt = (
                    f"{self.SYSTEM_PROMPT}\n\n"
                    f"当前事件动作: {event_record.action_type.value}, 目标类别: {getattr(event_record, 'target_category', 'person')}, "
                    f"滞留时间: {dwell_s:.1f}秒, 空间距离: {dist_m:.1f}米, 是否有物理接触信号: {event_record.has_contact_signal}。\n"
                    "请输出判定 JSON："
                )
                output_str = self._py_runner.generate(prompt, keyframes=keyframes[:2] if keyframes else None)
                if output_str:
                    raw_text = output_str.strip()
                    if "```" in raw_text:
                        raw_text = raw_text.split("```")[1]
                        if raw_text.startswith("json"):
                            raw_text = raw_text[4:]
                    if "{" in raw_text and "}" in raw_text:
                        s = raw_text.find("{")
                        e = raw_text.rfind("}") + 1
                        parsed = json.loads(raw_text[s:e])
                        if "summary_text" in parsed:
                            parsed["summary_text"] = parsed["summary_text"][:50]
                        return parsed
        except Exception:
            pass
        return None

    def infer(
        self,
        event_record: ThreatEventRecord,
        keyframes: Optional[List[np.ndarray]] = None
    ) -> Dict[str, Any]:
        """端侧 VLM 视觉推理与语义战报生成入口"""
        self._probe_paths()

        # 1. 优先尝试最高效的 C++ 原生按需推理 (零常驻守护进程、即时显存回收)
        res_cpp = self._infer_native_cpp(event_record, keyframes)
        if res_cpp is not None:
            return res_cpp

        # 2. 备选尝试 Python 原生绑定
        res_py = self._infer_python_bindings(event_record, keyframes)
        if res_py is not None:
            return res_py

        # 3. 开发测试/离线主机兜底：挂载端侧高保真语义复核规则 (融合滞留时间与空间距离)
        res = self.fallback_mock.infer(event_record, keyframes)
        dwell_s = getattr(event_record, "target_dwell_time", 0.0)
        dist_m = getattr(event_record, "target_distance", 0.0)
        if dwell_s > 10.0 and res.get("summary_text"):
            if "徘徊" not in res["summary_text"]:
                res["summary_text"] = f"目标在车身 {dist_m:.1f}m 处持续徘徊 {dwell_s:.1f} 秒，{res['summary_text']}"[:50]
        return res


class AsyncVLMReviewer:
    """
    异步多线程 VLM 语义复核引擎 (阶段 4 慢通路主控)
    维护低优先级后台工作线程队列，彻底解耦 30ms 快通路防御与 0.8~1.5s 大模型复核
    """
    def __init__(
        self,
        backend: Optional[VLMBackendBase] = None,
        enable_worker_thread: bool = True
    ):
        self.backend = backend or MockVLMBackend()
        self.review_history: List[Dict[str, Any]] = []
        self.task_queue: queue.Queue = queue.Queue()
        self.callbacks: List[Callable[[Dict[str, Any]], None]] = []

        self._active_tasks_count = 0
        self._lock = threading.Lock()
        self._is_running = enable_worker_thread
        self._worker_thread: Optional[threading.Thread] = None

        if enable_worker_thread:
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="VLM_Reviewer_Worker")
            self._worker_thread.start()

    def register_completion_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """注册异步完成后的事件回调"""
        self.callbacks.append(callback)

    def should_trigger_vlm(
        self,
        category_conf: float,
        is_unknown_obstacle: bool,
        fast_path_threat: ThreatLevel,
        action_type: ActionType,
        has_contact_signal: bool,
        is_body_roi_event: bool = False,
        is_zone3_roi_event: Optional[bool] = None
    ) -> bool:
        """
        唤醒门控逻辑 (步骤 2.2 门控条件 1~5)
        """
        # 条件 1: 检测置信度低 (<35%) 或未知障碍物
        if is_unknown_obstacle or category_conf < 0.35:
            return True

        # 条件 2: 快通路已触发 L2/L3，需要为车主生成人话总结
        if fast_path_threat in [ThreatLevel.L2_WARNING, ThreatLevel.L3_CRITICAL]:
            return True

        # 条件 3: 无接触传感器的疑似装备盗窃
        if action_type == ActionType.EQUIPMENT_THEFT and not has_contact_signal:
            return True

        # 条件 4: 车身固定 ROI (敏感部件/防碰撞区) 扰动事件，需要复核是否为人为接触或物理破坏
        roi_flag = is_body_roi_event if is_zone3_roi_event is None else (is_body_roi_event or is_zone3_roi_event)
        if roi_flag:
            return True

        # 条件 5: 车身碰撞/外物撞击事件 (邻车开门碰撞、树木倒塌、外物砸击) 必须唤醒 VLM 语义复核
        if action_type == ActionType.VEHICLE_COLLISION:
            return True

        return False

    def submit_review_async(
        self,
        event_record: ThreatEventRecord,
        keyframes: Optional[List[Any]] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> str:
        """
        非阻塞提交复核任务至后台工作线程队列 (耗时 < 1ms)
        """
        task_id = f"VLM_TASK_{event_record.event_id}_{int(time.time() * 1000)}"
        with self._lock:
            self._active_tasks_count += 1
        self.task_queue.put((task_id, event_record, keyframes, callback))
        return task_id

    def review_event_sync(
        self,
        event_record: ThreatEventRecord,
        keyframes: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        同步复核接口（用于离线测试或引擎立即获取保底战报）
        """
        raw_vlm_out = self.backend.infer(event_record, keyframes)
        return self._arbitrate_and_finalize(event_record, raw_vlm_out)

    def review_event_async(
        self,
        event_record: ThreatEventRecord,
        keyframes: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        向后兼容接口：提交异步队列，同时立即计算并返回仲裁结构
        确保既不阻塞快通路，又使外部调用能立即获取有效语义摘要
        """
        # 1. 提交至异步队列由后台工作线程执行（若工作线程开启）
        if self._is_running:
            self.submit_review_async(event_record, keyframes)

        # 2. 立即产出当前仲裁快照
        raw_vlm_out = self.backend.infer(event_record, keyframes)
        return self._arbitrate_and_finalize(event_record, raw_vlm_out)

    def _arbitrate_and_finalize(
        self,
        event_record: ThreatEventRecord,
        raw_vlm_out: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        核心安全权限沙箱：执行“只升不降铁律”与“纯视觉假阳性降级”
        """
        adjusted_threat = event_record.threat_level
        action_cat = raw_vlm_out.get("action_category", "")
        vlm_conf = raw_vlm_out.get("confidence", 0.8)
        summary = raw_vlm_out.get("summary_text", "")
        threat_escalation = raw_vlm_out.get("threat_escalation", "MAINTAIN")

        # 权限沙箱原则 1: “只升不降铁律” (Hard Rule)
        # 凡是有 CAN 门把手、车锁或车身 IMU 物理接触信号支撑的事件，VLM 绝对禁止降级！
        if event_record.has_contact_signal:
            if threat_escalation.startswith("DOWNGRADE") or raw_vlm_out.get("is_false_alarm", False):
                threat_escalation = "MAINTAIN_L3_CONTACT_LOCKED"
                adjusted_threat = event_record.threat_level
            elif threat_escalation == "CONFIRM_OR_UPGRADE" and event_record.threat_level == ThreatLevel.L2_WARNING:
                adjusted_threat = ThreatLevel.L3_CRITICAL

        # 权限沙箱原则 2: “纯视觉假阳性降级通道” (Soft Rule)
        # 仅针对纯视觉、完全无接触信号的 L2 事件，若 VLM 高置信度 (>=0.85) 明确判定为环境假阳性 (树叶/光影/飞鸟)，
        # 允许降为 L1，但必须保留录像与审计日志！
        elif event_record.threat_level == ThreatLevel.L2_WARNING and not event_record.has_contact_signal:
            if (raw_vlm_out.get("is_false_alarm", False) or threat_escalation == "DOWNGRADE_TO_L1") and vlm_conf >= 0.85:
                adjusted_threat = ThreatLevel.L1_NOTICE
                threat_escalation = "DOWNGRADE_TO_L1_ENVIRONMENT_FALSE_ALARM"
            elif threat_escalation == "CONFIRM_OR_UPGRADE":
                # 无接触信号但行为极其恶劣（如装备拆卸），升级至 L3
                if event_record.action_type == ActionType.EQUIPMENT_THEFT:
                    adjusted_threat = ThreatLevel.L3_CRITICAL

        # 战报长度硬性截断 <= 50 字
        summary = summary[:50]

        result = {
            "event_id": event_record.event_id,
            "original_threat": event_record.threat_level,
            "final_threat": adjusted_threat,
            "vlm_confidence": vlm_conf,
            "action_category": action_cat,
            "threat_escalation": threat_escalation,
            "summary_text": summary,
            "timestamp": time.time()
        }

        self.review_history.append(result)
        return result

    def _worker_loop(self):
        """后台低优先级异步推理循环"""
        while self._is_running:
            try:
                task = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            task_id, event_record, keyframes, callback = task
            try:
                raw_out = self.backend.infer(event_record, keyframes)
                result = self._arbitrate_and_finalize(event_record, raw_out)

                # 更新原记录
                event_record.vlm_summary = result["summary_text"]
                event_record.final_threat_level = result["final_threat"]

                # 分发特定回调与通用回调
                if callback:
                    callback(result)
                for cb in self.callbacks:
                    cb(result)
            except Exception:
                # 记录工作线程异常，不影响整车防御
                pass
            finally:
                with self._lock:
                    self._active_tasks_count = max(0, self._active_tasks_count - 1)
                self.task_queue.task_done()

    def wait_for_completion(self, timeout: float = 2.0) -> bool:
        """等待队列中所有任务执行完毕"""
        start_t = time.time()
        while True:
            with self._lock:
                if self._active_tasks_count == 0 and self.task_queue.empty():
                    return True
            time.sleep(0.01)
            if time.time() - start_t > timeout:
                return False


    def stop(self):
        """安全停止后台工作线程"""
        self._is_running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
