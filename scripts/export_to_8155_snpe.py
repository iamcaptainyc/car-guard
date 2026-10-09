"""
高通骁龙 8155 (SA8155P) 座舱平台模型导出与量化部署工具 (scripts/export_to_8155_snpe.py)
面向高通 SNPE (Snapdragon Neural Processing Engine) 与 QNN (Qualcomm AI Engine Direct)
生成静态张量 ONNX、准备 DSP 量化校准集 (raw_list.txt) 与转换指令
"""

import os
import sys
import numpy as np
import cv2

def generate_snpe_calibration_data(output_dir: str = "calib_data_8155", num_samples: int = 20):
    """
    生成符合高通 SNPE 要求的 INT8 量化浮点原始二进制文件 (RAW) 与文件索引列表
    高通 snpe-dlc-quantize 要求输入为 float32 格式的 raw 连续字节流
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_list_detector = os.path.join(output_dir, "yolo_raw_list.txt")
    raw_list_pose = os.path.join(output_dir, "rtmpose_raw_list.txt")

    print(f"[*] 正在为高通 8155 Hexagon DSP 生成量化校准数据样本 (目录: {output_dir})...")

    with open(raw_list_detector, "w") as f_det, open(raw_list_pose, "w") as f_pose:
        for i in range(num_samples):
            # 1. 检测器输入样本 (640x640x3, BGR to RGB, 0~1 float32)
            dummy_det = np.random.uniform(0.0, 1.0, (1, 3, 640, 640)).astype(np.float32)
            det_raw_path = os.path.join(output_dir, f"det_sample_{i:03d}.raw")
            dummy_det.tofile(det_raw_path)
            f_det.write(f"{det_raw_path}\n")

            # 2. RTMPose 姿态输入样本 (256x192x3, 归一化 (x-mean)/std)
            dummy_pose = np.random.normal(0.0, 1.0, (1, 3, 256, 192)).astype(np.float32)
            pose_raw_path = os.path.join(output_dir, f"pose_sample_{i:03d}.raw")
            dummy_pose.tofile(pose_raw_path)
            f_pose.write(f"{pose_raw_path}\n")

    print(f"[+] 8155 校准文件已生成:")
    print(f"    - 检测器校准列表: {raw_list_detector}")
    print(f"    - 姿态估计校准列表: {raw_list_pose}")
    return raw_list_detector, raw_list_pose

def print_8155_snpe_conversion_commands():
    """输出在高通 Linux/Ubuntu 交叉编译环境下的具体转换指令"""
    print("\n" + "=" * 75)
    print("【高通 SA8155P 模型量化与编译流水线操作指南】")
    print("环境要求: Qualcomm SNPE SDK (v1.68+ 或 v2.x) 或 QNN SDK")
    print("=" * 75)

    print("\n步骤 1: 将 ONNX 转换为高通原生 DLC 格式 (Non-quantized Float)")
    print("  # 1.1 目标检测模型转换:")
    print("  snpe-onnx-to-dlc -i yolo_nano.onnx -o yolo_nano.dlc")
    print("  # 1.2 RTMPose 姿态模型转换:")
    print("  snpe-onnx-to-dlc -i rtmpose-s.onnx -o rtmpose-s.dlc")

    print("\n步骤 2: 使用量化工具转为 Hexagon DSP (HTA) 专用的 INT8 DLC")
    print("  # 2.1 针对 8155 Hexagon 690 DSP 生成 INT8 模型:")
    print("  snpe-dlc-quantize --input_dlc yolo_nano.dlc \\")
    print("                    --input_list calib_data_8155/yolo_raw_list.txt \\")
    print("                    --output_dlc yolo_nano_quantized.dlc \\")
    print("                    --enable_hta")

    print("  snpe-dlc-quantize --input_dlc rtmpose-s.dlc \\")
    print("                    --input_list calib_data_8155/rtmpose_raw_list.txt \\")
    print("                    --output_dlc rtmpose_quantized.dlc \\")
    print("                    --enable_hta")

    print("\n步骤 3: 8155 实车端推断运行时加载方式 (C++ 代码片段)")
    print("""
    // 初始化 SNPE 运行时，优先绑定 DSP (Hexagon Tensor Accelerator)
    zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::DSP;
    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime)) {
        runtime = zdl::DlSystem::Runtime_t::GPU; // 降级为 Adreno GPU
    }
    std::unique_ptr<zdl::SNPE::SNPE> snpe = snpeBuilder.setOutputLayers(outputLayers)
                                                       .setRuntimeProcessor(runtime)
                                                       .build();
    """)
    print("=" * 75 + "\n")

if __name__ == "__main__":
    generate_snpe_calibration_data()
    print_8155_snpe_conversion_commands()
