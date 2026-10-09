#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
🚀 简易串行多模型原生 TensorRT 基准测试 (test_serial_models.py)
完全脱离外部算法工具包封装，采用原生 TensorRT API 进行底层引擎调用与精确延迟测定:
1. 测量口径严格对齐 trtexec:
   原生 TRT 延迟 = H2D (显存上传) + 纯 GPU 核心运算 + D2H (显存回传)
2. 串行依次执行: YOLO Det -> Face Det -> Face Rec -> OSNet ReID -> RTMPose
3. 逐帧在控制台实时打印各模型原生延迟 (ms) 与端到端帧率
4. 渲染可视化检测视频 (BBox, 人脸5点, 17骨骼连线, 性能HUD) 并安全导出 MP4
=============================================================================
"""

from test_serial_native_trt import main

if __name__ == "__main__":
    main()
