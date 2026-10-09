import onnxruntime as ort
import numpy as np

print("=== Verifying ONNX Models ===", flush=True)

models = {
    "Fisheye YOLO (YOLO11n-Fisheye)": "models/yolo/yolo11n-fisheye.onnx",
    "Fisheye YOLO (YOLOv8n-Fisheye)": "models/yolo/yolov8n-fisheye.onnx",
    "General YOLO (YOLO11s)": "models/yolo/yolo11s.onnx",
    "Pose Estimator (RTMPose-s)": "models/pose/rtmpose-s.onnx",
    "Face Detector (SCRFD)": "models/insightface/det_500m.onnx",
    "Face Recognizer (ArcFace MBF)": "models/insightface/w600k_mbf.onnx",
    "Face Pose/Landmarks (2d106det)": "models/insightface/2d106det.onnx",
    "Person ReID (OSNet)": "models/reid/osnet_x0_25.onnx"
}

for name, path in models.items():
    try:
        session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        print(f"[{name}] SUCCESS: {path}", flush=True)
        for i, inp in enumerate(session.get_inputs()):
            print(f"   Input {i}: name={inp.name}, shape={inp.shape}, type={inp.type}", flush=True)
        for i, out in enumerate(session.get_outputs()):
            print(f"   Output {i}: name={out.name}, shape={out.shape}, type={out.type}", flush=True)
    except Exception as e:
        print(f"[{name}] FAILED: {e}", flush=True)
