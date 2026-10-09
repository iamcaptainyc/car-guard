import os
import sys
import zipfile
import urllib.request
import json

print("=== Setting up models ===", flush=True)

# 1. Unzip buffalo_s.zip if needed
os.makedirs("models/insightface", exist_ok=True)
if os.path.exists("buffalo_s.zip"):
    print("Checking models/insightface contents...", flush=True)
    existing = os.listdir("models/insightface")
    if not existing:
        print("Extracting buffalo_s.zip...", flush=True)
        with zipfile.ZipFile("buffalo_s.zip", 'r') as zf:
            zf.extractall("models/insightface")
    print("InsightFace models ready:", os.listdir("models/insightface"), flush=True)

# 2. Let's find and download OSNet
# BoT-SORT or yolo_tracking or fast-reid models
# Let's test known direct download links for OSNet
osnet_urls = [
    # BoT-SORT official
    ("osnet_x0_25_market1501.onnx", "https://github.com/mikel-brostrom/yolo_tracking/releases/download/v9.0/osnet_x0_25_market1501.onnx"),
    ("osnet_x0_25_market1501.onnx", "https://github.com/mikel-brostrom/yolo_tracking/releases/download/v8.0/osnet_x0_25_market1501.onnx"),
    ("osnet_x0_25_market1501.onnx", "https://github.com/mikel-brostrom/yolo_tracking/releases/download/v7.0/osnet_x0_25_market1501.onnx"),
    ("osnet_x0_25_msmt17.onnx", "https://github.com/mikel-brostrom/yolo_tracking/releases/download/v9.0/osnet_x0_25_msmt17.onnx"),
    # FastReID / Boxmot
    ("osnet_x0_25_market1501.onnx", "https://github.com/mikel-brostrom/boxmot/releases/download/v10.0.64/osnet_x0_25_market1501.onnx"),
    ("osnet_x0_25_market1501.onnx", "https://github.com/mikel-brostrom/boxmot/releases/download/v10.0.1/osnet_x0_25_market1501.onnx")
]

os.makedirs("models/reid", exist_ok=True)
dest_osnet = "models/reid/osnet_x0_25_market1501.onnx"

if os.path.exists(dest_osnet) and os.path.getsize(dest_osnet) > 1000000:
    print(f"OSNet already downloaded: {dest_osnet} ({os.path.getsize(dest_osnet)} bytes)", flush=True)
else:
    for filename, url in osnet_urls:
        print(f"Trying download from {url}...", flush=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    data = resp.read()
                    with open(dest_osnet, "wb") as f:
                        f.write(data)
                    print(f"Successfully downloaded {dest_osnet} ({len(data)} bytes)!", flush=True)
                    break
        except Exception as e:
            print(f"Failed from {url}: {e}", flush=True)

print("=== Final Models Directory ===", flush=True)
for root, dirs, files in os.walk("models"):
    for file in files:
        fp = os.path.join(root, file)
        print(f"  {fp} ({os.path.getsize(fp)} bytes)", flush=True)
