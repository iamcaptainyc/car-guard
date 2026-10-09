import urllib.request
import os
import sys

models = {
    "osnet_x0_25_market1501.onnx": "https://github.com/mikel-brostrom/yolo_tracking/releases/download/v10.0/osnet_x0_25_market1501.onnx",
    "buffalo_s.zip": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
}

for name, url in models.items():
    dest = os.path.join(".", name)
    if os.path.exists(dest):
        print(f"Already exists: {dest}")
        continue
    print(f"Downloading {name} from {url}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as out_f:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
            print(f"Successfully downloaded {name} ({downloaded} bytes)")
    except Exception as e:
        print(f"Failed to download {name}: {e}")
