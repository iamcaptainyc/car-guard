from huggingface_hub import hf_hub_download
import shutil
import os

print("Downloading osnet_x0_25_msmt17.onnx from anriha/osnet_x0_25_msmt17...")
path = hf_hub_download(repo_id="anriha/osnet_x0_25_msmt17", filename="osnet_x0_25_msmt17.onnx")
os.makedirs("models/reid", exist_ok=True)
dest = "models/reid/osnet_x0_25.onnx"
shutil.copy(path, dest)
print(f"Successfully saved to {dest} ({os.path.getsize(dest)} bytes)")
