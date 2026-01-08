import torch
from torch.serialization import add_safe_globals

# ✅ Import YOLO classes that appear in the checkpoint
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.block import RepNCSPELAN4

# 1️⃣ Allow these classes for safe unpickling
add_safe_globals([DetectionModel, RepNCSPELAN4])

print("🔄 Loading YOLOv9 checkpoint (with safe globals)...")

# 2️⃣ Load PT file directly (NO YOLO(...) API here)
ckpt = torch.load("models/yolov9c.pt", map_location="cpu", weights_only=False)

print("💾 Converting state_dict...")

# 3️⃣ Extract actual weights
if isinstance(ckpt, dict):
    if "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
        sd = ckpt["model"].state_dict()
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        raise RuntimeError("❌ No valid state_dict found in YOLOv9 checkpoint!")
else:
    raise RuntimeError("❌ Unexpected checkpoint type, expected dict.")

# 4️⃣ Save safe version
out_path = "models/yolov9c-safe.pt"
torch.save(sd, out_path)

print(f"✅ Conversion successful → {out_path}")

