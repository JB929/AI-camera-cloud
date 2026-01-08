#!/usr/bin/env python3
import os, glob, random, json
root = "data/cctv_clean"
out = "data/cctv_splits"
os.makedirs(out, exist_ok=True)
splits = {"train": [], "val": [], "test": []}
RATIO = (0.7, 0.15, 0.15)  # train/val/test

for cls in sorted(os.listdir(root)):
    p = os.path.join(root, cls)
    if not os.path.isdir(p): continue
    files = sorted(glob.glob(os.path.join(p, "*.npy")))
    random.shuffle(files)
    n = len(files)
    n1 = int(n * RATIO[0])
    n2 = int(n * (RATIO[0] + RATIO[1]))
    splits["train"].extend(files[:n1])
    splits["val"].extend(files[n1:n2])
    splits["test"].extend(files[n2:])

with open(os.path.join(out, "splits.json"), "w") as f:
    json.dump(splits, f, indent=2)

print("splits written to", os.path.join(out, "splits.json"))
print("train:", len(splits["train"]), "val:", len(splits["val"]), "test:", len(splits["test"]))

