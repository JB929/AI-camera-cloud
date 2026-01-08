#!/usr/bin/env python3
"""
Clean CCTV keypoint dataset:
 - ensure shape (seq_len, 17, 3)
 - remove sequences with NaNs or near-zero keypoints
 - optional: convert (17,2) -> (17,3) by adding confidence=1
 - normalize x,y if values look like absolute pixels (heuristic)
 - save cleaned files to data/cctv_clean/<class>/*.npy
 - write a summary JSON (data/cctv_clean/summary.json)
"""
import os, sys, json, numpy as np
from glob import glob
from pathlib import Path

SRC_ROOT = "data/cctv_raw"
OUT_ROOT = "data/cctv_clean"
MIN_SEQ_LEN = 8   # sanity
EXPECTED_KEYPOINTS = 17
EXPECTED_DIM = 3

os.makedirs(OUT_ROOT, exist_ok=True)

summary = {"classes": {}, "total_in": 0, "total_out": 0, "removed": []}

def is_valid_array(arr):
    # arr shape (seq_len, 17, 3)
    if arr is None: 
        return False
    if not isinstance(arr, np.ndarray):
        return False
    if arr.ndim != 3:
        return False
    if arr.shape[1] != EXPECTED_KEYPOINTS:
        return False
    if arr.shape[2] < 2:
        return False
    return True

def normalize_xy_if_needed(seq):
    # seq: (L,17,3) or (L,17,2)
    # If any x or y > 1.5, treat as pixel coords and scale by max value observed
    xs = seq[:,:,0]
    ys = seq[:,:,1]
    maxv = max(xs.max(), ys.max())
    if maxv > 1.5:
        # scale into 0..1
        seq[:,:,0] = xs / (maxv + 1e-9)
        seq[:,:,1] = ys / (maxv + 1e-9)
    # clamp 0..1
    seq[:,:,0] = np.clip(seq[:,:,0], 0.0, 1.0)
    seq[:,:,1] = np.clip(seq[:,:,1], 0.0, 1.0)
    return seq

for class_dir in sorted(glob(os.path.join(SRC_ROOT, "*"))):
    if not os.path.isdir(class_dir):
        continue
    cls = os.path.basename(class_dir)
    out_dir = os.path.join(OUT_ROOT, cls)
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob(os.path.join(class_dir, "*.npy")))
    summary["classes"].setdefault(cls, {"in": 0, "out": 0})
    for fn in files:
        summary["total_in"] += 1
        summary["classes"][cls]["in"] += 1
        try:
            a = np.load(fn)
        except Exception as e:
            summary["removed"].append({"file": fn, "reason": f"load_err:{e}"})
            continue

        # Accept shapes: (L,17,3) or (L,17,2)
        if a.ndim != 3:
            summary["removed"].append({"file": fn, "reason": f"bad_dim:{a.shape}"})
            continue
        L, kpts, dim = a.shape
        if kpts != EXPECTED_KEYPOINTS:
            summary["removed"].append({"file": fn, "reason": f"bad_kpts:{a.shape}"})
            continue
        if L < MIN_SEQ_LEN:
            summary["removed"].append({"file": fn, "reason": f"short_seq:{L}"})
            continue

        # If dim == 2 convert to 3 (append 1.0 confidence)
        if dim == 2:
            c = np.ones((L, EXPECTED_KEYPOINTS, 1), dtype=np.float32)
            a = np.concatenate([a, c], axis=2)
            dim = 3

        # make float32
        a = a.astype(np.float32)

        # NaN / all zeros check
        if np.isnan(a).any():
            summary["removed"].append({"file": fn, "reason": "nan"})
            continue
        if np.allclose(a, 0.0):
            summary["removed"].append({"file": fn, "reason": "all_zero"})
            continue

        # Normalize x,y if needed
        a = normalize_xy_if_needed(a)

        # Basic sanity: confidence column in 0..1 (if not, clamp)
        a[:,:,2] = np.clip(a[:,:,2], 0.0, 1.0)

        # Save with same seq_len (we will NOT change length here)
        base = Path(fn).stem
        out_fn = os.path.join(out_dir, base + ".npy")
        np.save(out_fn, a)
        summary["total_out"] += 1
        summary["classes"][cls]["out"] += 1

# Write summary
with open(os.path.join(OUT_ROOT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("CLEANING COMPLETE")
print(json.dumps(summary, indent=2))

