#!/usr/bin/env python3
import os, glob
root = "data/cctv_clean"
for c in sorted(os.listdir(root)):
    p = os.path.join(root, c)
    if os.path.isdir(p):
        files = glob.glob(os.path.join(p, "*.npy"))
        print(f"{c:12s}: {len(files)}")

