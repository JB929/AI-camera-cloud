# src/core/augment_keypoints.py
import os, numpy as np, argparse, math
from glob import glob

def ensure_dir(p): os.makedirs(p, exist_ok=True)

# Transform helpers (operate on (seq,17,3))
def jitter(seq, sigma=0.02):
    noise = np.random.normal(0, sigma, size=seq[:, :, :2].shape)
    seq2 = seq.copy()
    seq2[:, :, :2] = np.clip(seq2[:, :, :2] + noise, 0.0, 1.0)
    return seq2

def scale_center(seq, scale_range=(0.9, 1.1)):
    s = np.random.uniform(*scale_range)
    # scale relative to center of torso
    cx = np.mean(seq[:, [5,6,11,12], 0])
    cy = np.mean(seq[:, [5,6,11,12], 1])
    seq2 = seq.copy()
    seq2[:, :, 0] = (seq2[:, :, 0] - cx) * s + cx
    seq2[:, :, 1] = (seq2[:, :, 1] - cy) * s + cy
    seq2[:, :, :2] = np.clip(seq2[:, :, :2], 0.0, 1.0)
    return seq2

def rotate(seq, ang_deg=8):
    theta = np.deg2rad(np.random.uniform(-ang_deg, ang_deg))
    cos, sin = math.cos(theta), math.sin(theta)
    # center
    cx = np.mean(seq[:, :, 0])
    cy = np.mean(seq[:, :, 1])
    seq2 = seq.copy()
    x = seq2[:, :, 0] - cx
    y = seq2[:, :, 1] - cy
    seq2[:, :, 0] = x * cos - y * sin + cx
    seq2[:, :, 1] = x * sin + y * cos + cy
    seq2[:, :, :2] = np.clip(seq2[:, :, :2], 0.0, 1.0)
    return seq2

def drop_joints(seq, drop_prob=0.1):
    seq2 = seq.copy()
    mask = np.random.rand(seq2.shape[1]) < drop_prob
    for j, m in enumerate(mask):
        if m:
            seq2[:, j, :2] = seq2[:, j, :2] * 0.0
            seq2[:, j, 2] = 0.0
    return seq2

def time_warp(seq, max_scale=0.2):
    # simple time warp by linear interpolation between stretched index positions
    L = seq.shape[0]
    s = np.random.uniform(1 - max_scale, 1 + max_scale)
    new_len = L
    src = np.linspace(0, L - 1, L)
    dst = np.linspace(0, L - 1, new_len) * s
    dst = np.clip(dst, 0, L - 1)
    seq2 = np.empty((new_len, seq.shape[1], seq.shape[2]), dtype=seq.dtype)
    for i in range(new_len):
        lo = int(np.floor(dst[i]))
        hi = min(lo + 1, L - 1)
        a = dst[i] - lo
        seq2[i] = (1 - a) * seq[lo] + a * seq[hi]
    return seq2

def hflip(seq):
    seq2 = seq.copy()
    seq2[:, :, 0] = 1.0 - seq2[:, :, 0]
    # swap left/right joints: pairs (5,6),(7,8),(9,10),(11,12),(13,14)
    pairs = [(5,6),(7,8),(9,10),(11,12),(13,14)]
    for a,b in pairs:
        tmp = seq2[:, a, :].copy()
        seq2[:, a, :] = seq2[:, b, :]
        seq2[:, b, :] = tmp
    return seq2

AUG_FUNCS = [jitter, scale_center, rotate, drop_joints, time_warp, hflip]

def augment_one(seq, n_aug=5):
    out = []
    for _ in range(n_aug):
        s = seq.copy()
        # apply random 2-3 transforms
        funcs = np.random.choice(AUG_FUNCS, size=np.random.randint(2,4), replace=False)
        for f in funcs:
            # random params for some functions
            if f is jitter:
                s = f(s, sigma=np.random.uniform(0.01, 0.05))
            elif f is scale_center:
                s = f(s, scale_range=(0.92, 1.08))
            elif f is rotate:
                s = f(s, ang_deg=np.random.uniform(3,12))
            elif f is drop_joints:
                s = f(s, drop_prob=np.random.uniform(0.05, 0.2))
            elif f is time_warp:
                s = f(s, max_scale=np.random.uniform(0.08, 0.25))
            elif f is hflip:
                if np.random.rand() < 0.5: s = f(s)
        out.append(s)
    return out

def main(root="data/cctv_fixed", out="data/cctv_augmented", multiplier=3):
    ensure_dir(out)
    classes = sorted([d for d in os.listdir(root) if not d.startswith(".") and os.path.isdir(os.path.join(root, d))])
    for cls in classes:
        in_dir = os.path.join(root, cls)
        out_dir = os.path.join(out, cls)
        ensure_dir(out_dir)
        files = [f for f in os.listdir(in_dir) if f.endswith(".npy")]
        for fn in files:
            arr = np.load(os.path.join(in_dir, fn))
            # save original
            np.save(os.path.join(out_dir, fn), arr)
            # augment
            aug_list = augment_one(arr, n_aug=multiplier)
            base = os.path.splitext(fn)[0]
            for i, a in enumerate(aug_list):
                np.save(os.path.join(out_dir, f"{base}_aug{i:02d}.npy"), a)
        print("Augmented class", cls, "->", len(files), "-> out:", len(os.listdir(out_dir)))
    print("Done augmentation")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/cctv_fixed")
    p.add_argument("--out", default="data/cctv_augmented")
    p.add_argument("--multiplier", type=int, default=3)
    args = p.parse_args()
    main(root=args.root, out=args.out, multiplier=args.multiplier)

