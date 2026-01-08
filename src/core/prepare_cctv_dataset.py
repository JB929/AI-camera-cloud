# src/core/prepare_cctv_dataset.py
import os, numpy as np, argparse

def ensure_dir(p): 
    os.makedirs(p, exist_ok=True)

def fix_sequence(arr, seq_len=16):
    # arr: (seq_len, 17, 2) or (seq_len,17,3) or sometimes (seq_len,17)
    arr = np.array(arr, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == 17:  # weird shape
        arr = arr.reshape(arr.shape[0], 17, 1)
    if arr.ndim == 2 and arr.shape[1] == 2:
        arr = arr.reshape(arr.shape[0], 17, 2)
    if arr.ndim == 3 and arr.shape[2] == 2:
        # add confidence = 1.0
        conf = np.ones((arr.shape[0], arr.shape[1], 1), dtype=np.float32)
        arr = np.concatenate([arr, conf], axis=2)
    if arr.ndim == 3 and arr.shape[2] == 3:
        pass
    # enforce seq_len
    if arr.shape[0] < seq_len:
        last = arr[-1]
        pad = np.repeat(last[None, :, :], seq_len - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.shape[0] > seq_len:
        arr = arr[:seq_len]
    return arr.astype(np.float32)

def normalize_by_image(arr, maybe_pixels_threshold=1.0):
    # If coords > 1 anywhere, assume pixel coords and normalize to [0,1] by max dimension unknown:
    # We can't reliably detect width/height; so here we just clip >1 to 1 and leave as-is if already normalized.
    # This is conservative: if coordinates are pixels this won't fully normalize, so prefer normalized data when
    # collecting. We only clip values to [0,1].
    arr[:, :, 0] = np.clip(arr[:, :, 0], 0.0, 1.0)
    arr[:, :, 1] = np.clip(arr[:, :, 1], 0.0, 1.0)
    return arr

def main(root="data/cctv_raw", out="data/cctv_fixed", seq_len=16):
    ensure_dir(out)
    classes = sorted([d for d in os.listdir(root) if not d.startswith(".") and os.path.isdir(os.path.join(root, d))])
    print("Classes:", classes)
    total = 0
    for cls in classes:
        in_dir = os.path.join(root, cls)
        out_dir = os.path.join(out, cls)
        ensure_dir(out_dir)
        files = [f for f in os.listdir(in_dir) if f.endswith(".npy")]
        for fn in files:
            try:
                arr = np.load(os.path.join(in_dir, fn))
                fixed = fix_sequence(arr, seq_len=seq_len)
                fixed = normalize_by_image(fixed)
                np.save(os.path.join(out_dir, fn), fixed)
                total += 1
            except Exception as e:
                print("SKIP", cls, fn, e)
    print("Processed total:", total)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/cctv_raw")
    p.add_argument("--out", default="data/cctv_fixed")
    p.add_argument("--seq_len", type=int, default=16)
    args = p.parse_args()
    main(root=args.root, out=args.out, seq_len=args.seq_len)

