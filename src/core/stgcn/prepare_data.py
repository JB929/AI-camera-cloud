import os
import argparse
import numpy as np

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def load_npy(path):
    try:
        arr = np.load(path)
        if arr.ndim != 3:
            return None
        T, K, C = arr.shape

        # Convert (17,2) → (17,3)
        if C == 2:
            conf = np.ones((T, K, 1), dtype=np.float32)
            arr = np.concatenate([arr, conf], axis=2)

        return arr
    except:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path: raw CCTV keypoints")
    parser.add_argument("--out", required=True, help="Output sanitized dataset folder")
    args = parser.parse_args()

    in_root = args.data
    out_root = args.out
    ensure_dir(out_root)

    classes = sorted([d for d in os.listdir(in_root) if not d.startswith(".")])
    print("Classes:", classes)

    for cls in classes:
        src_dir = os.path.join(in_root, cls)
        dst_dir = os.path.join(out_root, cls)
        ensure_dir(dst_dir)

        files = [f for f in os.listdir(src_dir) if f.endswith(".npy")]
        print(f"[{cls}] {len(files)} files")

        idx = 0
        for f in files:
            seq = load_npy(os.path.join(src_dir, f))
            if seq is None:
                continue

            T, K, C = seq.shape

            if K != 17:
                continue
            if C != 3:
                continue

            out_path = os.path.join(dst_dir, f"{cls}_{idx:05d}.npy")
            np.save(out_path, seq)
            idx += 1

        print(f"Saved {idx} cleaned sequences for class {cls} → {dst_dir}")

if __name__ == "__main__":
    main()

