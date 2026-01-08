import os
import numpy as np

RAW = "data/cctv_raw"
OUT = "data_clean"
os.makedirs(OUT, exist_ok=True)

def normalize(seq):
    seq = seq.copy().astype(np.float32)

    # detect range
    mn = seq.min()
    mx = seq.max()

    # if already 0..1
    if 0 <= mn and mx <= 1.0 and mx > 0.95:
        return seq

    # if pixels 0..640 or 0..1080
    if mx > 2:
        seq[:, :, 0] /= seq[:, :, 0].max()
        seq[:, :, 1] /= seq[:, :, 1].max()
        if seq.shape[2] == 3:
            conf = seq[:, :, 2:3]
            seq = np.concatenate([seq[:, :, :2], conf], axis=2)
        return seq

    # fallback: min/max normalization
    seq = (seq - mn) / (mx - mn + 1e-6)
    return seq


for cls in os.listdir(RAW):
    if cls.startswith("."):
        continue
    in_dir = os.path.join(RAW, cls)
    if not os.path.isdir(in_dir):
        continue

    out_dir = os.path.join(OUT, cls)
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(in_dir) if f.endswith(".npy")]

    for f in files:
        seq = np.load(os.path.join(in_dir, f))

        # normalize
        seq = normalize(seq)

        # smooth temporal noise (optional)
        seq[:, :, :2] = np.apply_along_axis(
            lambda col: np.convolve(col, np.ones(3)/3, mode='same'),
            axis=0,
            arr=seq[:, :, :2]
        )

        # save
        np.save(os.path.join(out_dir, f), seq)

print("✔ Dataset normalization complete → data_clean")

