import os
import numpy as np

RAW = "data/cctv_raw"
OUT = "data_fixed_final"
os.makedirs(OUT, exist_ok=True)

def fix_sequence(seq):
    seq = seq.astype(np.float32)

    # --- 1. If shape is (T,17,2) → add dummy confidence 0.5 ---
    if seq.shape[2] == 2:
        conf = np.ones((seq.shape[0], seq.shape[1], 1), dtype=np.float32) * 0.5
        seq = np.concatenate([seq, conf], axis=2)

    # --- 2. Extract xy, conf ---
    xy = seq[:, :, :2]
    conf = seq[:, :, 2]

    # --- 3. Detect raw pixel scales ---
    max_xy = xy.max()
    min_xy = xy.min()

    # If xy > 1.5, assume pixel coords → normalize
    if max_xy > 1.5:
        max_val = max_xy  # global max normalizer
        xy = xy / (max_val + 1e-6)

    # --- 4. Force xy into 0..1 range ---
    xy = np.clip(xy, 0, 1)

    # --- 5. Normalize confidence separately to 0.3–1.0 range ---
    conf = np.clip(conf, 0.3, 1.0)

    # Combine
    seq_fixed = np.concatenate([xy, conf[..., None]], axis=2)

    return seq_fixed


for cls in os.listdir(RAW):
    if cls.startswith("."):
        continue
    in_dir = os.path.join(RAW, cls)
    if not os.path.isdir(in_dir):
        continue

    out_dir = os.path.join(OUT, cls)
    os.makedirs(out_dir, exist_ok=True)

    for f in os.listdir(in_dir):
        if not f.endswith(".npy"):
            continue

        seq = np.load(os.path.join(in_dir, f))

        seq_fixed = fix_sequence(seq)

        np.save(os.path.join(out_dir, f), seq_fixed)

print("✔ STRICT normalization complete → data_fixed_final")

