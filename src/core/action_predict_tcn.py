# src/core/action_predict_tcn.py
import torch
import numpy as np
import os
import argparse
from train_tcn_bigrun import TCN_BiGRU

def predict(model_path, npy_path, data_root="data/cctv_augmented", device="cpu"):
    # Load classes in alphabetical order (training order)
    classes = sorted([
        d for d in os.listdir(data_root)
        if not d.startswith(".") and os.path.isdir(os.path.join(data_root, d))
    ])

    # Load sequence
    arr = np.load(npy_path).astype(np.float32)

    # Fix shapes: add conf column if missing
    if arr.shape[2] == 2:
        conf = np.ones((arr.shape[0], arr.shape[1], 1), dtype=np.float32)
        arr = np.concatenate([arr, conf], axis=2)

    # Flatten to (seq_len, 51)
    x = arr[:, :, :3].reshape(arr.shape[0], -1)
    x = torch.from_numpy(x[None]).float().to(device)

    # Load model
    model = TCN_BiGRU(
        input_dim=51,
        tcn_channels=[64, 128],
        gru_hidden=128,
        num_classes=len(classes)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    pred_conf = float(probs[pred_idx])

    return pred_label, pred_conf, probs, classes

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--npy", required=True)
    p.add_argument("--data_root", default="data/cctv_augmented")
    args = p.parse_args()

    label, conf, probs, classes = predict(
        args.model,
        args.npy,
        args.data_root
    )

    print("\n📌 Prediction:", label)
    print("🔢 Confidence:", conf)
    print("📊 All class probs:")
    for c, p in zip(classes, probs):
        print(f"   {c:>10} : {p:.3f}")

