# src/core/auto_retrain.py
"""
Auto retrain pipeline for CCTV keypoint action model.

Usage:
  # Dry run (no model save)
  python3 src/core/auto_retrain.py --dry-run

  # Full retrain (merge autotrain_buffer into dataset, train, export model)
  python3 src/core/auto_retrain.py --merge-buffer --epochs 30 --batch 16

Notes:
- Expects main dataset under `data/cctv_raw/<Label>/*.npy` (shape (seq_len,17,3) or (seq_len,17,2))
- Expects autotrain_buffer/*.npy (saved sequences, shape maybe (<=16,17,3))
- Model saved to models/cctv_action_gru.pt and models/cctv_action_gru_ts.pt (atomic write)
"""

import os
import time
import argparse
import random
from glob import glob
import shutil
import tempfile
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# -----------------------
# Config / paths
# -----------------------
DATA_DIR = "data/cctv_raw"
BUFFER_DIR = "autotrain_buffer"
MODELS_DIR = "models"
MODEL_OUT = os.path.join(MODELS_DIR, "cctv_action_gru.pt")
MODEL_TS_OUT = os.path.join(MODELS_DIR, "cctv_action_gru_ts.pt")
TMP_MODEL = os.path.join(MODELS_DIR, "tmp_model.pt")

MIN_SAMPLES_PER_CLASS = 5  # refuse to train if below this for any class (tuneable)

# -----------------------
# Simple dataset loader
# -----------------------
LABELS = None  # computed dynamically

def find_classes(data_root):
    classes = []
    for p in sorted(os.listdir(data_root)):
        if os.path.isdir(os.path.join(data_root, p)) and p[0] != ".":
            classes.append(p)
    return sorted(classes)

class KeypointSeqDataset(Dataset):
    def __init__(self, root_dirs, seq_len=16, augment=False):
        """
        root_dirs: list of directories to search (each directory should contain label subfolders)
        """
        self.seq_len = seq_len
        self.files = []      # tuples (path, label_idx)
        global LABELS
        LABELS = find_classes(root_dirs[0]) if LABELS is None else LABELS

        for root in root_dirs:
            for i,lab in enumerate(LABELS):
                folder = os.path.join(root, lab)
                if not os.path.isdir(folder):
                    continue
                for f in glob(os.path.join(folder, "*.npy")):
                    self.files.append((f, i))
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        arr = np.load(path)  # expected (seq_len, 17, 2|3)
        # normalize shape to (seq_len, 17, 3)
        if arr.ndim == 2 and arr.shape[1] == 34:
            arr = arr.reshape(arr.shape[0], 17, 2)
        if arr.shape[-1] == 2:
            # append confidence 1.0
            conf = np.ones((arr.shape[0], arr.shape[1], 1), dtype=np.float32)
            arr = np.concatenate([arr, conf], axis=2)
        # pad or trim in time
        if arr.shape[0] < self.seq_len:
            pad = np.repeat(arr[-1:,:,:], self.seq_len - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        else:
            arr = arr[-self.seq_len:,:,:]

        if self.augment:
            arr = augment_sequence(arr)

        # convert to (seq_len, 17*3) or keep (seq_len,17,3) depending on model input
        # we'll provide (seq_len, 17, 3)
        return torch.from_numpy(arr).float(), label

# -----------------------
# Augmentations (light)
# -----------------------
def augment_sequence(seq):
    # seq: (T,17,3), x,y normalized or px values
    seq = seq.copy()
    T, J, C = seq.shape
    # small additive jitter on x,y
    jitter = (np.random.randn(T, J, 2) * 0.01) * np.max(seq[..., :2])  # scale
    seq[..., :2] = seq[..., :2] + jitter
    # random horizontal flip
    if random.random() < 0.4:
        seq[..., 0] = 1.0 - seq[..., 0] if np.max(seq[...,0]) <= 1.0 else -seq[..., 0]
        # swap left/right joints if using consistent indexing (optional)
        # simple swap map (for COCO-like): L_sh(5)<->R_sh(6), L_elbow(7)<->R_elbow(8) etc.
        swap_pairs = [(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
        for a,b in swap_pairs:
            seq[:, [a,b], :] = seq[:, [b,a], :]
    # small scaling
    if random.random() < 0.25:
        s = 1.0 + (np.random.randn() * 0.02)
        seq[..., :2] *= s
    return seq

# -----------------------
# Model (GRU-based)
# -----------------------
class GRUActionNet(nn.Module):
    def __init__(self, input_dim=17*3, hid=128, num_classes=6):
        super().__init__()
        self.input_dim = input_dim
        self.gru = nn.GRU(input_dim, hid, num_layers=1, batch_first=True, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(hid, hid//2),
            nn.ReLU(),
            nn.Linear(hid//2, num_classes)
        )

    def forward(self, x):
        # x: (B, T, 17, 3)
        B,T,_,_ = x.shape
        x = x.view(B, T, -1)  # (B,T,51)
        out,_ = self.gru(x)   # (B,T,hid)
        out = out[:, -1, :]   # last timestep
        return self.fc(out)

# -----------------------
# Helpers: training, eval
# -----------------------
def collate_batch(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, ys

def compute_class_counts(root):
    counts = Counter()
    for lab in os.listdir(root):
        p = os.path.join(root, lab)
        if os.path.isdir(p):
            counts[lab] = len(glob(os.path.join(p, "*.npy")))
    return counts

def train_model(train_loader, val_loader, device, epochs=20, lr=1e-3):
    num_classes = len(LABELS)
    model = GRUActionNet(input_dim=17*3, hid=128, num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_val = -1.0
    best_state = None
    for ep in range(1, epochs+1):
        model.train()
        total=0; correct=0; loss_acc=0.0
        for x,y in train_loader:
            x = x.to(device); y = y.to(device)
            out = model(x)
            loss = crit(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_acc += float(loss.item())
            preds = out.argmax(dim=1)
            total += y.size(0)
            correct += (preds==y).sum().item()
        train_acc = correct/total if total>0 else 0.0

        # val
        model.eval()
        vtotal=0; vcorrect=0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv = xv.to(device); yv = yv.to(device)
                outv = model(xv)
                preds = outv.argmax(dim=1)
                vtotal += yv.size(0)
                vcorrect += (preds==yv).sum().item()
        val_acc = vcorrect / vtotal if vtotal>0 else 0.0

        print(f"Epoch {ep}/{epochs} Train Acc: {train_acc:.3f} Val Acc: {val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val

# -----------------------
# Main pipeline
# -----------------------
def main(args):
    os.makedirs(MODELS_DIR, exist_ok=True)

    # If requested, move buffer into DATA_DIR (merge)
    if args.merge_buffer:
        if not os.path.isdir(BUFFER_DIR):
            print("No buffer directory found, skipping merge.")
        else:
            # we expect buffer files named like camera_ts.npy or label_*.npy
            for f in sorted(glob(os.path.join(BUFFER_DIR, "*.npy"))):
                try:
                    arr = np.load(f)
                    # we expect the original buffer file to have label in name camera_Label_ts.npy
                    # fallback: place in "Other"
                    parts = os.path.basename(f).split("_")
                    possible_label = parts[0]
                    # if label not in dataset, append to 'Other' or create label
                    # For safety, require that label is present in DATA_DIR
                    label_found = None
                    for lab in find_classes(DATA_DIR):
                        if lab.lower() == possible_label.lower():
                            label_found = lab
                            break
                    if label_found is None:
                        # if only one label exists, push there; else create 'Other'
                        label_found = "Other" if "Other" in find_classes(DATA_DIR) else find_classes(DATA_DIR)[0]
                    target_dir = os.path.join(DATA_DIR, label_found)
                    os.makedirs(target_dir, exist_ok=True)
                    outp = os.path.join(target_dir, f"{int(time.time())}_{os.path.basename(f)}")
                    np.save(outp, arr)
                    print(f"[MERGE] {f} -> {outp}")
                    os.remove(f)
                except Exception as e:
                    print(f"[MERGE ERROR] {f}: {e}")

    # list labels and class counts
    labels = find_classes(DATA_DIR)
    if not labels:
        print("No labeled data found in", DATA_DIR)
        return
    global LABELS
    LABELS = labels
    counts = compute_class_counts(DATA_DIR)
    print("[DATA] Class counts:", counts)

    # safety: ensure minimum samples per class
    for lab in LABELS:
        if counts.get(lab, 0) < max(MIN_SAMPLES_PER_CLASS, 3):
            print(f"[ABORT] Not enough samples for class {lab}: {counts.get(lab,0)}. Increase data and retry.")
            return

    # gather dataset roots (we only have DATA_DIR)
    all_roots = [DATA_DIR]
    # split train/val by simple file split
    dataset = KeypointSeqDataset(all_roots, seq_len=args.seq_len, augment=args.augment)
    total = len(dataset)
    if total == 0:
        print("No sequences found in dataset.")
        return

    # train/val split
    idxs = list(range(total))
    random.shuffle(idxs)
    split = int(total * 0.85)
    train_idx = idxs[:split]
    val_idx = idxs[split:]
    from torch.utils.data import Subset
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_batch, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[TRAIN] Device:", device)
    model, best_val = train_model(train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)

    print("[TRAIN] Best val acc:", best_val)

    if args.dry_run:
        print("[DRY RUN] Done.")
        return

    # export model atomically: save state_dict then torch.jit.trace
    tmpfile = TMP_MODEL
    try:
        torch.save(model.state_dict(), tmpfile)
        # load into a model wrapper for exporting
        export_model = GRUActionNet(input_dim=17*3, hid=128, num_classes=len(LABELS)).to(device)
        export_model.load_state_dict(torch.load(tmpfile, map_location=device))
        export_model.eval()
        # export script module using example input
        example = torch.randn(1, args.seq_len, 17, 3).to(device)
        traced = torch.jit.trace(export_model, example)
        tmp_ts = MODEL_TS_OUT + ".tmp"
        traced.save(tmp_ts)
        # atomic replace
        os.replace(tmpfile, MODEL_OUT)
        os.replace(tmp_ts, MODEL_TS_OUT)
        print("[SAVE] Model saved:", MODEL_OUT, "and", MODEL_TS_OUT)
    except Exception as e:
        print("[SAVE ERROR]", e)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--merge-buffer", action="store_true", help="Move autotrain buffer samples into dataset before training")
    p.add_argument("--dry-run", action="store_true", help="Don't save model, just run training.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--augment", action="store_true", help="Use augmentation during training")
    args = p.parse_args()
    main(args)

