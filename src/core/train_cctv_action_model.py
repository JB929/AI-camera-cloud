# src/core/train_cctv_action_model.py
import os
import random
import math
import argparse
from glob import glob
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# -----------------------
# Configuration defaults
# -----------------------
DEFAULT_SEQ_LEN = 16
NUM_JOINTS = 17
IN_CH = 3  # x,y,conf

# -----------------------
# Dataset
# -----------------------
def list_samples(root):
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    samples = []
    for idx, cls in enumerate(classes):
        for fn in sorted(os.listdir(os.path.join(root, cls))):
            if fn.endswith(".npy"):
                samples.append((os.path.join(root, cls, fn), idx, cls))
    return samples, classes

def random_rotate_translate_scale(kpts, rot_deg=10, translate_pix=0.02, scale_jitter=0.05):
    # kpts: (T,17,3) normalized
    T, J, C = kpts.shape
    out = kpts.copy()
    # center at 0.5,0.5
    cx, cy = 0.5, 0.5
    theta = np.deg2rad(random.uniform(-rot_deg, rot_deg))
    s = 1.0 + random.uniform(-scale_jitter, scale_jitter)
    tx = random.uniform(-translate_pix, translate_pix)
    ty = random.uniform(-translate_pix, translate_pix)
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta),  math.cos(theta)]], dtype=np.float32) * s
    for t in range(T):
        xy = out[t,:,:2] - np.array([cx,cy],dtype=np.float32)
        xy = (xy @ R.T) + np.array([cx,cy],dtype=np.float32) + np.array([tx,ty],dtype=np.float32)
        out[t,:,:2] = xy
    return out

def random_dropout_joints(kpts, p=0.05):
    out = kpts.copy()
    T,J,C = out.shape
    for t in range(T):
        for j in range(J):
            if random.random() < p:
                out[t,j,:] = 0.0
    return out

class CCTVSequenceDataset(Dataset):
    def __init__(self, samples, classes, seq_len=DEFAULT_SEQ_LEN, augment=False):
        self.samples = samples
        self.seq_len = seq_len
        self.augment = augment
        self.classes = classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_idx, label_name = self.samples[idx]
        arr = np.load(path).astype(np.float32)  # (seq_len,17,3)
        # Safety: if loaded seq length differs, pad or crop
        if arr.shape[0] < self.seq_len:
            pad = np.repeat(arr[-1:,...], self.seq_len - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        arr = arr[:self.seq_len]

        if self.augment:
            if random.random() < 0.7:
                arr = random_rotate_translate_scale(arr, rot_deg=12, translate_pix=0.03, scale_jitter=0.08)
            if random.random() < 0.6:
                arr = random_dropout_joints(arr, p=0.06)
            # small gaussian noise on xy only
            arr[:,:,:2] += np.random.normal(0, 0.003, size=arr[:,:,:2].shape).astype(np.float32)
            arr = np.clip(arr, 0.0, 1.0)

        # Normalize: flatten to (T, J*C) or keep (C,T,J) for conv1d
        # We'll return shape (C, T, J)
        arr = np.transpose(arr, (2,0,1))  # (3, T, 17)
        return torch.from_numpy(arr).float(), label_idx

# -----------------------
# Model: TemporalConv + BiGRU + MLP head
# -----------------------
class TemporalConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.1):
        super().__init__()
        pad = kernel_size//2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size,1), padding=(pad,0)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    def forward(self,x): return self.net(x)

class CCTVActionNet(nn.Module):
    def __init__(self, in_channels=3, seq_len=16, num_joints=17, hidden=128, nclasses=6):
        super().__init__()
        # Conv layers operate on (B, C, T, J)
        self.tconv1 = TemporalConvBlock(in_channels, 64, kernel_size=3)
        self.tconv2 = TemporalConvBlock(64, 128, kernel_size=3)
        # collapse joint dim via 1x1 conv
        self.collapse = nn.Conv2d(128, 128, kernel_size=(1,1))
        # GRU expects (B, T, features)
        self.gru = nn.GRU(input_size=128*num_joints, hidden_size=hidden, num_layers=1, bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, nclasses)
        )

    def forward(self, x):
        # x: (B, C, T, J)
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.collapse(x)     # (B,128,T,J)
        B, C, T, J = x.shape
        x = x.permute(0,2,1,3).reshape(B, T, C*J)  # (B,T, C*J)
        out, _ = self.gru(x)     # (B,T, 2*hidden)
        pooled = out.mean(dim=1) # temporal mean pooling
        logits = self.classifier(pooled)
        return logits

# -----------------------
# Utilities
# -----------------------
def collate_fn(batch):
    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]
    X = torch.stack(xs, dim=0)
    Y = torch.tensor(ys, dtype=torch.long)
    return X, Y

def train_loop(model, device, train_loader, val_loader, epochs, lr, save_path, classes):
    opt = optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.to(device)
    best_val = 0.0
    for ep in range(1, epochs+1):
        model.train()
        total, correct = 0, 0
        for X,Y in train_loader:
            X = X.to(device); Y = Y.to(device)
            opt.zero_grad()
            logits = model(X)
            loss = crit(logits, Y)
            loss.backward()
            opt.step()
            preds = logits.argmax(dim=1)
            total += Y.size(0)
            correct += (preds==Y).sum().item()
        train_acc = correct/total if total>0 else 0.0

        # validation
        model.eval()
        v_total, v_correct = 0, 0
        ys_true, ys_pred = [], []
        with torch.no_grad():
            for X,Y in val_loader:
                X = X.to(device); Y = Y.to(device)
                logits = model(X)
                preds = logits.argmax(dim=1)
                v_total += Y.size(0)
                v_correct += (preds==Y).sum().item()
                ys_true += Y.cpu().tolist()
                ys_pred += preds.cpu().tolist()
        val_acc = v_correct / v_total if v_total>0 else 0.0
        print(f"Epoch {ep:02d}/{epochs}  Train Acc: {train_acc:.3f}  Val Acc: {val_acc:.3f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), save_path)
    print("Training complete. Best val:", best_val)
    # final report
    print("Validation report:")
    print(classification_report(ys_true, ys_pred, target_names=classes))

# -----------------------
# Main
# -----------------------
def main(args):
    root = args.data
    samples, classes = list_samples(root)
    print("Classes:", classes)
    print("Samples:", len(samples))
    # split
    train_s, val_s = train_test_split(samples, test_size=0.15, stratify=[s[1] for s in samples], random_state=42)
    # dataset & loaders
    train_ds = CCTVSequenceDataset(train_s, classes, seq_len=args.seq_len, augment=True)
    val_ds = CCTVSequenceDataset(val_s, classes, seq_len=args.seq_len, augment=False)
    # balanced sampler
    counts = Counter([s[1] for s in train_s])
    weights = [1.0/counts[s[1]] for s in train_s]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CCTVActionNet(in_channels=IN_CH, seq_len=args.seq_len, num_joints=NUM_JOINTS, hidden=args.hidden, nclasses=len(classes))
    print("Model params:", sum(p.numel() for p in model.parameters()))
    train_loop(model, device, train_loader, val_loader, epochs=args.epochs, lr=args.lr, save_path=args.out, classes=classes)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/cctv_raw", help="root folder with class subfolders")
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--out", default="models/cctv_action_net.pt")
    args = p.parse_args()
    main(args)

