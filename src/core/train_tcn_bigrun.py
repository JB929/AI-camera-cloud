# src/core/train_tcn_bigrun.py
import os, glob, random, numpy as np, argparse, time
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ------------------------
# Dataset
# ------------------------
class PoseSeqDataset(Dataset):
    def __init__(self, root, seq_len=16, classes=None, transform=None):
        self.root = root
        self.seq_len = seq_len
        self.samples = []  # list of (path, class_idx)
        self.classes = sorted([d for d in os.listdir(root) if not d.startswith(".") and os.path.isdir(os.path.join(root, d))])
        if classes:
            self.classes = classes
        for i, cls in enumerate(self.classes):
            p = os.path.join(root, cls)
            files = [f for f in os.listdir(p) if f.endswith(".npy")]
            for f in files:
                self.samples.append((os.path.join(p, f), i))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = np.load(path).astype(np.float32)  # (seq, 17, 3)
        # safety checks
        if arr.shape != (self.seq_len, 17, 3):
            # try to fix quickly
            if arr.ndim == 3 and arr.shape[1] == 17 and arr.shape[2] == 2:
                conf = np.ones((arr.shape[0], arr.shape[1], 1), dtype=np.float32)
                arr = np.concatenate([arr, conf], axis=2)
            arr = arr[:self.seq_len]
            if arr.shape[0] < self.seq_len:
                pad = np.repeat(arr[-1][None,:,:], self.seq_len - arr.shape[0], axis=0)
                arr = np.concatenate([arr, pad], axis=0)
        # features: use x,y,conf -> flatten to (seq_len, 51)
        x = arr[:, :, :3].reshape(self.seq_len, -1)  # (seq_len, 51)
        if self.transform:
            x = self.transform(x)
        # convert to torch
        return torch.from_numpy(x).float(), int(label)

# ------------------------
# Model: small TCN + BiGRU
# ------------------------
class TemporalConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm1d(out_ch),
            nn.Dropout(dropout)
        )
    def forward(self, x):  # x: (B, C, T)
        return self.net(x)

class TCN_BiGRU(nn.Module):
    def __init__(self, input_dim=51, tcn_channels=[64,128], gru_hidden=128, num_classes=6, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.tcn1 = TemporalConvBlock(input_dim, tcn_channels[0], kernel_size=3, dilation=1, dropout=dropout)
        self.tcn2 = TemporalConvBlock(tcn_channels[0], tcn_channels[1], kernel_size=3, dilation=2, dropout=dropout)
        self.gru = nn.GRU(tcn_channels[1], gru_hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.LayerNorm(gru_hidden*2),
            nn.Linear(gru_hidden*2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):  # x: (B, T, input_dim)
        x = x.permute(0,2,1)  # (B, C, T)
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = x.permute(0,2,1)  # (B, T, C)
        out, _ = self.gru(x)  # (B, T, 2H)
        out = out[:, -1, :]   # last timestep
        return self.head(out)

# ------------------------
# Training utils
# ------------------------
def make_loader(root, batch_size=32, seq_len=16, train=True, sampler=None):
    ds = PoseSeqDataset(root=root, seq_len=seq_len)
    if sampler is None:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=2, pin_memory=True)
    else:
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    return loader, ds

def compute_class_weights(dataset):
    counts = Counter([label for _, label in dataset.samples])
    cls_order = dataset.classes
    weights = []
    for i in range(len(cls_order)):
        weights.append(1.0 / (counts.get(i,1)))
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum() * len(weights)
    return torch.tensor(w, dtype=torch.float32)

def train_loop(model, opt, criterion, loader, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x,y in loader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / max(1,total), correct/total if total>0 else 0.0

def eval_loop(model, criterion, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += float(loss.item()) * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / max(1,total), correct/total if total>0 else 0.0

# ------------------------
# CLI / training orchestration
# ------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare dataset (assumes you have run prepare + augment)
    train_root = os.path.join(args.data, "train")
    val_root = os.path.join(args.data, "val")
    # if val doesn't exist, we'll split
    if not os.path.isdir(val_root):
        # create temporary split
        all_root = args.data
        files = []
        classes = sorted([d for d in os.listdir(all_root) if not d.startswith(".") and os.path.isdir(os.path.join(all_root, d))])
        for cls in classes:
            p = os.path.join(all_root, cls)
            fns = [os.path.join(p, f) for f in os.listdir(p) if f.endswith(".npy")]
            random.shuffle(fns)
            k = int(0.85 * len(fns))
            train_dir = os.path.join(all_root + "_split/train", cls)
            val_dir = os.path.join(all_root + "_split/val", cls)
            os.makedirs(train_dir, exist_ok=True); os.makedirs(val_dir, exist_ok=True)
            for i,f in enumerate(fns):
                if i <= k:
                    dest = train_dir
                else:
                    dest = val_dir
                import shutil
                shutil.copy(f, dest)
        train_root = all_root + "_split/train"
        val_root = all_root + "_split/val"

    train_loader, train_ds = make_loader(train_root, batch_size=args.batch, seq_len=args.seq_len, train=True)
    val_loader, val_ds = make_loader(val_root, batch_size=args.batch, seq_len=args.seq_len, train=False)

    print("[DATA] Train samples:", len(train_ds), "Val samples:", len(val_ds))
    num_classes = len(train_ds.classes)
    model = TCN_BiGRU(input_dim=51, tcn_channels=[64,128], gru_hidden=128, num_classes=num_classes, dropout=args.dropout)
    model.to(device)

    # class weights
    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = 0.0
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        train_loss, train_acc = train_loop(model, opt, criterion, train_loader, device)
        val_loss, val_acc = eval_loop(model, criterion, val_loader, device)
        scheduler.step()
        took = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} Train Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f} Val Acc: {val_acc:.3f} | {took:.1f}s")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), args.out)
            print("🎉 Saved best ->", args.out)
    print("Best val:", best_val)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/cctv_augmented", help="root folder with class subfolders")
    p.add_argument("--out", default="models/cctv_action_tcn_gru.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.3)
    args = p.parse_args()
    main(args)

