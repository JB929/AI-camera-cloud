import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

CLASSES = ["Standing", "Sitting", "Running", "Falling", "Lying", "Waving"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# ----------------------------------------------------
# DATASET
# ----------------------------------------------------
class KeypointSequenceDataset(Dataset):
    def __init__(self, root, seq_len=16):
        self.samples = []   # list of (np file path, label_idx)
        self.seq_len = int(seq_len)

        for label in CLASSES:
            label_dir = os.path.join(root, label)
            if not os.path.isdir(label_dir):
                continue

            for fn in os.listdir(label_dir):
                if fn.endswith(".npy"):
                    full = os.path.join(label_dir, fn)
                    self.samples.append((full, CLASS_TO_IDX[label]))

        print(f"📦 Loaded {len(self.samples)} total samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        seq = np.load(path)   # expected shape: (SEQ, 17, 3)

        # Fix: ensure correct numeric type
        seq = seq.astype(np.float32)

        # Some sequences may be shorter → pad
        if seq.shape[0] < self.seq_len:
            pad_len = self.seq_len - seq.shape[0]
            pad = np.zeros((pad_len, seq.shape[1], seq.shape[2]), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)
        else:
            seq = seq[:self.seq_len]

        # Flatten (17×3 = 51)
        seq = seq.reshape(self.seq_len, -1)  # (seq_len, 51)

        return torch.tensor(seq), torch.tensor(label)


# ----------------------------------------------------
# MODEL
# ----------------------------------------------------
class ActionModel(nn.Module):
    def __init__(self, input_dim=51, hidden=128, classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 16, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, classes)
        )

    def forward(self, x):
        # x: (B, seq_len, 51)
        x = x.reshape(x.size(0), -1)  # flatten sequence
        return self.net(x)


# ----------------------------------------------------
# TRAINING
# ----------------------------------------------------
def train(data_root, epochs, batch, seq_len, out_file):
    ds = KeypointSequenceDataset(data_root, seq_len=seq_len)

    train_idx, val_idx = train_test_split(
        np.arange(len(ds)), test_size=0.15, shuffle=True, random_state=42
    )

    train_loader = DataLoader(torch.utils.data.Subset(ds, train_idx),
                              batch_size=batch,
                              shuffle=True)

    val_loader = DataLoader(torch.utils.data.Subset(ds, val_idx),
                            batch_size=batch)

    model = ActionModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
        model.train()
        train_correct = train_total = 0
        for x, y in train_loader:
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

            train_correct += (pred.argmax(1) == y).sum().item()
            train_total += y.size(0)

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                val_correct += (pred.argmax(1) == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch {ep}/{epochs}  Train Acc: {train_acc:.3f}  Val Acc: {val_acc:.3f}")

        torch.save(model.state_dict(), out_file)

    print("🎉 Training complete. Saved →", out_file)


# ----------------------------------------------------
# MAIN ENTRY
# ----------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--out", default="action_model_movenet_ultra.pt")
    args = p.parse_args()

    train(args.data, args.epochs, args.batch, args.seq_len, args.out)

