import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import random

DATA_DIR = "data/cctv_raw"
LABELS = ["Standing", "Sitting", "Lying", "Running", "Waving", "Falling"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------
# DATASET
# -------------------------------------------------
class CCTVActionDataset(Dataset):
    def __init__(self, sequences, labels, augment=False):
        self.sequences = sequences
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.sequences)

    def augment_keypoints(self, seq):
        # random noise - makes model robust
        noise = np.random.normal(0, 0.01, seq.shape)
        seq = seq + noise

        # small random scaling
        scale = np.random.uniform(0.95, 1.05)
        seq = seq * scale

        return seq

    def __getitem__(self, idx):
        seq = self.sequences[idx]  # (16,17,3)
        label = self.labels[idx]

        if self.augment:
            seq = self.augment_keypoints(seq)

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# -------------------------------------------------
# MODEL — BiGRU (BIG improvement over old GRU)
# -------------------------------------------------
class BiGRUActionModel(nn.Module):
    def __init__(self, classes=6):
        super().__init__()
        self.gru = nn.GRU(
            input_size=51,   # 17×3
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, classes)
        )

    def forward(self, x):
        B, T, V, C = x.shape   # (B,16,17,3)
        x = x.reshape(B, T, V*C)  # flatten joints
        out, _ = self.gru(x)
        out = out[:, -1]  # final timestep
        return self.fc(out)


# -------------------------------------------------
# LOAD ALL DATA
# -------------------------------------------------
def load_dataset():
    all_seq = []
    all_labels = []

    for idx, class_name in enumerate(LABELS):
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            continue

        files = sorted([f for f in os.listdir(class_dir) if f.endswith(".npy")])
        print(f"[LOAD] {class_name}: {len(files)} samples")

        for f in files:
            arr = np.load(os.path.join(class_dir, f))
            if arr.shape == (16, 17, 3):
                all_seq.append(arr)
                all_labels.append(idx)

    return np.array(all_seq), np.array(all_labels)


# -------------------------------------------------
# TRAINING
# -------------------------------------------------
def train():
    sequences, labels = load_dataset()

    # balance dataset using oversampling
    count = Counter(labels)
    print("\n[COUNTS] Before balancing:", count)

    max_count = max(count.values())
    balanced_data = []
    balanced_labels = []

    for cls in range(6):
        cls_data = sequences[labels == cls]
        if len(cls_data) == 0:
            continue

        repeats = max_count // len(cls_data) + 1
        extended = np.tile(cls_data, (repeats,1,1,1))[:max_count]

        balanced_data.append(extended)
        balanced_labels += [cls] * max_count

    sequences = np.concatenate(balanced_data)
    labels = np.array(balanced_labels)

    print("[COUNTS] After balancing:", Counter(labels))

    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.15, stratify=labels, random_state=42
    )

    train_dataset = CCTVActionDataset(X_train, y_train, augment=True)
    val_dataset = CCTVActionDataset(X_val, y_val, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = BiGRUActionModel(classes=6).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0

    print("\n[TRAIN] Starting training...\n")
    for epoch in range(1, 61):

        model.train()
        correct, total = 0, 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            opt.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += len(y)

        train_acc = correct / total

        # ---- validation ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                out = model(X)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += len(y)

        val_acc = correct / total

        print(f"Epoch {epoch:02d}/60 | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "models/cctv_bigru_model.pt")
            print("🔥 Saved new best model!")

    print("\n🎉 Training complete")
    print(f"🏆 Best Validation Accuracy: {best_val:.3f}")


if __name__ == "__main__":
    train()

