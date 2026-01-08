import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random


# ====================================================
# ACTION LABELS
# ====================================================
ACTION_LABELS = ["Standing", "Sitting", "Running", "Falling", "Waving", "Other"]
NUM_CLASSES = len(ACTION_LABELS)
FEATURES = 51      # 17 keypoints × (x,y,confidence)


# ====================================================
# GENERATE SYNTHETIC MOVENET POSES
# ====================================================
def generate_pose(action):
    """
    Creates realistic synthetic MoveNet-style keypoints
    scaled between 0–1.
    """
    base = np.random.rand(17, 3)  # random noise base

    if action == "Standing":
        base[:, 0] = 0.45 + np.random.randn(17)*0.01
        base[:, 1] = np.linspace(0.1, 0.9, 17) + np.random.randn(17)*0.02

    elif action == "Sitting":
        base[:, 1] = np.clip(np.linspace(0.4, 0.7, 17) +
                             np.random.randn(17)*0.03, 0, 1)

    elif action == "Running":
        base[:, 1] = np.linspace(0.1, 0.9, 17)
        base[:, 0] += np.random.randn(17) * 0.1  # lateral motion

    elif action == "Falling":
        base[:, 1] = 0.9 + np.random.rand(17)*0.05  # LOW vertical alignment
        base[:, 0] += np.random.randn(17)*0.1

    elif action == "Waving":
        base[:, 1] = np.linspace(0.1, 0.9, 17)
        base[9,1] -= 0.4  # right wrist up
        base[10,1] -= 0.4 # left wrist up

    else:
        base = np.random.rand(17, 3)

    return np.clip(base, 0, 1)


# ====================================================
# DATASET CLASS
# ====================================================
class MoveNetDataset(Dataset):
    def __init__(self, size=5000):
        self.x = []
        self.y = []
        for _ in range(size):
            action = random.choice(ACTION_LABELS)
            kp = generate_pose(action).flatten()
            self.x.append(kp)
            self.y.append(ACTION_LABELS.index(action))

        self.x = torch.tensor(np.array(self.x), dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ====================================================
# MODEL DEFINITION
# ====================================================
class MoveNetActionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURES, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)


# ====================================================
# TRAINING LOOP
# ====================================================
def train():
    dataset = MoveNetDataset(size=8000)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = MoveNetActionNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("🚀 Training MoveNet Action Model...")

    for epoch in range(20):
        total_loss = 0
        correct = 0
        for x, y in loader:
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (pred.argmax(1) == y).sum().item()

        acc = correct / len(dataset)
        print(f"Epoch {epoch+1}/20  Loss={total_loss:.4f}  Acc={acc:.3f}")

    torch.save(model.state_dict(), "action_model_movenet.pt")
    print("\n🎉 Training done!")
    print("📦 Saved → action_model_movenet.pt")


if __name__ == "__main__":
    train()

