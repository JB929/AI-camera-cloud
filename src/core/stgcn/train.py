# src/core/stgcn/train.py

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import KeypointDataset
from model import STGCN_LSTM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to prepared dataset (data_ready)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--map", type=str, default="kp_map.json")
    args = p.parse_args()

    classes = ["Standing", "Sitting", "lying", "running", "waving", "falling"]

    print("[INFO] Loading dataset…")
    dataset = KeypointDataset(
        root_dir=args.data,
        classes=classes,
        seq_len=16,
        normalize=True,
        map_path=args.map,
        augment=True,
    )

    total = len(dataset)
    val_size = max(1, int(total * 0.2))
    train_size = total - val_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = STGCN_LSTM(num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)

    print("[INFO] Starting training…")

    for epoch in range(1, args.epochs + 1):
        model.train()
        correct = 0
        total_train = 0
        train_loss = 0.0

        for seq, lbl in train_loader:
            seq, lbl = seq.to(device), lbl.to(device)

            optimizer.zero_grad()
            pred = model(seq)
            loss = loss_fn(pred, lbl)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * seq.size(0)
            correct += (pred.argmax(dim=1) == lbl).sum().item()
            total_train += lbl.size(0)

        train_acc = correct / total_train
        train_loss /= total_train

        # ---- validation ----
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for seq, lbl in val_loader:
                seq, lbl = seq.to(device), lbl.to(device)

                pred = model(seq)
                loss = loss_fn(pred, lbl)

                val_loss += loss.item() * seq.size(0)
                val_correct += (pred.argmax(dim=1) == lbl).sum().item()
                val_total += lbl.size(0)

        val_acc = val_correct / val_total
        val_loss /= val_total

        print(f"Epoch {epoch:02d}/{args.epochs} "
              f"| Train Acc: {train_acc:.3f} Val Acc: {val_acc:.3f} "
              f"| Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/action_model_stgcn.pt")
            print(f"🏆 Best model saved → models/action_model_stgcn.pt")

    print(f"Training complete. Best val acc: {best_val_acc}")


if __name__ == "__main__":
    main()

