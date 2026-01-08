import argparse
import numpy as np
import torch
import json
from model import STGCN_LSTM

def load_mapping(path):
    m = json.load(open(path))
    return m["map"], m["coco_names"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--map", default="kp_map.json")
    args = parser.parse_args()

    seq = np.load(args.seq)
    if seq.shape[2] == 2:
        conf = np.ones((seq.shape[0], seq.shape[1], 1), dtype=np.float32)
        seq = np.concatenate([seq, conf], axis=2)

    seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

    with open(args.classes) as f:
        cls = [x.strip() for x in f.readlines()]

    model = STGCN_LSTM(len(cls))
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        out = model(seq)
        probs = torch.softmax(out, dim=1)[0].numpy()

    pred = cls[np.argmax(probs)]
    print("\n📌 Prediction:", pred)
    print("🔢 Confidence:", float(np.max(probs)))
    print("📊 All class probs:")
    for c, p in zip(cls, probs):
        print(f"  {c:10s}: {p:.3f}")

if __name__ == "__main__":
    main()

