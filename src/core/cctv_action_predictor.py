import os
import numpy as np
import torch
import torch.nn as nn

class CCTVActionGRU(nn.Module):
    def __init__(self, input_dim=51, hidden_dim=128, num_layers=1, num_classes=6):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out,_ = self.gru(x)
        last = out[:, -1, :]
        return self.fc(last)


CKPT_PATH = "models/cctv_action_gru.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=device)
    labels = ckpt["labels"]
    seq_len = ckpt["seq_len"]
    model = CCTVActionGRU(input_dim=ckpt["input_dim"], hidden_dim=128, num_layers=1, num_classes=len(labels))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

else:
    print("[WARN] Action model missing!")
    model = None
    labels = []
    seq_len = 16


def predict_cctv_action(sequence):
    if model is None:
        return "Unknown", 0.0

    seq = np.array(sequence, dtype=np.float32)

    # Fix shape: (T,17,3)
    if seq.ndim == 2 and seq.shape == (17,3):
        seq = seq[None,:,:]

    if seq.shape[1:] != (17,3):
        return "Unknown", 0.0

    T = seq.shape[0]

    # pad/crop
    if T < seq_len:
        pad = np.repeat(seq[-1][None,:,:], seq_len-T, axis=0)
        seq = np.concatenate([seq,pad],axis=0)
    elif T > seq_len:
        seq = seq[-seq_len:]

    seq = seq.reshape(seq_len, 51)

    x = torch.from_numpy(seq).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))
    return labels[idx], float(probs[idx])

