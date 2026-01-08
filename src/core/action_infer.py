# src/core/action_infer.py
import torch, numpy as np
from src.core.train_cctv_action_model import CCTVActionNet

ACTION_LABELS = None

def load_model(weights="models/cctv_action_net.pt", device=None, classes=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = CCTVActionNet(in_channels=3, seq_len=16, num_joints=17, hidden=128, nclasses=len(classes))
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device

def predict_sequence(model, device, seq):
    # seq: list/np.array of shape (T,17,3) normalized
    arr = np.array(seq, dtype=np.float32)[:16]
    if arr.shape[0] < 16:
        pad = np.repeat(arr[-1:,...], 16-arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    arr = np.transpose(arr, (2,0,1))  # (3,T,17)
    x = torch.from_numpy(arr).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return idx, float(probs[idx]), probs

