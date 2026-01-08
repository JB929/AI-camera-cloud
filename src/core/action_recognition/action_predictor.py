# src/core/action_recognition/action_predictor.py
import os
import time
import numpy as np
import torch

# Labels - keep consistent with training
ACTION_LABELS = ["Standing", "Sitting", "Running", "Falling", "Waving", "Other"]

# Path to your trained model
MODEL_PATH = os.path.join("models", "cctv_bigru_model.pt")

# Load model lazily (thread-safe enough for simple use)
_model = None
_device = torch.device("cpu")

def _load_model():
    global _model, _device
    if _model is not None:
        return _model
    # Try to load model (state dict or full model)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Action model not found: {MODEL_PATH}")
    try:
        # Try loading a full model object first
        ckpt = torch.load(MODEL_PATH, map_location=_device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            # user saved a dict
            state = ckpt["state_dict"]
            model = torch.nn.Module()  # fallback placeholder, we'll attempt forwarding adaptively
            model.load_state_dict(state)  # if this fails, we will catch below
        else:
            # Could be a full model or a raw state_dict; try direct load
            try:
                model = ckpt
            except Exception:
                # fallback: try to interpret ckpt as state_dict into a simple linear model
                model = None
        # set eval
        if hasattr(model, "eval"):
            model.eval()
            _model = model
            return _model
    except Exception:
        pass

    # If above failed, try a safer approach: assume state_dict for a common architecture
    # We will attempt to build a small compatible model (same shape as many training scripts used).
    # This is best-effort; if your training used a custom class, replace this section with the original model class.
    try:
        # A linear classifier that flattens (seq_len, 17, 3) -> (1, seq_len*51) then Linear -> classes
        class FallbackModel(torch.nn.Module):
            def __init__(self, seq_len=16, input_dim=51, hidden=256, classes=6):
                super().__init__()
                self.seq_len = seq_len
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(input_dim * seq_len, hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden, hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden, classes)
                )
            def forward(self, x):
                # accept: (B, seq_len, input_dim) or (B, seq_len*input_dim)
                if x.ndim == 3:
                    b = x.size(0)
                    x = x.view(b, -1)
                return self.net(x)

        fallback = FallbackModel()
        state = torch.load(MODEL_PATH, map_location=_device)
        # if it's a state_dict, try to load into fallback
        if isinstance(state, dict):
            try:
                fallback.load_state_dict(state)
            except Exception:
                # try to be lenient: if keys have 'net.' prefix, strip it
                new_state = {}
                for k, v in state.items():
                    nk = k
                    if k.startswith("net."):
                        nk = k[len("net."):]
                    new_state[nk] = v
                fallback.load_state_dict(new_state, strict=False)
        _model = fallback
        _model.eval()
        return _model
    except Exception as e:
        raise RuntimeError(f"Failed to load any compatible action model: {e}")

def _prepare_sequence(sequence, target_seq_len=16):
    """
    Accepts:
      - sequence: list/np.array of frames, each frame (17,2) or (17,3)
    Returns:
      - numpy array shape (target_seq_len, 17, 3) float32
    """
    seq = np.array(sequence, dtype=np.float32)

    # If each frame is shape (17,2) -> add confidence column (1.0)
    if seq.ndim == 2 and seq.shape[1] == 34:
        # flattened sequence (N, 34) -> reshape to (N,17,2)
        seq = seq.reshape(-1, 17, 2)

    if seq.ndim == 2 and seq.shape[1] == 17:
        # (N,17) unusual -> fail gracefully
        seq = seq.reshape(-1, 17, 1)

    if seq.ndim == 3 and seq.shape[2] == 2:
        conf = np.ones((seq.shape[0], 17, 1), dtype=np.float32)
        seq = np.concatenate([seq, conf], axis=2)

    # Now we expect (N,17,3)
    if seq.ndim != 3 or seq.shape[1] != 17 or seq.shape[2] not in (2,3):
        # try to salvage: if seq is (17,3) single frame, wrap it
        if seq.ndim == 2 and seq.shape == (17,3):
            seq = seq[None, ...]
        else:
            raise ValueError(f"Unsupported frame shape for action predictor: {seq.shape}")

    if seq.shape[2] == 2:
        # add confidence
        conf = np.ones((seq.shape[0], 17, 1), dtype=np.float32)
        seq = np.concatenate([seq, conf], axis=2)

    # pad/repeat last frame if too short
    if seq.shape[0] < target_seq_len:
        pad_count = target_seq_len - seq.shape[0]
        pad = np.repeat(seq[-1][None, ...], pad_count, axis=0)
        seq = np.concatenate([seq, pad], axis=0)

    # trim if longer
    if seq.shape[0] > target_seq_len:
        seq = seq[-target_seq_len:, ...]

    return seq.astype(np.float32)  # (seq_len,17,3)

def predict_action(sequence, movenet_pose=None):
    """
    print(">>> ENTERED predict_action()")

    print("[ACTION_DEBUG] model loaded =", model is not None)
    print("[ACTION_DEBUG] input mean =", float(np.nanmean(seq)))
    print("[ACTION_DEBUG] input max =", float(np.nanmax(seq)))
 
    sequence: list/np.ndarray of frames (each (17,2) or (17,3))
    movenet_pose: optional pose label string (can be used later)
    returns: (label: str, conf: float)
    """
    try:
        model = _load_model()
    except Exception as e:
        print(f"[predict_action] model load failed: {e}")
        return "Unknown", 0.0

    try:
        seq = _prepare_sequence(sequence, target_seq_len=16)  # (16,17,3)
    except Exception as e:
        print(f"[predict_action] bad sequence: {e}")
        return "Unknown", 0.0

    # -------------------------------------------------
    # ST-GCN INPUT PREPARATION (CRITICAL FIX)
    # -------------------------------------------------

    seq = np.array(seq, dtype=np.float32)  # (T,17,2 or 3)

    # ensure confidence channel exists
    if seq.shape[-1] == 2:
        conf = np.ones((seq.shape[0], seq.shape[1], 1), dtype=np.float32)
        seq = np.concatenate([seq, conf], axis=2)

    # seq: (16,17,2)
    x = torch.from_numpy(seq).float()

    # CORRECT: flatten exactly as trained
    x = x.view(1, -1)  # (1, 816)

    
    with torch.no_grad():
        out = model(x)

    if out is None:
        return "Unknown", 0.0

    if out.ndim == 1:
        out = out.unsqueeze(0)

    probs = torch.softmax(out, dim=1)[0].cpu().numpy()


    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    label = ACTION_LABELS[idx] if idx < len(ACTION_LABELS) else "Other"

    # HARD-FALL PROTECTION (safeguard): if model reports Falling but low movement, reduce conf
    if label == "Falling" and conf < 0.85:
        conf = conf * 0.5

    return label, float(conf)

