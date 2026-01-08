# src/core/stgcn/dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset

# Load keypoint mapping if provided
def load_map(path=None):
    if path is None or not os.path.isfile(path):
        return None
    import json
    j = json.load(open(path, "r"))
    return j["map"]


def reorder_kp(seq, map_dict):
    """
    seq: (T, 17, C)
    map_dict: detector_index -> coco_index
    Returns reordered sequence in COCO order.
    """
    if map_dict is None:
        return seq  # identity mapping

    T, K, C = seq.shape
    out = np.zeros((T, 17, C), dtype=seq.dtype)
    for d_idx, c_idx in map_dict.items():
        out[:, c_idx, :] = seq[:, d_idx, :]
    return out


class KeypointDataset(Dataset):
    def __init__(self, root_dir, classes, seq_len=16, normalize=True, map_path=None, augment=False):
        self.root_dir = root_dir
        self.classes = classes
        self.seq_len = seq_len
        self.normalize = normalize
        self.augment = augment

        # Load mapping
        self.map_dict = load_map(map_path)

        # Load all files from all classes
        self.samples = []
        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for f in os.listdir(cls_dir):
                if f.endswith(".npy"):
                    self.samples.append((os.path.join(cls_dir, f), cls))

        # Build label index lookup
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls_name = self.samples[idx]
        seq = np.load(path)  # (T,17,3)

        # Reorder into COCO order
        seq = reorder_kp(seq, self.map_dict)

        # Normalize (because Z = 1 always)
        if self.normalize:
            seq[:, :, :2] = seq[:, :, :2]  # already normalized 0–1

        # Optional simple augmentation
        if self.augment:
            noise = np.random.normal(0, 0.003, seq.shape)
            seq[:, :, :2] += noise[:, :, :2]

        # Convert to torch
        seq = torch.tensor(seq, dtype=torch.float32)  # (T,17,3)
        label = torch.tensor(self.class_to_idx[cls_name])

        return seq, label

