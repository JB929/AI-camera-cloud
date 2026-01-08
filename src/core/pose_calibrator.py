# -----------------------------
# Camera-specific learned pose classifier (centroid-based)
# -----------------------------
import os
import time
import glob
import json
from typing import Dict
import numpy as np

POSE_CENTROID_PATH = "pose_centroids.json"

def _extract_features_from_kpts(kpts, frame_height):
    """Return vector of normalized features from one keypoint array (17,3).
    Features:
      - torso_v_norm (shoulder_y - hip_y) / frame_h
      - hip_to_knee_n
      - hip_to_ankle_n
      - body_width / body_height
    """
    if kpts is None:
        return None
    k = np.array(kpts, dtype=np.float32)
    if k.shape[0] < 17:
        return None
    # keypoints indices assumed COCO order
    L_sh = k[5]; R_sh = k[6]; L_hp = k[11]; R_hp = k[12]
    L_k = k[13]; R_k = k[14]; L_a = k[15]; R_a = k[16]

    mid_sh_y = (L_sh[1] + R_sh[1]) / 2.0
    mid_hp_y = (L_hp[1] + R_hp[1]) / 2.0
    vertical = abs(mid_sh_y - mid_hp_y)
    torso_v_norm = vertical / (frame_height + 1e-9)

    hip_y = (L_hp[1] + R_hp[1]) / 2.0
    knee_y = (L_k[1] + R_k[1]) / 2.0
    ankle_y = (L_a[1] + R_a[1]) / 2.0
    hip_to_knee_n = abs(hip_y - knee_y) / (frame_height + 1e-9)
    hip_to_ankle_n = abs(hip_y - ankle_y) / (frame_height + 1e-9)

    body_w = (np.nanmax(k[:, 0]) - np.nanmin(k[:, 0])) + 1e-6
    body_h = (np.nanmax(k[:, 1]) - np.nanmin(k[:, 1])) + 1e-6
    body_ratio = float(body_w / body_h)

    return np.array([torso_v_norm, hip_to_knee_n, hip_to_ankle_n, body_ratio], dtype=np.float32)


def calibrate_pose_centroids(data_root="data/cctv_raw", min_per_class=20):
    """
    Scan data_root/<class>/*.npy sequences (each seq shape (T,17,3))
    Compute per-sequence median features, aggregate per-class centroid.
    Save centroids to POSE_CENTROID_PATH JSON.
    Returns dict of class->centroid and counts.
    """
    classes = {}
    counts = {}
    # iterate class folders
    for cls_dir in glob.glob(os.path.join(data_root, "*")):
        if not os.path.isdir(cls_dir):
            continue
        cls_name = os.path.basename(cls_dir)
        feats = []
        for p in glob.glob(os.path.join(cls_dir, "*.npy")):
            try:
                seq = np.load(p)
                # seq shape can be (T,17,3) or (T,17,2) etc.
                # compute median frame features (reduce noisy frames)
                if seq.ndim == 3 and seq.shape[1] >= 17:
                    # choose median over frames to represent sequence
                    med = np.nanmedian(seq, axis=0)  # (17,3)
                    # If coords normalized (0..1) we need frame_height guess -> use 1.0 for normalized
                    # We'll keep 'frame_h' unknown; calibration assumes pixel coords if >1
                    # We'll compute features in normalized space if values <=1
                    # Detect whether coords look normalized (max <=1.01)
                    maxxy = np.nanmax(med[:, :2])
                    if maxxy <= 1.01:
                        # normalized -> pass frame_height = 1.0
                        f = _extract_features_from_kpts(med, frame_height=1.0)
                    else:
                        # pixel coords -> normalize by a surrogate frame height: use max Y - min Y
                        # approximate frame height as median body height * 3 (heuristic)
                        approx_frame_h = (np.nanmax(med[:,1]) - np.nanmin(med[:,1])) * 3.0 + 1.0
                        f = _extract_features_from_kpts(med, frame_height=approx_frame_h)
                    if f is not None and not np.any(np.isnan(f)):
                        feats.append(f)
            except Exception:
                continue
        if len(feats) > 0:
            classes[cls_name] = np.stack(feats, axis=0)
            counts[cls_name] = classes[cls_name].shape[0]

    centroids = {}
    usable_counts = {}
    for cls, arr in classes.items():
        if arr.shape[0] >= min_per_class:
            centroid = np.median(arr, axis=0).tolist()
            centroids[cls] = centroid
            usable_counts[cls] = int(arr.shape[0])
        else:
            # still compute centroid but mark low-count
            centroid = np.median(arr, axis=0).tolist()
            centroids[cls] = centroid
            usable_counts[cls] = int(arr.shape[0])

    meta = {"centroids": centroids, "counts": usable_counts, "generated_at": time.time()}
    try:
        with open(POSE_CENTROID_PATH, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    safe_print(f"[CALIBRATE] Pose centroids saved: {POSE_CENTROID_PATH} (classes: {list(centroids.keys())})")
    return meta


# load centroids (if exists)
_pose_centroids = None
def load_pose_centroids(path=POSE_CENTROID_PATH):
    global _pose_centroids
    try:
        if _pose_centroids is not None:
            return _pose_centroids
        if os.path.exists(path):
            j = json.load(open(path, "r"))
            _pose_centroids = j
            return j
    except Exception:
        pass
    return None


def classify_by_centroid(kpts, frame_height, fallback_fn=None):
    """
    Use centroids to classify single kpts -> label, confidence
    Returns (label, conf) where conf in [0..1] (higher = closer)
    """
    meta = load_pose_centroids()
    if meta is None or "centroids" not in meta:
        # no centroids available
        if fallback_fn is not None:
            return fallback_fn(kpts, frame_height), 0.0
        return "Unknown", 0.0

    feat = _extract_features_from_kpts(kpts, frame_height)
    if feat is None:
        if fallback_fn is not None:
            return fallback_fn(kpts, frame_height), 0.0
        return "Unknown", 0.0

    # compute distances to centroids
    best = None
    bestd = float("inf")
    for cls, arr in meta["centroids"].items():
        c = np.array(arr, dtype=np.float32)
        d = float(np.linalg.norm(feat - c))
        if d < bestd:
            bestd = d
            best = cls

    # convert distance to confidence: smaller distance → higher conf
    # we scale by a heuristically-chosen gamma
    gamma = 0.5
    conf = float(np.exp(-bestd / (gamma + 1e-6)))

    return best, conf
