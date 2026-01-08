# src/core/collect_cctv_keypoints.py

import os
import time
import argparse
import numpy as np
import cv2

# Use your YOLO pose detector
from src.core.yolov8_pose_detector import detect_pose_yolo as detect_pose


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def open_camera(source):
    """Handle both webcam index (0, 1, ...) and RTSP/HTTP URL"""
    try:
        src = int(source)
        return cv2.VideoCapture(src)
    except ValueError:
        return cv2.VideoCapture(source)


def capture_cctv_keypoints(
    camera_source,
    label,
    out_dir="data/cctv_raw",
    duration_sec=60,
    fps=8,
    seq_len=16,
):
    """
    Records keypoint sequences from a CCTV camera / webcam.

    Saves them as:
        data/cctv_raw/<label>/<label>_00001.npy
    """

    label_dir = os.path.join(out_dir, label)
    ensure_dir(label_dir)

    cap = open_camera(camera_source)
    if not cap.isOpened():
        print(f"❌ Could not open camera source: {camera_source}")
        return

    print(f"🎥 Source: {camera_source}")
    print(f"🏷️ Label: {label}")
    print(f"⏱️ Duration: {duration_sec}s, FPS: {fps}, seq_len: {seq_len}")

    # ------------------------------------------
    # NEW: continue sequence numbering
    # ------------------------------------------
    existing = sorted([f for f in os.listdir(label_dir) if f.endswith(".npy")])
    if existing:
        try:
            last = existing[-1]
            seq_idx = int(last.split("_")[-1].split(".")[0]) + 1
        except Exception:
            seq_idx = len(existing)
    else:
        seq_idx = 0

    buffer = []
    start_time = time.time()
    last_capture = 0
    interval = 1.0 / fps

    while time.time() - start_time < duration_sec:

        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.03)
            continue

        now = time.time()
        if now - last_capture < interval:
            continue
        last_capture = now

        # -----------------------
        # YOLO pose detection
        # -----------------------
        k = detect_pose(frame)
        if k is None:
            print("⚠️ NO KEYPOINTS DETECTED")
            continue

        k = np.array(k, dtype=np.float32)  # shape (17,3)

        # Normalization
        h, w = frame.shape[:2]
        k[:, 0] /= max(w, 1)
        k[:, 1] /= max(h, 1)

        buffer.append(k)

        # -----------------------
        # Save sequence
        # -----------------------
        if len(buffer) >= seq_len:
            seq = np.stack(buffer[-seq_len:])  # (seq_len, 17, 3)
            out_path = os.path.join(label_dir, f"{label}_{seq_idx:05d}.npy")
            np.save(out_path, seq)

            print(f"💾 Saved {label} sequence #{seq_idx} → {out_path}")

            seq_idx += 1
            buffer = []  # no overlap for now

    cap.release()
    print(f"✅ DONE recording label={label}")
    print(f"Total sequences saved: {seq_idx}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--camera", required=True, help="0 for webcam, or RTSP/HTTP URL")
    p.add_argument("--label", required=True, help="Standing / Sitting / Lying / Running / Waving / Falling")
    p.add_argument("--out", default="data/cctv_raw")
    p.add_argument("--duration", type=int, default=60)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=16)

    args = p.parse_args()

    capture_cctv_keypoints(
        camera_source=args.camera,
        label=args.label,
        out_dir=args.out,
        duration_sec=args.duration,
        fps=args.fps,
        seq_len=args.seq_len,
    )

