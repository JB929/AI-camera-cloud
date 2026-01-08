# src/core/collect_movenet_keypoints.py
import os
import time
import json
import argparse
import cv2
import numpy as np

from src.core.movenet_detector import movenet_detect


def ensure(path):
    os.makedirs(path, exist_ok=True)


def save_sequence(out_dir, label, seq_idx, sequences):
    fn = os.path.join(out_dir, label, f"{label}_{seq_idx:05d}.npy")
    np.save(fn, sequences)


def open_camera(source):
    """
    Converts string "0" → camera index 0
    Otherwise uses RTSP/HTTP/FILE source
    """
    if isinstance(source, str) and source.isdigit():
        source = int(source)
        print(f"🎥 Using WEBCAM index {source}")
    else:
        print(f"🌐 Using video source: {source}")

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("❌ ERROR: Failed to open camera/video stream.")
    else:
        print("✅ Camera stream opened successfully.")
    return cap


def capture(camera_source, label, out_dir, duration_sec=60, fps=10, seq_len=32):
    ensure(os.path.join(out_dir, label))

    cap = open_camera(camera_source)
    if not cap.isOpened():
        print("❌ Could not open camera. Aborting.")
        return

    start = time.time()
    seq_buffer = []
    seq_idx = 0
    last_capture = 0
    interval = 1.0 / fps

    print(f"📸 Start capturing {label} for {duration_sec} sec")

    while time.time() - start < duration_sec:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("⚠️ Frame read failed — retrying...")
            time.sleep(0.05)
            continue

        now = time.time()
        if now - last_capture < interval:
            continue

        last_capture = now

        # 🔵 Run MoveNet
        try:
            keypoints = movenet_detect(frame)
        except Exception as e:
            print(f"⚠️ MoveNet error: {e}")
            continue

        if keypoints is None:
            print("⚠️ No keypoints — skipping frame.")
            continue

        seq_buffer.append(keypoints.astype(np.float32))

        if len(seq_buffer) >= seq_len:
            save_sequence(out_dir, label, seq_idx, np.stack(seq_buffer[-seq_len:]))
            seq_idx += 1
            seq_buffer = []
            print(f"💾 Saved {label} sample #{seq_idx}")

    cap.release()
    print(f"✅ Done collecting: {label}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--camera", default="0", help="Camera index or URL")
    p.add_argument("--label", required=True)
    p.add_argument("--out", default="data/movenet_raw")
    p.add_argument("--duration", type=int, default=60)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=16)
    args = p.parse_args()

    capture(args.camera, args.label, args.out, args.duration, args.fps, args.seq_len)

