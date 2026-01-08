import cv2
import time
import base64
import requests

CAMERA_NAME = "Front_Yard"
SOURCE = "http://192.0.0.4:8080/video"

# 🔴 TEMP: send to local server (we’ll cloudify later)
RELAY_URL = "http://127.0.0.1:8000/api/relay/frame"
print("[RELAY] Posting to:", RELAY_URL)
print("[RELAY] Starting relay sender...")
print(f"[RELAY] Camera: {CAMERA_NAME}")
print(f"[RELAY] Source: {SOURCE}")
print(f"[RELAY] Target: {RELAY_URL}")

cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    print("[RELAY ERROR] Failed to open source")
    exit(1)

print("[RELAY] Video source opened")

frame_count = 0
last_log = time.time()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[RELAY WARN] Frame grab failed")
        time.sleep(0.1)
        continue

    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if not ok:
        continue

    payload = {
        "camera": CAMERA_NAME,
        "ts": time.time(),
        "jpeg": base64.b64encode(jpg).decode("utf-8"),
    }

    try:
        requests.post(RELAY_URL, json=payload, timeout=1)
        frame_count += 1
    except Exception as e:
        print(f"[RELAY ERROR] {e}")
        time.sleep(0.2)

    # heartbeat log every 2 seconds
    if time.time() - last_log > 2:
        print(f"[RELAY] Frames sent: {frame_count}")
        last_log = time.time()
