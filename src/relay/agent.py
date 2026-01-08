# src/relay/agent.py
import cv2
import time
import requests
import socket

CLOUD_URL = "http://localhost:8000"  # later → cloud domain
CAMERA_NAME = "Front_Yard"
FRAME_INTERVAL = 2.0  # seconds

def main():
    cap = cv2.VideoCapture(0)  # webcam / RTSP / phone stream

    if not cap.isOpened():
        print("[AGENT ERROR] Cannot open camera")
        return

    print("[AGENT] Relay started")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[AGENT WARN] Frame read failed")
            time.sleep(1)
            continue

        _, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

        try:
            res = requests.post(
                f"{CLOUD_URL}/ingest/frame",
                files={"frame": jpg.tobytes()},
                data={
                    "camera": CAMERA_NAME,
                    "host": socket.gethostname()
                },
                timeout=5
            )
            if res.status_code != 200:
                print("[AGENT WARN] Upload failed:", res.status_code)

        except Exception as e:
            print("[AGENT ERROR]", e)

        time.sleep(FRAME_INTERVAL)

if __name__ == "__main__":
    main()
