import cv2
from ultralytics import YOLO
import requests
import threading
import time
import os

# === CONFIG ===
MODEL_PATH = "yolov8n.pt"
CAMERA_INDEX = 0  # change if using external webcam
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
DETECTION_ZONE = (100, 100, 500, 400)  # x1, y1, x2, y2
OVERLAP_THRESHOLD = 0.2  # trigger if 20% of person inside zone

# === SETUP ===
print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAMERA_INDEX)
os.makedirs("snapshots", exist_ok=True)

def send_telegram_alert(frame):
    timestamp = int(time.time())
    snapshot_path = f"snapshots/snapshot_{timestamp}.jpg"
    cv2.imwrite(snapshot_path, frame)
    files = {"photo": open(snapshot_path, "rb")}
    data = {"chat_id": CHAT_ID, "caption": "🚨 Person detected inside zone!"}
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        requests.post(url, data=data, files=files)
        print(f"✅ Telegram alert sent at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"[ERROR] Telegram failed: {e}")

print("[INFO] Starting camera...")
if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame)[0]
    alert_triggered = False

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        label = model.names[int(cls)]
        if label != "person":
            continue

        bx1, by1, bx2, by2 = map(int, box)
        zx1, zy1, zx2, zy2 = DETECTION_ZONE

        # Calculate overlap area
        ix = max(0, min(bx2, zx2) - max(bx1, zx1))
        iy = max(0, min(by2, zy2) - max(by1, zy1))
        overlap_area = ix * iy
        person_area = (bx2 - bx1) * (by2 - by1)
        overlap_ratio = overlap_area / (person_area + 1e-6)

        # Draw detection box
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}", (bx1, by1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if overlap_ratio >= OVERLAP_THRESHOLD:
            alert_triggered = True

    # Draw detection zone
    zx1, zy1, zx2, zy2 = DETECTION_ZONE
    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 0, 0), 2)

    if alert_triggered:
        threading.Thread(target=send_telegram_alert, args=(frame,), daemon=True).start()

    cv2.imshow("AI Camera Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting...")
        break

cap.release()
cv2.destroyAllWindows()

