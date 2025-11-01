import cv2
import torch
import threading
import time
import requests
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import requests
camera_frames = {}

# ==============================
# ✅ CONFIGURATION
# ==============================
CAMERAS = {
    "Front_Yard": 0,     # Local webcam
    "Back_Yard": 1,      # Second USB camera (or IP camera URL)
    # Add more cameras as needed:
    # "Garage": "rtsp://192.168.0.105:554/stream"
}

CLOUD_URL = "https://ai-camera-cloud.onrender.com"
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Load YOLO model once
print("🔍 Loading YOLOv5 model...")
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
requests.get(f"{CLOUD_URL}/health", timeout=5)
print("✅ YOLO model loaded successfully!")


# ==============================
# ✅ ALERT SENDER
# ==============================
import requests
from requests.adapters import HTTPAdapter, Retry

def send_alert_to_cloud(camera_name, frame):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"Person detected by {camera_name} at {timestamp}"
    snapshot_path = f"/tmp/{camera_name}_{int(time.time())}.jpg"
    cv2.imwrite(snapshot_path, frame)

    # ✅ Resilient HTTP session
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=2,
                    status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        with open(snapshot_path, "rb") as img_file:
            files = {"snapshot": img_file}
            data = {
                "camera_name": camera_name,
                "timestamp": timestamp,
                "message": message
            }
            response = session.post(
                f"{CLOUD_URL}/api/alerts",
                data=data,
                files=files,
                timeout=30  # ⏱ longer timeout
            )
            if response.status_code == 200:
                print(f"[{camera_name}] ✅ Alert sent successfully.")
            else:
                print(f"[{camera_name}] ⚠️ Failed to send alert: {response.status_code} {response.text}")
    except requests.exceptions.Timeout:
        print(f"[{camera_name}] ⚠️ Cloud request timed out — will retry next detection.")
    except Exception as e:
        print(f"[{camera_name}] ❌ Error sending alert: {e}")
    finally:
        # Clean up temp snapshot
        try:
            os.remove(snapshot_path)
        except:
            pass


# ==============================
# ✅ CAMERA MONITOR FUNCTION
# ==============================
def monitor_camera(camera_name, source):
    print(f"[INFO] Starting camera stream: {camera_name} ({source})")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera: {camera_name}")
        return

    alert_sent_recently = False
    last_detection_time = 0
    detection_interval = 3  # seconds between detections

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[ERROR] Failed to read from {camera_name}")
            break

        camera_frames[camera_name] = frame.copy()

        # Perform detection
        results = model(frame)
        persons = results.pandas().xyxy[0]
        persons = persons[persons['name'] == 'person']

        if len(persons) > 0:
            now = time.time()
            if not alert_sent_recently or (now - last_detection_time > detection_interval):
                print(f"[{camera_name}] Person detected at {datetime.now().strftime('%H:%M:%S')}")
                send_alert_to_cloud(camera_name, frame)
                alert_sent_recently = True
                last_detection_time = now

        # Display locally (optional)
        try:
            cv2.imshow(f"Camera: {camera_name}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except Exception:
            pass

    cap.release()
    cv2.destroyAllWindows()


# ==============================
# ✅ RUN MULTIPLE CAMERAS IN THREADS
# ==============================
if __name__ == "__main__":
    print("[INFO] Multi-camera monitoring started. Press Ctrl+C to quit.")

    threads = []
    for name, source in CAMERAS.items():
        t = threading.Thread(target=monitor_camera, args=(name, source), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(1)  # small delay to avoid camera init conflict

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping all cameras...")

