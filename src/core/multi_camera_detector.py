import cv2
import torch
import requests
import time
import warnings
from datetime import datetime

# ✅ Suppress noisy YOLO FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ✅ Cloud API endpoint
CLOUD_URL = "https://ai-camera-cloud.onrender.com/api/alerts"

# ✅ Load YOLOv5 model
print("Loading YOLOv5 model...")
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
print("✅ YOLO model loaded successfully!")

# ✅ Track last alert timestamp
last_alert_time = 0

# ✅ Send detection to cloud
def send_alert_to_cloud(camera_name, frame):
    try:
        # Save snapshot temporarily
        snapshot_path = f"/tmp/{camera_name}_{int(time.time())}.jpg"
        cv2.imwrite(snapshot_path, frame)

        # Prepare payload
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"Person detected by {camera_name} at {timestamp}"

        # Send with snapshot
        with open(snapshot_path, "rb") as f:
            files = {"snapshot": (f"{camera_name}.jpg", f, "image/jpeg")}
            data = {
                "camera_name": camera_name,
                "timestamp": timestamp,
                "message": message
            }
            response = requests.post(CLOUD_URL, data=data, files=files)

        if response.status_code == 200:
            print(f"[CLOUD] ✅ Alert from {camera_name} sent successfully.")
        else:
            print(f"[CLOUD] ⚠️ Failed to send alert: {response.status_code} {response.text}")

    except Exception as e:
        print(f"[CLOUD] ❌ Error sending alert: {e}")

# ✅ Monitor camera and run detections
def monitor_camera(camera_name, camera_id=0):
    global last_alert_time
    print(f"[INFO] Starting camera: {camera_name} ({camera_id})")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_name}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[ERROR] Failed to read from {camera_name}")
            break

        # Run YOLO detection
        results = model(frame)
        persons = [det for det in results.xyxy[0] if int(det[5]) == 0]  # class 0 = person

        # Alert if a person is detected and cooldown passed
        current_time = time.time()
        if len(persons) > 0 and (current_time - last_alert_time > 10):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{camera_name}] Person detected at {timestamp}")
            send_alert_to_cloud(camera_name, frame)
            last_alert_time = current_time

        # ✅ Safe OpenCV display (doesn’t crash on Mac)
        try:
            cv2.imshow(f"Camera: {camera_name}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except cv2.error:
            pass  # Skip frame if GUI not supported

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("[INFO] Multi-camera monitoring started. Press Ctrl+C to quit.")
    monitor_camera("Front_Yard", 0)

