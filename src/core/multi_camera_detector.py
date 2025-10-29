import cv2
import torch
import threading
import requests
from datetime import datetime
import time

# ✅ YOLOv5 model load
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
model.conf = 0.45  # confidence threshold
print("✅ YOLOv5 model loaded")

# ✅ Cloud endpoint
CLOUD_URL = "https://ai-camera-cloud.onrender.com/api/alerts"


# ✅ Send alert to cloud
def send_alert_to_cloud(camera_name):
    """Send a person detection alert to the cloud dashboard"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "camera_name": camera_name,
        "timestamp": timestamp,
        "message": f"Person detected by {camera_name} at {timestamp}",
    }

    try:
        response = requests.post(CLOUD_URL, data=data)
        if response.status_code == 200:
            print(f"[CLOUD] ✅ Alert from {camera_name} sent successfully.")
        else:
            print(f"[CLOUD] ⚠️ Failed to send alert: {response.status_code} {response.text}")
    except Exception as e:
        print(f"[CLOUD] ❌ Error sending alert: {e}")


# ✅ Camera monitoring function
def monitor_camera(camera_name, camera_source=0):
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"❌ Could not open camera: {camera_name}")
        return

    alert_sent_recently = False

    print(f"[INFO] Starting camera: {camera_name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Failed to grab frame from {camera_name}")
            break

        # YOLO inference
        results = model(frame)
        detections = results.pandas().xyxy[0]
        persons = detections[detections["name"] == "person"]

        # ✅ Trigger alert when a person is detected
        if len(persons) > 0 and not alert_sent_recently:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{camera_name}] Person detected at {timestamp}")
            send_alert_to_cloud(camera_name)  # ✅ Correct call
            alert_sent_recently = True

            # Reset alert cooldown
            threading.Timer(10, lambda: globals().update({"alert_sent_recently": False})).start()

        # Optional: display frame (comment out if headless)
        # cv2.imshow(f"Camera: {camera_name}", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    cap.release()
    cv2.destroyAllWindows()


# ✅ Start multi-camera monitoring
if __name__ == "__main__":
    print("[INFO] Multi-camera monitoring started. Press Ctrl+C to quit.")
    cameras = {"Front_Yard": 0}  # add more sources if needed

    threads = []
    for name, src in cameras.items():
        t = threading.Thread(target=monitor_camera, args=(name, src))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

