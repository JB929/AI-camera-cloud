import cv2
import torch
import numpy as np
import requests
import threading
from requests.adapters import HTTPAdapter, Retry
import time
import os
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def should_send_alert(camera_name, x, y):
    """Check if this camera should send a new alert."""
    global last_alert_time, last_detection_pos
    now = time.time()

    # Never alerted before → allow
    if camera_name not in last_alert_time:
        last_alert_time[camera_name] = 0
        last_detection_pos[camera_name] = (x, y)
        return True

    # Check cooldown time
    if now - last_alert_time[camera_name] < COOLDOWN:
        return False

    # Check movement distance (Euclidean)
    last_x, last_y = last_detection_pos[camera_name]
    dist = np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)
    if dist < MOVEMENT_THRESHOLD:
        return False  # not much movement → skip

    # Passed both checks → allow alert
    last_alert_time[camera_name] = now
    last_detection_pos[camera_name] = (x, y)
    return True

import requests
import threading
import os
import cv2
from datetime import datetime
import time

CLOUD_URL = "https://ai-camera-cloud.onrender.com"
print(f"🌐 CLOUD_URL set to: {CLOUD_URL}")

def send_alert_background(camera_name, frame, message):
    def send_alert_background(camera_name, frame, message):
        print(f"[DEBUG] send_alert_background() CALLED for {camera_name} → {message}")

    """Send alert + snapshot to the cloud reliably with synchronous save + threaded upload."""

    def task():
        snapshot_path = None
        try:
            # --- Step 1: Save snapshot (synchronously, guaranteed) ---
            os.makedirs("temp_snapshots", exist_ok=True)
            filename = f"{camera_name}_{int(time.time())}.jpg"
            snapshot_path = os.path.join("temp_snapshots", filename)

            success = cv2.imwrite(snapshot_path, frame)
            if not success:
                print(f"[{camera_name}] ❌ Failed to save snapshot image (cv2.imwrite returned False).")
                return

            # Confirm file exists
            if not os.path.exists(snapshot_path):
                print(f"[{camera_name}] ❌ Snapshot file missing after save.")
                return

            # --- Step 2: Prepare payload ---
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data = {
                "camera_name": camera_name,
                "timestamp": timestamp,
                "message": message or f"Alert from {camera_name} at {timestamp}",
            }

            # --- Step 3: Upload snapshot + alert ---
            with open(snapshot_path, "rb") as img:
                files = {"snapshot": ("snapshot.jpg", img, "image/jpeg")}
                print(f"[DEBUG] 📤 Uploading alert from {camera_name} → {CLOUD_URL}/api/alerts)...")

                response = requests.post(
                    f"{CLOUD_URL}/api/alerts",
                    data=data,
                    files=files,
                    timeout=30,
                )

            # --- Step 4: Log response ---
            print(f"[{camera_name}] 🌐 Cloud response: {response.status_code}")
            print(f"[{camera_name}] 🌐 Response body: {response.text[:300]}")

            if response.status_code == 200:
                print(f"[{camera_name}] ☁️ Alert successfully sent to cloud.")
            else:
                print(f"[{camera_name}] ⚠️ Cloud rejected alert: {response.text}")

        except Exception as e:
            print(f"[{camera_name}] ❌ Fatal error while sending alert: {e}")

        finally:
            # --- Step 5: Clean temp snapshot ---
            if snapshot_path and os.path.exists(snapshot_path):
                try:
                    os.remove(snapshot_path)
                except Exception as e:
                    print(f"[{camera_name}] ⚠️ Could not delete snapshot: {e}")

    # Start threaded upload
    threading.Thread(target=task, daemon=True).start()




import time
from datetime import datetime, timezone
import numpy as np

# Track camera activity
last_alert_time = {}
last_detection_pos = {}
COOLDOWN = 10  # seconds before same camera can alert again
MOVEMENT_THRESHOLD = 50  # pixels (distance movement before retrigger)


# ==============================
# 🧠 SMART ALERT LOGIC
# ==============================
from datetime import datetime, timedelta

# Dictionary to store last alert timestamps per camera & class
last_alert_times = {}

def should_send_alert(camera_name, class_name, cooldown=10):
    """
    Determines whether a new alert should be sent to avoid spam.
    cooldown: minimum seconds between alerts for same camera & class.
    """
    key = f"{camera_name}_{class_name}"
    now = datetime.now()

    # Check last alert time
    last_time = last_alert_times.get(key)

    if last_time is None or (now - last_time).total_seconds() > cooldown:
        last_alert_times[key] = now
        return True  # ✅ Send alert
    else:
        return False  # ⏸ Skip (still cooling down)

# ========================================
# ✅ CAMERA CONFIGURATION
# ========================================
# You can add more cameras here easily
CAMERAS = {
    "Front_Yard": 0,  # webcam index (for USB camera)
    # Example RTSP/HTTP camera URLs:
    # "Garage": "rtsp://username:password@192.168.1.50:554/stream",
    # "Backyard": "http://192.168.1.51:8080/video"
}

# ==================================
# 🌐 CLOUD CONNECTION CONFIG
# ==================================
CLOUD_URL = "https://ai-camera-cloud.onrender.com"

# ✅ Step 1: Cloud health check function
def check_cloud_health(CLOUD_URL):
    """Safely check if cloud server is online before continuing."""
    print("🌐 Checking cloud server status...")

    session = requests.Session()
    retries = Retry(total=3, backoff_factor=2, status_forcelist=[502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        response = session.get(f"{CLOUD_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Cloud server is online.")
        else:
            print(f"⚠️ Cloud responded with status {response.status_code}")
    except requests.exceptions.ReadTimeout:
        print("⚠️ Cloud health check timed out. Continuing anyway...")
    except requests.exceptions.ConnectionError:
        print("⚠️ Could not connect to cloud — will continue offline mode.")
    except Exception as e:
        print(f"❌ Unexpected error checking cloud: {e}")

# 🚀 Run the cloud connectivity test once before model load
check_cloud_health(CLOUD_URL)
# Load YOLO model once
print("🔍 Loading YOLOv5 model...")
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
requests.get(f"{CLOUD_URL}/health", timeout=5)
print("✅ YOLO model loaded successfully!")

# ==============================
# 🌐 CLOUD CONNECTION CHECK
# ==============================
import time
import requests

CLOUD_URL = "https://ai-camera-cloud.onrender.com"

def check_cloud_connection():
    """Check cloud connection with retries and long timeout."""
    for attempt in range(5):
        try:
            print(f"🌐 Checking cloud connection (attempt {attempt + 1})...")
            r = requests.get(f"{CLOUD_URL}/health", timeout=15)
            if r.status_code == 200:
                print("✅ Cloud connection OK!")
                return True
            else:
                print(f"⚠️ Unexpected response: {r.status_code}")
        except requests.exceptions.ReadTimeout:
            print("⚠️ Timeout — server took too long to respond.")
        except Exception as e:
            print(f"⚠️ Connection failed: {e}")
        time.sleep(5)  # wait before retrying
    print("❌ Could not connect to cloud after 5 attempts.")
    return False

# ✅ Run connection check at startup
if not check_cloud_connection():
    print("⚠️ Continuing without cloud check (network unstable).")

# ==============================
# ✅ ALERT SENDER
# ==============================
import requests
from requests.adapters import HTTPAdapter, Retry

# ✅ Define restricted zones per camera
ZONES = {
    "Front_Yard": [(100, 200), (400, 200), (400, 400), (100, 400)],  # example rectangle
    "Garage": [(50, 50), (500, 50), (500, 350), (50, 350)]
}

def is_inside_zone(x, y, zone_points):
    """Simple polygon check using OpenCV."""
    import cv2
    contour = np.array(zone_points, dtype=np.int32)
    return cv2.pointPolygonTest(contour, (int(x), int(y)), False) >= 0


def send_alerts_from_detections(camera_name, detections, frame):
    """Iterate detections, check zones, send alerts to cloud."""
    for _, det in detections.iterrows():
        class_name = det["name"]
        confidence = det["confidence"]
        x_center = (det["xmin"] + det["xmax"]) / 2
        y_center = (det["ymin"] + det["ymax"]) / 2

        if confidence > 0.45 and class_name == "person":
            print(f"[{camera_name}] 👀 Detected {class_name} ({confidence:.2f})")

            # ✅ Always send alert for each detection
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"{class_name.capitalize()} detected in {camera_name}"

            snapshot_path = f"/tmp/{camera_name}_{int(datetime.now().timestamp())}.jpg"
            # Save compressed snapshot to reduce upload size
            cv2.imwrite(snapshot_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            import time
            time.sleep(1.0)  # prevent back-to-back rapid uploads

            try:
                with open(snapshot_path, "rb") as img_file:
                    files = {"snapshot": img_file}
                    data = {
                        "camera_name": camera_name,
                        "timestamp": timestamp,
                        "message": message
                    }

                    response = requests.post(
                        f"{CLOUD_URL}/api/alerts",
                        data=data,
                        files=files,
                        timeout=10
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
                try:
                    os.remove(snapshot_path)
                except Exception:
                    pass



# ==============================
# ✅ CAMERA MONITOR FUNCTION
# ==============================
def monitor_camera(camera_name, camera_url):
    import cv2
    import numpy as np
    import threading
    from datetime import datetime

    cap = cv2.VideoCapture(camera_url)
    print(f"[INFO] Starting camera: {camera_name}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"[{camera_name}] ⚠️ Unable to read frame.")
            break

                # ==========================
        # 🚀 YOLOv5 Detection Block
        # ==========================
        try:
            results = model(frame)
            detections = results.pandas().xyxy[0]
            
            # === Overlay detected objects on frame ===
            for _, row in detections.iterrows():
                x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
                confidence = row["confidence"]
                label = row["name"]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Label text (e.g., "person 0.85")
                text = f"{label} {confidence:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # === End overlay block ===

         
            # ✅ Send detections to cloud
            send_alerts_from_detections(camera_name, detections, frame)

            if detections.empty:
                continue  # no detections, skip frame

            for _, det in detections.iterrows():
                class_name = det["name"]
                confidence = float(det["confidence"])

                if confidence < 0.45:
                    continue  # skip low confidence

                if class_name == "person":
                    print(f"[{camera_name}] 👀 Detected person ({confidence:.2f})")

                    # Compute center point for zone logic
                    x_center = (det["xmin"] + det["xmax"]) / 2
                    y_center = (det["ymin"] + det["ymax"]) / 2

                    # --- Check if inside restricted zone (optional)
                    if camera_name in ZONES and is_inside_zone(x_center, y_center, ZONES[camera_name]):
                        message = f"Person entered restricted area in {camera_name}"

                        if should_send_alert(camera_name, "person"):
                            print(f"[{camera_name}] 🚨 Sending alert: {message}")
                            send_alert_background(camera_name, frame, message)
                        else:
                            print(f"[{camera_name}] ⏳ Skipping duplicate alert (cooldown active).")

        except Exception as e:
            print(f"[{camera_name}] ❌ YOLO detection error: {e}")


        # 🚨 Detection and alert logic
        for _, det in detections.iterrows():
            try:
                class_name = det["name"]
                confidence = det["confidence"]
                x_center = (det["xmin"] + det["xmax"]) / 2
                y_center = (det["ymin"] + det["ymax"]) / 2

                if confidence > 0.45 and class_name == "person":
                    print(f"[{camera_name}] Detected {class_name} ({confidence:.2f})")

                    # Zone check (optional)
                    if camera_name in ZONES and is_inside_zone(x_center, y_center, ZONES[camera_name]):
                        if should_send_alert(camera_name, class_name):
                            message = f"{class_name.capitalize()} entered restricted area in {camera_name}"
                            print(f"[{camera_name}] 🚨 Sending alert: {message}")
                            send_alert_background(camera_name, frame, message)
                        else:
                            print(f"[{camera_name}] ⏳ Skipping duplicate alert (cooldown active).")

            except Exception as e:
                print(f"[{camera_name}] ❌ Detection error: {e}")

        # --- Optional Display ---
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

