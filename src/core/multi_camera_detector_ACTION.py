import os
import sys
import time
import json
import threading
from collections import deque, Counter
from datetime import datetime
from collections import defaultdict
from src.core.events import event_bus
from src.core.events.event_bus import emit_event
from src.core.config import get_camera_config
from src.core.config import save_camera_config
from src.core.cameras.rtsp_builder import build_rtsp_url
from src.core.onvif.rtsp_resolver import resolve_rtsp_url as onvif_rtsp_url
from src.core.subscription.enforcement import get_subscription_state
import numpy as np
import cv2
import requests
import hashlib
AUTOLEARN_ENABLED = False
USE_CLOUD_RELAY = True

import requests
import base64

SERVER_BASE_URL = "http://127.0.0.1:8000"

from src.core.config import load_config
from src.core.config import load_config, get_camera_config
CONFIG = load_config()

from datetime import datetime, time as dtime

CONFIG_PATH = "config/runtime_config.json"

def load_runtime_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {
            "schedule": {
                "enabled": False
            }
        }

RUNTIME_CONFIG = load_runtime_config()

def is_within_schedule():
    sched = RUNTIME_CONFIG.get("schedule", {})
    if not sched.get("enabled", False):
        return True  # always active

    now = datetime.now().time()

    start = dtime.fromisoformat(sched["start"])
    end   = dtime.fromisoformat(sched["end"])

    # overnight window (e.g. 20:00 → 09:00)
    if start > end:
        return now >= start or now <= end
    else:
        return start <= now <= end

# Try to import ultralytics YOLO (v8/v9)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*torch.cuda.amp.autocast.*"
)

# Try to import torch for yolov5 via hub (optional)
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

import torch

yolo5_model = torch.hub.load(
    "src/core/models/yolov5",
    "yolov5s",
    source="local"
)
yolo5_model.eval()

# =============================
# MODEL HEALTH (Phase 15B)
# =============================
MODEL_COOLDOWN_SEC = 30

model_health = {
    "pose": {
        "ok": True,
        "last_fail": 0.0,
    },
    "yolo": {
        "ok": True,
        "last_fail": 0.0,
    },
}

# Action predictor 
from src.core.action_recognition.action_predictor import predict_action
assert predict_action.__code__.co_filename.endswith(
    "action_predictor.py"
), f"Wrong predict_action loaded: {predict_action.__code__.co_filename}"


# Person identifier (optional). Keep or stub.
try:
    from person_identifier import identify_person
except Exception:
    def identify_person(crop):
        return "Unknown"

# --------------------------
# Configuration
# --------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.cameras.rtsp_builder import build_rtsp_url
from src.core.config import get_camera_config

MODELS = {
    "pose": "models/yolov8s-pose.pt",   # path to ultralytics pose model
    "yolov5_det": None,                  # if None, we try torch.hub load
}

CLOUD_URL = os.environ.get("AI_CAMERA_CLOUD_URL", "https://ai-camera-cloud.onrender.com")
AUTOTRAIN_SAVE_DIR = "autotrain_buffer"
os.makedirs(AUTOTRAIN_SAVE_DIR, exist_ok=True)
DATA_LOG_DIR = "data/logs"
SNAPSHOT_DIR = CONFIG["paths"]["snapshots_dir"]
os.makedirs(DATA_LOG_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
final_log_last = defaultdict(float)

# Intervals & thresholds
POSE_INTERVAL = 0.6       # seconds between pose detections
ACTION_INTERVAL = 5.0     # seconds between action recognitions
SMOOTH_WINDOW = 5         # fusion smoothing window
ALERT_COOLDOWN = 15       # seconds per camera+alert_type
FALL_CONF_THRESHOLD = 0.75
MOVEMENT_THRESHOLD = 0.02  # normalized coordinate movement threshold
AUTOTRAIN_CONF_TH = 0.60
MAX_POSE_INFER_TIME = 0.35     # seconds
MAX_ACTION_INFER_TIME = 0.25
MAX_YOLO_INFER_TIME = 0.40
PRESENT_HEARTBEAT_SECONDS = 45
# -----------------------------
# EVENT DEDUPLICATION / COOLDOWN
# -----------------------------
event_last_fired = {}  
# key = (camera_name, event_type)
# value = timestamp

# -----------------------------
# MOTION THRESHOLDS (PIXEL SPACE)
# -----------------------------
MOTION_IDLE_EPS = CONFIG["motion"]["idle_eps"]
MOTION_MOVING = CONFIG["motion"]["moving_threshold"]

# -----------------------------
# PRESENCE ALERT CONFIG
# -----------------------------
ENABLE_PRESENCE_ALERT = True

PRESENCE_ALERT_COOLDOWN = CONFIG["presence"]["cooldown_seconds"]

# -----------------------------
# PHASE 2: ACTIVE TIME WINDOW
# -----------------------------

ENABLE_TIME_WINDOW = True

# 24-hour format
ACTIVE_START = "04:00"   # 4 AM
ACTIVE_END   = "21:00"   # 9 PM


# -----------------------------
# PRESENCE TRACKING (PHASE 3.5)
# -----------------------------
presence_first_seen = {}     # camera -> timestamp
presence_last_seen = {}      # camera -> timestamp
presence_alert_last = {}     # camera -> timestamp
last_seen_ts = {}
last_enter_ts = {}
last_detected_ts = {}
last_present_ts = {}
detected_fired = {}
presence_started_ts = {}
CAMERA_HEALTH = {} 
presence_state = {}
presence_started_ts = {}
last_seen_ts = {}
last_present_emit = {}
detected_fired = {}

# thresholds (seconds)
PRESENCE_EXIT_TIMEOUT = 6.0      # person gone for 6s = exit
PRESENCE_ALERT_COOLDOWN = 30.0   # no spam
LOITERING_TIME = 300.0           # 5 minutes
PRESENCE_EXIT_DELAY = 3.0  # seconds (tune 2–5)
ENTER_COOLDOWN_SEC = 10        # min seconds between ENTERs
DETECTED_COOLDOWN_SEC = 15    # min seconds between PERSON_DETECTED
MIN_PRESENCE_SEC = 3.0        # must be present this long before EXIT allowed
PRESENT_INTERVAL_SEC = 45
DETECTED_INTERVAL_SEC = 60
EXIT_ABSENCE_SEC = 2.5

# --------------------------
# CAMERA FAILURE HARDENING
# --------------------------
camera_failures = {}
camera_disabled = {}
CAMERA_MAX_FAILURES = 5
CAMERA_BACKOFF_SECONDS = 10

# --------------------------
# Global state (per-camera dicts)
# --------------------------
last_pose_time = {}
last_action_time = {}
last_fusion_status = {}
prev_keypoints = {}
action_sequence = {}
unified_history = {}
alert_last_sent = {}
last_snapshot_time = {}
display_frames = {}
action_buffer = {}
prev_keypoints_pixels = {}
motion_log_last = {}
last_presence_alert = {}
last_frame_hash = {}
freeze_counter = {}
last_present_emit = {}
last_frame_ts_seen = {}

CRITICAL_ACTIONS = {
    "Falling",
    "Collapse",
    "Fighting",
    "Attack",
}

# Load models
pose_model = None
yolo5_model = None
USE_POSE = False
USE_YOLO5 = False

if YOLO_AVAILABLE:
    pose_path = MODELS.get("pose")
    if pose_path and os.path.exists(pose_path):
        try:
            pose_model = YOLO(pose_path)
            USE_POSE = True
            print(f"[MODEL] Loaded pose model: {pose_path}")
        except Exception as e:
            print(f"[MODEL] Failed to load pose model {pose_path}: {e}")
    else:
        print(f"[MODEL] Pose model not found at {pose_path}; pose disabled.")
else:
    print("[MODEL] ultralytics YOLO not available; pose disabled.")

# try to load YOLOv5 (object detection) via torch.hub as optional
if TORCH_AVAILABLE:
    try:
        yolo5_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        yolo5_model.conf = 0.25
        USE_YOLO5 = True
        print("[MODEL] Loaded YOLOv5s (object detection) via torch.hub")
    except Exception as e:
        print(f"[MODEL] YOLOv5 load failed or skipped: {e}")
else:
    print("[MODEL] torch not available; YOLOv5 disabled.")

# --------------------------
# Utilities
# --------------------------

    def _worker():
        while True:
            try:
                for name, frm in list(display_frames.items()):
                    if frm is None:
                        continue
                    try:
                        cv2.imshow(f"Camera: {name}", frm)
                    except cv2.error:
                        pass
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Quit requested (display)")
                    os._exit(0)
                time.sleep(0.03)
            except Exception as e:
                print(f"[display_worker] error: {e}")
                time.sleep(0.5)

    t = threading.Thread(target=_worker, daemon=True, name="display_worker")
    t.start()


def detect_pose_wrapper(frame):
    """
    Robust pose extraction from ultralytics YOLO pose result.
    Returns:
      - np.array shape (17,3) with [x, y, conf] NORMALIZED to 0..1,
      - or None if no person / keypoints found.
    """
    global pose_model, USE_POSE
    if not USE_POSE or pose_model is None:
        return None

    # -----------------------------
    # SAFE YOLO POSE INFERENCE
    # -----------------------------
    if not model_health["pose"]["ok"]:
        # cooldown check
        if time.time() - model_health["pose"]["last_fail"] < MODEL_COOLDOWN_SEC:
            return None
        else:
            print("[RECOVERY] Reloading pose model...")
            try:
                pose_model = load_pose_model()  # use your existing loader
                model_health["pose"]["ok"] = True
            except Exception as e:
                print(f"[POSE RELOAD FAILED] {e}")
                model_health["pose"]["last_fail"] = time.time()
                return None

    try:
        res = pose_model(frame, verbose=False)
    except Exception as e:
        print(f"[POSE ERROR] {e}")
        model_health["pose"]["ok"] = False
        model_health["pose"]["last_fail"] = time.time()
        return None


    if not res or len(res) == 0:
        return None

    r = res[0]

    # Guarantee a Keypoints-like object exists
    if not hasattr(r, "keypoints") or r.keypoints is None:
        return None

    # Try safe access patterns in order (most robust)
    try:
        # Preferred: r.keypoints.data -> tensor (N,17,3)
        if hasattr(r.keypoints, "data"):
            kp_tensor = r.keypoints.data  # torch tensor
            kp = kp_tensor.cpu().numpy()  # (N,17,3) or (N,17,2)
            if kp is None or kp.size == 0 or kp.shape[0] == 0:
                return None
            k = kp[0].astype(np.float32)
        elif hasattr(r.keypoints, "xy"):
            arr = r.keypoints.xy  # maybe list of arrays
            if arr is None or len(arr) == 0:
                return None
            k = np.array(arr[0], dtype=np.float32)
        elif hasattr(r.keypoints, "xyn"):
            arr = r.keypoints.xyn
            if arr is None or len(arr) == 0:
                return None
            # xyn is normalized coords -> convert to pixels below
            k = np.array(arr[0], dtype=np.float32)
        else:
            # fallback try r.keypoints.numpy() etc
            try:
                k = np.array(r.keypoints, dtype=np.float32)
            except Exception:
                return None

        # At this point k is (17,2) or (17,3). If normalized (x in [0..1]) convert to pixel coords
        h, w = frame.shape[:2]

        # If shape (17,2) -> add conf column = 1.0
        if k.ndim == 2 and k.shape[1] == 2:
            confcol = np.ones((k.shape[0], 1), dtype=np.float32)
            k = np.concatenate([k, confcol], axis=1)


        # final sanity: ensure shape (17,3)
        if k.shape[0] != 17:
            return None
        if k.shape[1] == 2:
            confcol = np.ones((17, 1), dtype=np.float32)
            k = np.concatenate([k, confcol], axis=1)

        # --------------------------------------------------
        # 🔒 FINAL NORMALIZATION GUARANTEE (CRITICAL)
        # --------------------------------------------------
        h, w = frame.shape[:2]

        # If values look like pixels, normalize to 0..1
        if np.nanmax(k[:, 0]) > 1.5 or np.nanmax(k[:, 1]) > 1.5:
            if w > 0 and h > 0:
                k[:, 0] = k[:, 0] / float(w)
                k[:, 1] = k[:, 1] / float(h)

        # Clamp for safety
        k[:, 0] = np.clip(k[:, 0], 0.0, 1.0)
        k[:, 1] = np.clip(k[:, 1], 0.0, 1.0)

        return k.astype(np.float32)

    except Exception as e:
        # Print debug for weird Keypoints types
        print("[POSE ERROR] detect_pose_wrapper failed:", e)
        # Helpful debug: show Keypoints attrs
        try:
            attrs = [a for a in dir(r.keypoints) if not a.startswith("_")]
            print("Keypoints attrs:", attrs)
        except Exception:
            pass
        return None


def send_alert_background(camera_name, frame, message):
    def task():
        try:
            os.makedirs('temp_snapshots', exist_ok=True)
            fname = f"{camera_name}_{int(time.time())}.jpg"
            path = os.path.join('temp_snapshots', fname)
            cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        except Exception as e:
            print(f"[alert] snapshot save failed: {e}")
            return
        try:
            with open(path, 'rb') as f:
                files = {'snapshot': f}
                data = {'camera_name': camera_name, 'message': message}
                r = requests.post(f"{CLOUD_URL}/api/alerts", data=data, files=files, timeout=8)
                print(f"[alert] cloud response: {r.status_code}")
        except Exception as e:
            print(f"[alert] send failed: {e}")
    threading.Thread(target=task, daemon=True).start()


def should_send_alert(camera_name, alert_type, cooldown=ALERT_COOLDOWN):
    key = (camera_name, alert_type)
    now = time.time()
    last = alert_last_sent.get(key, 0)
    if now - last > cooldown:
        alert_last_sent[key] = now
        return True
    return False


def draw_overlay(frame, keypoints, label, conf, person_bbox=None):
    try:
        h, w = frame.shape[:2]
        txt = f"{label} {conf:.2f}"
        cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        if person_bbox is None:
            return frame

        if keypoints is None:
            return frame
        kp = np.array(keypoints, dtype=np.float32)
        if kp.ndim == 2 and (kp.shape[0] == 17) and (kp.shape[1] in (2,3)):
            kp_px = kp.copy()
            # If normalized (0..1) convert to pixel coords
            if np.nanmax(kp_px[:,0]) <= 1.01 and np.nanmax(kp_px[:,1]) <= 1.01:
                kp_px[:,0] = kp_px[:,0] * w
                kp_px[:,1] = kp_px[:,1] * h
            skeleton = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(11,12),(5,11),(6,12)]
            for a,b in skeleton:
                if not (np.isnan(kp_px[a,0]) or np.isnan(kp_px[b,0])):
                    xa,ya = int(kp_px[a,0]), int(kp_px[a,1])
                    xb,yb = int(kp_px[b,0]), int(kp_px[b,1])
                    cv2.line(frame, (xa,ya), (xb,yb), (0,200,0), 2)
            for x,y,*_ in kp_px:
                if not (np.isnan(x) or np.isnan(y)):
                    cv2.circle(frame, (int(x), int(y)), 3, (0,120,255), -1)
    except Exception as e:
        print(f"[overlay] {e}")
    return frame

def compute_movement_score(prev_px, curr_px):
    """
    Computes mean pixel displacement between consecutive keypoints.
    Inputs:
        prev_px, curr_px: (17,3) or (17,2) arrays in PIXEL space
    Returns:
        float motion score
    """
    try:
        if prev_px is None or curr_px is None:
            return 0.0

        prev = np.array(prev_px, dtype=np.float32)
        curr = np.array(curr_px, dtype=np.float32)

        if prev.shape != curr.shape:
            return 0.0

        # use x,y only
        diffs = np.linalg.norm(curr[:, :2] - prev[:, :2], axis=1)

        # ignore NaNs
        diffs = diffs[~np.isnan(diffs)]

        # clamp unrealistic per-joint motion (noise protection)
        diffs = np.clip(diffs, 0, 50)  # pixels

        if len(diffs) == 0:
            return 0.0

        # normalize by shoulder width if possible
        try:
            LSH, RSH = curr[5], curr[6]  # shoulders
            shoulder_width = abs(LSH[0] - RSH[0])
            if shoulder_width > 1:
                return float(np.mean(diffs) / shoulder_width)
        except Exception:
            pass

        return float(np.mean(diffs))


    except Exception:
        return 0.0

from datetime import datetime, time as dtime

def is_within_active_window():
    """
    Returns True if current time is inside active window.
    Handles overnight windows correctly (e.g. 20:00 → 09:00)
    """
    if not ENABLE_TIME_WINDOW:
        return True

    now = datetime.now().time()

    start = datetime.strptime(ACTIVE_START, "%H:%M").time()
    end   = datetime.strptime(ACTIVE_END, "%H:%M").time()

    if start < end:
        # same-day window (e.g. 09:00 → 18:00)
        return start <= now <= end
    else:
        # overnight window (e.g. 20:00 → 09:00)
        return now >= start or now <= end

# -----------------------------
# SNAPSHOT SAVE HELPER
# -----------------------------
def save_snapshot(frame, camera_name, reason):
    """
    Saves a snapshot image and returns the file path.
    """
    try:
        os.makedirs("snapshots", exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{camera_name}_{reason}_{ts}.jpg"
        path = os.path.join("snapshots", fname)

        cv2.imwrite(path, frame)
        return path

    except Exception as e:
        print(f"[SNAPSHOT ERROR] {e}")
        return None


def is_valid_person_skeleton(kpts, frame):
    """
    Robust human skeleton validator for CCTV.
    kpts: (17,3) [x,y,conf] in PIXEL space
    """
    if kpts is None or len(kpts) != 17:
        return False

    k = np.array(kpts, dtype=np.float32)

    # 1️⃣ confidence gate (lowered, CCTV-friendly)
    conf = k[:, 2]
    valid = conf >= 0.25

    if np.sum(valid) < 5:   # 👈 key fix (was too high before)
        return False

    # 2️⃣ spatial extent (bounding box from valid joints)
    xs = k[valid, 0]
    ys = k[valid, 1]

    if xs.size == 0 or ys.size == 0:
        return False

    w = float(xs.max() - xs.min())
    h = float(ys.max() - ys.min())

    frame_h, frame_w = frame.shape[:2]

    # 🔑 AUTO-DETECT NORMALIZED vs PIXEL
    if max(xs.max(), ys.max()) <= 1.5:
        # normalized → convert to pixels
        w *= frame_w
        h *= frame_h

    # 3️⃣ reject tiny blobs (RELAXED)
    if h < 0.12 * frame_h:
        return False

    # 4️⃣ aspect ratio sanity (human-like)
    if h / (w + 1e-6) < 1.1:
        return False

    return True



def is_valid_person_bbox(bbox, frame_shape):
    """
    bbox: (x1, y1, x2, y2) in pixels
    """
    if bbox is None:
        return False

    x1, y1, x2, y2 = bbox
    h, w = frame_shape[:2]

    bw = x2 - x1
    bh = y2 - y1

    # Reject tiny / flat objects (bed edges, blankets)
    if bw < 0.15 * w:
        return False
    if bh < 0.25 * h:
        return False

    # Reject extreme aspect ratios
    aspect = bh / (bw + 1e-6)
    if aspect < 0.6:
        return False

    return True

def can_fire_event(camera_name, event_type, cooldown_seconds):
    """
    Returns True if event is allowed to fire (cooldown passed).
    """
    now = time.time()
    key = (camera_name, event_type)

    last_ts = event_last_fired.get(key, 0)

    if now - last_ts >= cooldown_seconds:
        event_last_fired[key] = now
        return True

    return False

def load_pose_model():
    print("[LOAD] Pose model")
    return YOLO("yolov8s-pose.pt")  


def load_yolo5_model():
    print("[LOAD] YOLOv5 model")
    return torch.hub.load(
        "ultralytics/yolov5",
        "yolov5s",
        pretrained=True
    )

# ===============================
# RELAY FRAME FETCHER (GLOBAL)
# ===============================
def get_relay_frame(camera_name):
    RELAY_GET_URL = f"{SERVER_BASE_URL}/api/relay/latest?camera={camera_name}"

    while True:
        try:
            r = requests.get(RELAY_GET_URL, timeout=1)
            if r.status_code != 200:
                time.sleep(0.05)
                continue

            data = r.json()
            ts = data.get("ts", 0)
            if time.time() - ts > 1.0:
                continue

            jpeg_b64 = data.get("jpeg")
            if not jpeg_b64:
                continue

            frame = cv2.imdecode(
                np.frombuffer(base64.b64decode(jpeg_b64), np.uint8),
                cv2.IMREAD_COLOR,
            )

            if frame is not None:
                return frame, ts

        except Exception as e:
            print(f"[RELAY] wait error: {e}")
            time.sleep(0.1)

# --------------------------
# Main per-camera monitor
# --------------------------
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
def monitor_camera(camera_name: str):
  
    sub = get_subscription_state()

    if not sub["active"]:
        print(f"[SUBSCRIPTION] Inactive → camera {camera_name} not allowed to start")
        return  # 🚫 HARD STOP

    allowed = sub["limits"]["cameras"]

    # get enabled cameras from config
    enabled_cams = [
        name for name, cfg in CONFIG.get("cameras", {}).items()
        if cfg.get("enabled", True)
    ]

    # deterministic order
    enabled_cams.sort()

    if camera_name not in enabled_cams[:allowed]:
        print(
            f"[SUBSCRIPTION] Camera limit exceeded → "
            f"{camera_name} blocked ({allowed} allowed)"
        )
        return  # 🚫 HARD STOP

    print(
        f"[PLAN] camera={camera_name} "
        f"plan={sub['plan']} "
        f"active={sub['active']} "
        f"features={sub['limits']['features']}"
    )

    print(f"[INFO] Starting monitor for {camera_name} -> RELAY MODE")
  
    # ---- FEATURE FLAGS ----
    if not sub["limits"]["features"]["actions"]:
        print(f"[FEATURE] Actions DISABLED for {camera_name}")

    if not sub["limits"]["features"]["presence"]:
        print(f"[FEATURE] Presence DISABLED for {camera_name}")

    # -----------------------------
    # LOAD CAMERA CONFIG (ONCE)
    # -----------------------------
    cam_cfg = get_camera_config(camera_name)
    presence_cfg = cam_cfg.get("presence", {})

    RELAY_GET_URL = f"{SERVER_BASE_URL}/api/relay/latest?camera={camera_name}"
    last_warn = 0

    # init per-camera state ONCE
    last_seen_ts.setdefault(camera_name, 0.0)
    presence_state[camera_name] = False
    detected_fired[camera_name] = False
    last_seen_ts[camera_name] = 0.0
    last_present_emit[camera_name] = 0.0    

    http = requests.Session()
    http.trust_env = False
    
    while True:
        frame, ts = get_relay_frame(camera_name)

        sub = get_subscription_state()

        if not sub["active"]:
            print(f"[SUBSCRIPTION] Inactive → stopping detection for {camera_name}")
            time.sleep(5)
            continue

        # -----------------------------
        # YOLOv5 PERSON DETECTION
        # -----------------------------
        detections = None
            
        try:    
            results = yolo5_model(frame)
            df = results.pandas().xyxy[0]
            detections = df[
                (df["name"] == "person") &
                (df["confidence"] > 0.4)
            ]
            
        except Exception as e:
            print(f"[YOLO ERROR] {camera_name}: {e}")
            

        # ===============================
        # MAIN RELAY PROCESSING LOOP
        # ===============================
    

        camera_failures.setdefault(camera_name, 0)
        camera_disabled.setdefault(camera_name, False)
        
        # -----------------------------
        # INIT CAMERA HEALTH
        # -----------------------------
        CAMERA_HEALTH[camera_name] = {
            "last_frame_ts": time.time(),
            "freeze_counter": 0,
            "restart_counter": 0,
            "last_restart_ts": 0.0,
        }
  
        # ---------------------------------------
        # Per-camera configuration (SAFE SCOPE)
        # ---------------------------------------
        cam_cfg = get_camera_config(camera_name)

        presence_cfg = cam_cfg.get("presence", {})

        PRESENCE_CONFIRM_FRAMES = presence_cfg.get("presence_confirm_frames", 5)
        ABSENCE_CONFIRM_FRAMES  = presence_cfg.get("absence_confirm_frames", 8)
        PRESENCE_ALERT_COOLDOWN = presence_cfg.get("cooldown_seconds", 30)

        last_frame_time = time.time()
        frame_count = 0
        fps_window_start = time.time()
        now = time.time()

        # init per-camera state
        last_pose_time.setdefault(camera_name, 0.0)
        last_action_time.setdefault(camera_name, 0.0)
        last_fusion_status.setdefault(camera_name, "Unknown")
        prev_keypoints.setdefault(camera_name, None)
        action_sequence.setdefault(camera_name, deque(maxlen=32))
        unified_history.setdefault(camera_name, deque(maxlen=SMOOTH_WINDOW))
        last_snapshot_time.setdefault(camera_name, 0.0)
        frame_counter = 0
        action_buffer.setdefault(camera_name, deque(maxlen=8))
        action_sequence.setdefault(camera_name, deque(maxlen=32))
        motion_log_last.setdefault(camera_name, 0.0)
        final_log_last.setdefault(camera_name, 0.0)
        last_seen_ts.setdefault(camera_name, 0.0)
        presence_first_seen.setdefault(camera_name, 0.0)
        presence_last_seen.setdefault(camera_name, 0.0)
        presence_alert_last.setdefault(camera_name, 0.0)
        last_presence_alert.setdefault(camera_name, 0.0)
        last_present_emit.setdefault(camera_name, 0)
        last_frame_ts_seen.setdefault(camera_name, 0.0)
        last_enter_ts.setdefault(camera_name, 0.0)
        last_detected_ts.setdefault(camera_name, 0.0)
        presence_started_ts.setdefault(camera_name, 0.0)
        presence_started_ts.setdefault(camera_name, 0.0)
        last_seen_ts.setdefault(camera_name, 0.0)
        last_present_emit.setdefault(camera_name, 0.0)
        detected_fired.setdefault(camera_name, False)

        if ts <= last_frame_ts_seen[camera_name]:
            time.sleep(0.02)
            continue

        last_frame_ts_seen[camera_name] = ts

        # ----------------------------------
        # STREAM FREEZE DETECTION (RELAY SAFE)
        # ----------------------------------
        small = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        freeze_hash = hashlib.md5(gray.tobytes()).hexdigest()

        # per-camera freeze state
        freeze_state = CAMERA_HEALTH[camera_name].setdefault(
            "freeze",
            {
                "last_hash": None,
                "freeze_counter": 0,
            }
        )

        if freeze_hash == freeze_state["last_hash"]:
            freeze_state["freeze_counter"] += 1
        else:
            freeze_state["freeze_counter"] = 0

        freeze_state["last_hash"] = freeze_hash

        if freeze_state["freeze_counter"] >= 15:
            print(f"[WARN] {camera_name}: relay frames frozen")
            freeze_state["freeze_counter"] = 0  

        # -----------------------------
        # PHASE 2: TIME WINDOW GATE
        # -----------------------------
        if not is_within_active_window():
            if frame_counter % 300 == 0:  # log every ~10 sec
                print(f"[{camera_name}] ⏸ Outside active time window")
            frame_counter += 1
            continue
       
        # -----------------------------
        # SCHEDULE GATE
        # -----------------------------
        if not is_within_schedule():
            # still show video, but do nothing else
            if frame_counter % 300 == 0:
                print(f"[{camera_name}] ⏸ Outside active schedule")
            continue
 
      
        # STEP 1: Pose detection (slow interval)
        keypoints = None
        pose_label = last_fusion_status.get(camera_name, "Unknown")

        if now - last_pose_time[camera_name] >= POSE_INTERVAL:
            last_pose_time[camera_name] = now

            _pose_t0 = time.time()

            try:
                keypoints = detect_pose_wrapper(frame)
            except Exception as e:
                 print(f"[POSE ERROR] {camera_name}: {e}")

            _pose_dt = time.time() - _pose_t0
            if _pose_dt > MAX_POSE_INFER_TIME:
                print(
                    f"[WARN] {camera_name} pose inference slow: {_pose_dt:.2f}s → skipped"
                )
                keypoints = None

            if keypoints is None:
                keypoints = prev_keypoints.get(camera_name, None)
                pose_label = last_fusion_status.get(camera_name, "Unknown")

            else:
                keypoints = keypoints

                # store for motion & action
                prev_keypoints[camera_name] = keypoints.copy()

                try:
                    L_sh = keypoints[5]; R_sh = keypoints[6]
                    L_hp = keypoints[11]; R_hp = keypoints[12]
                    mid_sh = (L_sh + R_sh) / 2.0
                    mid_hp = (L_hp + R_hp) / 2.0
                    vertical = abs(mid_sh[1] - mid_hp[1])
                    horiz = abs(mid_sh[0] - mid_hp[0])

                    # If coordinates look normalized (values <=1), scale heuristics
                    if vertical <= 1.01:
                        s = max(frame.shape[:2])
                        vert_px = vertical * s
                        horiz_px = horiz * s
                    else:
                        vert_px = vertical
                        horiz_px = horiz

                    ratio = horiz_px / (vert_px + 1e-6)

                    if ratio > 1.6:
                        pose_label = "Lying"
                    elif vert_px < 40:
                        pose_label = "Sitting"
                    else:
                        pose_label = "Standing"

                except Exception as e:
                    print(f"[{camera_name}] pose calc error:", e)
                    pose_label = "Unknown"

        else:
            # reuse previous pose/keypoints
            keypoints = prev_keypoints.get(camera_name, None)
            pose_label = last_fusion_status.get(camera_name, "Unknown")

        if keypoints is None:
            action_sequence[camera_name].clear()

        # -----------------------------
        # STEP 2 — MOTION (PIXEL SPACE)  ✅ FINAL
        # -----------------------------
        movement_score = 0.0
        motion_state = "Idle"

        if keypoints is not None and len(keypoints) >= 5:
            frame_h, frame_w = frame.shape[:2]

            # 🔑 convert NORMALIZED → PIXELS
            curr_px = keypoints[:, :2].copy()
            curr_px[:, 0] *= frame_w
            curr_px[:, 1] *= frame_h

            prev_px = prev_keypoints_pixels.get(camera_name)

            if prev_px is not None:
                movement_score = compute_movement_score(prev_px, curr_px)

            # ✅ CRITICAL: update AFTER computing motion
            prev_keypoints_pixels[camera_name] = curr_px

            if movement_score < MOTION_IDLE_EPS:
                motion_state = "Idle"
            elif movement_score > MOTION_MOVING:
                motion_state = "Moving"
            else:
                motion_state = "Subtle"

        # -----------------------------
        # ACTION DEFAULTS (PER FRAME)
        # -----------------------------
        action_label = "Unknown"
        action_conf = 0.0	
                
        # STEP 3: Action recognition (every ACTION_INTERVAL)
        action_label = "Unknown"
        action_conf = 0.0
        ACTION_CONF_MIN = 0.0  
        kp_norm = None

        # -----------------------------
        # ACTION FEATURE GATE
        # -----------------------------
        sub = get_subscription_state()
        actions_enabled = sub["limits"]["features"]["actions"]

        if keypoints is not None:
            mean_conf = float(np.nanmean(keypoints[:, 2]))  
      
            # --- ACTION SEQUENCE UPDATE (NO CONF GATE) ---
            h, w = frame.shape[:2]
            kp_norm = keypoints.copy()

            # normalize ONLY if pixel space
            if np.nanmax(kp_norm[:, :2]) > 1.5:
                kp_norm[:, 0] /= w
                kp_norm[:, 1] /= h

            action_sequence[camera_name].append(kp_norm[:, :2])                      
           
            # KEEP ONLY IF ACTION IS IMPORTANT
            if action_label in ("Falling", "Running", "Intrusion"):
                print(
                    f"[ACTION] {camera_name} | {action_label} ({action_conf:.2f})"
            )


            actions_enabled = sub["limits"]["features"]["actions"]

            # RUN MODEL
            if actions_enabled and len(action_sequence[camera_name]) >= 16:
                seq = list(action_sequence[camera_name])[-16:]

                raw_action, raw_conf = predict_action(seq)

                if raw_conf > 0.12:
                    action_label = raw_action
                    action_conf = raw_conf

                _action_t0 = time.time()

                try:
                    raw_action, raw_conf = predict_action(seq)
                
                except Exception as e:
                    print(f"[ACTION ERROR] {camera_name}: {e}")

                _action_dt = time.time() - _action_t0
                if _action_dt > MAX_ACTION_INFER_TIME:
                    print(
                        f"[WARN] {camera_name} action inference slow: {_action_dt:.2f}s → skipped"
                    )
                    raw_action, raw_conf = "Unknown", 0.0

                    # -----------------------------
                    # HARD GATE: FALLING REQUIRES MOTION
                    # -----------------------------
                    if raw_action == "Falling":
                        if motion_state == "Idle" or raw_conf < 0.4:
                            raw_action = "Unknown"
                            raw_conf = 0.0

                    # -----------------------------
                    # ACTION BUFFER + VOTING (FIXED)
                    # -----------------------------
                    try:
                        raw_action, raw_conf = predict_action(seq)
                    
                        if raw_action != "Unknown" and raw_conf > 0.12:
                            action_buffer[camera_name].append(raw_action)

                        # majority vote if buffer not empty
                        if len(action_buffer[camera_name]) >= 4:
                            votes = Counter(action_buffer[camera_name])
                            action_label = votes.most_common(1)[0][0]
                            action_conf = raw_conf
                        else:
                            action_label = "unknown"
                            action_conf = 0.0
                    
                    except Exception as e:
                        print(f"[ERROR] {camera_name}: action model failed: {e}")
                        raw_action, raw_conf = "Unknown", 0.0                 
                    
                    # -----------------------------
                    # ACTION EVENT (THROTTLED)
                    # -----------------------------
                    if (
                        action_label not in ("Unknown", "Standing")
                        and action_conf > 0.15
                    ):
                        from src.core.alerts.alert_throttle import can_send_alert

                        if can_send_alert(camera_name, "ACTION"):
                            snap = save_snapshot(frame, camera_name, f"ACTION_{action_label}")

                            emit_event(
                                event_type="ACTION",
                                camera=camera_name,
                                payload={
                                    "action": action_label,
                                    "confidence": round(action_conf, 2),
                                    "pose": pose_label,
                                    "snapshot": snap,
                                },
                            )


                    # -----------------------------
                    # CRITICAL ACTION ESCALATION
                    # -----------------------------
                    if (
                        action_label in CRITICAL_ACTIONS
                        and action_conf >= 0.25
                    ):
                        from src.core.alerts.alert_throttle import can_send_alert

                        if can_send_alert(camera_name, "CRITICAL"):
                            snap = save_snapshot(frame, camera_name, f"CRITICAL_{action_label}")

                            emit_event(
                                event_type="CRITICAL",
                                camera=camera_name,
                                payload={
                                    "action": action_label,
                                    "confidence": round(action_conf, 2),
                                    "pose": pose_label,
                                    "motion": motion_state,
                                    "snapshot": snap,
                                },
                            )

                            print(
                                f"[CRITICAL] {camera_name} | "
                                f"ACTION={action_label} | "
                                f"CONF={action_conf:.2f}"
                            )
 
                # Autotrain: save only rare or uncertain samples
                if AUTOLEARN_ENABLED:
                    try:
                        # Save only if action_conf is extremely low AND pose_label is not Unknown
                        if (action_conf < 0.35) and (pose_label != "Unknown") and (len(seq) >= 12):
                            fname = os.path.join(
                                AUTOTRAIN_SAVE_DIR,
                                f"{camera_name}_{int(time.time())}.npy"
                            )
                            np.save(fname, np.array(seq[-16:], dtype=np.float32))
                            print(f"[AUTOLEARN] saved {fname}")
                    except Exception as e:
                        print(f"[autolearn] {e}")

                
        # STEP 4: Fusion & smoothing
        final_label = pose_label  # default = pose

        # action can override pose ONLY if strong
        if action_label != "Unknown" and action_conf >= 0.4:
            final_label = action_label

        # ---- 1. HARD FALL OVERRIDE ----
        if action_label == "Falling":
            final_label = "Falling"

        # ---- 2. MOTION-BASED ACTION OVERRIDE ----
        elif action_label in ("Walking", "Running"):
            if movement_score > MOTION_MOVING:
                final_label = action_label

        # ---- 3. IGNORE ACTIONS THAT ARE POSES ----
        elif action_label in ("Standing", "Sitting", "Lying"):
            final_label = pose_label

        # ---- 4. UNKNOWN ACTION → POSE ONLY ----
        elif action_label == "Unknown":
            final_label = pose_label

        # ---- 5. SAFETY FALLBACK ----
        if final_label is None:
            final_label = "Unknown"
        
        # --------------------------------
        # MOTION ↔ ACTION CONSISTENCY GATE
        # --------------------------------

        if motion_state == "Idle" and action_label in ("Running", "Walking"):
            action_label = "Standing"
            action_conf = min(action_conf, 0.3)

        last_fusion_status[camera_name] = final_label
        final_status = final_label

        
        
        # -----------------------------
        # FIX-3: POSE EXTRACTION (YOLOv8 ONLY)
        # -----------------------------
        keypoints = detect_pose_wrapper(frame)

        # ---- frame-level skeleton debounce ----
        skeleton_ok = False

        # =============================
        # PRESENCE STATE MACHINE (FINAL)
        # =============================

        now = time.time()

        # ---- SINGLE SOURCE OF TRUTH ----
        person_present = (
            keypoints is not None
            and is_valid_person_skeleton(keypoints, frame)
        )

        print(
            f"[DEBUG PRESENCE] "
            f"kp={keypoints is not None} | "
            f"skeleton={person_present} | "
            f"state={presence_state[camera_name]}"
        )

        # ---- ENTER ----
        now = time.time()

        if person_present:
            last_seen_ts[camera_name] = now

            if not presence_state[camera_name]:
                # 🔔 NEW SESSION START
                presence_state[camera_name] = True
                presence_started_ts[camera_name] = now
                detected_fired[camera_name] = False
                last_present_emit[camera_name] = now

                snap = save_snapshot(frame, camera_name, "ENTER")

                print(f"[PRESENCE] {camera_name} | ENTER")

                emit_event(
                    event_type="PERSON_ENTER",
                    camera=camera_name,
                    payload={
                        "pose": pose_label,
                        "action": action_label,
                        "snapshot": snap,
                    }
                )

            # 🔔 PERSON_DETECTED — ONCE PER SESSION
            if not detected_fired[camera_name]:
                snap = save_snapshot(frame, camera_name, "PersonDetected")

                emit_event(
                    event_type="PERSON_DETECTED",
                    camera=camera_name,
                    payload={
                        "pose": pose_label,
                        "action": action_label,
                        "snapshot": snap,
                    }
                )

                detected_fired[camera_name] = True

        # ---- HEARTBEAT (PERSON_PRESENT) ----
        PRESENT_INTERVAL = 45

        if presence_state[camera_name]:
            if now - last_present_emit[camera_name] >= PRESENT_INTERVAL:
                snap = save_snapshot(frame, camera_name, "PRESENT")

                emit_event(
                    event_type="PERSON_PRESENT",
                    camera=camera_name,
                    payload={
                        "pose": pose_label,
                        "action": action_label,
                        "snapshot": snap,
                    }
                )

                last_present_emit[camera_name] = now

        # ---- EXIT ----
        EXIT_ABSENCE_SEC = 2.5

        if presence_state[camera_name] and not person_present:
            if now - last_seen_ts[camera_name] >= EXIT_ABSENCE_SEC:
                presence_state[camera_name] = False

                print(f"[PRESENCE] {camera_name} | EXIT")

                emit_event(
                    event_type="PERSON_EXIT",
                    camera=camera_name,
                    payload={}
                )

        # STEP 5: High-priority fall alert
        try:
            conf = float(action_conf or 0.0)
            if final_status == "Falling" and conf > FALL_CONF_THRESHOLD and movement_score > 0.12:
                if should_send_alert(camera_name, "fall"):
                    print(f"[{camera_name}] ALERT hard fall (conf={conf:.2f})")
                    send_alert_background(camera_name, frame, f"Fall detected ({conf:.2f})")
        except Exception as e:
            print(f"[fall_alert] {e}")
            emit_event(
                event_type="FALL",
                camera=camera_name,
                payload={
                    "pose": pose_label,
                    "action": action_label,
                    "snapshot": snap_path,
                }
            )

        # Optional: YOLOv5 object detection for environment
        # -----------------------------
        # YOLOv5 PERSON DETECTION
        # -----------------------------
        #detections = None

        #try:
            #results = yolo5_model(frame)
            #df = results.pandas().xyxy[0]

            #detections = df[
                #(df["name"] == "person") &
                #(df["confidence"] > 0.4)
            #]

        #except Exception as e:
            #print(f"[YOLO ERROR] {camera_name}: {e}")
            #detections = None

        # Person identification (best-effort)
        person_name = "Unknown"
        try:
            if detections is not None and not detections.empty:
                rows = detections[detections['name'] == 'person']
                if len(rows) > 0:
                    r = rows.iloc[0]
                    x1,y1,x2,y2 = int(r['xmin']), int(r['ymin']), int(r['xmax']), int(r['ymax'])
                    crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                    if crop is not None and crop.size > 0:
                        person_name = identify_person(crop)
        except Exception:
            person_name = "Unknown"

        # Visualization + logging
        try:
            display = frame.copy()
            display = draw_overlay(display, keypoints, final_status, float(action_conf or 0.0))
            if person_name != "Unknown":
                cv2.putText(display, f"ID: {person_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,0), 2)
            display_frames[camera_name] = display

            # log to JSONL
            log_entry = {
                'time': time.time(),
                'camera': camera_name,
                'pose': pose_label,
                'action': action_label,
                'final': final_status,
                'person': person_name,
                'confidence': float(action_conf or 0.0),
                'motion': float(movement_score)
            }
            day = datetime.now().strftime('%Y-%m-%d')
            with open(os.path.join(DATA_LOG_DIR, f"{day}_{camera_name}.jsonl"), 'a') as f:
                f.write(json.dumps(log_entry) + "\n")

            nowt = time.time()
            if nowt - last_snapshot_time.get(camera_name, 0) > 5.0:
                fname = os.path.join(SNAPSHOT_DIR, f"{camera_name}_{int(nowt)}.jpg")
                cv2.imwrite(fname, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                last_snapshot_time[camera_name] = nowt
        except Exception as e:
            print(f"[FATAL LOOP ERROR] {camera_name} → {e}")
            time.sleep(1)
            continue

            # ==========================
            # 14B-3: FPS WATCHDOG
            # ==========================
            if time.time() - fps_window_start >= 5.0:
                fps = frame_count / (time.time() - fps_window_start)

                if fps < 2.0:
                    print(f"[WARN] {camera_name} relay FPS low ({fps:.2f}) — waiting for frames")

                frame_count = 0
                fps_window_start = time.time()

            time.sleep(0.02)
    
# --------------------------
# Helpers: zone check
# --------------------------

def is_inside_zone(x, y, pts):
    try:
        contour = np.array(pts, dtype=np.int32)
        return cv2.pointPolygonTest(contour, (int(x), int(y)), False) >= 0
    except Exception:
        return False


# --------------------------
# Entrypoint
# --------------------------
if __name__ == "__main__":
    print("[INFO] Multi-camera monitor starting...")

    CONFIG = load_config()
    threads = []

    for camera_name, cam_cfg in CONFIG.get("cameras", {}).items():
        if not cam_cfg.get("enabled", True):
            print(f"[INFO] {camera_name} disabled, skipping")
            continue

        t = threading.Thread(
            target=monitor_camera,
            args=(camera_name,),
            daemon=True
        )
        t.start()
        threads.append(t)
        time.sleep(0.3)

    for t in threads:
        t.join()    

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[INFO] Shutdown requested. Exiting.")


    # === MAIN THREAD DISPLAY LOOP (MAC-SAFE) ===
    while True:
        try:
            for cam_name in list(display_frames.keys()):
                frame = display_frames.get(cam_name)
                if frame is not None and frame.size > 0:
                    cv2.imshow(f"Camera: {cam_name}", frame)

            if cv2.waitKey(1) == ord('q'):
                print("[INFO] Quit requested. Closing...")
                break

        except KeyboardInterrupt:
            print("[INFO] Keyboard interrupt received. Exiting...")
            break
        except Exception as e:
            print(f"[MAIN DISPLAY] Error: {e}")
            time.sleep(0.1)

    cv2.destroyAllWindows()

import signal
import sys

def shutdown_handler(sig, frame):
    print("[SHUTDOWN] Graceful shutdown requested")
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)


