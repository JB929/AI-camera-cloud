import os
import sys
import time
import json
import threading
from collections import deque, Counter
from datetime import datetime

import numpy as np

# Try to import ultralytics YOLO — required for pose detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# Action predictor (trained model wrapper) — expected to be at src/core/action_recognition/action_predictor.py
try:
    from src.core.action_recognition.action_predictor import predict_action
except Exception:
    def predict_action(seq, movenet_pose=None):
        # Safe fallback
        return "Unknown", 0.0

# Optional person identifier
try:
    from person_identifier import identify_person
except Exception:
    def identify_person(crop):
        return "Unknown"

# --------------------
# Config (user choices)
# --------------------
POSE_MODEL_PATH = os.environ.get("POSE_MODEL", "models/yolov8s-pose.pt")  # Q1: YOLOv8 default
CAMERAS = {
    "Front_Yard": 0,
}

# Intervals (Q3: detect every 5s)
POSE_INTERVAL = 5.0
ACTION_INTERVAL = 5.0

# Operation mode (headless per Q4=B)
DISPLAY = False

# Fusion / thresholds
SMOOTH_WINDOW = 5
FALL_CONF_THRESHOLD = 0.85
RUN_CONF_THRESHOLD = 0.65
WAVE_CONF_THRESHOLD = 0.60
MIN_ACTION_CONF_SAVE = 0.15
AUTOTRAIN_CONF_TH = 0.60
ALERT_COOLDOWN = 15  # seconds
MOVEMENT_THRESHOLD = 0.02  # normalized coords

# Storage
AUTOTRAIN_DIR = "autotrain_buffer"
LOG_DIR = "data/logs"
SNAPSHOT_DIR = "data/snapshots"
os.makedirs(AUTOTRAIN_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# --------------------
# Global state (thread-safe per camera dicts)
# --------------------
last_pose_time = {}
last_action_time = {}
last_fusion_status = {}
prev_keypoints = {}
action_sequence = {}
unified_history = {}
alert_last_sent = {}

# Load pose model
pose_model = None
if YOLO_AVAILABLE and os.path.exists(POSE_MODEL_PATH):
    try:
        pose_model = YOLO(POSE_MODEL_PATH)
        print(f"[MODEL] Loaded pose model: {POSE_MODEL_PATH}")
    except Exception as e:
        print(f"[MODEL] Failed to load pose model {POSE_MODEL_PATH}: {e}")
        pose_model = None
else:
    print(f"[MODEL] ultralytics YOLO not available or model not found at {POSE_MODEL_PATH}. Pose disabled.")

# --------------------
# Helpers
# --------------------

def safe_norm_coords(kp, frame_shape=None):
    """Return numpy array (17,3) with x,y normalized to 0..1 if frame_shape is provided.
    kp may be (17,2) or (17,3)
    """
    k = np.array(kp, dtype=np.float32)
    if k.size == 0:
        return None
    if k.ndim != 2 or k.shape[0] < 17:
        # try to reshape if it's flattened
        return None
    # If only x,y provided
    if k.shape[1] == 2:
        conf = np.ones((k.shape[0], 1), dtype=np.float32)
        k = np.concatenate([k, conf], axis=1)

    # Normalize if frame_shape (h,w) given and coords look like pixels (max>1.01)
    if frame_shape is not None and (k[:, :2].max() > 1.01):
        h, w = frame_shape[0], frame_shape[1]
        # avoid division by zero
        denom_x = float(w) if w > 0 else 1.0
        denom_y = float(h) if h > 0 else 1.0
        k[:, 0] = k[:, 0] / denom_x
        k[:, 1] = k[:, 1] / denom_y
    return k


def extract_motion(prev_kp, curr_kp):
    """Compute simple motion score (mean joint displacement) assuming normalized coords.
    Returns movement_score (float)
    """
    try:
        if prev_kp is None or curr_kp is None:
            return 0.0
        prev = np.array(prev_kp, dtype=np.float32)
        curr = np.array(curr_kp, dtype=np.float32)
        if prev.shape != curr.shape:
            return 0.0
        # only x,y
        diffs = np.linalg.norm(curr[:, :2] - prev[:, :2], axis=1)
        # ignore NaNs
        mask = ~np.isnan(diffs)
        if mask.sum() == 0:
            return 0.0
        return float(np.mean(diffs[mask]))
    except Exception:
        return 0.0


def classify_pose_from_kp(kp):
    """Simple rule-based classifier for Standing/Sitting/Lying using shoulders & hips.
    kp expected normalized (17,3)
    """
    try:
        if kp is None:
            return "Unknown"
        # use indexes: left_shoulder=5 right_shoulder=6 left_hip=11 right_hip=12 (COCO)
        L_sh = kp[5]
        R_sh = kp[6]
        L_hp = kp[11]
        R_hp = kp[12]
        if np.isnan(L_sh[0]) or np.isnan(L_hp[0]):
            return "Unknown"
        mid_sh = (L_sh[:2] + R_sh[:2]) / 2.0
        mid_hp = (L_hp[:2] + R_hp[:2]) / 2.0
        vertical = abs(mid_sh[1] - mid_hp[1])
        horiz = abs(mid_sh[0] - mid_hp[0])
        # heuristics (normalized space)
        def classify_pose_cctv(keypoints):
            """
            Much more tolerant CCTV-specific pose classifier.
            Works even with knees folded, sitting on floor, lying sideways.
            """

            # Key joints
            L_sh = keypoints[5]; R_sh = keypoints[6]
            L_hp = keypoints[11]; R_hp = keypoints[12]

            mid_sh = (L_sh + R_sh) / 2
            mid_hp = (L_hp + R_hp) / 2

            v = abs(mid_sh[1] - mid_hp[1])   # vertical distance
            h = abs(mid_sh[0] - mid_hp[0])   # horizontal distance

            # Normalize by image height so different cameras behave same
            v_norm = v / 480   # assuming typical CCTV height
            h_norm = h / 480

            # New ratio – more tolerant for sitting
            ratio = h_norm / (v_norm + 1e-6)

            # RULES:
            if ratio > 1.1:
                return "Lying"

            # Sitting tolerance expanded
            if v_norm < 0.35:
                return "Sitting"

            return "Standing"
    except Exception:
        return "Unknown"


def detect_pose_wrapper(frame):
    """Runs the loaded pose model (if available) and returns (17,3) normalized keypoints or None.
    Robust to ultralytics Keypoints API variations.
    """
    if pose_model is None:
        return None
    try:
        results = pose_model(frame, verbose=False)
    except Exception as e:
        print(f"[POSE ERROR] model inference failed: {e}")
        return None

    if not results or len(results) == 0:
        return None

    r = results[0]
    # r.keypoints may be an ultralytics Keypoints object
    try:
        if hasattr(r, 'keypoints') and r.keypoints is not None:
            kp_obj = r.keypoints
            # Keypoints object may provide .xy, .conf, or .data
            if hasattr(kp_obj, 'xy'):
                kp_xy = kp_obj.xy  # tensor or numpy
            elif hasattr(kp_obj, 'xyn'):
                kp_xy = kp_obj.xyn
            else:
                # fallback to .data
                kp_xy = getattr(kp_obj, 'data', None)

            if kp_xy is None:
                # sometimes Keypoints has .numpy() like structure
                try:
                    arr = np.array(kp_obj)
                    if arr.size == 0:
                        return None
                    kp_xy = arr
                except Exception:
                    return None

            # kp_xy could be tensor with shape (N,17,2) or (N,17,3) or (17,2)
            kparr = np.array(kp_xy)
            # If shape (N, 17, 2) and N>=1 pick first
            if kparr.ndim == 3 and kparr.shape[0] >= 1:
                k = kparr[0]
            else:
                k = kparr

            # check confidences
            confs = None
            if hasattr(kp_obj, 'conf'):
                try:
                    confs = np.array(kp_obj.conf)
                except Exception:
                    confs = None

            # if k has only x,y
            if k.ndim == 2 and k.shape[1] == 2:
                # try to get confidences from kp_obj.conf or set to 1.0
                if confs is not None and confs.shape[0] >= k.shape[0]:
                    conf = confs[:k.shape[0]]
                    k = np.concatenate([k, conf.reshape(-1, 1)], axis=1)
                else:
                    confcol = np.ones((k.shape[0], 1), dtype=np.float32)
                    k = np.concatenate([k, confcol], axis=1)

            # final check
            if k.shape[0] < 17:
                return None

            # normalize to 0..1 by frame shape if needed
            k_norm = safe_norm_coords(k, frame_shape=frame.shape[:2])
            return k_norm

    except Exception as e:
        # as fallback, try r.keypoints.data -> tensor
        try:
            if hasattr(r, 'keypoints') and hasattr(r.keypoints, 'data'):
                data = r.keypoints.data.cpu().numpy()
                if data.ndim == 3 and data.shape[0] >= 1:
                    k = data[0]
                    k_norm = safe_norm_coords(k, frame_shape=frame.shape[:2])
                    return k_norm
        except Exception:
            pass

    return None


def save_autotrain_sample(camera_name, seq):
    try:
        t = int(time.time())
        path = os.path.join(AUTOTRAIN_DIR, f"{camera_name}_{t}.npy")
        np.save(path, np.array(seq, dtype=np.float32))
        print(f"[AUTOLEARN] saved: {path}")
    except Exception as e:
        print(f"[AUTOLEARN ERROR] {e}")


def log_detection(camera_name, pose_label, action_label, final_status, conf, motion):
    ts = time.time()
    day = datetime.now().strftime("%Y-%m-%d")
    out_file = os.path.join(LOG_DIR, f"{day}_{camera_name}.jsonl")
    entry = {
        "time": ts,
        "camera": camera_name,
        "pose": pose_label,
        "action": action_label,
        "final": final_status,
        "confidence": float(conf),
        "motion": float(motion),
    }
    try:
        with open(out_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[LOG ERROR] {e}")

# --------------------
# Main camera loop
# --------------------

def monitor_camera(camera_name, camera_source):
    import cv2

    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"[{camera_name}] ❌ Cannot open camera source: {camera_source}")
        return

    print(f"[INFO] Starting monitor for {camera_name} -> {camera_source}")

    # init per-camera state
    last_pose_time[camera_name] = 0.0
    last_action_time[camera_name] = 0.0
    last_fusion_status[camera_name] = "Unknown"
    prev_keypoints[camera_name] = None
    action_sequence[camera_name] = deque(maxlen=32)
    unified_history[camera_name] = deque(maxlen=SMOOTH_WINDOW)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.2)
            continue

        now = time.time()

        # POSE: run only every POSE_INTERVAL seconds
        pose_label = last_fusion_status.get(camera_name, "Unknown")
        keypoints = prev_keypoints.get(camera_name)
        if now - last_pose_time.get(camera_name, 0.0) >= POSE_INTERVAL:
            last_pose_time[camera_name] = now
            kp = detect_pose_wrapper(frame)
            if kp is None:
                print(f"[{camera_name}] ⚠️ NO KEYPOINTS DETECTED")
                # keep previous keypoints & pose
            else:
                keypoints = kp
                pose_label = classify_pose_from_kp(keypoints)
                prev_keypoints[camera_name] = keypoints
                print(f"[{camera_name}] 🧍 YOLO Pose → {pose_label}")

        # MOTION
        movement_score = extract_motion(prev_keypoints.get(camera_name), keypoints)

        # ACTION: run only every ACTION_INTERVAL seconds
        action_label = "Unknown"
        action_conf = 0.0
        if now - last_action_time.get(camera_name, 0.0) >= ACTION_INTERVAL:
            last_action_time[camera_name] = now
            if keypoints is not None:
                action_sequence[camera_name].append(keypoints)
            seq = list(action_sequence[camera_name])
            if len(seq) >= 6:
                try:
                    action_label, action_conf = predict_action(seq, movenet_pose=pose_label)
                    print(f"[{camera_name}] 🧠 Action → {action_label} ({action_conf:.2f})")
                except Exception as e:
                    print(f"[{camera_name}] ⚠️ Action block failed: {e}")
                    action_label, action_conf = "Unknown", 0.0

                # Auto-learn uncertain/disagree samples
                try:
                    if (action_conf < AUTOTRAIN_CONF_TH) or (action_label != "Unknown" and action_label != pose_label and action_conf > MIN_ACTION_CONF_SAVE):
                        save_autotrain_sample(camera_name, seq[-16:])
                except Exception as e:
                    print(f"[AUTOLEARN ERROR] {e}")

        # FUSION + SMOOTHING
        final_status = pose_label
        try:
            if action_label == "Falling" and action_conf >= FALL_CONF_THRESHOLD and movement_score > 0.12:
                final_status = "Falling"
            elif action_label == "Running" and action_conf >= RUN_CONF_THRESHOLD and movement_score > 0.2:
                final_status = "Running"
            elif action_label == "Waving" and action_conf >= WAVE_CONF_THRESHOLD and pose_label != "Lying":
                final_status = "Waving"
            elif action_label == "Sitting" and action_conf >= 0.7 and pose_label == "Sitting":
                final_status = "Sitting"
            else:
                final_status = pose_label

            # smoothing (mode of last N)
            unified_history[camera_name].append(final_status)
            most = Counter(unified_history[camera_name]).most_common(1)
            if most:
                final_status = most[0][0]

            last_fusion_status[camera_name] = final_status
            print(f"[{camera_name}] 🔥 Unified Status (STABLE) → {final_status}")

        except Exception as e:
            print(f"[{camera_name}] ⚠️ Fusion failed: {e}")
            final_status = pose_label
            last_fusion_status[camera_name] = final_status

        # FALL ALERT (high priority)
        try:
            if final_status == "Falling" and action_conf >= FALL_CONF_THRESHOLD and movement_score > 0.12:
                print(f"[{camera_name}] 🚨 HARD FALL detected — alerting! conf={action_conf:.2f} motion={movement_score:.3f}")
                # send_alert_background(camera_name, frame, f"Fall detected ({action_conf:.2f})")
        except Exception as e:
            print(f"[{camera_name}] ⚠️ Fall alert failed: {e}")

        # LOG
        try:
            log_detection(camera_name, pose_label, action_label, final_status, action_conf, movement_score)
        except Exception as e:
            print(f"[{camera_name}] ⚠️ Logging failed: {e}")

        # sleep a bit to avoid tight loop when headless
        time.sleep(0.1)

    # cleanup
    cap.release()


# --------------------
# Run threads
# --------------------
if __name__ == "__main__":
    print("[INFO] Multi-camera monitor starting...")
    threads = []
    for name, src in CAMERAS.items():
        t = threading.Thread(target=monitor_camera, args=(name, src), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.5)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[INFO] Stopping monitors...")
        # threads are daemon; program will exit

