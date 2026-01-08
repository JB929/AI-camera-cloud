import os
import sys
import time
import torch
import threading
import collections
from collections import deque, Counter
from datetime import datetime
from src.core.pose_calibrator import calibrate_pose_centroids, classify_by_centroid
from src.core.multi_camera_detector_BASELINE import monitor_camera
import numpy as np
import contextlib

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*torch.cuda.amp.autocast.*"
)

from collections import defaultdict

log_throttle = defaultdict(lambda: 0.0)
motion_log_last = defaultdict(lambda: 0.0)
LOG_THROTTLE_SECS = 0.5  # print no more than twice per second per key


def throttle_print(key, msg, now=None):
    if now is None:
        now = time.time()
    if now - log_throttle[key] >= LOG_THROTTLE_SECS:
        safe_print(msg)
        log_throttle[key] = now


# === GLOBAL POSE CALIBRATION STATE ===
POSE_CALIBRATION_SAMPLES = 120
pose_calibration_data = {}  # temporary samples for calibration
camera_pose_thresholds = {}  # final thresholds per camera

last_torso_norm = {}  # store torso_v_norm per camera
CURRENT_CAMERA_NAME = "default"  # live camera name for classify()


# -----------------------------
# ACTION DEBOUNCE STATE
# -----------------------------
last_action_reported = defaultdict(lambda: None)
last_action_time = defaultdict(lambda: 0.0)

ACTION_REPORT_COOLDOWN = 5.0  # seconds

# -----------------------------
# Optional imports (safe)
# -----------------------------
try:
    import cv2
except Exception:
    cv2 = None

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# -----------------------------
# Project path
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -----------------------------
# Action predictor wrapper
# -----------------------------
try:
    from src.core.action_recognition.action_predictor import predict_action
except Exception:

    def predict_action(seq, movenet_pose=None):
        # Fallback stub
        return "Unknown", 0.0


# Optional person identifier (we won't use it in logic, but keep stub)
try:
    from person_identifier import identify_person
except Exception:

    def identify_person(crop):
        return "Unknown"


# -----------------------------
# CONFIG
# -----------------------------
CAMERAS = {
    "Front_Yard": 0,  # change to RTSP URL if needed
}

MODELS = {
    "pose": "models/yolov8s-pose.pt",  # YOLOv8 pose model
    "yolov5_det": None,  # None -> load yolov5s via torch.hub if available
}

SHOW_WINDOWS = False  # set True if you want OpenCV windows (may fail on mac/headless)
USE_YOLO5_FOR_ZONES = True  # optional zone detection
AUTOTRAIN_SAVE_DIR = "autotrain_buffer"
os.makedirs(AUTOTRAIN_SAVE_DIR, exist_ok=True)

RUNNING_TH = 7.0  # strong full-body locomotion
WALKING_TH = 2.5  # controlled step movement
POSE_INTERVAL = 0.5  # run pose detector every 5 seconds
ACTION_INTERVAL = 0.25  # run action classifier every 5 seconds
SMOOTH_WINDOW = 5  # fused status smoothing window
ALERT_COOLDOWN = 15.0
FALL_CONF_THRESHOLD = 0.75
MIN_ACTION_CONF = 0.60
MIN_WAVE_MOTION = 6.0  # minimum movement for waving
MOTION_STILL_THRESHOLD = 0.005  # for normalized/pixel scale (tune later)
LOG_INTERVAL = 5.0  # print summary every 5 seconds
FUSION_LOG_INTERVAL = 5.0
POSE_STABLE_REQUIRED = 3
FUSION_LOG_INTERVAL = 3.0
MOTION_IDLE_EPS = 0.5  # below this -> absolutely idle
MOTION_MOVING = 3.0
SMOOTH_WINDOW = 8
POSE_IN_THRESH = {"standing": 0.24, "sitting": 0.20}  # example
POSE_OUT_THRESH = {"standing": 0.22, "sitting": 0.18}  # slightly lower to avoid bounce
HYSTERESIS_WINDOW = 6
MOTION_WINDOW = 6
MOTION_THRESHOLD = 3.0

# Optional zones: camera_name -> list of (x,y)
ZONES = {
    # "Front_Yard": [(100, 100), (400, 100), (400, 400), (100, 400)]
}

# -----------------------------
# GLOBAL STATE (per camera)
# -----------------------------
last_pose_time = {}
last_action_time = {}
last_fusion_status = {}
action_sequence = {}  # camera_name → deque(maxlen=32)
unified_history = {}
alert_last_sent = {}
display_frames = {}
last_pose_time = {}
last_torso_norm = {}
camera_pose_thresholds = {}
pose_stable_count = {}
last_fusion_log_time = {}
prev_motion_score = {}
motion_debug_last = {}
fall_detected = {}
autolearn_last_saved = {}
frame_id = {}
pose_switch_timers = {}  # (optional) track how long a pose candidate remains stable
prev_keypoints = {}
prev_keypoints_pixels = {}
motion_buffer = {}
pose_change_buffer = {}
pose_exit_count = {}
prev_pose_label = {}


# small utility: safe-get-or-init deque for camera
def _ensure_action_deque(camera_name):
    global action_sequence
    if camera_name not in action_sequence:
        action_sequence[camera_name] = deque(maxlen=32)
    return action_sequence[camera_name]


# stability / pose inertia
POSE_STABLE_REQUIRED = 2  # how many "same pose" detections to accept

# models
pose_model = None
yolo5_model = None


def safe_print(*args, **kwargs):
    """Thread-safe print."""
    print(*args, **kwargs)


# -----------------------------
# MODEL LOADING
# -----------------------------
def load_models():
    global pose_model, yolo5_model

    # Pose model (YOLOv8)
    if YOLO_AVAILABLE and MODELS.get("pose") and os.path.exists(MODELS["pose"]):
        try:
            pose_model = YOLO(MODELS["pose"])
            safe_print(f"[MODEL] Loaded pose model: {MODELS['pose']}")
        except Exception as e:
            safe_print(f"[MODEL] Failed to load pose model: {e}")
            pose_model = None
    else:
        safe_print(
            "[MODEL] Pose model not found or ultralytics unavailable; pose disabled."
        )
        pose_model = None

    # YOLOv5 detection for zones (optional)
    if TORCH_AVAILABLE and USE_YOLO5_FOR_ZONES:
        try:
            yolo5_model = torch.hub.load(
                "ultralytics/yolov5", "yolov5s", pretrained=True
            )
            yolo5_model.eval()
            safe_print("[MODEL] Loaded YOLOv5s (object detection) via torch.hub")
        except Exception as e:
            safe_print(f"[MODEL] YOLOv5 load failed or skipped: {e}")
            yolo5_model = None
    else:
        yolo5_model = None

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# -----------------------------
# CCTV-tuned standing/sitting/lying classifier
# -----------------------------

def classify_pose_from_kpts(kpts, frame_shape):
    """
    Distance-invariant CCTV pose classifier.
    Input:
        kpts: (17,3) keypoints (pixel OR normalized)
        frame_shape: (H,W)
    Output:
        "Standing" | "Sitting" | "Lying" | "Unknown"
    """
    try:
        if kpts is None or len(kpts) < 17:
            return "Unknown"

        if isinstance(frame_shape, (tuple, list)):
            H, W = float(frame_shape[0]), float(frame_shape[1])
        else:
            H = float(frame_shape)
            W = H * 16.0 / 9.0

        k = np.asarray(kpts, dtype=np.float32)

        # normalize if needed
        if np.nanmax(k[:, 0]) <= 1.1 and np.nanmax(k[:, 1]) <= 1.1:
            k[:, 0] *= W
            k[:, 1] *= H

        # -----------------------------
        # KEY JOINTS (COCO ORDER)
        # -----------------------------
        LSH, RSH = k[5], k[6]
        LHP, RHP = k[11], k[12]
        LK, RK = k[13], k[14]
        LA, RA = k[15], k[16]

        mid_sh = (LSH + RSH) / 2.0
        mid_hp = (LHP + RHP) / 2.0

        # -----------------------------
        # FEATURE 1: TORSO HEIGHT (NORMALIZED)
        # -----------------------------
        torso_h = abs(mid_sh[1] - mid_hp[1]) / (H + 1e-6)

        # -----------------------------
        # FEATURE 2: LEG EXTENSION RATIO
        # -----------------------------
        hip_y = mid_hp[1]
        knee_y = (LK[1] + RK[1]) / 2.0
        ankle_y = (LA[1] + RA[1]) / 2.0

        hip_to_knee = abs(hip_y - knee_y)
        knee_to_ankle = abs(knee_y - ankle_y)
        hip_to_ankle = abs(hip_y - ankle_y)

        leg_extension = hip_to_ankle / (hip_to_knee + knee_to_ankle + 1e-6)

        # -----------------------------
        # FIX-2: KNEE BEND CONFIDENCE
        # -----------------------------
        def angle(a, b, c):
            ba = a - b
            bc = c - b
            cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

        left_knee_angle = angle(LHP, LK, LA)
        right_knee_angle = angle(RHP, RK, RA)

        knee_bent = min(left_knee_angle, right_knee_angle) < 150

        # -----------------------------
        # FEATURE 3: BODY ASPECT RATIO
        # -----------------------------
        body_w = np.nanmax(k[:, 0]) - np.nanmin(k[:, 0]) + 1e-6
        body_h = np.nanmax(k[:, 1]) - np.nanmin(k[:, 1]) + 1e-6
        body_ratio = body_h / body_w
        
        # -----------------------------
        # FIX-6: TOP-DOWN BODY SPREAD (ALWAYS DEFINED)
        # -----------------------------
        x_spread = body_w
        y_spread = body_h
        spread_ratio = max(x_spread, y_spread) / min(x_spread, y_spread)

        # -----------------------------
        # FEATURE 4: TORSO ANGLE
        # -----------------------------
        dx = mid_sh[0] - mid_hp[0]
        dy = mid_hp[1] - mid_sh[1]  # hip below shoulder

        torso_angle = abs(np.degrees(np.arctan2(dx, dy)))

        # -----------------------------
        # DISTANCE ESTIMATION (VERY SIMPLE)
        # -----------------------------
        pixel_height = body_h
        far_person = pixel_height < (0.18 * H)  # tune: 15–20%

        # -----------------------------
        # VISIBILITY CHECK
        # -----------------------------
        valid_pts = np.isfinite(k[:, 0]) & np.isfinite(k[:, 1]) & (k[:, 2] > 0.25)
        visible_ratio = np.sum(valid_pts) / len(k)

        # If person is barely visible → Unknown
        if visible_ratio < 0.4:
            return "Unknown"

        # -----------------------------
        # POSE CLASSIFICATION (FINAL – CCTV HARD RULED)
        # -----------------------------

        pose_label = "Standing"  # SAFE DEFAULT

        # 1️⃣   LYING — horizontal body (very strict)
        if (
            (torso_angle > 60 and body_ratio < 1.2)
            or
            (spread_ratio > 2.4 and torso_h < 0.10)
        ):
            pose_label = "Lying"


        # 2️⃣  SITTING — LOW TORSO HEIGHT (PRIMARY)
        elif knee_bent and torso_h < 0.20:
            pose_label = "Sitting"

        # else → keep Standing (do NOT fall back to Unknown)

        # 2.5  SITTING — sofa / chair style (legs vertical, torso compact)
        elif (
            torso_h < 0.11
            and leg_extension > 0.95
            and body_ratio > 2.0
        ):
            pose_label = "Sitting"

        # 3️⃣  SATNDING — tall torso
        elif torso_h >= 0.12:
            pose_label = "Standing"

        safe_print(
            f"[POSE_DBG] vis={visible_ratio:.2f} torso={torso_h:.3f} "
            f"leg={leg_extension:.2f} ratio={body_ratio:.2f} angle={torso_angle:.1f}"
        )

        return pose_label

    except Exception as e:
        safe_print(f"[POSE_DBG] classify error: {e}")
        return "Unknown"


# -----------------------------
# Movement score
# -----------------------------
def compute_movement_score(prev_px, curr_px, max_reasonable_px=1e4):
    """
    Returns amplified mean L2 pixel displacement across valid keypoints.
    """
    try:
        if prev_px is None or curr_px is None:
            return 0.0

        prev = np.asarray(prev_px, dtype=np.float32)
        curr = np.asarray(curr_px, dtype=np.float32)

        if prev.shape != curr.shape or prev.shape[0] == 0:
            return 0.0

        # use x,y only
        prev_xy = prev[:, :2]
        curr_xy = curr[:, :2]

        # sanitize
        prev_xy = np.nan_to_num(prev_xy, nan=0.0, posinf=max_reasonable_px, neginf=0.0)
        curr_xy = np.nan_to_num(curr_xy, nan=0.0, posinf=max_reasonable_px, neginf=0.0)

        diffs = np.linalg.norm(curr_xy - prev_xy, axis=1)

        # ignore insane jumps
        valid = np.isfinite(diffs) & (diffs < max_reasonable_px * 0.25)
        if valid.sum() == 0:
            return 0.0

        raw_motion = float(np.mean(diffs[valid]))

        # 🔥 amplify for CCTV-scale movement
        amplified = raw_motion * 200.0

        # clamp to keep fusion stable
        return float(np.clip(amplified, 0.0, 10.0))

    except Exception:
        return 0.0


def save_autotrain_sample(camera_name, seq):
    try:
        ts = int(time.time())
        path = os.path.join(AUTOTRAIN_SAVE_DIR, f"{camera_name}_{ts}.npy")
        np.save(path, np.array(seq, dtype=np.float32))
        safe_print(f"[AUTOLEARN] saved: {path}")
    except Exception as e:
        safe_print("[AUTOLEARN ERROR]", e)


def is_inside_zone(x, y, zone_pts):
    try:
        contour = np.array(zone_pts, dtype=np.int32)
        return cv2.pointPolygonTest(contour, (int(x), int(y)), False) >= 0
    except Exception:
        return False


# -----------------------------------------
# MOTION GRADIENT CLASSIFIER (very robust)
# -----------------------------------------
def classify_motion(movement_score, prev_movement_score):
    """
    Returns: "Idle", "Walking", "Running"
    Uses smooth temporal motion to prevent jittery false walking.
    """

    # temporal smoothing
    motion = 0.6 * movement_score + 0.4 * prev_movement_score

    # Walking thresholds (tuned for CCTV + YOLO keypoints)
    if motion < 0.008:
        return "Idle"
    elif motion < 0.035:
        return "Walking"
    else:
        return "Running"


def detect_pose_wrapper(frame):
    try:
        results = pose_model(frame)
        if not results:
            return None

        r = results[0]

        if r.keypoints is None:
            return None

        # Prefer normalized keypoints
        if hasattr(r.keypoints, "xyn") and r.keypoints.xyn is not None:
            xy = r.keypoints.xyn[0].cpu().numpy()  # (17,2)
        elif hasattr(r.keypoints, "xy") and r.keypoints.xy is not None:
            xy = r.keypoints.xy[0].cpu().numpy()  # (17,2)
        else:
            return None

        # confidence
        if hasattr(r.keypoints, "conf") and r.keypoints.conf is not None:
            conf = r.keypoints.conf[0].cpu().numpy()
        else:
            conf = np.ones((xy.shape[0],), dtype=np.float32)

        kpts = np.concatenate([xy, conf.reshape(-1, 1)], axis=1)  # (17,3)
        return kpts

    except Exception as e:
        return None


# -----------------------------
# Display worker (optional)
# -----------------------------
def start_display_worker():
    if not SHOW_WINDOWS or cv2 is None:
        return

    def worker():
        while True:
            try:
                for cam, frm in list(display_frames.items()):
                    if frm is None:
                        continue
                    try:
                        cv2.imshow(f"Camera: {cam}", frm)
                    except Exception as e:
                        safe_print("[display_worker] error:", e)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    safe_print("[INFO] Quit requested (display)")
                    os._exit(0)
                time.sleep(0.02)
            except Exception as e:
                safe_print("[display_worker] unexpected error:", e)
                time.sleep(0.5)

    t = threading.Thread(target=worker, daemon=True)
    t.start()


SMOOTH_WINDOW = 8

log_throttle = defaultdict(lambda: 0.0)


def normalize_keypoints_to_unit(kp, frame_w, frame_h):
    """Return Nx3 np.float32 keypoints normalized to [0,1] in x,y and keep score in col2.
    Safe against NaN/inf and will clip values into [0,1]. If kp already appears to be normalized
    (max x <= 1.1) we still clip to [0,1]."""
    if kp is None:
        return None
    kp = np.asarray(kp, dtype=np.float32)
    if kp.size == 0:
        return None
    # shape (N,>=2)
    try:
        xs = kp[:, 0].astype(np.float32)
        ys = kp[:, 1].astype(np.float32)
    except Exception:
        return None

    # detect if already normalized (likely <=1.1)
    if np.nanmax(xs) <= 1.1 and np.nanmax(ys) <= 1.1:
        xs = np.nan_to_num(xs, nan=0.0, posinf=1.0, neginf=0.0)
        ys = np.nan_to_num(ys, nan=0.0, posinf=1.0, neginf=0.0)
    else:
        # convert pixels -> normalized
        # defend division by zero
        frame_w = max(1.0, float(frame_w))
        frame_h = max(1.0, float(frame_h))
        xs = np.nan_to_num(xs / frame_w, nan=0.0, posinf=1.0, neginf=0.0)
        ys = np.nan_to_num(ys / frame_h, nan=0.0, posinf=1.0, neginf=0.0)

    xs = np.clip(xs, 0.0, 1.0)
    ys = np.clip(ys, 0.0, 1.0)
    out = kp.copy().astype(np.float32)
    out[:, 0] = xs
    out[:, 1] = ys
    # ensure score col exists
    if out.shape[1] < 3:
        sc = np.ones((out.shape[0], 1), dtype=np.float32)
        out = np.concatenate([out, sc], axis=1)
    return out


def scale_keypoints_to_pixels(kp_norm, frame_w, frame_h):
    """Return a pixel Nx3 keypoints array. Handles None or already-pixel input."""
    if kp_norm is None:
        return None
    kp = np.asarray(kp_norm, dtype=np.float32)
    # if values appear already in pixel space (max>2*min(frame dimension)), assume already pixels
    if np.nanmax(kp[:, :2]) > max(frame_w, frame_h) * 1.5:
        return kp  # already pixels
    frame_w = max(1.0, float(frame_w))
    frame_h = max(1.0, float(frame_h))
    out = kp.copy()
    out[:, 0] = np.nan_to_num(out[:, 0], nan=0.0, posinf=1.0, neginf=0.0) * frame_w
    out[:, 1] = np.nan_to_num(out[:, 1], nan=0.0, posinf=1.0, neginf=0.0) * frame_h
    out[:, 0] = np.clip(out[:, 0], 0, frame_w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, frame_h - 1)
    return out


prev_keypoints_pixels = defaultdict(lambda: None)
last_fusion_status = defaultdict(lambda: "Unknown")


# -----------------------------
# SAFE POSE DETECTION WRAPPER
# -----------------------------
def safe_detect_pose(frame, camera_name=None):
    try:
        kp_raw = detect_pose_wrapper(frame)
    except Exception as e:
        safe_print(f"[{camera_name}] ⚠️ detect_pose_wrapper error: {e}")
        return None

    if kp_raw is None:
        return None

    try:
        if hasattr(kp_raw, "cpu"):
            arr = kp_raw.cpu().numpy()
        else:
            arr = np.array(kp_raw)

        if arr.size == 0:
            return None

        k = arr[0] if arr.ndim == 3 else arr
        k = k.astype(np.float32)

        if k.shape[1] < 2:
            return None

        xy = k[:, :2]
        conf = k[:, 2] if k.shape[1] >= 3 else np.ones((xy.shape[0],), dtype=np.float32)

        h, w = frame.shape[:2]
        xy_norm = np.zeros_like(xy, dtype=np.float32)
        xy_norm[:, 0] = xy[:, 0] / w if w > 0 else 0.0
        xy_norm[:, 1] = xy[:, 1] / h if h > 0 else 0.0

        conf = np.asarray(conf, dtype=np.float32)
        if conf.ndim == 0:
            conf = np.full((xy_norm.shape[0],), float(conf), dtype=np.float32)

        kpts = np.concatenate([xy_norm, conf.reshape(-1, 1)], axis=1)

        if kpts.shape[0] < 17:
            pad = np.full((17 - kpts.shape[0], 3), np.nan, dtype=np.float32)
            kpts = np.vstack([kpts, pad])

        return kpts[:17, :3]

    except Exception as e:
        safe_print(f"[{camera_name}] [POSE ERROR] {e}")
        return None


def extract_action_features(kpts_seq):
    """
    kpts_seq: list of (17,3) normalized keypoints
    Returns dict of temporal features
    """
    if len(kpts_seq) < 5:
        return None

    kpts = np.array(kpts_seq, dtype=np.float32)

    # torso vertical movement
    torso_y = (kpts[:, 5, 1] + kpts[:, 6, 1]) / 2.0
    torso_dy = np.diff(torso_y)

    # overall motion
    motion_energy = np.mean(np.linalg.norm(np.diff(kpts[:, :, :2], axis=0), axis=2))

    # hand movement (for waving/fighting)
    lw = kpts[:, 9, :2]
    rw = kpts[:, 10, :2]
    hand_motion = np.mean(
        np.linalg.norm(np.diff(lw, axis=0), axis=1)
        + np.linalg.norm(np.diff(rw, axis=0), axis=1)
    )

    return {
        "torso_drop": np.sum(torso_dy > 0.04),
        "motion_energy": motion_energy,
        "hand_motion": hand_motion,
    }


def classify_action_from_features(feat, pose, motion_score):
    if feat is None:
        return None, 0.0

    # ------------------
    # FALLING (event)
    # ------------------
    if feat["torso_drop"] >= 2 and motion_score > 1.2:
        return "Falling", 0.95

    # ------------------
    # RUNNING / WALKING
    # ------------------
    if motion_score > 6.0:
        return "Running", 0.85
    elif motion_score > 1.5:
        return "Walking", 0.75

    # ------------------
    # BENDING / PICKING
    # ------------------
    if pose == "Standing" and feat["torso_drop"] >= 1 and motion_score < 1.5:
        return "Bending", 0.70

    # ------------------
    # WAVING
    # ------------------
    if feat["hand_motion"] > 1.0 and motion_score < 2.0:
        return "Waving", 0.75

    return None, 0.0


# -----------------------------
# Main monitor loop per camera
# -----------------------------


def monitor_camera(camera_name, camera_url):
    """Single-camera monitor loop (replace your current monitor_camera block with this).
    This version:
      - ensures variables like deque are available
      - uses safe_detect_pose to avoid 'kp' undefined
      - stabilizes pose fusion buffers
      - avoids unclosed try blocks
    """

    cap = None
    if cv2 is not None:
        cap = cv2.VideoCapture(camera_url)

    if cap is None or (hasattr(cap, "isOpened") and not cap.isOpened()):
        safe_print(f"[ERROR] Could not open camera {camera_name}: {camera_url}")
        return

    safe_print(f"[INFO] Camera {camera_name} started loop")

    # ensure per-camera globals exist
    prev_motion_score.setdefault(camera_name, 0.0)
    pose_calibration_data.setdefault(camera_name, [])
    last_torso_norm.setdefault(camera_name, None)
    camera_pose_thresholds.setdefault(camera_name, None)
    autolearn_last_saved.setdefault(camera_name, 0.0)
    fall_detected.setdefault(camera_name, False)
    last_fusion_log_time.setdefault(camera_name, 0.0)
    motion_debug_last.setdefault(camera_name, 0.0)
    alert_last_sent.setdefault((camera_name, "fall"), 0.0)
    display_frames.setdefault(camera_name, None)
    pose_change_buffer.setdefault(camera_name, deque(maxlen=8))
    action_sequence.setdefault(camera_name, deque(maxlen=32))
    pose_stable_count.setdefault(camera_name, 0)
    unified_history.setdefault(camera_name, deque(maxlen=64))
    motion_log_last.setdefault(camera_name, 0.0)

    # --- REQUIRED: initialize all per-camera buffers ---
    unified_history.setdefault(camera_name, deque(maxlen=8))
    action_sequence.setdefault(camera_name, deque(maxlen=32))
    prev_keypoints.setdefault(camera_name, None)
    prev_keypoints_pixels.setdefault(camera_name, None)
    pose_stable_count.setdefault(camera_name, 0)
    last_action_time.setdefault(camera_name, 0.0)
    # ---------------------------------------------------

    # -----------------------------
    # FRAME-LOCAL DEFAULTS (MUST EXIST)
    # -----------------------------
    motion_label = "Idle"  # <-- DEFAULT (prevents UnboundLocalError)
    motion_state = "Idle"
    movement_score = 0.0

    # ensure pose_accepted always exists
    pose_accepted = False

    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        now = time.time()
        frame_counter += 1

        # -----------------------------
        # INIT CONTEXT OBJECTS (ALWAYS SAFE)
        # -----------------------------
        context_objects = []
        
        # -----------------------------
        # PERSON BBOX (SAFE INIT)
        # -----------------------------
        person_bbox = None

        # frame-local state defaults
        keypoints = None
        action_label = None
        action_conf = 0.0

        pose_label = "Unknown"          # 🔒 NEVER read fused state here
        candidate_pose = "Unknown"

        pose_conf = 0.0
        movement_score = 0.0

        # -----------------------------
        # STEP 0: POSE EXTRACTION (SAFE)
        # -----------------------------
        keypoints = safe_detect_pose(frame, camera_name)

        if keypoints is None:
            safe_print(f"[{camera_name}] ⚠️ NO PERSON / NO KEYPOINTS")

        # -----------------------------
        # CONTEXT OBJECTS (YOLOv5)
        # -----------------------------
        context_objects = []
        person_bbox = None

        try:
            if yolo5_model is not None:
                results = yolo5_model(frame)   # AutoShape handles everything
                df = results.pandas().xyxy[0]

                for _, row in df.iterrows():
                    conf = float(row["confidence"])
                    if conf < 0.45:
                        continue

                    label = row["name"]
                    bbox = (
                        float(row["xmin"]),
                        float(row["ymin"]),
                        float(row["xmax"]),
                        float(row["ymax"]),
                    )

                    if label == "person":
                        person_bbox = bbox

                    elif label in ("sofa", "chair", "bench", "bed", "stool"):
                        context_objects.append({
                            "label": label,
                            "bbox": bbox
                        })

        except Exception as e:
            safe_print(f"[{camera_name}] ⚠️ YOLO context failed: {e}")

        # -----------------------------
        # STEP 1: POSE CLASSIFICATION
        # -----------------------------
        pose_model = None
        pose_conf = 0.0

        if keypoints is not None:
            try:
                pose_model, pose_conf = predict_pose_from_sequence(
                    action_sequence[camera_name]  # or whatever buffer you trained on
                )
            except Exception as e:
                safe_print(f"[{camera_name}] ⚠️ pose model failed: {e}")
                pose_model = None                
                
                # =========================================================
                # SUPPORT CONTACT REASONING (GEOMETRY-FREE, STATELESS)
                # =========================================================
                # --- SUPPORT STATE SAFETY GUARD ---
                if keypoints is None or len(keypoints) < 17:
                    raise ValueError("Insufficient keypoints for support reasoning")


                # --- SAFE PIXEL KEYPOINTS FOR SUPPORT ---
                frame_h, frame_w = frame.shape[:2]

                k = np.asarray(keypoints, dtype=np.float32).copy()
                k[:, 0] *= frame_w
                k[:, 1] *= frame_h

                support_state = {
                    "torso":  "free",
                    "pelvis": "free",
                    "feet":   "free",
                }

                try:
                    # -------- keypoints (pixel space) --------
                    LH, RH = k[11], [12]   # hip
                    LS, RS = k[5],  [6]    # shoulders
                    LA, RA = k[15], [16]   # ankles

                    hip_y      = (LH[1] + RH[1]) / 2.0
                    shoulder_y = (LS[1] + RS[1]) / 2.0
                    ankle_y    = (LA[1] + RA[1]) / 2.0

                    body_h = np.nanmax(kpts_px[:, 1]) - np.nanmin(kpts_px[:, 1]) + 1e-6
  
                    # -------- object support --------
                    for obj in context_objects or []:
                        ox1, oy1, ox2, oy2 = obj["bbox"]
                        obj_top = oy1

                        # pelvis support (chair / sofa / stool)
                        if abs(hip_y - obj_top) < 0.30 * body_h:
                            support_state["pelvis"] = "supported"

                        # torso support (bed / sofa)
                        if abs(shoulder_y - obj_top) < 0.20 * body_h:
                            support_state["torso"] = "supported"

                    # -------- feet / ground --------
                    if ankle_y > (0.90 * frame_h) and (ankle_y - hip_y) > (0.25 * body_h):
                        support_state["feet"] = "supported"

                except Exception as e:
                    safe_print(f"[{camera_name}] ⚠️ support-state calc failed: {e}")

                    # OPTIONAL: learned pose override
                    try:
                        learned_pose, learned_conf = predict_pose_from_sequence(
                            action_sequence[camera_name]
                        )

                        if learned_conf > 0.75:
                            pose_label = learned_pose
                    except Exception:
                        pass

                    # -----------------------------
                    # POSE FROM SUPPORT (AUTHORITATIVE)
                    # =========================================================

                    pose_label = pose_model  # start with model output

                    # refinement only
                    if pose_model == "Standing" and pelvis_supported:
                        pose_label = "Sitting"

                    if pose_model == "Standing" and torso_supported:
                        pose_label = "Lying"

                    if pose_model == "Sitting" and torso_supported:
                        pose_label = "Lying"

                except Exception as e:
                    safe_print(f"[{camera_name}] ⚠️ support-contact failed: {e}")
                    support_state = None 
                   
                    # =========================================================
                    # SUPPORT TEMPORAL CONFIRMATION (ANTI-JITTER)
                    # =========================================================

                    support_hist = unified_history.setdefault(
                        (camera_name, "support"), deque(maxlen=7)
                    )

                    support_hist.append(support_state)

                    def confirmed_support(hist):
                        for s in ("torso", "pelvis", "feet"):
                            if hist.count(s) >= 3:
                                return s
                        return None

                    stable_support = confirmed_support(support_hist)

                    # =========================================================
                    # POSE ELIGIBILITY — WHAT IS PHYSICALLY POSSIBLE?
                    # =========================================================

                    eligible_poses = set()

                    if support_state["torso"] == "supported":
                        eligible_poses.add("Lying")

                    if support_state["pelvis"] == "supported":
                        eligible_poses.add("Sitting")

                    if support_state["feet"] == "supported":
                        eligible_poses.add("Standing")

                    # IMPORTANT:
                    # If nothing is eligible, DO NOT FORCE ANYTHING

                    # =========================================================
                    # FINAL SUPPORT CONSTRAINTS (STRICT, MULTI-SIGNAL)
                    # =========================================================

                    allowed_poses = {"Standing", "Sitting", "Lying"}

                    # ---------------------------------------------------------
                    # LYING: must satisfy MULTIPLE conditions
                    # ---------------------------------------------------------
                    lying_confirmed = (
                        torso_supported
                        and not feet_load_bearing          # feet not supporting weight
                        and leg_extension < 0.35          # CRITICAL: legs collapsed
                        and body_ratio < 1.2               # body spread horizontally
                    )
                 
                    if lying_confirmed:
                        allowed_poses = {"Lying"}

                    # ---------------------------------------------------------
                    # SITTING: pelvis supported, but torso NOT fully supported
                    # ---------------------------------------------------------
                    elif pelvis_supported:
                        allowed_poses &= {"Sitting", "Standing"}

                    # ---------------------------------------------------------
                    # STANDING: requires feet load-bearing
                    # ---------------------------------------------------------
                    if not (feet_on_ground and feet_load_bearing):
                        allowed_poses.discard("Standing")

                    # =========================================================
                    # FINAL POSE SELECTION (NO UNKNOWN, NO GEOMETRY OVERRIDE)
                    # =========================================================

                    if lying_confirmed:
                        pose_label = "Lying"

                    elif pelvis_supported:
                        pose_label = "Sitting"

                    elif feet_on_ground and feet_load_bearing:
                        pose_label = "Standing"

                    else:
                        # Free flow: do NOT force Unknown or Lying
                        pose_label = candidate_pose

                # =========================================================
                # STEP 1C: TEMPORAL POSE STABILITY (ANTI-FLIP)
                # =========================================================
                pose_hist = unified_history.setdefault(
                    (camera_name, "pose"), deque(maxlen=5)
                )

                pose_hist.append(pose_label)
              
                if pose_label is None:
                    pose_label = last_fusion_status.get(camera_name, "Unknown")

            except Exception as e:
                safe_print(f"[{camera_name}] ⚠️ pose classify failed: {e}")
                pose_label = "Unknown"


        # -----------------------------
        # POSE STABILITY / ACCEPTANCE
        # -----------------------------
        prev_pose_label.setdefault(camera_name, None)

        if pose_label in ("Standing", "Sitting", "Lying"):
            if pose_label == prev_pose_label[camera_name]:
                pose_stable_count[camera_name] += 1
            else:
                pose_stable_count[camera_name] = 1
                prev_pose_label[camera_name] = pose_label
        else:
            pose_stable_count[camera_name] = 0
            prev_pose_label[camera_name] = None

        pose_accepted = pose_stable_count[camera_name] >= POSE_STABLE_REQUIRED


        # -----------------------------
        # THROTTLED POSE LOG (RESTORED)
        # -----------------------------
        if now - log_throttle[(camera_name, "pose")] > 10.0:
            safe_print(f"[{camera_name}] 🧍 YOLO Pose -> {pose_label}")
            log_throttle[(camera_name, "pose")] = now

        # -----------------------------
        # STEP 2 — MOTION (PIXEL SPACE)
        # -----------------------------
        movement_score = 0.0
        motion_state = "Idle"

        if keypoints is not None and len(keypoints) >= 5:
            frame_h, frame_w = frame.shape[:2]

            curr_px = scale_keypoints_to_pixels(keypoints, frame_w, frame_h)

            prev_px = prev_keypoints_pixels.get(camera_name)

            if prev_px is not None:
                movement_score = compute_movement_score(prev_px, curr_px)

            # ✅ correct update
            prev_keypoints_pixels[camera_name] = curr_px

            if movement_score < MOTION_IDLE_EPS:
                motion_state = "Idle"
            elif movement_score > MOTION_MOVING:
                motion_state = "Moving"
            else:
                motion_state = "Subtle"

            now_ts = time.time()
            if now_ts - motion_log_last[camera_name] >= 10.0:
                motion_log_last[camera_name] = now_ts
                safe_print(
                    f"[{camera_name}] MOTION={movement_score:.3f} → {motion_state}"
                )

        # -----------------------------
        # CONTEXT OBJECT EXTRACTION (SAFE)
        # -----------------------------
        context_objects = []

        try:
            if yolo5_model is not None:
                res = yolo5_model(frame, verbose=False)
                df = res.pandas().xyxy[0]

                for _, row in df.iterrows():
                    if row["name"] in ("chair", "sofa", "bench", "bed"):
                        context_objects.append({
                            "label": row["name"],
                            "bbox": (
                                float(row["xmin"]),
                                float(row["ymin"]),
                                float(row["xmax"]),
                                float(row["ymax"]),
                            ),
                        })
        except Exception:
            context_objects = []

        # -----------------------------
        # SUPPORT-AWARE POSE CORRECTION
        # -----------------------------
        pose_label = candidate_pose  # start from geometric pose

        if person_bbox is not None and context_objects:

            px1, py1, px2, py2 = person_bbox
            person_h = py2 - py1

            for obj in context_objects:
                ox1, oy1, ox2, oy2 = obj["bbox"]

                horizontal_overlap = (min(px2, ox2) - max(px1, ox1)) > 0.3 * (px2 - px1)
                vertical_support = abs(py2 - oy1) < 0.25 * person_h

                # 🪑 Sitting support (sofa / chair / bench)
                if (
                    pose_label == "Standing"
                    and obj["label"] in ("chair", "sofa", "bench")
                    and horizontal_overlap
                    and vertical_support
                ):
                    pose_label = "Sitting"
                    break

                # 🛏 Lying support (bed)
                if (
                    obj["label"] == "bed"
                    and horizontal_overlap
                    and abs((py1 + py2) / 2 - (oy1 + oy2) / 2) < 0.3 * person_h
                ):
                    pose_label = "Lying"
                    break

        # -----------------------------
        # PHYSICAL SUPPORT REASONING (POST-POSE)
        # -----------------------------
        final_pose_override = candidate_pose

        if person_bbox is not None and context_objects and keypoints is not None:

            px1, py1, px2, py2 = person_bbox
            person_h = py2 - py1

            # Key joints (pixel space)
            LHP, RHP = kpts_px[11], kpts_px[12]   # hips
            LA, RA   = kpts_px[15], kpts_px[16]   # ankles

            pelvis_y = (LHP[1] + RHP[1]) / 2.0
            feet_y   = (LA[1] + RA[1]) / 2.0
            torso_y  = (kpts_px[5][1] + kpts_px[6][1]) / 2.0  # shoulders mid

            for obj in context_objects:
                ox1, oy1, ox2, oy2 = obj["bbox"]

                horizontal_overlap = (min(px2, ox2) - max(px1, ox1)) > 0.3 * (px2 - px1)

                # -----------------------------
                # SITTING (pelvis supported)
                # -----------------------------
                if (
                    obj["label"] in ("chair", "sofa", "bench")
                    and horizontal_overlap
                    and abs(pelvis_y - oy1) < 0.25 * person_h
                ):
                    final_pose_override = "Sitting"
                    break

                # -----------------------------
                # LYING (torso supported)
                # -----------------------------
                if (
                    obj["label"] in ("bed", "sofa")
                    and horizontal_overlap
                    and abs(torso_y - oy1) < 0.3 * person_h
                ):
                    final_pose_override = "Lying"
                    break

            # -----------------------------
            # STANDING VALIDATION (feet on ground)
            # -----------------------------
            ground_y = frame_h * 0.95
            if final_pose_override == "Standing":
                if abs(feet_y - ground_y) > 0.15 * person_h:
                    final_pose_override = "Unknown"

        # overwrite pose
        candidate_pose = final_pose_override
       
        # -----------------------------
        # SOFA / CHAIR SITTING OVERRIDE
        # -----------------------------
        if (
            candidate_pose == "Standing"
            and context_objects
            and not feet_load_bearing
        ):
            for obj in context_objects:
                if obj["label"] in ("sofa", "chair", "bench"):
                    pose_label = "Sitting"
                    break

        # -----------------------------
        # POSE STICKINESS (ANTI-FLIP)
        # -----------------------------
        pose_mem = unified_history.setdefault(
            (camera_name, "pose_stable"), deque(maxlen=8)
        )

        pose_mem.append(candidate_pose)

        # Prevent Sitting → Standing flip if recently sitting
        if candidate_pose == "Standing":
            if pose_mem.count("Sitting") >= 4:
               candidate_pose = "Sitting"

        # -----------------------------
        # STEP 3: ACTION RECOGNITION
        # -----------------------------
        if keypoints is not None:
            action_sequence[camera_name].append(keypoints)

        if now - last_action_time.get(camera_name, 0.0) >= ACTION_INTERVAL:
            last_action_time[camera_name] = now
            seq = list(action_sequence[camera_name])

            raw_action_label = None
            action_label = None
            action_conf = 0.0

        # -----------------------------
        # RULE-BASED ACTION DETECTION
        # -----------------------------
        if len(action_sequence[camera_name]) >= 8:
            feats = extract_action_features(action_sequence[camera_name])
            action_label, action_conf = classify_action_from_features(
                feats, final_pose, movement_score
            )

            # if frame_counter % 30 == 0:
            #    safe_print(f"[ACTION_DIAG] seq_len={len(seq)}, last_kpt_mean={float(last_mean)}")
            if len(seq) >= 6:
                try:
                    raw_action_label, action_conf = predict_action(
                        seq, movenet_pose=candidate_pose
                    )
                except Exception as e:
                    safe_print(f"[{camera_name}] ⚠️ Action block failed: {e}")
                    raw_action_label, action_conf = None, 0.0

            # confidence floor
            if action_label and action_conf < 0.65:
                action_label = None

            # sanitize pose-like responses from action model
            POSE_NAMES = {"Standing", "Sitting", "Lying", "Unknown", None}
            if raw_action_label in POSE_NAMES:
                #   safe_print(f"[ACTION_DBG] action returned pose-name '{raw_action_label}' → ignoring as action")
                action_label = None
            else:
                action_label = (
                    str(raw_action_label).strip().capitalize()
                    if raw_action_label is not None
                    else None
                )

            # -----------------------------
            # WAVING DETECTION (ROBUST)
            # -----------------------------
            try:
                # COCO indices
                LW, RW = keypoints[9], keypoints[10]  # wrists
                LE, RE = keypoints[7], keypoints[8]  # elbows
                LK, RK = keypoints[13], keypoints[14]  # knees

                # arm motion
                arm_motion = (abs(LW[0] - LE[0]) + abs(RW[0] - RE[0])) * frame_w

                # leg motion (proxy for walking/running)
                leg_motion = (abs(LK[1] - LE[1]) + abs(RK[1] - RE[1])) * frame_h

                if (
                    candidate_pose == "Standing"
                    and arm_motion > 20
                    and arm_motion > leg_motion * 1.5
                ):
                    action_label = "Waving"
                    action_conf = 0.75

            except Exception:
                pass

            # -----------------------------
            # POSE CONFIDENCE GATE
            # -----------------------------
            pose_ok_for_locomotion = (
                candidate_pose == "Standing"
                and pose_stable_count.get(camera_name, 0) >= POSE_STABLE_REQUIRED
            )

            # -----------------------------
            # ACTION FROM MOTION (SMOOTHED)
            # -----------------------------

            # keep per-camera action memory
            action_hist = unified_history.setdefault(
                (camera_name, "action"), deque(maxlen=6)
            )

            new_action = None
            new_conf = 0.0

            # -----------------------------
            # LEG CADENCE (ANTI-FALSE WALK)
            # -----------------------------
            try:
                LK, RK = keypoints[13], keypoints[14]  # knees (COCO)

                knee_delta = abs(LK[1] - RK[1]) * frame_h

                cadence_hist = unified_history.setdefault(
                    (camera_name, "knee_delta"), deque(maxlen=6)
                )
                cadence_hist.append(knee_delta)

                knee_var = (
                    float(np.std(cadence_hist)) if len(cadence_hist) >= 3 else 0.0
                )

            except Exception:
                knee_var = 0.0

            # -----------------------------
            # ACTION DECISION (POSE-AWARE)
            # -----------------------------
            if pose_ok_for_locomotion:
                if avg_motion >= RUNNING_TH:
                    new_action = "Running"
                    new_conf = 0.80
                elif avg_motion >= WALKING_TH and knee_var > 8.0:
                    new_action = "Walking"
                    new_conf = 0.60
                elif avg_motion < 0.5:
                    new_action = "Idle"
                    new_conf = 0.50

            else:
                # pose not reliable → freeze locomotion
                new_action = "Idle"
                new_conf = 0.50

            # push into history
            if new_action:
                action_hist.append(new_action)

            # majority vote
            if len(action_hist) >= 3:
                action_label = Counter(action_hist).most_common(1)[0][0]
                action_conf = new_conf
            else:
                action_label = None

            # cadence estimation (simple)
            motion_buffer.setdefault(camera_name, deque(maxlen=6))
            motion_buffer[camera_name].append(movement_score)

            cadence = np.std(motion_buffer[camera_name])

            # -----------------------------
            # ACTION CONFIDENCE SANITY
            # -----------------------------
            if action_label in ("Running", "Walking") and action_conf < 0.5:
                action_label = "Other"
                action_conf = 0.0

            # -----------------------------
            # WALKING FALSE-POSITIVE FILTER
            # -----------------------------
            walk_counter = unified_history.setdefault((camera_name, "walk_frames"), 0)

            if action_label == "Walking":
                walk_counter += 1
            else:
                walk_counter = 0

            unified_history[(camera_name, "walk_frames")] = walk_counter

            # require sustained walking
            if action_label == "Walking" and walk_counter < 6:
                action_label = None
                action_conf = 0.0

            # -----------------------------
            # MOTION LABEL (FOR ACTION LOGIC)
            # -----------------------------
            if motion_state == "Idle":
                motion_label = "Idle"
            elif motion_state == "Moving":
                motion_label = "Walking"  # default human motion
            else:
                motion_label = "Unknown"

            # filter low confidence
            if action_conf < MIN_ACTION_CONF:
                action_label = None

            # -----------------------------
            # ACTION TEMPORAL STABILITY
            # -----------------------------
            action_hist = unified_history.setdefault(
                (camera_name, "action_final"), deque(maxlen=5)
            )

            if action_label:
                action_hist.append(action_label)

            if len(action_hist) >= 3:
                action_label = Counter(action_hist).most_common(1)[0][0]

            # waving sanity
            if action_label == "Waving" and movement_score < MIN_WAVE_MOTION:
                action_label = None

            # allow locomotion actions even if pose is unstable
            if not pose_accepted and action_label not in ("Walking", "Running"):
                action_label = None

            # normalize into validated set
            VALID_ACTIONS = {
                "Walking",
                "Running",
                "Waving",
                "Falling",
                "Bending",
                "Picking",
                "Jumping",
                "Idle",
                "Other",
            }
            if action_label not in VALID_ACTIONS:
                action_label = "Other" if action_label else None

            # -----------------------------
            # ACTION DEBOUNCE (ANTI-SPAM)
            # -----------------------------
            now_ts = time.time()
            prev_action = last_action_reported[camera_name]
            prev_time = last_action_time[camera_name]

            action_changed = action_label != prev_action
            cooldown_passed = (now_ts - prev_time) >= ACTION_REPORT_COOLDOWN

            if action_label and action_label not in ("Other", None):
                if action_changed or cooldown_passed:
                    # report action
                    safe_print(
                        f"[{camera_name}] 🟢 ACTION → {action_label} ({action_conf:.2f})"
                    )

                    # update state
                    last_action_reported[camera_name] = action_label
                    last_action_time[camera_name] = now_ts

                    # -----------------------------
                    # AUTOTRAIN SAVE (DEBOUNCED)
                    # -----------------------------
                    if (
                        action_label in ("Walking", "Running")
                        and action_conf >= 0.75
                        and pose_ok_for_locomotion
                    ):
                        try:
                            np.save(
                                f"autotrain_buffer/{camera_name}_{int(now_ts)}.npy",
                                np.array(
                                    action_sequence[camera_name], dtype=np.float32
                                ),
                            )
                            safe_print(
                                f"[AUTOLEARN] saved: autotrain_buffer/{camera_name}_{int(now_ts)}.npy"
                            )
                        except Exception as e:
                            safe_print(f"[AUTOLEARN_ERR] {e}")

        # --- EXTRACT TORSO FEATURE (simple vertical torso length) ---
        try:
            # indices may change depending on your model; adjust if needed:
            LSH = keypoints[11]  # left shoulder (x, y, conf)
            LHP = keypoints[23]  # left hip (x, y, conf)

            torso_v_norm = abs(LSH[1] - LHP[1])  # vertical distance only
            torso_conf = min(LSH[2], LHP[2])  # confidence = weakest of the two
        except Exception:
            torso_v_norm = 0.0
            torso_conf = 0.0

        #  if action_label and action_label != "Other":
        #      safe_print(f"[{camera_name}] 🟢 ACTION → {action_label} ({action_conf:.2f})")

        # -----------------------------
        # SITTING SURFACE IDENTIFICATION
        # -----------------------------
        if pose_label == "Sitting" and context_objects:

            px1, py1, px2, py2 = person_bbox
            person_height = py2 - py1

            for obj in context_objects:
                ox1, oy1, ox2, oy2 = obj["bbox"]

                vertical_contact = abs(py2 - oy1) < 0.2 * person_height
                horizontal_overlap = (min(px2, ox2) - max(px1, ox1)) > 0.3 * (px2 - px1)

                if vertical_contact and horizontal_overlap:
                    if obj["label"] == "sofa":
                        pose_label = "Sitting_On_Sofa"
                    elif obj["label"] == "chair":
                        pose_label = "Sitting_On_Chair"
                    elif obj["label"] == "bench":
                        pose_label = "Sitting_On_Bench"
                    elif obj["label"] == "bed":
                        pose_label = "Sitting_On_Bed"
                    break

        # -----------------------------
        # STEP 4: POSE FUSION (STATE ONLY)
        # -----------------------------
        # Pose is state — accept immediately if valid
        if pose_label in ("Standing", "Sitting", "Lying"):
            final_pose = pose_label
        else:
            final_pose = last_fusion_status.get(camera_name, "Unknown")

        # --- Temporal smoothing ---   
        pose_hist = unified_history.setdefault(
            (camera_name, "pose"), deque(maxlen=5)
        )
        pose_hist.append(final_pose)

        if len(pose_hist) >= 3:
            final_pose = Counter(pose_hist).most_common(1)[0][0]
        else:
            final_pose = pose_label

        last_fusion_status[camera_name] = final_pose

        # -----------------------------
        # LOG (THROTTLED)
        # -----------------------------
        if now - log_throttle[(camera_name, "fused")] > 10.0:
            safe_print(f"[{camera_name}] 🔥 Final FUSED → {final_pose}")
            log_throttle[(camera_name, "fused")] = now

        # -----------------------------
        # STEP 5: FALL ALERT (action-driven)
        # -----------------------------
        try:
            if (
                final_pose == "Falling"
                and action_conf > FALL_CONF_THRESHOLD
                and movement_score > 0.02
            ):
                key = (camera_name, "fall")
                if now - alert_last_sent.get(key, 0.0) > ALERT_COOLDOWN:
                    alert_last_sent[key] = now
                    safe_print(
                        f"[{camera_name}] 🚨 HARD FALL detected — conf={action_conf:.2f}, motion={movement_score:.3f}"
                    )
        except Exception as e:
            safe_print(f"[{camera_name}] ⚠️ Fall alert error: {e}")

        pose_hist = unified_history.setdefault((camera_name, "pose"), deque(maxlen=5))
        pose_hist.append(pose_label)

        if pose_hist.count(pose_label) < 3:
            pose_label = pose_hist[-2] if len(pose_hist) > 1 else pose_label

        person_bbox = None
        context_objects = []

        # -----------------------------
        # STEP 6: YOLOv5 ZONE DETECTION (optional)
        # -----------------------------
        try:
            if yolo5_model is not None and camera_name in ZONES and cv2 is not None:
                res = yolo5_model(frame, verbose=False)
                detections = res.pandas().xyxy[0]
                for _, row in detections.iterrows():
                    label = row["name"]
                    conf = float(row["confidence"])

                    if conf < 0.45:
                        continue

                    xmin, ymin = float(row["xmin"]), float(row["ymin"])
                    xmax, ymax = float(row["xmax"]), float(row["ymax"])
                    bbox = (xmin, ymin, xmax, ymax)

                    # PERSON
                    if label == "person":
                        person_bbox = bbox

                        # Existing zone logic (unchanged)
                        x_center = (xmin + xmax) / 2.0
                        y_center = (ymin + ymax) / 2.0
                        if is_inside_zone(x_center, y_center, ZONES[camera_name]):
                            key = (camera_name, "zone")
                            if now - alert_last_sent.get(key, 0.0) > ALERT_COOLDOWN:
                                alert_last_sent[key] = now
                                safe_print(
                                    f"[{camera_name}] 🚨 Person in restricted zone"
                                )

                    # CONTEXT OBJECTS (NEW)
                    elif label in ("chair", "sofa", "bench", "bed"):
                        context_objects.append({"label": label, "bbox": bbox})

        except Exception:
            pass

        # -----------------------------
        # CONTEXT-AWARE SITTING REFINEMENT
        # -----------------------------
        if pose_label == "Standing" and person_bbox and context_objects:
            px1, py1, px2, py2 = person_bbox
            person_h = py2 - py1

            for obj in context_objects:
                ox1, oy1, ox2, oy2 = obj["bbox"]

                vertical_contact = abs(py2 - oy1) < 0.25 * person_h
                horizontal_overlap = (min(px2, ox2) - max(px1, ox1)) > 0.4 * (px2 - px1)

                if vertical_contact and horizontal_overlap:
                    pose_label = f"Sitting_On_{obj['label'].capitalize()}"
                    break

        # -----------------------------
        # STEP 7: THROTTLED LOGGING + DISPLAY
        # -----------------------------
        try:
            if time.time() - last_log_time.get(camera_name, 0.0) >= LOG_INTERVAL:
                last_log_time[camera_name] = time.time()
                safe_print(
                    f"[{camera_name}] 🧍 Final Pose: {final_pose} | "
                    f"action={action_label if action_label else 'unknown'} "
                )

            if SHOW_WINDOWS and cv2 is not None:
                disp = frame.copy()
                cv2.putText(
                    disp,
                    f"{final_pose}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )
                display_frames[camera_name] = disp

        except Exception:
            pass

        time.sleep(0.01)

    cap.release()


# -----------------------------
# ENTRY POINT
# -----------------------------
def main():
    load_models()
    start_display_worker()

    threads = []
    for name, src in CAMERAS.items():
        t = threading.Thread(target=monitor_camera, args=(name, src), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.5)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        safe_print("[INFO] Shutting down...")


if __name__ == "__main__":
    main()
