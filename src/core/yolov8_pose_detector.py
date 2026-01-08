# src/core/yolov8_pose_detector.py

from ultralytics import YOLO
import numpy as np

pose_model = YOLO("models/yolov8s-pose.pt")

def detect_pose_yolo(frame):
    """
    Returns keypoints as (17,3):
        x,y,confidence
    or None if no person detected.
    """

    try:
        results = pose_model(frame, verbose=False)
    except Exception as e:
        print("YOLO pose error:", e)
        return None

    if not results or len(results) == 0:
        return None

    r = results[0]

    # KEYPOINT FIX — modern Ultralytics stores keypoints here:
    if r.keypoints is None or r.keypoints.data is None:
        return None

    arr = r.keypoints.data  # tensor: (num_people, 17, 3)

    if arr is None or len(arr) == 0:
        return None

    k = arr.cpu().numpy()[0]   # first person → (17,3)

    if k.shape != (17,3):
        return None

    return k.astype(np.float32)

