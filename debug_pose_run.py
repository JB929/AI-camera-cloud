# debug_pose_run.py
from ultralytics import YOLO
import cv2
import numpy as np

m = "models/yolov8s-pose.pt"
print("Loading:", m)
model = YOLO(m)
print("Model loaded. Running single inference on webcam frame...")
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
if not ret:
    print("No frame from camera")
else:
    res = model(frame, verbose=False)
    print("Results len:", len(res))
    if len(res) and hasattr(res[0],'keypoints') and res[0].keypoints is not None:
        try:
            kp = res[0].keypoints.data.cpu().numpy()
            print("KP shape (tensor->numpy):", kp.shape)
        except Exception:
            print("Keypoints exists but couldn't access data; attributes:", dir(res[0].keypoints))
    else:
        print("No keypoints found in result")

