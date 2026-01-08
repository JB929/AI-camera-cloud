from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("models/yolov8s-pose.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    r = results[0]

    if r.keypoints is not None:
        print("KP:", r.keypoints.data.shape)

    cv2.imshow("pose", results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

