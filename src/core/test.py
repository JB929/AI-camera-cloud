import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")
frame = cv2.imread("test_person.jpg")
results = model(frame)
results[0].show()
