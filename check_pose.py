from ultralytics import YOLO
import cv2

model = YOLO("models/your_model.pt")

img = cv2.imread("test.jpg")  # any image with a person
res = model(img)[0]

print("Has keypoints:", hasattr(res, "keypoints"))
if hasattr(res, "keypoints"):
    print("Keypoints array:", res.keypoints.shape if res.keypoints is not None else None)

