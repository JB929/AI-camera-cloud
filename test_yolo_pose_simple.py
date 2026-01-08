from ultralytics import YOLO
import cv2

model = YOLO("models/yolov8s-pose.pt")
cap = cv2.VideoCapture(0)

print("Loaded model. Starting webcam...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame...")
        continue

    results = model(frame, verbose=False)

    if len(results) == 0:
        print("No results")
        continue

    r = results[0]

    if hasattr(r, "keypoints") and r.keypoints is not None:
        kp = r.keypoints.data.cpu().numpy()
        print("Detected keypoints:", kp.shape)
    else:
        print("NO KEYPOINTS in result")

    cv2.imshow("YOLOv8 Pose Test", results[0].plot())

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

