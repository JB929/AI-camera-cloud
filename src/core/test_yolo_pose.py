from src.core.yolov8_pose_detector import detect_pose_yolo as detect_pose
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ cannot read frame")
        continue

    k = detect_pose(frame)

    if k is None:
        print("⚠️ NO KEYPOINTS DETECTED")
    else:
        print("✅ Keypoints:", k.shape)
        print(k[:5])  # print first 5 pts

    cv2.imshow("test", frame)
    if cv2.waitKey(1) == ord('q'):
        break


