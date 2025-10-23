import cv2
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = "yolov8n.pt"
CAMERA_INDEX = 0  # change to 1 if external webcam

print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)

print("[INFO] Starting camera...")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not captured")
        break

    # Run YOLO detection
    results = model(frame)

    # Loop through detections
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("AI Camera Monitor", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting...")
        break

cap.release()
cv2.destroyAllWindows()

