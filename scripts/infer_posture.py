# scripts/infer_posture.py
import cv2, time, argparse
from ultralytics import YOLO
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--source", default=0)
args = p.parse_args()

model = YOLO(args.model)
cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)[0]
    # results.boxes.xyxy, results.boxes.conf, results.boxes.cls, results.boxes.names
    for box in results.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())
        name = results.names.get(cls, str(cls))
        x1,y1,x2,y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0), 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)
    cv2.imshow("posture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release(); cv2.destroyAllWindows()

