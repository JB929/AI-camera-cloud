import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("Camera opened. Reading frames...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame read failed")
        exit()

    print("Frame OK", frame.shape)
    break

