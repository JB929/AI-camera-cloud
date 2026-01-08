# debug_keypoint_order.py
import cv2
import numpy as np
from src.core.yolov8_pose_detector import detect_pose_yolo as detect_pose
import time

def draw_kps(img, kps):
    img2 = img.copy()
    h, w = img2.shape[:2]
    # if normalized (0..1) convert
    if kps.max() <= 1.01:
        kps_vis = np.array(kps.copy(), dtype=np.float32)
        kps_vis[:,0] *= w
        kps_vis[:,1] *= h
    else:
        kps_vis = np.array(kps, dtype=np.float32)
    for i,(x,y) in enumerate(kps_vis):
        if np.isnan(x) or np.isnan(y):
            continue
        cv2.circle(img2, (int(x), int(y)), 5, (0,255,0), -1)
        cv2.putText(img2, str(i), (int(x)+6, int(y)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
    return img2

def main(camera_src=0):
    cap = cv2.VideoCapture(int(camera_src) if str(camera_src).isdigit() else camera_src)
    if not cap.isOpened():
        print("Can't open source:", camera_src); return
    print("Press q to quit. Press s to sample a frame and print keypoints.")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05); continue
        cv2.imshow("preview (press s to sample)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            kp = detect_pose(frame)  # expected numpy array or None
            print("RAW keypoints (shape, dtype):", None if kp is None else (kp.shape, kp.dtype))
            print(kp)
            if kp is not None:
                viz = draw_kps(frame, kp)
                cv2.imshow("keypoints sample", viz)
                cv2.imwrite("debug_keypoints_sample.jpg", viz)
                print("Wrote debug_keypoints_sample.jpg — check the overlay labels for index -> body part")
            else:
                print("⚠️ NO KEYPOINTS DETECTED")
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    main(src)

