import cv2
import numpy as np
import os
import time

# --- Pose model files ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
protoFile = os.path.join(PROJECT_ROOT, "pose_models", "pose_deploy_linevec.prototxt")
weightsFile = os.path.join(PROJECT_ROOT, "pose_models", "pose_iter_440000.caffemodel")

print("🔍 Checking pose model paths:")
print("   prototxt =", protoFile, "→ exists:", os.path.exists(protoFile))
print("   caffemodel =", weightsFile, "→ exists:", os.path.exists(weightsFile))

# --- Load Caffe model ---
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("✅ Pose model loaded successfully (OpenCV DNN backend CPU).")

# Optional: optimize backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# --- Load OpenPose COCO model ---
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
protoFile = os.path.join(BASE_DIR, "../../pose_models/pose_deploy_linevec.prototxt")
weightsFile = os.path.join(BASE_DIR, "../../pose_models/pose_iter_440000.caffemodel")
nPoints = 18
POSE_PAIRS = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
    [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]
]

print("🧠 Loading pose model (this can take 15–45 seconds on CPU)...")

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    protoFile = os.path.join(BASE_DIR, "../../pose_models/pose_deploy_linevec.prototxt")
    weightsFile = os.path.join(BASE_DIR, "../../pose_models/pose_iter_440000.caffemodel")

    print(f"🔍 Checking model paths:")
    print(f"   prototxt = {protoFile}  → exists: {os.path.exists(protoFile)}")
    print(f"   caffemodel = {weightsFile}  → exists: {os.path.exists(weightsFile)}")

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    # Force CPU to avoid Apple Metal / GPU crashes
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    print("✅ Pose model loaded successfully and backend set to CPU!")

except Exception as e:
    print(f"❌ Pose model loading failed: {e}")
    import sys
    sys.exit(1)

def detect_pose(frame):
    try:
        if frame is None or not hasattr(frame, "shape"):
            print("❌ Frame is empty — skipping pose detection.")
            return "Unknown", np.zeros((18, 2))

        print(f"✅ Frame received for pose detection: {frame.shape}")
        inputBlob = cv2.dnn.blobFromImage(
            frame, 1.0 / 255, (368, 368), (0, 0, 0),
            swapRB=False, crop=False
        )
        net.setInput(inputBlob)
        print(f"🧠 Input blob shape: {inputBlob.shape}")

        # --- Warm-up pass (fixes layer initialization issue) ---
        if not getattr(detect_pose, "_warmup_done", False):
            _ = net.forward()
            detect_pose._warmup_done = True
            print("🔥 Warm-up forward pass done.")

        print("⚙️ Running forward pass...")
        output = net.forward()
        print(f"✅ Forward pass complete! Output shape: {output.shape}")

        # --- Extract keypoints from output map ---
        H, W = output.shape[2], output.shape[3]
        points = []
        for i in range(18):  # 18 keypoints
            probMap = output[0, i, :, :]
            _, prob, _, point = cv2.minMaxLoc(probMap)
            x = (frame.shape[1] * point[0]) / W
            y = (frame.shape[0] * point[1]) / H
            points.append((x, y) if prob > 0.1 else (np.nan, np.nan))

        keypoints = np.array(points, dtype=float)

        # --- Pose classification based on relative geometry ---
        valid_y = keypoints[~np.isnan(keypoints[:, 1]), 1]
        if len(valid_y) == 0:
            pose_label = "Unknown"
        else:
            vertical_span = (valid_y.max() - valid_y.min()) / frame.shape[0]
            if vertical_span > 0.25:
                pose_label = "Standing"
            elif 0.10 < vertical_span <= 0.25:
                pose_label = "Sitting"
            else:
                pose_label = "Lying"
        # --- 🧍 Draw Pose Skeleton on Frame ---
        # Define keypoint pairs that form human body connections
        POSE_PAIRS = [
            (1, 2), (2, 3), (3, 4),       # Right arm
            (1, 5), (5, 6), (6, 7),       # Left arm
            (1, 8), (8, 9), (9, 10),      # Right leg
            (1, 11), (11, 12), (12, 13),  # Left leg
            (1, 0), (0, 14), (14, 16),    # Neck to head
            (0, 15), (15, 17)             # Neck to head (other side)
        ]

        for pair in POSE_PAIRS:
            partA, partB = pair
            if partA < len(keypoints) and partB < len(keypoints):
                xA, yA = keypoints[partA]
                xB, yB = keypoints[partB]
                if not np.isnan([xA, yA, xB, yB]).any():
                    cv2.line(frame, (int(xA), int(yA)), (int(xB), int(yB)), (0, 255, 0), 2)
                    cv2.circle(frame, (int(xA), int(yA)), 4, (0, 0, 255), thickness=-1)
                    cv2.circle(frame, (int(xB), int(yB)), 4, (0, 0, 255), thickness=-1)

        # Add label (Standing / Sitting / Lying)
        cv2.putText(frame, f"Pose: {pose_label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Show visualization (optional, can be disabled in headless mode)
        try:
            # Log-only visualization (safe for headless mode)
            print(f"[POSE] 🧠 Pose detected successfully: {pose_label}")
        except Exception as e:
            print(f"[POSE] ⚠️ Visualization skipped due to: {e}")



        return pose_label, keypoints, frame

    except cv2.error as e:
        print(f"❌ OpenCV DNN forward() failed: {e}")
        return "Unknown", np.zeros((18, 2))

    except Exception as e:
        print(f"❌ Pose detection failed unexpectedly: {type(e).__name__}: {e}")
        return "Unknown", np.zeros((18, 2))



    # --- Normalize keypoints to a clean (N, 2) array ---
    keypoints = np.array(keypoints, dtype=float)

    # Filter only valid 2D points
    if keypoints.ndim == 1 or keypoints.shape[1] != 2:
        cleaned = []
        for pt in keypoints:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                cleaned.append(pt[:2])  # take only (x, y)
            else:
                cleaned.append([np.nan, np.nan])
        keypoints = np.array(cleaned, dtype=float)

    # ==========================================================
    # 🧩 STEP 2: Normalize keypoints for partial-body detection
    # ==========================================================
    def normalize_keypoints(keypoints):
        """Fix partial keypoints and scale for upper-body visibility."""
        keypoints = np.nan_to_num(keypoints, nan=0.0)

        # Compute torso height (neck to hip midpoint)
        if keypoints.shape[0] >= 9:
            neck = keypoints[1]
            left_hip, right_hip = keypoints[8], keypoints[11]
            torso_center = (
                (left_hip + right_hip) / 2
                if np.any(left_hip) and np.any(right_hip)
                else neck
            )
            torso_height = np.linalg.norm(neck - torso_center)
        else:
            torso_height = 1.0

        # If lower body missing → approximate legs by mirroring torso
        if np.all(keypoints[9:]) == 0:
            hip_y = (
                keypoints[8][1]
                if np.any(keypoints[8])
                else keypoints[1][1] + torso_height
            )
            for i in [9, 10, 11, 12, 13, 14]:
                keypoints[i] = [keypoints[1][0], hip_y + torso_height * 0.8]
      
    # --- Context-based posture correction ---
    if pose_label == "Standing":
        # If knees or ankles missing, and torso vertical but hips low → likely sitting
        visible_points = np.sum(~np.isnan(keypoints[:, 0]))
        if visible_points < 15:
            upper_y = np.nanmean(keypoints[1:5, 1])  # head/shoulders
            lower_y = np.nanmean(keypoints[8:12, 1])  # hips/knees (if visible)
            if not np.isnan(upper_y) and not np.isnan(lower_y) and (lower_y - upper_y) < frame.shape[0] * 0.25:
                pose_label = "Sitting"
                print("🧠 Context override → Adjusted 'Standing' → 'Sitting' (low vertical spread)")


        # Normalize scale
        max_val = np.max(np.abs(keypoints)) + 1e-6
        keypoints = keypoints / max_val
        return keypoints

    # ✅ Apply normalization before returning
    keypoints = normalize_keypoints(keypoints)
 
    if np.all(keypoints[9:]) == 0:
        print("⚙️ [Partial-body detected → Reconstructing lower limbs]")
    else:
        print("✅ [Full-body detected → No reconstruction needed]")
   
    if 'points' in locals() and points is not None:
        return pose_label, np.array(points)
    else:
        return pose_label, keypoints, frame

if __name__ == "__main__":
    import cv2, numpy as np

    print("🧠 Pose Normalization Test Mode Active...")

    # --- Load a test frame (torso-only) ---
    test_image_path = "ai_camera_clients/snapshots/snapshot_1760451551.jpg"
    frame = cv2.imread(test_image_path)

    if frame is None:
        print("⚠️ Could not load test image. Please update the path to an existing file.")
        exit()

    # --- Run detection ---
    from pose_detector import detect_pose

    pose_label, keypoints, annotated = detect_pose(frame)

    # --- Verify output ---
    print("\n✅ Pose Label:", pose_label)
    print("✅ Keypoints shape:", np.array(keypoints).shape)
    print("✅ First few keypoints (x,y):")
    print(np.array(keypoints)[:10])  # print first 10 for quick check

    # --- Visual sanity check (optional) ---
    try:
        for (x, y) in keypoints:
            if not np.isnan(x) and not np.isnan(y):
                cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 3, (0, 255, 0), -1)
        cv2.imshow("Normalized Keypoints", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("⚠️ Display skipped:", e)





