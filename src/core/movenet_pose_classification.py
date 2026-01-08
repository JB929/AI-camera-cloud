import numpy as np

def classify_pose_movenet(keypoints):

    if keypoints is None or np.isnan(keypoints).all():
        return "Unknown"

    neck = keypoints[0] 
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]

    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    hip_y = (left_hip[1] + right_hip[1]) / 2
    body_vertical = abs(hip_y - shoulder_y)

    shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
    hip_x = (left_hip[0] + right_hip[0]) / 2
    body_horizontal = abs(hip_x - shoulder_x)

    if body_vertical < 0.05 and body_horizontal > 0.2:
        return "Lying"
    if body_vertical < 0.14:
        return "Sitting"
    return "Standing"

