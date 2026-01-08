# map_to_coco.py
import numpy as np
import json
"""
Small helper:
1) Run debug_keypoint_order.py and note which index is which joint.
2) Edit the "detector_to_coco" dict below to map your detector index -> COCO index.
   Example: if your detector index 5 is Left Shoulder (COCO index 5), map 5->5.
3) Run this script to save a mapping file 'kp_map.json'.
"""
    
# COCO joint names (index -> name):
COCO_NAMES = [ 
 "nose","left_eye","right_eye","left_ear","right_ear",
 "left_shoulder","right_shoulder","left_elbow","right_elbow",
 "left_wrist","right_wrist","left_hip","right_hip",
 "left_knee","right_knee","left_ankle","right_ankle"
]

# --- EDIT this mapping after you inspect debug image ---
# detector_index : coco_index
# If you are certain detector uses COCO order already, set identity 0:0,1:1...
detector_to_coco = {
    # example identity mapping:
    0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15,16:16
}
    
if __name__ == "__main__":
    # validate mapping
    if set(detector_to_coco.keys()) != set(range(17)):
        print("Warning: detector_to_coco doesn't map all 0..16 indices currently.")
    name_map = {int(k): int(v) for k,v in detector_to_coco.items()}
    import os
    out_path = os.path.join(os.path.dirname(__file__), "kp_map.json")
    with open(out_path, "w") as f:
        json.dump({"map": name_map, "coco_names": COCO_NAMES}, f, indent=2)
    print("Saved →", out_path)

    print("Saved kp_map.json. Example usage:")
    print("  from map_to_coco import load_map; m = load_map(); reorder(seq, m)")

def load_map():
    import json, os
    path = os.path.join(os.path.dirname(__file__), "kp_map.json")
    j = json.load(open(path, "r"))
    return j["map"], j["coco_names"]


def reorder_sequence(seq, map_dict):
    """
    seq: np.array (T, 17, C)
    map_dict: detector_index -> coco_index
    returns seq_reordered in COCO order
    """
    T, K, C = seq.shape
    out = np.zeros((T, 17, C), dtype=seq.dtype) + np.nan
    for det_idx, coco_idx in map_dict.items():
        if det_idx < K and coco_idx < 17:
            out[:, coco_idx, :] = seq[:, det_idx, :]  
    return out
