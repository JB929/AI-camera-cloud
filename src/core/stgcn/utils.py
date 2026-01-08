import json
import numpy as np


def load_kp_map(path='kp_map.json'):
if not os.path.exists(path):
return None
j = json.load(open(path,'r'))
return {int(k):int(v) for k,v in j['map'].items()}


# reorder sequence helper
def reorder_sequence(seq, map_dict):
T,K,C = seq.shape
out = np.zeros((T,17,C), dtype=seq.dtype) + np.nan
for det_idx, coco_idx in map_dict.items():
if det_idx < K and coco_idx < 17:
out[:, coco_idx, :] = seq[:, det_idx, :]
return out
