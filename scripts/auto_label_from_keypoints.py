# scripts/auto_label_from_keypoints.py
import os, argparse, numpy as np, glob, cv2
from tqdm import tqdm

def bbox_from_keypoints(kp):
    # kp: (T, 17, 2 or 3) or (17,2/3)
    if kp.ndim == 3:
        mid = kp[kp.shape[0]//2]
    else:
        mid = kp
    # mid: (17,2/3)
    xs = mid[:,0]; ys = mid[:,1]
    valid = (~np.isnan(xs)) & (~np.isnan(ys))
    if valid.sum() == 0:
        return None
    x1 = float(np.min(xs[valid])); x2 = float(np.max(xs[valid]))
    y1 = float(np.min(ys[valid])); y2 = float(np.max(ys[valid]))
    # expand box by 10%
    w = x2 - x1; h = y2 - y1
    pad_x = 0.1 * w; pad_y = 0.1 * h
    return max(0, x1 - pad_x), max(0, y1 - pad_y), x2 + pad_x, y2 + pad_y

def save_yolo_label(lbl_path, cls_idx, bbox, img_w, img_h):
    x1,y1,x2,y2 = bbox
    # normalize
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    with open(lbl_path, "w") as f:
        f.write(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npy-dir", required=True, help="root dir with class subdirs of .npy sequences")
    p.add_argument("--frames-dir", default=None, help="optional directory containing raw frames (matching npy base names)")
    p.add_argument("--out-img-dir", required=True)
    p.add_argument("--out-label-dir", required=True)
    p.add_argument("--classes", required=True, help="comma-separated ordered class names (Sitting,Standing,Lying,... )")
    args = p.parse_args()

    classes = [c.strip() for c in args.classes.split(",")]
    os.makedirs(args.out_img_dir, exist_ok=True)
    os.makedirs(args.out_label_dir, exist_ok=True)

    for i, cls in enumerate(classes):
        npy_files = sorted(glob.glob(os.path.join(args.npy_dir, cls, "*.npy")))
        for pth in tqdm(npy_files, desc=cls):
            try:
                seq = np.load(pth)  # expected (T,17,2) or (T,17,3)
            except Exception as e:
                print("skip", pth, e); continue
            bbox = bbox_from_keypoints(seq)
            if bbox is None:
                continue
            # pick a synthetic image name or try to find a frame
            base = os.path.splitext(os.path.basename(pth))[0]
            if args.frames_dir:
                # try to find an image with base in frames dir
                candidates = sorted(glob.glob(os.path.join(args.frames_dir, f"*{base}*.jpg")))
                if candidates:
                    img_path = candidates[0]
                    img = cv2.imread(img_path)
                else:
                    img = None
            else:
                img = None
            # if no image, create blank image sized 640x480 (not ideal)
            if img is None:
                img_h, img_w = 480, 640
                img = 255 * np.ones((img_h, img_w, 3), dtype=np.uint8)
            else:
                img_h, img_w = img.shape[:2]
            # draw bbox on image (optional)
            x1,y1,x2,y2 = map(int, bbox)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            out_img = os.path.join(args.out_img_dir, f"{cls}_{base}.jpg")
            out_lbl = os.path.join(args.out_label_dir, f"{cls}_{base}.txt")
            cv2.imwrite(out_img, img)
            save_yolo_label(out_lbl, i, (x1,y1,x2,y2), img_w, img_h)

