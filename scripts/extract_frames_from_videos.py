import os, cv2, argparse, time
from tqdm import tqdm

def extract(video_path, out_dir, frame_step=15, max_frames=None):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    saved = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    pbar = tqdm(total=total, desc=os.path.basename(video_path))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_step == 0:
            fname = os.path.join(out_dir, f"{int(time.time())}_{idx:06d}.jpg")
            cv2.imwrite(fname, frame)
            saved += 1
            if max_frames and saved >= max_frames:
                break
        idx += 1
        pbar.update(1)
    cap.release()
    pbar.close()
    print(f"Saved {saved} frames to {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--step", type=int, default=15, help="save every Nth frame")
    p.add_argument("--max", type=int, default=0)
    args = p.parse_args()
    extract(args.video, args.out, frame_step=args.step, max_frames=(args.max or None))
