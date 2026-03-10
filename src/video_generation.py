import os
import cv2
import numpy as np

# Input / output
IMAGE_FOLDER = r"C:\DREAM_final\figsave_DRIFT_dataset_rounD_02"  # folder with numeric PNG frames
VIDEO_NAME = r"C:\DREAM_final\figsave_DRIFT_dataset_rounD_02\rounD_drift_02.mp4"
FPS = 20

# Keep figure aspect ratio. Frames will be uniformly scaled to fit within
# these bounds (no horizontal/vertical squeeze).
MAX_W = 1920
MAX_H = 1080

# If True, adds black bars when a frame ratio differs from output ratio.
# If False, resizes directly to output size (can distort).
LETTERBOX = True


def _even(v):
    return int(v) if int(v) % 2 == 0 else int(v) - 1


def _fit_size(src_w, src_h, max_w, max_h):
    scale = min(max_w / float(src_w), max_h / float(src_h), 1.0)
    out_w = max(2, _even(round(src_w * scale)))
    out_h = max(2, _even(round(src_h * scale)))
    return out_w, out_h


def _letterbox(frame, out_w, out_h):
    h, w = frame.shape[:2]
    scale = min(out_w / float(w), out_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x0 = (out_w - new_w) // 2
    y0 = (out_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def main():
    images = [
        img for img in os.listdir(IMAGE_FOLDER)
        if img.endswith(".png") and os.path.splitext(img)[0].isdigit()
    ]
    images.sort(key=lambda x: int(os.path.splitext(x)[0]))

    if not images:
        raise FileNotFoundError(f"No numeric PNG frames found in {IMAGE_FOLDER}")

    first = cv2.imread(os.path.join(IMAGE_FOLDER, images[0]))
    if first is None:
        raise RuntimeError(f"Cannot read first frame: {images[0]}")

    src_h, src_w = first.shape[:2]
    out_w, out_h = _fit_size(src_w, src_h, MAX_W, MAX_H)

    print(f"Frames: {len(images)}  |  FPS: {FPS}")
    print(f"Source frame: {src_w}x{src_h}")
    print(f"Output video: {out_w}x{out_h}  (letterbox={LETTERBOX})")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(VIDEO_NAME, fourcc, FPS, (out_w, out_h))

    for image in images:
        frame = cv2.imread(os.path.join(IMAGE_FOLDER, image))
        if frame is None:
            print(f"[WARN] Cannot read {image}, skipping.")
            continue

        if LETTERBOX:
            frame_out = _letterbox(frame, out_w, out_h)
        else:
            frame_out = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(frame_out)

    writer.release()
    print(f"Video created: {VIDEO_NAME}")


if __name__ == "__main__":
    main()
