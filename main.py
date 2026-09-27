import argparse
import threading
import time

import cv2
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor

PROMPT = "toy car"

# The camera runs in its own thread and keeps ONLY the newest frame here.
# SAM3 is slower than the camera, so old frames must be thrown away, otherwise
# we would always be looking at the past.
camera_state = {"frame": None, "stop": False}


# ---------------------------------------------------------------- camera

def open_camera(source, width, height, fps):
    """Open the camera and start the background reader thread."""
    # "/dev/video0" -> index 0; V4L2 cannot open cameras by name
    src = str(source).replace("/dev/video", "")
    cap = cv2.VideoCapture(int(src) if src.isdigit() else str(source), cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # needed for 1080p30
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {source}")
    threading.Thread(target=read_camera_forever, args=(cap,), daemon=True).start()
    return cap


def read_camera_forever(cap):
    """Background thread: keep reading, always overwrite with the newest frame."""
    while not camera_state["stop"]:
        ok, frame = cap.read()
        if not ok:
            camera_state["stop"] = True
            return
        camera_state["frame"] = frame


def get_frame():
    """Newest camera frame, or None if nothing has arrived yet."""
    frame = camera_state["frame"]
    return None if frame is None else frame.copy()


def wait_for_first_frame():
    while camera_state["frame"] is None and not camera_state["stop"]:
        time.sleep(0.05)
    return get_frame()


# ---------------------------------------------------------------- model

def load_model(weights, conf, imgsz):
    return SAM3SemanticPredictor(
        overrides=dict(conf=conf, task="segment", mode="predict", model=weights,
                       quantize=16, imgsz=imgsz, save=False, verbose=False),
    )


def detect(predictor, image):
    """Best match for PROMPT. Returns (box_xyxy, score, mask) or None."""
    predictor.set_image(image)
    result = predictor(text=[PROMPT])[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    best = int(result.boxes.conf.argmax())
    mask = None
    if result.masks is not None:
        mask = result.masks.data[best].cpu().numpy() > 0.5
    return result.boxes.xyxy[best].cpu().numpy(), float(result.boxes.conf[best]), mask


# ---------------------------------------------------------------- tracking helpers

def crop_window(center, size, frame_shape):
    """Square around center, clipped to the frame. Returns (x0, y0, x1, y1)."""
    height, width = frame_shape[:2]
    half = int(size / 2)
    cx, cy = int(center[0]), int(center[1])
    return (max(0, cx - half), max(0, cy - half),
            min(width, cx + half), min(height, cy + half))


def choose_region(frame, track, args):
    """Pick where to look. Returns (image_to_search, offset, window_or_None)."""
    if track["center"] is None:
        return frame, np.zeros(2, dtype=int), None  # SEARCH: whole frame

    # TRACK: aim at where the car should be now, widen the crop for each miss
    size = max(args.min_crop,
               args.crop_scale * track["size"] * (1 + 0.5 * track["misses"]))
    x0, y0, x1, y1 = crop_window(track["center"] + track["velocity"], size, frame.shape)
    return frame[y0:y1, x0:x1], np.array([x0, y0]), (x0, y0, x1, y1)


def update_track(track, box, max_miss=None):
    """Fold a new detection into the tracking state."""
    center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    if track["center"] is not None and track["misses"] == 0:
        # smooth the movement so a jittery box doesn't wreck the prediction
        track["velocity"] = 0.6 * track["velocity"] + 0.4 * (center - track["center"])
    track["center"] = center
    track["size"] = max(box[2] - box[0], box[3] - box[1])
    track["misses"] = 0


def count_miss(track, max_miss):
    """No detection this frame: count it, and give up after max_miss."""
    track["misses"] += 1
    if track["misses"] > max_miss:
        track["center"] = None
        track["velocity"] = np.zeros(2)


# ---------------------------------------------------------------- drawing

def draw_detection(frame, box, score, mask, offset):
    if mask is not None:
        full = np.zeros(frame.shape[:2], bool)
        h, w = mask.shape
        full[offset[1]:offset[1] + h, offset[0]:offset[0] + w] = mask
        frame[full] = (0.5 * frame[full] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    b = box.astype(int)
    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
    cv2.putText(frame, f"rc car {score:.2f}", (b[0], max(30, b[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)


def draw_overlay(frame, window, trail, state, fps):
    if window is not None:
        cv2.rectangle(frame, window[:2], window[2:], (255, 200, 0), 1)
    for a, b in zip(trail[:-1], trail[1:]):
        cv2.line(frame, a, b, (0, 0, 255), 2)
    cv2.putText(frame, f"{state}  {fps:.1f} FPS", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 200, 0), 2)


# ---------------------------------------------------------------- main

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/dev/video0")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--cam-fps", type=int, default=30)
    ap.add_argument("--weights", default="/home/dokterkepin/models/sam3.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=644,
                    help="448 ~17 FPS, 644 ~10 FPS, 1008 ~3.7 FPS")
    ap.add_argument("--crop-scale", type=float, default=4.0)
    ap.add_argument("--min-crop", type=int, default=320)
    ap.add_argument("--max-miss", type=int, default=5)
    ap.add_argument("--record", default=None)
    ap.add_argument("--display-scale", type=float, default=0.5)
    return ap.parse_args()


def main():
    args = parse_args()

    predictor = load_model(args.weights, args.conf, args.imgsz)
    cap = open_camera(args.source, args.width, args.height, args.cam_fps)
    first = wait_for_first_frame()
    detect(predictor, first)  # warm-up: the first CUDA call is always slow

    track = {"center": None, "velocity": np.zeros(2), "size": 0, "misses": 0}
    trail, frame_times = [], []
    writer = None
    cv2.namedWindow("sam3 rc car", cv2.WINDOW_NORMAL)

    while not camera_state["stop"]:
        frame = get_frame()
        if frame is None:
            break
        started = time.time()

        searching = track["center"] is None
        region, offset, window = choose_region(frame, track, args)
        found = detect(predictor, region)

        if found is not None:
            box, score, mask = found
            box = box + np.tile(offset, 2)  # crop coords -> full frame coords
            update_track(track, box)
            trail.append(tuple(track["center"].astype(int)))
            trail[:] = trail[-90:]
            draw_detection(frame, box, score, mask, offset)
        else:
            count_miss(track, args.max_miss)

        frame_times.append(time.time() - started)
        frame_times[:] = frame_times[-30:]
        fps = 1 / np.mean(frame_times)
        draw_overlay(frame, window, trail, "SEARCH" if searching else "TRACK", fps)

        if args.record:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                         max(1.0, fps), (w, h))
            writer.write(frame)

        cv2.imshow("sam3 rc car",
                   cv2.resize(frame, None, fx=args.display_scale, fy=args.display_scale))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            cv2.imwrite(f"snapshot_{int(time.time())}.jpg", frame)
        if key == ord("c"):
            trail.clear()

    camera_state["stop"] = True
    time.sleep(0.05)
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
