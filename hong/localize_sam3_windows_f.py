"""Original SAM3 pipeline with automatic front/rear heading.

Put car_heading_references.npz beside this script. Blue is front in the supplied
blue/red car references. No PCA, point selection, markers or extra training.
Yellow rectangle = fitted four corners; cyan arrow = selected front.
Heading status explains missing theta. FRONT UNCERTAIN keeps theta=-1000;
image axis is still shown separately and is not a signed heading.
The green/yellow car needs its own labelled references before validation.
SAM3 loading, quantize=16, tracking, calibration and UDP format retain the
uploaded original settings. Only CPU half-precision NMS inputs are converted to
FP32 during prediction. Reference matching has been tested offline;
real SAM3 masks and camera performance need live validation.
"""
import argparse
import socket
import time
from pathlib import Path
from unittest.mock import patch


import cv2
import numpy as np
import torch
import torchvision
from ultralytics.models.sam import SAM3SemanticPredictor


PROMPT = "toy car"
MISSING = -1000.0


def load_predictor(weights, conf, imgsz):
    return SAM3SemanticPredictor(
        overrides=dict(
            conf=conf,
            task="segment",
            mode="predict",
            model=weights,
            quantize=16,
            imgsz=imgsz,
            save=False,
            verbose=False,
        )
    )


def detect_car(predictor, frame):
    predictor.set_image(frame)
    # Limit the FP32 compatibility conversion to CPU NMS inputs. The SAM3 model
    # stays at the uploaded original precision; restore the NMS function afterward.
    original_nms = torchvision.ops.nms
    def cpu_compatible_nms(boxes, scores, iou_threshold):
        if boxes.device.type == "cpu" and boxes.dtype in (torch.float16, torch.bfloat16):
            boxes, scores = boxes.float(), scores.float()
        return original_nms(boxes, scores, iou_threshold)
    with patch.object(torchvision.ops, "nms", cpu_compatible_nms):
        result = predictor(text=[PROMPT])[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None

    best = int(result.boxes.conf.argmax())
    box = result.boxes.xyxy[best].cpu().numpy()
    score = float(result.boxes.conf[best])
    mask = None
    if result.masks is not None:
        raw = result.masks.data[best].cpu().numpy() > 0.5
        if raw.shape == frame.shape[:2]:
            mask = raw
        elif tuple(result.masks.orig_shape) == tuple(frame.shape[:2]):
            # Ultralytics xy uses original-image coordinates and removes letterbox
            # scaling/padding. Do not discard or directly stretch a model-size mask.
            polygon = np.asarray(result.masks[best].xy[0], dtype=np.float32)
            if polygon.ndim == 2 and len(polygon) >= 3 and np.isfinite(polygon).all():
                restored = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(restored, [np.rint(polygon).astype(np.int32)], 1)
                mask = restored > 0
    return box, score, mask


def crop_window(center, size, frame_shape):
    height, width = frame_shape[:2]
    half = int(size / 2)
    center_x, center_y = np.asarray(center, dtype=int)
    return (
        max(0, center_x - half),
        max(0, center_y - half),
        min(width, center_x + half),
        min(height, center_y + half),
    )


def choose_region(frame, track, args):
    if track["center"] is None:
        return frame, np.zeros(2, dtype=int), None

    size = max(
        args.min_crop,
        args.crop_scale * track["size"] * (1 + 0.5 * track["misses"]),
    )
    window = crop_window(
        track["center"] + track["velocity"],
        size,
        frame.shape,
    )
    x0, y0, x1, y1 = window
    return frame[y0:y1, x0:x1], np.array([x0, y0]), window


def update_track(track, box):
    center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    if track["center"] is not None and track["misses"] == 0:
        track["velocity"] = (
            0.6 * track["velocity"]
            + 0.4 * (center - track["center"])
        )
    track["center"] = center
    track["size"] = max(box[2] - box[0], box[3] - box[1])
    track["misses"] = 0


def count_miss(track, max_miss):
    track["misses"] += 1
    if track["misses"] > max_miss:
        track["center"] = None
        track["velocity"] = np.zeros(2)


def world_position(homography, box):
    # The assignment reports the real-world center of the detected car.
    image_point = np.array(
        [[[(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]]],
        dtype=np.float32,
    )
    return cv2.perspectiveTransform(image_point, homography)[0, 0]


def mask_axis(mask, expected_shape):
    """Return rectangle center, unsigned long axis, length and width (no PCA)."""
    if mask is None or mask.shape != expected_shape[:2]:
        return None  # Never silently interpret model-resolution pixels as image pixels.
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 20:
        return None
    corners = cv2.boxPoints(cv2.minAreaRect(contour))
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    index = int(np.argmax(lengths))
    long_side, short_side = float(lengths[index]), float(np.min(lengths))
    if short_side < 3 or long_side / short_side < 1.15:
        return None  # Near-square silhouette has an unreliable long axis.
    return corners.mean(axis=0), edges[index] / long_side, long_side, short_side

def canonical_car(region, mask, geometry, sign):
    """Affine crop with candidate front pointing UP; preserve appearance and mask."""
    center, axis, length, width = geometry
    front = axis * sign
    right = np.array([-front[1], front[0]], dtype=np.float32)
    length, width = length * 1.12, width * 1.12
    source = np.float32([center - right * width / 2 + front * length / 2,
                         center + right * width / 2 + front * length / 2,
                         center - right * width / 2 - front * length / 2])
    target = np.float32([[0, 0], [63, 0], [0, 127]])
    transform = cv2.getAffineTransform(source, target)
    crop = cv2.warpAffine(region, transform, (64, 128))
    valid = cv2.warpAffine(mask.astype(np.uint8), transform, (64, 128),
                           flags=cv2.INTER_NEAREST) > 0
    return crop, valid

def color_layout(crop, valid):
    """Coarse spatial chromaticity; avoid fragile matching of tiny pixel details."""
    values = crop.astype(np.float32)
    chroma = values / (values.sum(axis=2, keepdims=True) + 1e-6) - 1.0 / 3.0
    interior = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    feature = []
    for index in range(4):
        patch = chroma[index * 32:(index + 1) * 32, 12:52]
        support = interior[index * 32:(index + 1) * 32, 12:52]
        if np.count_nonzero(support) < 25:
            return None
        feature.extend(patch[support].mean(axis=0))
    return np.asarray(feature, dtype=np.float32)

def appearance_score(first, first_mask, second, second_mask):
    a, b = color_layout(first, first_mask), color_layout(second, second_mask)
    if a is None or b is None:
        return -1.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-5 else -1.0

def direction_angle(homography, center, direction, offset, length):
    points = np.float32([center, center + direction * length / 2]) + offset
    mapped = cv2.perspectiveTransform(points.reshape(1, 2, 2), homography)[0]
    delta = mapped[1] - mapped[0]
    if not np.isfinite(delta).all() or np.linalg.norm(delta) < 1e-6:
        return None
    return float(np.degrees(np.arctan2(delta[1], delta[0])))

class CarHeading:
    """Rectangle geometry + automatic front/rear reference matching, no PCA."""
    def __init__(self, reference_path, min_score=0.55, margin=0.12):
        self.min_score, self.margin = min_score, margin
        self.features = []
        reference_path = Path(reference_path)
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"Missing car reference file: {reference_path}. "
                "Place car_heading_references.npz beside this Python script.")
        with np.load(reference_path, allow_pickle=False) as data:
            for crop, valid in zip(data['crops'], data['masks']):
                feature = color_layout(crop, valid.astype(bool))
                if feature is not None and np.linalg.norm(feature) > 1e-5:
                    self.features.append(feature / np.linalg.norm(feature))
        if not self.features:
            raise ValueError(f"No usable car references in {reference_path}")
        self.reset()
        self.geometry = None
        self.front = None
        self.status = 'WAITING FOR CAR'
        self.axis_angle = None
        self.scores = None

    def reset(self):
        self.last_angle = None
        self.last_confirmed = None

    def estimate(self, region, mask, offset, homography, now):
        self.front = None
        self.scores = None
        self.axis_angle = None
        self.geometry = mask_axis(mask, region.shape)
        if mask is None:
            self.status = 'NO MASK'
            return None
        if self.geometry is None:
            self.status = 'INVALID/SQUARE MASK AXIS'
            return None
        center, axis, length, _ = self.geometry
        self.axis_angle = float(np.degrees(np.arctan2(axis[1], axis[0])) % 180)
        candidates, scores = [], []
        for sign in (1, -1):
            crop, valid = canonical_car(region, mask, self.geometry, sign)
            feature = color_layout(crop, valid)
            norm = np.linalg.norm(feature) if feature is not None else 0
            score = (max(float(np.dot(feature / norm, ref)) for ref in self.features)
                     if norm > 1e-5 else -1.0)
            scores.append(score)
            candidates.append(direction_angle(homography, center, sign * axis, offset, length))
        self.scores = tuple(scores)
        best = int(np.argmax(scores))
        supported = (scores[best] >= self.min_score and
                     scores[best] - scores[1 - best] >= self.margin and
                     candidates[best] is not None)
        # Temporal fallback is short, and only follows an appearance-confirmed track.
        if supported:
            selected = best
            self.last_confirmed = now
            self.status = 'FRONT MATCH'
        elif self.last_confirmed is not None and now - self.last_confirmed <= 0.5:
            distances = [abs(wrapped_angle_difference(a, self.last_angle))
                         if a is not None else float('inf') for a in candidates]
            selected = int(np.argmin(distances))
            if distances[selected] > 45:
                self.status = 'FRONT UNCERTAIN'
                return None
            self.status = 'TEMPORAL (<=0.5s)'
        else:
            self.status = 'FRONT UNCERTAIN'
            return None
        self.front = axis * (1 if selected == 0 else -1)
        self.last_angle = candidates[selected]
        return self.last_angle

    def draw(self, frame, offset):
        axis_text = ('--' if self.axis_angle is None else f'{self.axis_angle:.1f}')
        score_text = ('--/--' if self.scores is None else
                      f'{self.scores[0]:.2f}/{self.scores[1]:.2f}')
        cv2.putText(frame, f'Heading: {self.status}', (20, 225),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        cv2.putText(frame, f'Image axis: {axis_text} deg | front scores: {score_text}',
                    (20, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        if self.geometry is None:
            return
        center, axis, length, width = self.geometry
        normal = np.array([-axis[1], axis[0]])
        corners = np.array([center + a * axis * length / 2 + b * normal * width / 2
                            for a, b in ((-1,-1),(-1,1),(1,1),(1,-1))]) + offset
        cv2.polylines(frame, [np.rint(corners).astype(np.int32)], True, (0,255,255), 2)
        if self.front is not None:
            start = tuple(np.rint(center + offset).astype(int))
            end = tuple(np.rint(center + offset + self.front * length / 2).astype(int))
            cv2.arrowedLine(frame, start, end, (255,255,0), 2, tipLength=0.35)


def wrapped_angle_difference(current, previous):
    return (current - previous + 180.0) % 360.0 - 180.0


def open_camera(source, width, height, fps):
    source_text = str(source).replace("/dev/video", "")
    camera_index = int(source_text) if source_text.isdigit() else source
    camera = cv2.VideoCapture(camera_index)

    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        raise RuntimeError(f"could not open camera {source}")
    return camera


def draw_result(
    frame,
    box,
    score,
    mask,
    mask_offset,
    car_id,
    image_point,
    field_point,
    angle,
    velocity,
    angular_velocity,
    fps,
):
    if mask is not None:
        mask_height, mask_width = mask.shape
        x0, y0 = mask_offset
        full_mask = np.zeros(frame.shape[:2], dtype=bool)
        full_mask[y0:y0 + mask_height, x0:x0 + mask_width] = mask
        frame[full_mask] = (
            0.5 * frame[full_mask] + 0.5 * np.array([0, 255, 0])
        ).astype(np.uint8)

    corners = box.astype(int)
    cv2.rectangle(
        frame,
        (corners[0], corners[1]),
        (corners[2], corners[3]),
        (0, 255, 0),
        3,
    )
    u, v = image_point
    x_mm, y_mm = field_point
    vx, vy = velocity
    cv2.circle(frame, (u, v), 8, (0, 0, 255), -1)
    cv2.putText(
        frame,
        f"car {car_id}  conf={score:.2f}",
        (corners[0], max(30, corners[1] - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    status_lines = (
        f"ID: {car_id}  pixel: ({u}, {v})",
        f"world (x, y): ({x_mm:.1f}, {y_mm:.1f}) mm",
        f"velocity (vx, vy): ({vx:.1f}, {vy:.1f}) mm/s",
        f"theta: {angle:.1f} deg  angular: {angular_velocity:.1f} deg/s",
        f"FPS: {fps:.1f}",
    )
    for line_index, line in enumerate(status_lines):
        cv2.putText(
            frame,
            line,
            (20, 35 + line_index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )


def draw_missing_status(frame, car_id, fps):
    status_lines = (
        f"ID: {car_id}  pixel: (-1, -1)",
        "world (x, y): (-1000.0, -1000.0) mm",
        "velocity (vx, vy): (-1000.0, -1000.0) mm/s",
        "theta: -1000.0 deg  angular: -1000.0 deg/s",
        f"FPS: {fps:.1f}",
    )
    for line_index, line in enumerate(status_lines):
        cv2.putText(
            frame,
            line,
            (20, 35 + line_index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 255),
            2,
        )


def main():
    parser = argparse.ArgumentParser(description="SAM3 car localization on a calibrated field.")
    parser.add_argument("--source", default="0")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--cam-fps", type=int, default=30)
    parser.add_argument("--camera-calibration", default="calibration/matrix/c920_charuco_calibration.npz")
    parser.add_argument("--homography", default="calibration/matrix/homography_v2.npz")
    parser.add_argument("--weights", default="sam3.pt")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--imgsz", type=int, default=448)
    parser.add_argument("--crop-scale", type=float, default=4.0)
    parser.add_argument("--min-crop", type=int, default=320)
    parser.add_argument("--max-miss", type=int, default=5)
    parser.add_argument("--display-scale", type=float, default=1.0)
    parser.add_argument("--udp-host", default=None)
    parser.add_argument("--udp-port", type=int, default=5000)
    parser.add_argument("--car-id", type=int, default=1)
    parser.add_argument("--heading-references", default=str(
        Path(__file__).resolve().with_name("car_heading_references.npz")))
    parser.add_argument("--heading-min-score", type=float, default=0.55)
    parser.add_argument("--heading-margin", type=float, default=0.12)
    args = parser.parse_args()
    if not -1 <= args.heading_min_score <= 1 or not 0 <= args.heading_margin <= 2:
        parser.error("invalid heading similarity threshold/margin")
    heading = CarHeading(args.heading_references, args.heading_min_score, args.heading_margin)

    camera_data = np.load(args.camera_calibration)
    homography_data = np.load(args.homography)
    camera_matrix = camera_data["camera_matrix"]
    dist_coeffs = camera_data["dist_coeffs"]
    homography = homography_data["homography"]
    calibration_size = (
        int(camera_data["image_width"]),
        int(camera_data["image_height"]),
    )
    if (args.width, args.height) != calibration_size:
        raise ValueError(
            f"camera resolution {(args.width, args.height)} does not match "
            f"calibration resolution {calibration_size}"
        )

    predictor = load_predictor(args.weights, args.conf, args.imgsz)
    camera = open_camera(args.source, args.width, args.height, args.cam_fps)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args.udp_host else None
    udp_address = (args.udp_host, args.udp_port) if udp_socket else None

    ok, first_frame = camera.read()
    if not ok:
        raise RuntimeError("could not read the first camera frame")
    predictor.set_image(cv2.undistort(first_frame, camera_matrix, dist_coeffs))

    previous_time = None
    previous_position = None
    previous_angle = None
    track = {
        "center": None,
        "velocity": np.zeros(2),
        "size": 0,
        "misses": 0,
    }
    frame_times = []
    started = time.monotonic_ns()
    print("Press q to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            loop_started = time.perf_counter()
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
            searching = track["center"] is None
            region, offset, window = choose_region(frame, track, args)
            found = detect_car(predictor, region)
            timestamp_us = (time.monotonic_ns() - started) // 1_000
            now = time.monotonic()

            x_mm = y_mm = theta = vx = vy = angular_velocity = MISSING
            u = v = -1
            if found is not None:
                box, score, mask = found
                box = box + np.tile(offset, 2)
                u = int((box[0] + box[2]) / 2)
                v = int(box[3])
                current_position = world_position(homography, box)
                x_mm, y_mm = map(float, current_position)
                theta = heading.estimate(region, mask, offset, homography, now)
                if theta is None:
                    theta = MISSING

                if previous_time is not None and previous_position is not None:
                    dt = now - previous_time
                    if dt > 0:
                        vx = (x_mm - previous_position[0]) / dt
                        vy = (y_mm - previous_position[1]) / dt
                        if previous_angle != MISSING and theta != MISSING:
                            angular_velocity = wrapped_angle_difference(theta, previous_angle) / dt
                update_track(track, box)
                previous_time = now
                previous_position = (x_mm, y_mm)
                previous_angle = theta
                draw_result(
                    frame, box, score, mask, offset, args.car_id,
                    (u, v), current_position, theta,
                    (vx, vy), angular_velocity,
                    1.0 / max(time.perf_counter() - loop_started, 1e-9),
                )
            else:
                count_miss(track, args.max_miss)
                heading.geometry = None
                heading.front = None
                heading.axis_angle = None
                heading.scores = None
                heading.status = 'NO CAR'
                heading.reset()
                previous_time = None
                previous_position = None
                previous_angle = None

            elapsed = time.perf_counter() - loop_started
            frame_times.append(elapsed)
            frame_times = frame_times[-30:]
            fps = 1.0 / np.mean(frame_times)
            if window is not None:
                cv2.rectangle(frame, window[:2], window[2:], (255, 200, 0), 2)
            state = "SEARCH" if searching else "TRACK"
            cv2.putText(
                frame, state, (20, 190), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 200, 0), 2,
            )
            if found is None:
                draw_missing_status(frame, args.car_id, fps)

            heading.draw(frame, offset)
            message = (
                f'{timestamp_us}:{args.car_id},{x_mm:.1f},{y_mm:.1f},'
                f'{theta:.1f},{vx:.1f},{vy:.1f},{angular_velocity:.1f},{u},{v}\n'
            )
            if udp_socket:
                udp_socket.sendto(message.encode("utf-8"), udp_address)
            print(message, end="")

            cv2.imshow("SAM3 world localization", cv2.resize(frame, None, fx=args.display_scale, fy=args.display_scale))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        if udp_socket:
            udp_socket.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
