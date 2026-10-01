import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor


PROMPT = "toy car"

START_BUTTON = (20, 210, 240, 260)
start_requested = False


def handle_mouse(event, x, y, flags, parameter):
    global start_requested
    display_scale = parameter if parameter else 1.0
    x = int(x / display_scale)
    y = int(y / display_scale)
    x0, y0, x1, y1 = START_BUTTON
    if event == cv2.EVENT_LBUTTONDOWN and x0 <= x <= x1 and y0 <= y <= y1:
        start_requested = True


def load_predictor(weights, imgsz):
    return SAM3SemanticPredictor(
        overrides=dict(
            conf=0.3,
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
    result = predictor(text=[PROMPT])[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None

    best = int(result.boxes.conf.argmax())
    box = result.boxes.xyxy[best].cpu().numpy()
    score = float(result.boxes.conf[best])
    mask = None
    if result.masks is not None:
        mask = result.masks.data[best].cpu().numpy() > 0.5
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


def choose_region(frame, track, crop_scale, min_crop):
    if track["center"] is None:
        return frame, np.zeros(2, dtype=int), None

    size = max(
        min_crop,
        crop_scale * track["size"] * (1 + 0.5 * track["misses"]),
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
    image_point = np.array(
        [[[(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]]],
        dtype=np.float32,
    )
    return cv2.perspectiveTransform(image_point, homography)[0, 0]


def world_orientation(homography, mask, offset):
    if mask is None:
        return None
    y_coordinates, x_coordinates = np.nonzero(mask)
    if len(x_coordinates) < 10:
        return None

    points = np.column_stack((x_coordinates, y_coordinates)).astype(np.float32)
    mean, eigenvectors, _ = cv2.PCACompute2(points, mean=None)
    axis = eigenvectors[0] * 100.0
    center = mean[0]
    local_points = np.array([[center, center + axis]], dtype=np.float32)
    full_points = local_points + np.asarray(offset, dtype=np.float32)
    field_points = cv2.perspectiveTransform(
        full_points.reshape(1, 2, 2),
        homography,
    )[0]
    direction = field_points[1] - field_points[0]
    return float(np.degrees(np.arctan2(direction[1], direction[0])))


def angle_error(estimated, true_angle, modulo_180):
    if estimated is None:
        return None
    period = 180.0 if modulo_180 else 360.0
    return abs((estimated - true_angle + period / 2) % period - period / 2)


def open_camera(source, width, height, fps):
    source_text = str(source).replace("/dev/video", "")
    camera = cv2.VideoCapture(
        int(source_text) if source_text.isdigit() else source,
        cv2.CAP_V4L2,
    )
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        raise RuntimeError(f"could not open camera {source}")
    return camera


def draw_overlay(frame, box, score, image_point, estimated_position,
                 estimated_angle, velocity, fps):
    if box is not None:
        corners = box.astype(int)
        cv2.rectangle(frame, tuple(corners[:2]), tuple(corners[2:]), (0, 255, 0), 3)
        cv2.putText(frame, f"score={score:.3f}", (corners[0], max(30, corners[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    pixel_text = "(-1, -1)" if image_point is None else f"({image_point[0]}, {image_point[1]})"
    world_text = "missing" if estimated_position is None else (
        f"({estimated_position[0]:.1f}, {estimated_position[1]:.1f}) mm"
    )
    theta_text = "missing" if estimated_angle is None else f"{estimated_angle:.1f} deg"
    lines = [
        f"pixel: {pixel_text}",
        f"world (y, x): {world_text}",
        f"velocity: ({velocity[0]:.1f}, {velocity[1]:.1f}) mm/s",
        f"theta: {theta_text}",
        f"FPS: {fps:.1f}",
    ]
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (20, 35 + index * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)


def main():
    global start_requested
    parser = argparse.ArgumentParser(description="Evaluate stationary-car field mapping accuracy.")
    parser.add_argument("--source", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--cam-fps", type=int, default=30)
    parser.add_argument("--camera-calibration", default="calibration/matrix/c920_charuco_calibration.npz")
    parser.add_argument("--homography", default="calibration/matrix/homography_v2.npz")
    parser.add_argument("--weights", default="/home/dokterkepin/models/sam3.pt")
    parser.add_argument("--imgsz", type=int, default=448)
    parser.add_argument("--crop-scale", type=float, default=4.0)
    parser.add_argument("--min-crop", type=int, default=320)
    parser.add_argument("--max-miss", type=int, default=5)
    parser.add_argument("--true-y-mm", type=float, default=103.0)
    parser.add_argument("--true-x-mm", type=float, default=18311.0)
    parser.add_argument("--true-theta-deg", type=float, default=45.0)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--display-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", default="evaluation/mapping/data")
    args = parser.parse_args()

    true_y_mm = args.true_y_mm
    true_x_mm = args.true_x_mm
    true_theta_deg = args.true_theta_deg
    sample_count = args.samples

    camera_data = np.load(args.camera_calibration)
    homography_data = np.load(args.homography)
    camera_matrix = camera_data["camera_matrix"]
    dist_coeffs = camera_data["dist_coeffs"]
    homography = homography_data["homography"]
    expected_size = (int(camera_data["image_width"]), int(camera_data["image_height"]))
    if (args.width, args.height) != expected_size:
        raise ValueError(f"camera resolution {(args.width, args.height)} does not match calibration {expected_size}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"y{true_y_mm:g}_x{true_x_mm:g}_theta{true_theta_deg:g}"
    samples_path = output_dir / f"{stem}.csv"
    summary_path = output_dir / f"{stem}_summary.csv"

    predictor = load_predictor(args.weights, args.imgsz)
    camera = open_camera(args.source, args.width, args.height, args.cam_fps)
    true_position = np.array([true_y_mm, true_x_mm])
    sample_rows = []
    position_errors = []
    orientation_errors = []
    detected_count = 0
    frame_number = 0
    collection_started = False
    previous_estimated_position = None
    previous_estimated_time = None
    track = {
        "center": None,
        "velocity": np.zeros(2),
        "size": 0,
        "misses": 0,
    }
    print("Keep the car still. Press q to stop early.")

    ok, warmup_frame = camera.read()
    if not ok:
        raise RuntimeError("could not read the warm-up camera frame")
    detect_car(predictor, cv2.undistort(warmup_frame, camera_matrix, dist_coeffs))

    try:
        with samples_path.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "frame", "truth_y_mm", "truth_x_mm", "truth_theta_deg",
                "score", "estimated_y_mm", "estimated_x_mm", "estimated_theta_deg",
                "position_error_mm", "orientation_error_deg", "accepted",
            ])
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                frame_started = time.perf_counter()
                frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
                if start_requested:
                    collection_started = True
                searching = track["center"] is None
                region, offset, window = choose_region(
                    frame,
                    track,
                    args.crop_scale,
                    args.min_crop,
                )
                found = detect_car(predictor, region)
                box = score = estimated_position = estimated_angle = None
                position_error = orientation_error = None
                image_point = None
                velocity = np.zeros(2)
                accepted = False

                if found is not None:
                    box, score, mask = found
                    box = box + np.tile(offset, 2)
                    image_point = (
                        int((box[0] + box[2]) / 2),
                        int((box[1] + box[3]) / 2),
                    )
                    estimated_position = world_position(homography, box)
                    estimated_angle = world_orientation(homography, mask, offset)
                    current_time = time.monotonic()
                    if previous_estimated_position is not None and previous_estimated_time is not None:
                        delta_time = current_time - previous_estimated_time
                        if delta_time > 0:
                            velocity = (
                                estimated_position - previous_estimated_position
                            ) / delta_time
                    previous_estimated_position = estimated_position.copy()
                    previous_estimated_time = current_time
                    accepted = collection_started
                    if accepted:
                        detected_count += 1
                        position_error = float(np.linalg.norm(estimated_position - true_position))
                        orientation_error = angle_error(
                            estimated_angle, true_theta_deg, True
                        )
                        position_errors.append(position_error)
                        if orientation_error is not None:
                            orientation_errors.append(orientation_error)
                    update_track(track, box)
                else:
                    count_miss(track, args.max_miss)
                    previous_estimated_position = None
                    previous_estimated_time = None

                if collection_started and len(sample_rows) < sample_count:
                    row = [
                        frame_number, true_y_mm, true_x_mm, true_theta_deg,
                        "" if score is None else f"{score:.8f}",
                        "" if estimated_position is None else f"{estimated_position[0]:.4f}",
                        "" if estimated_position is None else f"{estimated_position[1]:.4f}",
                        "" if estimated_angle is None else f"{estimated_angle:.4f}",
                        "" if position_error is None else f"{position_error:.4f}",
                        "" if orientation_error is None else f"{orientation_error:.4f}",
                        int(accepted),
                    ]
                    writer.writerow(row)
                    csv_file.flush()
                    sample_rows.append(row)

                draw_overlay(
                    frame, box, 0.0 if score is None else score,
                    image_point, estimated_position, estimated_angle,
                    velocity,
                    1.0 / max(time.perf_counter() - frame_started, 1e-9),
                )
                if window is not None:
                    cv2.rectangle(frame, window[:2], window[2:], (255, 200, 0), 2)
                cv2.putText(
                    frame,
                    "SEARCH" if searching else "TRACK",
                    (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 200, 0),
                    2,
                )
                if not collection_started:
                    cv2.rectangle(frame, START_BUTTON[:2], START_BUTTON[2:], (0, 180, 0), -1)
                    cv2.putText(frame, "START", (45, 244),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(frame, "click START or press s", (20, 290),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, "COLLECTING", (20, 290),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                display = cv2.resize(
                    frame,
                    None,
                    fx=args.display_scale,
                    fy=args.display_scale,
                )
                cv2.imshow("mapping evaluation", display)
                cv2.setMouseCallback(
                    "mapping evaluation",
                    handle_mouse,
                    args.display_scale,
                )
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    start_requested = True
                if key == ord("q"):
                    break
                frame_number += 1
                if collection_started and len(sample_rows) >= sample_count:
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    with summary_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        writer.writerow(["frames_evaluated", len(sample_rows)])
        writer.writerow(["accepted_detections", detected_count])
        writer.writerow(["detection_rate", detected_count / max(len(sample_rows), 1)])
        writer.writerow(["mean_position_error_mm", np.mean(position_errors) if position_errors else ""])
        writer.writerow(["max_position_error_mm", np.max(position_errors) if position_errors else ""])
        writer.writerow(["mean_orientation_error_deg", np.mean(orientation_errors) if orientation_errors else ""])
        writer.writerow(["max_orientation_error_deg", np.max(orientation_errors) if orientation_errors else ""])

    print(f"Saved samples: {samples_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Accepted detections: {detected_count}/{len(sample_rows)}")
    if position_errors:
        print(f"Mean/max position error: {np.mean(position_errors):.2f} / {np.max(position_errors):.2f} mm")
    if orientation_errors:
        print(f"Mean/max orientation error: {np.mean(orientation_errors):.2f} / {np.max(orientation_errors):.2f} deg")


if __name__ == "__main__":
    main()
