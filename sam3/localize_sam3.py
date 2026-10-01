import argparse
import socket
import time

import cv2
import numpy as np
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


def mask_orientation(mask):
    if mask is None:
        return None
    y_coordinates, x_coordinates = np.nonzero(mask)
    if len(x_coordinates) < 10:
        return None

    points = np.column_stack((x_coordinates, y_coordinates)).astype(np.float32)
    _, eigenvectors, _ = cv2.PCACompute2(points, mean=None)
    axis = eigenvectors[0]
    return float(np.degrees(np.arctan2(axis[1], axis[0])))


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


def wrapped_angle_difference(current, previous):
    return (current - previous + 180.0) % 360.0 - 180.0


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
    y_mm, x_mm = field_point
    dy, dx = velocity
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
        f"world (y, x): ({y_mm:.1f}, {x_mm:.1f}) mm",
        f"velocity (dy, dx): ({dy:.1f}, {dx:.1f}) mm/s",
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
        "world (y, x): (-1000.0, -1000.0) mm",
        "velocity (dy, dx): (-1000.0, -1000.0) mm/s",
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
    parser.add_argument("--source", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--cam-fps", type=int, default=30)
    parser.add_argument("--camera-calibration", default="calibration/matrix/c920_charuco_calibration.npz")
    parser.add_argument("--homography", default="calibration/matrix/homography_v2.npz")
    parser.add_argument("--weights", default="/home/dokterkepin/models/sam3.pt")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--imgsz", type=int, default=448)
    parser.add_argument("--crop-scale", type=float, default=4.0)
    parser.add_argument("--min-crop", type=int, default=320)
    parser.add_argument("--max-miss", type=int, default=5)
    parser.add_argument("--display-scale", type=float, default=1.0)
    parser.add_argument("--udp-host", default=None)
    parser.add_argument("--udp-port", type=int, default=5000)
    parser.add_argument("--car-id", type=int, default=1)
    args = parser.parse_args()

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

            y_mm = x_mm = theta = dy = dx = angular_velocity = MISSING
            u = v = -1
            if found is not None:
                box, score, mask = found
                box = box + np.tile(offset, 2)
                u = int((box[0] + box[2]) / 2)
                v = int(box[3])
                current_position = world_position(homography, box)
                y_mm, x_mm = map(float, current_position)
                theta = world_orientation(homography, mask, offset)
                if theta is None:
                    theta = MISSING

                if previous_time is not None and previous_position is not None:
                    dt = now - previous_time
                    if dt > 0:
                        dy = (y_mm - previous_position[0]) / dt
                        dx = (x_mm - previous_position[1]) / dt
                        if previous_angle != MISSING and theta != MISSING:
                            angular_velocity = wrapped_angle_difference(theta, previous_angle) / dt
                update_track(track, box)
                previous_time = now
                previous_position = (y_mm, x_mm)
                previous_angle = theta
                draw_result(
                    frame, box, score, mask, offset, args.car_id,
                    (u, v), current_position, theta,
                    (dy, dx), angular_velocity,
                    1.0 / max(time.perf_counter() - loop_started, 1e-9),
                )
            else:
                count_miss(track, args.max_miss)
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

            message = (
                f'{timestamp_us}:{args.car_id},{y_mm:.1f},{x_mm:.1f},'
                f'{theta:.1f},{dy:.1f},{dx:.1f},{angular_velocity:.1f},{u},{v}\n'
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
