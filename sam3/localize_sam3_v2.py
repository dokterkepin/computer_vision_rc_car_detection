import argparse
import socket
import time

import cv2
import numpy as np

from localize_sam3 import (
    MISSING,
    choose_region,
    count_miss,
    load_predictor,
    open_camera,
    update_track,
    wrapped_angle_difference,
    world_orientation,
    world_position,
)


NUM_OBJECTS = 2
# BGR colors: (box and mask overlay, status text) for the left and right object.
CAR_COLORS = (
    ((0, 255, 0), (0, 255, 255)),
    ((255, 0, 255), (255, 255, 0)),
)
MISSING_COLOR = (0, 200, 255)


def box_iou(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return float(intersection / union)


def box_center_size(box):
    center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
    return center, max(box[2] - box[0], box[3] - box[1])


def is_taken(center, size, taken):
    # Two detections closer than half an object length are the same object.
    return any(np.linalg.norm(center - c) < 0.5 * max(size, s) for c, s in taken)


def detect_objects(predictor, frame, prompts):
    # The image is encoded once; each distinct prompt is then queried separately,
    # so every detection is known to belong to the prompt that produced it.
    predictor.set_image(frame)
    return {
        prompt: parse_result(predictor(text=[prompt])[0])
        for prompt in dict.fromkeys(prompts)
    }


def parse_result(result):
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    masks = None
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy() > 0.5

    detections = []
    for index in np.argsort(-scores):
        box = boxes[index]
        if any(box_iou(box, kept[0]) > 0.5 for kept in detections):
            continue
        detections.append((box, float(scores[index]), None if masks is None else masks[index]))
    return detections


def pick_tracked(detections, offset, track, taken):
    expected = track["center"] + track["velocity"]
    gate = 1.5 * track["size"] * (1 + track["misses"])
    best = None
    best_distance = gate
    for box, score, mask in detections:
        full_box = box + np.tile(offset, 2)
        center, size = box_center_size(full_box)
        if is_taken(center, size, taken):
            continue
        distance = np.linalg.norm(center - expected)
        if distance < best_distance:
            best = (full_box, score, mask, offset)
            best_distance = distance
    return best


def new_object(object_id, prompt):
    return {
        "id": object_id,
        "prompt": prompt,
        "track": {
            "center": None,
            "velocity": np.zeros(2),
            "size": 0,
            "misses": 0,
        },
        "previous_time": None,
        "previous_position": None,
        "previous_angle": None,
        "found": None,
        "window": None,
        "searching": True,
        "measurement": None,
    }


def find_objects(frame, cars, predictor, args):
    # A tracked object is searched only inside its own crop, with its own
    # prompt; a lost object needs one full-frame search shared by all lost ones.
    taken = []
    for car in cars:
        track = car["track"]
        car["found"] = None
        car["window"] = None
        car["searching"] = track["center"] is None
        if car["searching"]:
            continue
        region, offset, window = choose_region(frame, track, args)
        car["window"] = window
        detections = detect_objects(predictor, region, [car["prompt"]])
        car["found"] = pick_tracked(detections[car["prompt"]], offset, track, taken)
        if car["found"] is not None:
            taken.append(box_center_size(car["found"][0]))
        else:
            taken.append((track["center"], track["size"]))

    lost = [car for car in cars if car["searching"]]
    if not lost:
        return
    zero_offset = np.zeros(2, dtype=int)
    detections = detect_objects(predictor, frame, [car["prompt"] for car in lost])
    for car in lost:
        for box, score, mask in detections[car["prompt"]]:
            center, size = box_center_size(box)
            if is_taken(center, size, taken):
                continue
            car["found"] = (box, score, mask, zero_offset)
            taken.append((center, size))
            break


def measure(car, homography, now, max_miss):
    track = car["track"]
    found = car["found"]
    if found is None:
        count_miss(track, max_miss)
        car["previous_time"] = None
        car["previous_position"] = None
        car["previous_angle"] = None
        car["measurement"] = None
        return

    box, score, mask, offset = found
    u = int((box[0] + box[2]) / 2)
    v = int(box[3])
    position = world_position(homography, box)
    x_mm, y_mm = map(float, position)
    theta = world_orientation(homography, mask, offset)
    if theta is None:
        theta = MISSING

    vx = vy = angular_velocity = MISSING
    if car["previous_time"] is not None and car["previous_position"] is not None:
        dt = now - car["previous_time"]
        if dt > 0:
            vx = (x_mm - car["previous_position"][0]) / dt
            vy = (y_mm - car["previous_position"][1]) / dt
            if car["previous_angle"] != MISSING and theta != MISSING:
                angular_velocity = wrapped_angle_difference(theta, car["previous_angle"]) / dt

    update_track(track, box)
    car["previous_time"] = now
    car["previous_position"] = (x_mm, y_mm)
    car["previous_angle"] = theta
    car["measurement"] = {
        "box": box,
        "score": score,
        "mask": mask,
        "offset": offset,
        "pixel": (u, v),
        "world": (x_mm, y_mm),
        "theta": theta,
        "velocity": (vx, vy),
        "angular_velocity": angular_velocity,
    }


def draw_status(frame, lines, color, side):
    for line_index, line in enumerate(lines):
        x = 20
        if side == "right":
            (text_width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            x = frame.shape[1] - 20 - text_width
        cv2.putText(
            frame,
            line,
            (x, 35 + line_index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )


def draw_object(frame, car, fps, side):
    box_color, text_color = CAR_COLORS[0 if side == "left" else 1]
    car_id = car["id"]
    state = "SEARCH" if car["searching"] else "TRACK"
    m = car["measurement"]
    if m is None:
        draw_status(
            frame,
            (
                f"ID: {car_id}  pixel: (-1, -1)",
                "world (x, y): (-1000.0, -1000.0) mm",
                "velocity (vx, vy): (-1000.0, -1000.0) mm/s",
                "theta: -1000.0 deg  angular: -1000.0 deg/s",
                f"FPS: {fps:.1f}",
                state,
            ),
            MISSING_COLOR,
            side,
        )
        return

    mask = m["mask"]
    if mask is not None:
        mask_height, mask_width = mask.shape
        x0, y0 = m["offset"]
        full_mask = np.zeros(frame.shape[:2], dtype=bool)
        full_mask[y0:y0 + mask_height, x0:x0 + mask_width] = mask
        frame[full_mask] = (
            0.5 * frame[full_mask] + 0.5 * np.array(box_color)
        ).astype(np.uint8)

    corners = m["box"].astype(int)
    cv2.rectangle(frame, (corners[0], corners[1]), (corners[2], corners[3]), box_color, 3)
    cv2.circle(frame, m["pixel"], 8, (0, 0, 255), -1)
    cv2.putText(
        frame,
        f"{car['prompt']} {car_id}  conf={m['score']:.2f}",
        (corners[0], max(30, corners[1] - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        box_color,
        2,
    )
    (u, v), (x_mm, y_mm) = m["pixel"], m["world"]
    vx, vy = m["velocity"]
    draw_status(
        frame,
        (
            f"ID: {car_id}  pixel: ({u}, {v})",
            f"world (x, y): ({x_mm:.1f}, {y_mm:.1f}) mm",
            f"velocity (vx, vy): ({vx:.1f}, {vy:.1f}) mm/s",
            f"theta: {m['theta']:.1f} deg  angular: {m['angular_velocity']:.1f} deg/s",
            f"FPS: {fps:.1f}",
            state,
        ),
        text_color,
        side,
    )


def format_message(car, timestamp_us):
    m = car["measurement"]
    if m is None:
        x_mm = y_mm = theta = vx = vy = angular_velocity = MISSING
        u = v = -1
    else:
        x_mm, y_mm = m["world"]
        theta = m["theta"]
        vx, vy = m["velocity"]
        angular_velocity = m["angular_velocity"]
        u, v = m["pixel"]
    return (
        f'{timestamp_us}:{car["id"]},{x_mm:.1f},{y_mm:.1f},'
        f'{theta:.1f},{vx:.1f},{vy:.1f},{angular_velocity:.1f},{u},{v}\n'
    )


def main():
    parser = argparse.ArgumentParser(description="SAM3 localization of up to two objects, each found by a text prompt, on a calibrated field.")
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
    parser.add_argument("--display-scale", type=float, default=0.5)
    parser.add_argument("--udp-host", default=None)
    parser.add_argument("--udp-port", type=int, default=5000)
    parser.add_argument(
        "--prompts", nargs="+", default=["yellow duck", "yellow cup lid"],
        help="one text prompt per object (max 2); a single prompt is used for both objects",
    )
    parser.add_argument("--ids", type=int, nargs=NUM_OBJECTS, default=[1, 2])
    args = parser.parse_args()
    if len(args.prompts) > NUM_OBJECTS:
        parser.error(f"at most {NUM_OBJECTS} prompts are supported")
    prompts = [args.prompts[min(i, len(args.prompts) - 1)] for i in range(NUM_OBJECTS)]

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

    cars = [new_object(object_id, prompt) for object_id, prompt in zip(args.ids, prompts)]
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
            find_objects(frame, cars, predictor, args)
            timestamp_us = (time.monotonic_ns() - started) // 1_000
            now = time.monotonic()
            for car in cars:
                measure(car, homography, now, args.max_miss)

            fps = 1.0 / max(time.perf_counter() - loop_started, 1e-9)
            if frame_times:
                fps = 1.0 / np.mean(frame_times)
            for car, side in zip(cars, ("left", "right")):
                draw_object(frame, car, fps, side)
                if car["window"] is not None:
                    cv2.rectangle(frame, car["window"][:2], car["window"][2:], (255, 200, 0), 2)

                message = format_message(car, timestamp_us)
                if udp_socket:
                    udp_socket.sendto(message.encode("utf-8"), udp_address)
                print(message, end="")

            frame_times.append(time.perf_counter() - loop_started)
            frame_times = frame_times[-30:]

            cv2.imshow("SAM3 world localization (2 objects)", cv2.resize(frame, None, fx=args.display_scale, fy=args.display_scale))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        if udp_socket:
            udp_socket.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
