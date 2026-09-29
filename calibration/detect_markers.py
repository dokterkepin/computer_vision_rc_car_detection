import argparse
import time
from pathlib import Path

import cv2
import numpy as np


APRILTAG_FAMILIES = {
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "16h5": cv2.aruco.DICT_APRILTAG_16h5,
}
ARUCO_FAMILIES = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
}


def get_dictionary(family):
    if family.startswith("april:"):
        name = family.split(":", 1)[1]
        values = APRILTAG_FAMILIES
    elif family.startswith("aruco:"):
        name = family.split(":", 1)[1]
        values = ARUCO_FAMILIES
    else:
        raise ValueError("family must start with april: or aruco:")
    if name not in values:
        raise ValueError(f"unknown marker family: {name}")
    return cv2.aruco.getPredefinedDictionary(values[name])


def make_detector(dictionary):
    parameters = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters)
    return cv2.aruco.DetectorParameters_create(), dictionary


def detect_markers(detector, frame):
    if isinstance(detector, tuple):
        parameters, dictionary = detector
        return cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    return detector.detectMarkers(frame)


def marker_centers(corners, ids):
    centers = {}
    if ids is None:
        return centers
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        centers[int(marker_id)] = marker_corners[0].mean(axis=0)
    return centers


def field_points(field_width, field_height):
    return {
        0: (0.0, 0.0),
        1: (field_width, 0.0),
        2: (field_width, field_height),
        3: (0.0, field_height),
    }


def main():
    parser = argparse.ArgumentParser(description="Detect AprilTags/ArUco field references.")
    parser.add_argument("--source", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--camera-calibration", default="c920_charuco_calibration.npz")
    parser.add_argument("--family", default="april:36h11")
    parser.add_argument("--field-width-mm", type=float, default=280.0)
    parser.add_argument("--field-height-mm", type=float, default=340.0)
    parser.add_argument("--output", default="calibration/homography.npz")
    args = parser.parse_args()

    calibration = np.load(args.camera_calibration)
    camera_matrix = calibration["camera_matrix"]
    dist_coeffs = calibration["dist_coeffs"]
    calibration_size = (
        int(calibration["image_width"]),
        int(calibration["image_height"]),
    )
    if (args.width, args.height) != calibration_size:
        raise ValueError(
            f"camera resolution {(args.width, args.height)} does not match "
            f"calibration resolution {calibration_size}"
        )

    source = str(args.source).replace("/dev/video", "")
    camera = cv2.VideoCapture(
        int(source) if source.isdigit() else args.source,
        cv2.CAP_V4L2,
    )
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        raise RuntimeError(f"could not open camera {args.source}")

    detector = make_detector(get_dictionary(args.family))
    known_world = field_points(args.field_width_mm, args.field_height_mm)
    homography = None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_time = time.monotonic()

    print("Press s to save the current homography, q to quit.")
    while True:
        ok, frame = camera.read()
        if not ok:
            break

        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        corners, ids, _ = detect_markers(detector, frame)
        centers = marker_centers(corners, ids)
        image_points = []
        world_points = []
        for marker_id, world_point in known_world.items():
            if marker_id in centers:
                image_points.append(centers[marker_id])
                world_points.append(world_point)

        if len(image_points) >= 4:
            homography, _ = cv2.findHomography(
                np.asarray(image_points, dtype=np.float32),
                np.asarray(world_points, dtype=np.float32),
            )

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for marker_id, center in centers.items():
            point = tuple(np.rint(center).astype(int))
            cv2.putText(frame, f"ID {marker_id} {point}",
                        (point[0] + 8, point[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        status = f"markers {len(image_points)}/4"
        if homography is not None:
            status += "  HOMOGRAPHY READY"
        cv2.putText(frame, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 200, 255), 2)
        cv2.imshow("field marker calibration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and homography is not None:
            np.savez(
                output,
                homography=homography,
                field_width_mm=args.field_width_mm,
                field_height_mm=args.field_height_mm,
                family=args.family,
            )
            print(f"saved {output}")
        if key == ord("q"):
            break

        current_time = time.monotonic()
        if current_time - previous_time > 1.0:
            previous_time = current_time

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
