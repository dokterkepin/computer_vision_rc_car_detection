import argparse
from pathlib import Path

import cv2
import numpy as np


clicked_pixel = None


def on_mouse(event, x, y, flags, parameter):
    global clicked_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        scale = parameter
        clicked_pixel = np.array([[[x / scale, y / scale]]], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Test field homography with a known point.")
    parser.add_argument("--camera-calibration", default="c920_charuco_calibration.npz")
    parser.add_argument("--homography", default="homography.npz")
    parser.add_argument("--source", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--display-scale", type=float, default=0.5)
    parser.add_argument("--known-x-mm", type=float)
    parser.add_argument("--known-y-mm", type=float)
    args = parser.parse_args()

    camera_data = np.load(args.camera_calibration)
    homography_data = np.load(args.homography)
    camera_matrix = camera_data["camera_matrix"]
    dist_coeffs = camera_data["dist_coeffs"]
    homography = homography_data["homography"]

    expected_size = (
        int(camera_data["image_width"]),
        int(camera_data["image_height"]),
    )
    if (args.width, args.height) != expected_size:
        raise ValueError(
            f"camera resolution {(args.width, args.height)} does not match "
            f"calibration resolution {expected_size}"
        )

    source = str(args.source).replace("/dev/video", "")
    camera = cv2.VideoCapture(
        int(source) if source.isdigit() else args.source,
        cv2.CAP_V4L2,
    )
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not camera.isOpened():
        raise RuntimeError(f"could not open camera {args.source}")

    window_name = "homography test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse, args.display_scale)
    print("Click a known object or field point. Press q to quit.")

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        display = cv2.resize(
            frame,
            None,
            fx=args.display_scale,
            fy=args.display_scale,
        )

        if clicked_pixel is not None:
            field_point = cv2.perspectiveTransform(
                clicked_pixel,
                homography,
            )[0, 0]
            text = f"Field: ({field_point[0]:.1f}, {field_point[1]:.1f}) mm"
            cv2.circle(
                display,
                tuple(np.rint(clicked_pixel[0, 0] * args.display_scale).astype(int)),
                8,
                (0, 0, 255),
                2,
            )
            if args.known_x_mm is not None and args.known_y_mm is not None:
                error = np.linalg.norm(
                    field_point - np.array([args.known_x_mm, args.known_y_mm])
                )
                text += f"  error: {error:.1f} mm"
            cv2.putText(display, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 255), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
