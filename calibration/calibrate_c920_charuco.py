import time

import cv2
import numpy as np


# Downloaded board: 7 x 5 squares, 25 mm squares, 18 mm markers.
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LENGTH = 0.025
MARKER_LENGTH = 0.018

CAMERA_SOURCE = 0
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
OUTPUT_FILE = "c930_charuco_calibration.npz"


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    dictionary,
)
detector = cv2.aruco.CharucoDetector(board)

camera = cv2.VideoCapture(CAMERA_SOURCE)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

if not camera.isOpened():
    raise RuntimeError("Could not open the C930 camera")

charuco_corners = []
charuco_ids = []
image_size = None
last_capture_time = 0.0

print("Move the Charuco board to different positions and angles.")
print("Press s to save a calibration view.")
print("Press c to calibrate after collecting at least 15 views.")
print("Press q to quit.")

while True:
    ok, frame = camera.read()
    if not ok:
        print("Could not read a frame from the camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)

    display = frame.copy()
    corner_count = 0 if corners is None else np.asarray(corners).reshape(-1, 2).shape[0]
    id_count = 0 if ids is None else np.asarray(ids).reshape(-1).shape[0]
    valid_detection = corners is not None and ids is not None and corner_count == id_count
    detected_count = corner_count if valid_detection else 0

    if valid_detection and detected_count > 0:
        # OpenCV 5.0 can reject otherwise valid detector arrays in its
        # drawDetectedCornersCharuco helper, so draw the points directly.
        display_corners = np.asarray(corners).reshape(-1, 2)
        display_ids = np.asarray(ids).reshape(-1)
        for point, corner_id in zip(display_corners, display_ids):
            x, y = np.rint(point).astype(int)
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(display, str(corner_id), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    cv2.putText(
        display,
        f"Charuco corners: {detected_count} | saved views: {len(charuco_corners)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0) if detected_count >= 6 else (0, 0, 255),
        2,
    )
    cv2.putText(
        display,
        "s=save view  c=calibrate  q=quit",
        (20, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.imshow("C930 Charuco Calibration", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        now = time.monotonic()
        if not valid_detection or detected_count < 6:
            print("Not enough Charuco corners detected; view not saved")
        elif now - last_capture_time < 0.5:
            print("Wait briefly before saving another view")
        else:
            charuco_corners.append(corners.copy())
            charuco_ids.append(ids.copy())
            image_size = (frame.shape[1], frame.shape[0])
            last_capture_time = now
            print(
                f"Saved view {len(charuco_corners)} "
                f"with {len(ids)} Charuco corners"
            )

    elif key == ord("c"):
        if len(charuco_corners) < 15:
            print("Collect at least 15 views before calibrating")
            continue

        print("Calibrating...")
        error, camera_matrix, dist_coeffs, rvecs, tvecs = (
            cv2.aruco.calibrateCameraCharuco(
                charuco_corners,
                charuco_ids,
                board,
                image_size,
                None,
                None,
            )
        )

        np.savez(
            OUTPUT_FILE,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            image_width=image_size[0],
            image_height=image_size[1],
            reprojection_error=error,
        )

        print(f"Calibration saved to {OUTPUT_FILE}")
        print(f"Reprojection error: {error:.4f} pixels")
        print("Camera matrix:")
        print(camera_matrix)
        print("Distortion coefficients:")
        print(dist_coeffs.ravel())
        break

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
