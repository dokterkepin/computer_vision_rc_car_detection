import cv2
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor


predictor = SAM3SemanticPredictor(
    overrides=dict(
        conf=0.3,
        task="segment",
        mode="predict",
        model="/home/dokterkepin/models/sam3.pt",
        quantize=16,
        imgsz=644,
        save=False,
        verbose=False,
    )
)

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
camera.set(cv2.CAP_PROP_FPS, 30)

if not camera.isOpened():
    raise RuntimeError("Could not open camera 0")

while True:
    ok, frame = camera.read()

    if not ok:
        print("Could not read a frame from the camera")
        break

    predictor.set_image(frame)
    results = predictor(text=["toy car"])
    result = results[0]

    if result.boxes is not None and len(result.boxes) > 0:
        for index in range(len(result.boxes)):
            box = result.boxes.xyxy[index].cpu().numpy().astype(int)
            score = float(result.boxes.conf[index])

            if result.masks is not None:
                mask = result.masks.data[index].cpu().numpy() > 0.5
                frame[mask] = (
                    0.5 * frame[mask] + 0.5 * np.array([0, 255, 0])
                ).astype(np.uint8)

            cv2.rectangle(
                frame,
                (box[0], box[1]),
                (box[2], box[3]),
                (0, 255, 0),
                3,
            )
            cv2.putText(
                frame,
                f"toy car {score:.2f}",
                (box[0], max(30, box[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

    cv2.imshow("Simple SAM3 Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
