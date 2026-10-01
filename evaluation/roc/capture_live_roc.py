import argparse
import csv
import time

import cv2
from ultralytics.models.sam import SAM3SemanticPredictor


PROMPT = "toy car"


def load_predictor(weights, imgsz):
    return SAM3SemanticPredictor(
        overrides=dict(
            conf=0.001,
            task="segment",
            mode="predict",
            model=weights,
            quantize=16,
            imgsz=imgsz,
            save=False,
            verbose=False,
        )
    )


def get_score(predictor, frame):
    predictor.set_image(frame)
    results = predictor(text=[PROMPT])
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return 0.0, None

    best = int(result.boxes.conf.argmax())
    score = float(result.boxes.conf[best])
    box = result.boxes.xyxy[best].cpu().numpy().astype(int)
    return score, box


def main():
    parser = argparse.ArgumentParser(
        description="Collect live SAM3 scores and truth labels for ROC evaluation."
    )
    parser.add_argument("--source", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--cam-fps", type=int, default=30)
    parser.add_argument("--weights", default="/home/dokterkepin/models/sam3.pt")
    parser.add_argument("--imgsz", type=int, default=448)
    parser.add_argument("--output", default="evaluation/roc/data/roc_data.csv")
    args = parser.parse_args()

    source_text = str(args.source).replace("/dev/video", "")
    camera = cv2.VideoCapture(
        int(source_text) if source_text.isdigit() else args.source,
        cv2.CAP_V4L2,
    )
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    camera.set(cv2.CAP_PROP_FPS, args.cam_fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        raise RuntimeError(f"could not open camera {args.source}")

    predictor = load_predictor(args.weights, args.imgsz)
    ok, frame = camera.read()
    if not ok:
        raise RuntimeError("could not read the first camera frame")
    predictor.set_image(frame)

    output_path = args.output
    output_directory = output_path.rsplit("/", 1)[0] if "/" in output_path else "."
    import os
    os.makedirs(output_directory, exist_ok=True)

    truth = None
    frame_number = 0
    saved_rows = 0
    last_time = time.perf_counter()
    fps = 0.0

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "truth", "score"])

        print("Press 1 when the car is visible, 0 when it is absent.")
        print("Labels persist until changed. Press q to quit.")

        while True:
            ok, frame = camera.read()
            if not ok:
                break

            score, box = get_score(predictor, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("1"):
                truth = 1
                print("Truth label: car present")
            elif key == ord("0"):
                truth = 0
                print("Truth label: car absent")
            elif key == ord("q"):
                break

            if truth is not None:
                writer.writerow([frame_number, truth, f"{score:.8f}"])
                csv_file.flush()
                saved_rows += 1

            now = time.perf_counter()
            elapsed = now - last_time
            if elapsed > 0:
                fps = 1.0 / elapsed
            last_time = now

            if box is not None:
                cv2.rectangle(
                    frame,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (0, 255, 0),
                    3,
                )
                cv2.putText(
                    frame,
                    f"score={score:.3f}",
                    (box[0], max(30, box[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

            truth_text = "unset" if truth is None else str(truth)
            cv2.putText(
                frame,
                f"truth={truth_text}  score={score:.3f}  saved={saved_rows}  FPS={fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "1=car present  0=no car  q=quit",
                (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.imshow("SAM3 ROC data capture", frame)
            frame_number += 1

    camera.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved_rows} rows to {output_path}")


if __name__ == "__main__":
    main()
