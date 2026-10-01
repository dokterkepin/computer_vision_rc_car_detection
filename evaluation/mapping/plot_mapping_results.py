import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("evaluation/mapping")
OUTPUT_DIR = RESULTS_DIR / "plots"


def read_summary(path):
    values = {}
    with path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            values[row["metric"]] = float(row["value"]) if row["value"] else np.nan
    return values


def read_samples(path):
    position_errors = []
    orientation_errors = []
    with path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if row["position_error_mm"]:
                position_errors.append(float(row["position_error_mm"]))
            if row["orientation_error_deg"]:
                orientation_errors.append(float(row["orientation_error_deg"]))
    return position_errors, orientation_errors


def main():
    summary_files = sorted(RESULTS_DIR.glob("*_summary.csv"))
    sample_files = sorted(
        path for path in RESULTS_DIR.glob("*.csv")
        if not path.name.endswith("_summary.csv")
    )
    if not summary_files:
        raise FileNotFoundError(f"No summary CSV files found in {RESULTS_DIR}")

    summaries = [read_summary(path) for path in summary_files]
    all_position_errors = []
    all_orientation_errors = []
    for path in sample_files:
        position_errors, orientation_errors = read_samples(path)
        all_position_errors.extend(position_errors)
        all_orientation_errors.extend(orientation_errors)

    total_frames = sum(item["frames_evaluated"] for item in summaries)
    total_detections = sum(item["accepted_detections"] for item in summaries)
    detection_rate = total_detections / total_frames if total_frames else 0.0

    print("Mapping evaluation")
    print("==================")
    for path, summary in zip(summary_files, summaries):
        print(f"\n{path.name}")
        print(f"  frames: {int(summary['frames_evaluated'])}")
        print(f"  detection rate: {summary['detection_rate']:.3f}")
        print(f"  mean position error: {summary['mean_position_error_mm']:.3f} mm")
        print(f"  max position error: {summary['max_position_error_mm']:.3f} mm")
        print(f"  mean orientation error: {summary['mean_orientation_error_deg']:.3f} deg")
        print(f"  max orientation error: {summary['max_orientation_error_deg']:.3f} deg")

    print("\nOverall")
    print(f"  frames: {total_frames}")
    print(f"  detection rate: {detection_rate:.3f}")
    print(f"  mean position error: {np.mean(all_position_errors):.3f} mm")
    print(f"  max position error: {np.max(all_position_errors):.3f} mm")
    print(f"  mean orientation error: {np.mean(all_orientation_errors):.3f} deg")
    print(f"  max orientation error: {np.max(all_orientation_errors):.3f} deg")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = [path.stem.replace("_summary", "") for path in summary_files]
    mean_positions = [item["mean_position_error_mm"] for item in summaries]
    max_positions = [item["max_position_error_mm"] for item in summaries]
    mean_orientations = [item["mean_orientation_error_deg"] for item in summaries]
    max_orientations = [item["max_orientation_error_deg"] for item in summaries]

    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, mean_positions, width, label="mean")
    plt.bar(x + width / 2, max_positions, width, label="maximum")
    plt.axhline(np.mean(all_position_errors), color="black", linestyle="--",
                label=f"overall mean = {np.mean(all_position_errors):.2f} mm")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Position error (mm)")
    plt.title("Mapping position error by stationary test")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "position_error_summary.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, mean_orientations, width, label="mean")
    plt.bar(x + width / 2, max_orientations, width, label="maximum")
    plt.axhline(np.mean(all_orientation_errors), color="black", linestyle="--",
                label=f"overall mean = {np.mean(all_orientation_errors):.2f} deg")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Orientation error (degrees)")
    plt.title("Mapping orientation error by stationary test")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "orientation_error_summary.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(all_position_errors, bins=25, alpha=0.8, label="position error")
    plt.axvline(np.mean(all_position_errors), color="red", linestyle="--",
                label=f"mean = {np.mean(all_position_errors):.2f} mm")
    plt.xlabel("Position error (mm)")
    plt.ylabel("Frames")
    plt.title("Distribution of position error")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "position_error_distribution.png", dpi=160)
    plt.close()

    print(f"\nSaved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
