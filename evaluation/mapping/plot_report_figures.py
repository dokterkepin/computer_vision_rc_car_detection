"""Report figures for the stationary mapping evaluation.

Same data as plot_mapping_results.py, but every case is labelled in the report's
notation: (x, y) in millimetres and theta in degrees, with x along the tag
ID 0 -> ID 1 edge and y along ID 0 -> ID 3.  The CSV columns use the same names.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path("evaluation/mapping/data")
OUTPUT_DIR = Path("result/report_figures")
CASE_NAMES = "ABC"


def load_case(path):
    rows = list(csv.DictReader(path.open(newline="")))
    truth = np.array([float(rows[0]["truth_x_mm"]), float(rows[0]["truth_y_mm"])])
    theta_truth = float(rows[0]["truth_theta_deg"])
    estimate = np.array([[float(r["estimated_x_mm"]), float(r["estimated_y_mm"])]
                         for r in rows])
    position_error = np.array([float(r["position_error_mm"]) for r in rows])
    orientation_error = np.array([float(r["orientation_error_deg"]) for r in rows])
    return truth, theta_truth, estimate, position_error, orientation_error


def main():
    paths = sorted(p for p in DATA_DIR.glob("*.csv") if not p.name.endswith("_summary.csv"))
    # Order the cases as in the report: A=(724,145), B=(-103,75), C=(285,183)
    cases = sorted((load_case(p) for p in paths), key=lambda c: -c[0][0])
    cases = [cases[0], cases[2], cases[1]]
    labels = [f"{name}: ({c[0][0]:g}, {c[0][1]:g}) mm, {c[1]:g}°"
              for name, c in zip(CASE_NAMES, cases)]

    print("case  mean_pos  max_pos  mean_ang  max_ang   bias_x  bias_y  std_x  std_y")
    for name, (truth, theta, est, ep, eo) in zip(CASE_NAMES, cases):
        bias = est.mean(axis=0) - truth
        std = est.std(axis=0)
        print(f"{name}   {ep.mean():8.3f} {ep.max():8.3f} {eo.mean():9.3f} {eo.max():8.3f}"
              f"  {bias[0]:+7.2f} {bias[1]:+7.2f} {std[0]:6.2f} {std[1]:6.2f}")
    all_ep = np.concatenate([c[3] for c in cases])
    all_eo = np.concatenate([c[4] for c in cases])
    print(f"overall mean/max position error {all_ep.mean():.3f} / {all_ep.max():.3f} mm")
    print(f"overall mean/max orientation error {all_eo.mean():.3f} / {all_eo.max():.3f} deg")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(cases))
    width = 0.36

    for index, (ylabel, title, name) in enumerate([
        ("Position error (mm)", "Position error per test case", "mapping_position_error"),
        ("Orientation error (degrees)", "Orientation error per test case (axis, mod 180°)",
         "mapping_orientation_error"),
    ]):
        mean = [c[3 + index].mean() for c in cases]
        maximum = [c[3 + index].max() for c in cases]
        overall = np.concatenate([c[3 + index] for c in cases]).mean()
        plt.figure(figsize=(8, 4.6))
        plt.bar(x - width / 2, mean, width, label="mean")
        plt.bar(x + width / 2, maximum, width, label="maximum")
        plt.axhline(overall, color="black", linestyle="--",
                    label=f"overall mean = {overall:.2f}")
        plt.xticks(x, labels)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{name}.png", dpi=160)
        plt.close()

    # Offset of every estimate from the ground truth: shows bias versus jitter.
    plt.figure(figsize=(7, 4.6))
    for label, (truth, _, est, _, _) in zip(labels, cases):
        offset = est - truth
        plt.scatter(offset[:, 0], offset[:, 1], s=14, alpha=0.7, label=label)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlim(-4, 2.2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect("equal")
    plt.xlabel(r"$\hat{x}-x^{*}$ (mm)")
    plt.ylabel(r"$\hat{y}-y^{*}$ (mm)")
    plt.title("Estimate minus ground truth (200 frames per case)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mapping_offsets.png", dpi=160)
    plt.close()
    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
