import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    truth = []
    scores = []
    with open(path, newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            truth.append(int(row["truth"]))
            scores.append(float(row["score"]))
    return np.asarray(truth), np.asarray(scores)


def calculate_roc(truth, scores):
    positives = np.sum(truth == 1)
    negatives = np.sum(truth == 0)
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1]]
    true_positive_rate = []
    false_positive_rate = []

    for threshold in thresholds:
        predicted = scores >= threshold
        true_positives = np.sum(predicted & (truth == 1))
        false_positives = np.sum(predicted & (truth == 0))
        true_positive_rate.append(true_positives / positives)
        false_positive_rate.append(false_positives / negatives)

    false_positive_rate = np.asarray(false_positive_rate)
    true_positive_rate = np.asarray(true_positive_rate)
    order = np.argsort(false_positive_rate)
    false_positive_rate = false_positive_rate[order]
    true_positive_rate = true_positive_rate[order]
    thresholds = thresholds[order]
    curve_auc = np.trapezoid(true_positive_rate, false_positive_rate)
    return false_positive_rate, true_positive_rate, thresholds, curve_auc


def metrics_at_threshold(truth, scores, threshold):
    predicted = scores >= threshold
    positives = truth == 1
    negatives = truth == 0
    true_positive_rate = np.sum(predicted & positives) / np.sum(positives)
    false_positive_rate = np.sum(predicted & negatives) / np.sum(negatives)
    return true_positive_rate, false_positive_rate


def main():
    parser = argparse.ArgumentParser(description="Plot a ROC curve from live SAM3 scores.")
    parser.add_argument("--input", default="evaluation/roc/data/roc_data.csv")
    parser.add_argument("--output", default="evaluation/roc/roc_curve.png")
    parser.add_argument("--table-output", default="evaluation/roc/data/roc_thresholds.csv")
    args = parser.parse_args()

    truth, scores = load_csv(args.input)
    if len(truth) == 0:
        raise ValueError("the CSV contains no labeled rows")
    if len(np.unique(truth)) != 2:
        raise ValueError("ROC requires both truth=0 and truth=1 rows")

    false_positive_rate, true_positive_rate, thresholds, curve_auc = calculate_roc(
        truth,
        scores,
    )
    youden_j = true_positive_rate - false_positive_rate
    best_index = int(np.argmax(youden_j))

    print(f"Samples: {len(truth)}")
    print(f"Positive frames: {int(np.sum(truth == 1))}")
    print(f"Negative frames: {int(np.sum(truth == 0))}")
    print(f"AUC: {curve_auc:.4f}")
    print(f"Best threshold (Youden J): {thresholds[best_index]:.4f}")
    print(f"TPR at best threshold: {true_positive_rate[best_index]:.4f}")
    print(f"FPR at best threshold: {false_positive_rate[best_index]:.4f}")

    common_thresholds = np.arange(0.1, 1.0, 0.1)
    with open(args.table_output, "w", newline="") as table_file:
        writer = csv.writer(table_file)
        writer.writerow(["threshold", "tpr", "fpr"])
        for threshold in common_thresholds:
            tpr, fpr = metrics_at_threshold(truth, scores, threshold)
            writer.writerow([f"{threshold:.1f}", f"{tpr:.4f}", f"{fpr:.4f}"])
            print(f"threshold={threshold:.1f}  TPR={tpr:.4f}  FPR={fpr:.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.plot(false_positive_rate, true_positive_rate,
             label=f"SAM3 (AUC = {curve_auc:.3f})")
    plt.scatter(
        false_positive_rate[best_index],
        true_positive_rate[best_index],
        color="red",
        s=60,
        zorder=3,
        label=f"best threshold = {thresholds[best_index]:.4f}",
    )
    plt.annotate(
        f"threshold={thresholds[best_index]:.4f}",
        (false_positive_rate[best_index], true_positive_rate[best_index]),
        xytext=(10, -20),
        textcoords="offset points",
        color="red",
    )
    plt.plot([0, 1], [0, 1], "--", color="gray", label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("SAM3 Toy-Car Detection ROC")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    print(f"Saved {output}")
    print(f"Saved {args.table_output}")


if __name__ == "__main__":
    main()
