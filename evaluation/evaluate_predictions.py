"""Score predictions on a held-out CSV without depending on ML packages."""

import argparse
import csv
import json
import hashlib
from collections import Counter
from pathlib import Path


def score_rows(rows, labels=None):
    rows = list(rows)
    if not rows:
        raise ValueError("Prediction file has no data rows")
    observed = {row["true_label"] for row in rows} | {row["predicted_label"] for row in rows}
    labels = list(labels) if labels is not None else sorted(observed)
    if not labels or len(set(labels)) != len(labels) or not observed.issubset(labels):
        raise ValueError("Labels must be unique and include every observed label")
    if any(not row["true_label"] or not row["predicted_label"] for row in rows):
        raise ValueError("Labels must not be empty")
    counts = Counter((row["true_label"], row["predicted_label"]) for row in rows)
    per_class = {}
    for label in labels:
        tp = counts[label, label]
        fp = sum(counts[actual, label] for actual in labels if actual != label)
        fn = sum(counts[label, predicted] for predicted in labels if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": tp + fn, "precision": precision,
                            "recall": recall, "f1": f1}
    legitimate_support = sum(counts['legitimate', predicted] for predicted in labels)
    false_positives = sum(counts['legitimate', predicted] for predicted in labels if predicted != 'legitimate')
    return {
        "labels": labels,
        "legitimate_false_positive_rate": false_positives / legitimate_support if legitimate_support else None,
        "samples": len(rows),
        "accuracy": sum(counts[label, label] for label in labels) / len(rows),
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(labels),
        "per_class": per_class,
        "confusion_matrix": {actual: {predicted: counts[actual, predicted]
                                       for predicted in labels} for actual in labels},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path,
                        help="CSV with true_label and predicted_label columns")
    parser.add_argument("--output", type=Path, default=Path("evaluation.json"))
    parser.add_argument('--labels', type=Path, default=Path(__file__).with_name('labels.json'), help='Explicit ordered label universe JSON')
    args = parser.parse_args(argv)
    with args.predictions.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"true_label", "predicted_label"}.issubset(reader.fieldnames or []):
            parser.error("CSV must contain true_label and predicted_label columns")
        result = score_rows(reader, labels=json.loads(args.labels.read_text()))
    result['predictions_sha256'] = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
