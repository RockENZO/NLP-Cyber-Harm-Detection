"""Score predictions on a held-out CSV without depending on ML packages."""

import argparse
import csv
import json
import hashlib
import io
import tempfile
from collections import Counter
from pathlib import Path


def score_rows(rows, labels=None):
    rows = list(rows)
    if not rows:
        raise ValueError("Prediction file has no data rows")
    if any(not isinstance(row.get(field), str) or not row[field].strip()
           for row in rows for field in ("true_label", "predicted_label")):
        raise ValueError("Prediction labels must be nonempty strings")
    observed = {row["true_label"] for row in rows} | {row["predicted_label"] for row in rows}
    if labels is not None and not isinstance(labels, (list, tuple)):
        raise ValueError("Labels must be an ordered array of nonempty strings")
    labels = list(labels) if labels is not None else sorted(observed)
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("Labels must be nonempty strings")
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
    for source in (args.predictions, args.labels):
        if args.output.resolve() == source.resolve() or (
            args.output.exists() and source.exists() and args.output.samefile(source)
        ):
            parser.error("Output report must not overwrite predictions or labels")
    # Score and hash one byte snapshot, even if another process replaces the CSV.
    prediction_bytes = args.predictions.read_bytes()
    label_bytes = args.labels.read_bytes()
    with io.StringIO(prediction_bytes.decode("utf-8"), newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        fields = reader.fieldnames or []
        if any(not name or not name.strip() for name in fields) or len(set(fields)) != len(fields):
            parser.error("CSV column names must be nonempty and unique")
        if not {"true_label", "predicted_label"}.issubset(fields):
            parser.error("CSV must contain true_label and predicted_label columns")
        try:
            rows = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    parser.error(f"CSV row ending at line {reader.line_num} has the wrong number of columns")
                rows.append(row)
            universe = json.loads(label_bytes.decode("utf-8"))
            if not isinstance(universe, list):
                parser.error("Labels JSON must be an ordered array of nonempty strings")
            result = score_rows(rows, labels=universe)
        except (csv.Error, ValueError) as error:
            parser.error(str(error))
    result['predictions_sha256'] = hashlib.sha256(prediction_bytes).hexdigest()
    result['labels_sha256'] = hashlib.sha256(label_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                         prefix=".evaluation-", delete=False) as report:
            temporary = Path(report.name)
            report.write(json.dumps(result, indent=2) + "\n")
        # Replace the directory entry rather than following an output alias
        # created after the initial input-identity check.
        temporary.replace(args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
