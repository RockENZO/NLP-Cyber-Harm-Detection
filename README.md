# NLP Cyber Harm Detection

Research code and model artifacts for multiclass scam detection and explanation experiments. The repository includes classical baselines, BERT and DistilBERT classifiers, and notebooks for generative models. It is a research project; the advertised historical accuracy and explanation claims do not yet have a frozen, independently reproducible test report in this repository.

## Quick start: checked-in BERT classifier

The BERT checkpoint and tokenizer are tracked with Git LFS. From the repository root:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/RockENZO/NLP-Cyber-Harm-Detection.git
cd NLP-Cyber-Harm-Detection
git lfs install
git lfs pull --include="models/bert_model/*"
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demos/quick_demo.py --text "Your appointment is confirmed for tomorrow"
```

The demo uses the checked-in `models/bert_model` and `models/bert_tokenizer` folders and prints a predicted label and model score. It is a smoke check, not a measurement of accuracy. The checkpoint is approximately 438 MB, so downloading it and running inference need adequate storage and memory. If Git LFS files are missing, `git lfs pull --include="models/bert_model/*"` is required before inference.

## Evaluation

Follow [the evaluation protocol](docs/EVALUATION.md) before reporting a model score. It requires a source-disjoint held-out test set, frozen checkpoint and label mapping, exported predictions, and per-class metrics. The repository does not currently contain all of those artifacts, so the historical 94–96% accuracy and explanation-quality statements are unverified here. Classification scores alone cannot validate generated explanations.

To score an exported prediction file with `true_label,predicted_label` columns:

```bash
python evaluation/evaluate_predictions.py held_out_predictions.csv --output evaluation.json
```

## Repository map

- `demos/quick_demo.py`: BERT inference smoke check
- `demos/fraud_detection_demo.py`: interactive BERT exploration
- `training/`: baseline and model training scripts and notebooks
- `models/`: saved checkpoints and tokenizers, many via Git LFS
- `docs/`: architecture notes and evaluation protocol
- `reasoning/`: experimental explanation notebooks

See [historical project notes](docs/PROJECT_HISTORY.md) for previous experiments and reported results. Those notes include older commands and performance claims that have not been independently verified against a frozen test set.

## Reproducible source holdout baseline

The reviewed corpus has nine categories (eight fraud categories plus legitimate). `evaluation/labels.json` fixes their order for multiclass scoring, including zero-support categories. Unknown labels are rejected. Reports include legitimate false-positive rate and prediction-file SHA-256.

A new **binary** baseline was trained with the entire `difraud_` source family excluded from training. This is a fresh TF-IDF + balanced Logistic Regression baseline, not a measurement of the bundled neural checkpoints.

| Metric | Reviewed run |
| --- | ---: |
| Training rows | 158,909 |
| Held-out source rows | 36,004 |
| Accuracy | 0.7773 |
| Macro F1 (binary) | 0.7339 |
| Legitimate false-positive rate | 0.2455 |

See [full report](evaluation/reports/source_baseline_20260930.json) for per-source metrics, parameters, input/provenance hashes, split/prediction hashes and runtime versions. These results show substantial false-positive and cross-source generalization problems; they do not support a production-readiness claim.

```bash
# In the data repository, use the reviewed build from its README first.
# Then in this repository (Python 3.12):
pip install -r evaluation/requirements.txt
python evaluation/source_baseline.py \
  --data ../data/artifacts/verified/final_fraud_detection_dataset.csv \
  --provenance ../data/artifacts/verified/provenance.jsonl \
  --output-dir runs/source-baseline
python evaluation/evaluate_predictions.py runs/source-baseline/predictions.csv \
  --labels evaluation/binary_labels.json --output runs/source-baseline/scored.json
```

The partition verifies row order, content hashes, unique texts and provenance before fitting. No vocabulary is fitted on test texts. It writes exact train/test sample IDs in `split_manifest.json`; predictions contain sample IDs and source identifiers without message content. Test source identity is excluded from features. Source grouping does not prove template/semantic independence, and inherited spam/fraud labels require further adjudication.

For existing BERT/T5/BART/LLM checkpoints, their original training IDs and untouched held-out data must be recovered before publishing accuracy. The combined corpus may already have been used in training, so scoring these checkpoints on a newly chosen subset is not a valid held-out evaluation. Explanation faithfulness also requires a separate reviewed annotation protocol; this binary classifier report makes no explanation-quality claim.
