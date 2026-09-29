# NLP Cyber Harm Detection

Research code and model artifacts for multiclass scam detection and explanation experiments. The repository includes classical baselines, BERT and DistilBERT classifiers, and notebooks for generative models. It is a research project; the advertised historical accuracy and explanation claims do not yet have a frozen, independently reproducible test report in this repository.

## Quick start: checked-in BERT classifier

The BERT checkpoint and tokenizer are tracked with Git LFS. From the repository root:

```bash
git clone https://github.com/RockENZO/NLP-Cyber-Harm-Detection.git
cd NLP-Cyber-Harm-Detection
git lfs install
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demos/quick_demo.py --text "Your appointment is confirmed for tomorrow"
```

The demo uses the checked-in `models/bert_model` and `models/bert_tokenizer` folders and prints a predicted label and model score. It is a smoke check, not a measurement of accuracy. The checkpoint is approximately 438 MB, so downloading it and running inference need adequate storage and memory. If Git LFS files are missing, `git lfs pull` is required before inference.

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
