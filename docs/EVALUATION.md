# Reproducible evaluation protocol

The README distinguishes reproducible classical-classifier studies from historical neural/explanation claims. The source-family holdout and template-grouped nine-class studies publish their own protocols, hashes and reports; their scopes differ and neither validates the bundled neural checkpoints. Those historical checkpoint-specific accuracy and explanation claims still lack equivalent frozen held-out evidence. See the README for the current study commands and results.

## Required evidence for a model result

1. Record the dataset version and SHA-256, source licenses, label mapping, and the exact train/validation/test membership. Keep duplicate or near-duplicate messages and all variants from one source in the same split.
2. Choose the checkpoint using training and validation data only. Freeze the model, tokenizer, threshold, and generation settings before examining the test split.
3. Save one prediction per test item as UTF-8 CSV with `true_label,predicted_label` columns. Keep stable item IDs in an additional column so the predictions can be audited. Do not publish personal messages without permission.
4. Score the frozen file with:

   ```bash
   python evaluation/evaluate_predictions.py held_out_predictions.csv --output evaluation.json
   ```

   The script reports accuracy, macro F1, per-class precision/recall/F1 and a confusion matrix. The default label universe is `evaluation/labels.json`; use `--labels your_labels.json` for another task (for example a binary export). Supply an ordered JSON array of unique, nonempty label strings; absent classes remain in the macro-F1 denominator. CSV rows must match the header width, including any metadata columns, and malformed quoting is rejected. Invalid input does not replace an existing report. Reports record SHA-256 hashes of the exact prediction and label-file snapshots used for scoring, even if the paths change during evaluation. Record the command, Python/package versions, model checkpoint hash, and date alongside `evaluation.json`.
5. Review explanation quality separately with a blinded human rubric for factual grounding, label consistency, and unsupported claims. Classification accuracy alone does not establish explanation faithfulness.

The checked-in BERT `config.json` contains generic `LABEL_0` through `LABEL_8`; the demo's class mapping must be retained with any exported predictions. The short examples in the demo are a smoke check, not a test set.
