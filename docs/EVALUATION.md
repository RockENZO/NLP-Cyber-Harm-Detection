# Reproducible evaluation protocol

The headline accuracy claims in the README are historical project notes. The repository does not currently include a frozen, source-disjoint test manifest and prediction file sufficient to independently reproduce them. Do not present those figures as verified deployment results.

## Required evidence for a model result

1. Record the dataset version and SHA-256, source licenses, label mapping, and the exact train/validation/test membership. Keep duplicate or near-duplicate messages and all variants from one source in the same split.
2. Choose the checkpoint using training and validation data only. Freeze the model, tokenizer, threshold, and generation settings before examining the test split.
3. Save one prediction per test item as UTF-8 CSV with `true_label,predicted_label` columns. Keep stable item IDs in an additional column so the predictions can be audited. Do not publish personal messages without permission.
4. Score the frozen file with:

   ```bash
   python evaluation/evaluate_predictions.py held_out_predictions.csv --output evaluation.json
   ```

   The script reports accuracy, macro F1, per-class precision/recall/F1 and a confusion matrix. Record the command, Python/package versions, model checkpoint hash, and date alongside `evaluation.json`.
5. Review explanation quality separately with a blinded human rubric for factual grounding, label consistency, and unsupported claims. Classification accuracy alone does not establish explanation faithfulness.

The checked-in BERT `config.json` contains generic `LABEL_0` through `LABEL_8`; the demo's class mapping must be retained with any exported predictions. The short examples in the demo are a smoke check, not a test set.
