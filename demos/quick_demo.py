"""Run a BERT classifier smoke check from any working directory."""

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = [
    "job_scam", "legitimate", "phishing", "popup_scam", "refund_scam",
    "reward_scam", "sms_spam", "ssn_scam", "tech_support_scam",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="Your appointment is confirmed for tomorrow")
    args = parser.parse_args(argv)

    model_dir = ROOT / "models" / "bert_model"
    tokenizer_dir = ROOT / "models" / "bert_tokenizer"
    weights = model_dir / "model.safetensors"
    if not weights.is_file():
        parser.error("BERT weights are missing; run git lfs pull")
    with weights.open("rb") as handle:
        is_lfs_pointer = handle.read(7) == b"version"
    if is_lfs_pointer:
        parser.error("BERT weights are missing or are a Git LFS pointer; run git lfs pull")

    import torch
    from transformers import BertForSequenceClassification, BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(str(tokenizer_dir))
    model = BertForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    inputs = tokenizer(args.text, max_length=128, truncation=True, return_tensors="pt")
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]
    label_id = int(torch.argmax(probabilities))
    print(f"Label: {LABELS[label_id]}")
    print(f"Model score: {float(probabilities[label_id]):.3f}")
    print("Smoke check only; this score is not calibrated accuracy.")


if __name__ == "__main__":
    main()
