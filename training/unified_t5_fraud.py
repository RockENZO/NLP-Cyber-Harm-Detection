#!/usr/bin/env python3
"""
Unified classification + reasoning trainer using a text-to-text model (FLAN-T5).

Input CSV expected columns:
- text (str): the input message
- detailed_category (str): gold label from your dataset
- Optional columns for weak supervision of explanations: explanation or rationale

We format targets as: "label: <CATEGORY> | reason: <short explanation>"
When no explanation column exists, we auto-synthesize a concise rationale template
based on the label (label-specific patterns). You can later replace with
curated human rationales if available.

This script trains a single model that, at inference time, generates both the
predicted label and a short reasoning string in a single pass.

Example usage (macOS zsh):
  python training/unified_t5_fraud.py \
    --csv_path final_fraud_detection_dataset.csv \
    --output_dir models/unified-flan-t5-small \
    --model_name google/flan-t5-small \
    --max_train_samples 20000 \
    --num_train_epochs 3

Tip: On Apple Silicon, you can enable MPS acceleration with --device mps.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split


DEFAULT_LABELS = [
    'job_scam',
    'legitimate',
    'phishing',
    'popup_scam',
    'refund_scam',
    'reward_scam',
    'sms_spam',
    'ssn_scam',
    'tech_support_scam'
]


def default_reason_for_label(label: str) -> str:
    templates = {
        'legitimate': "Message content appears normal and lacks scam indicators.",
        'phishing': "Contains credential-stealing cues (links/requests for login or personal info).",
        'tech_support_scam': "Mentions fake support, urgent fixes, or remote access.",
        'reward_scam': "Promises winnings/prizes with urgency or fees.",
        'job_scam': "Unrealistic job offers, upfront payments, or unsolicited interviews.",
        'sms_spam': "Unwanted promo/bulk message characteristics present.",
        'popup_scam': "Fake security alerts/popups demanding immediate action.",
        'refund_scam': "Unexpected refund/chargeback claims with links or callbacks.",
        'ssn_scam': "Requests Social Security info or threats about your SSN.",
    }
    return templates.get(label, "Heuristic cues indicate this category.")


def build_target(label: str, reason: Optional[str]) -> str:
    if not reason or not isinstance(reason, str) or len(reason.strip()) == 0:
        reason = default_reason_for_label(label)
    # Keep concise target; helps generation be focused and fast
    return f"label: {label} | reason: {reason}"


@dataclass
class Seq2SeqExample:
    text: str
    target: str


def make_dataset(df: pd.DataFrame, text_col: str, label_col: str,
                 expl_cols: Optional[List[str]] = None) -> List[Seq2SeqExample]:
    examples: List[Seq2SeqExample] = []
    expl_cols = expl_cols or []
    for _, row in df.iterrows():
        text = str(row[text_col])
        label = str(row[label_col])
        # pick first available explanation-like column
        reason = None
        for c in expl_cols:
            if c in row and isinstance(row[c], str) and row[c].strip():
                reason = row[c]
                break
        target = build_target(label, reason)
        examples.append(Seq2SeqExample(text=text, target=target))
    return examples


def tokenize_examples(tokenizer, examples: List[Seq2SeqExample],
                      max_source_len: int, max_target_len: int) -> Dict[str, List]:
    inputs = tokenizer([e.text for e in examples],
                       max_length=max_source_len,
                       truncation=True,
                       padding=False)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer([e.target for e in examples],
                           max_length=max_target_len,
                           truncation=True,
                           padding=False)
    inputs["labels"] = labels["input_ids"]
    return inputs


def parse_device(arg_device: str) -> torch.device:
    d = arg_device.lower()
    if d == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if d == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def extract_label_from_text(text: str) -> str:
    t = " ".join(str(text).strip().lower().split())
    # very simple parse: look for 'label:' then take next token(s) until '|' or end
    if 'label:' in t:
        after = t.split('label:', 1)[1].strip()
        if '|' in after:
            after = after.split('|', 1)[0].strip()
        # clip to first 5 words to avoid long bleed
        return " ".join(after.split()[:5])
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to CSV with columns: text, detailed_category, optional explanation/rationale')
    parser.add_argument('--text_col', type=str, default='text')
    parser.add_argument('--label_col', type=str, default='detailed_category')
    parser.add_argument('--explanation_cols', type=str, default='explanation,rationale',
                        help='Comma-separated list of possible explanation columns')
    parser.add_argument('--model_name', type=str, default='google/flan-t5-small')
    parser.add_argument('--output_dir', type=str, default='models/unified-flan-t5-small')
    parser.add_argument('--max_source_length', type=int, default=256)
    parser.add_argument('--max_target_length', type=int, default=64)
    parser.add_argument('--input_prefix', type=str, default=(
        "Classify the message into one of these categories and explain briefly:\n"
        "Categories: job_scam, legitimate, phishing, popup_scam, refund_scam, reward_scam, sms_spam, ssn_scam, tech_support_scam\n"
        "Message: "
    ))
    parser.add_argument('--train_size', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--max_train_samples', type=int, default=-1,
                        help='Subsample training examples for quicker runs')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Device resolve
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = parse_device(args.device)

    print(f"Using device: {device}")

    # Load data
    df = pd.read_csv(args.csv_path)
    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found. Available: {list(df.columns)}")
    if args.text_col not in df.columns:
        raise ValueError(f"Text column '{args.text_col}' not found. Available: {list(df.columns)}")

    # Filter to known labels (avoid typos)
    known = set(DEFAULT_LABELS)
    before = len(df)
    df = df[df[args.label_col].isin(known)].copy()
    after = len(df)
    if after < before:
        print(f"Filtered {before - after} rows with unknown labels. Remaining: {after}")

    train_df, eval_df = train_test_split(df, test_size=1 - args.train_size, random_state=args.seed, stratify=df[args.label_col])

    if args.max_train_samples and args.max_train_samples > 0:
        train_df = train_df.sample(n=min(args.max_train_samples, len(train_df)), random_state=args.seed)
        print(f"Subsampled train set to {len(train_df)} rows")

    expl_cols = [c.strip() for c in args.explanation_cols.split(',') if c.strip()]

    # Build examples
    train_examples = make_dataset(train_df, args.text_col, args.label_col, expl_cols)
    eval_examples = make_dataset(eval_df, args.text_col, args.label_col, expl_cols)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Tokenize datasets lazily via map-like closures inside a HuggingFace Trainer-compatible Dataset
    class SimpleMapDataset:
        def __init__(self, examples: List[Seq2SeqExample]):
            self.examples = examples
        def __len__(self):
            return len(self.examples)
        def __getitem__(self, idx):
            e = self.examples[idx]
            source_text = f"{args.input_prefix}{e.text}"
            model_inputs = tokenizer(
                source_text,
                max_length=args.max_source_length,
                truncation=True,
                padding='max_length',
            )
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    e.target,
                    max_length=args.max_target_length,
                    truncation=True,
                    padding='max_length',
                )
            model_inputs['labels'] = labels['input_ids']
            return model_inputs

    train_ds = SimpleMapDataset(train_examples)
    eval_ds = SimpleMapDataset(eval_examples)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=50,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to=['none'],
        fp16=torch.cuda.is_available(),
        bf16=False,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
    )

    def compute_metrics(eval_pred):
        preds, labels_ids = eval_pred
        # Decode predictions
        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 with pad_token_id before decoding labels
        labels_ids = [[(lid if lid != -100 else tokenizer.pad_token_id) for lid in seq] for seq in labels_ids]
        label_texts = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        pred_labels = [extract_label_from_text(t) for t in pred_texts]
        gold_labels = [extract_label_from_text(t) for t in label_texts]

        # map to canonical labels
        canon = set(DEFAULT_LABELS)
        def canonize(s: str) -> str:
            s = s.strip()
            # choose exact match if present, else try loose contains
            if s in canon:
                return s
            for c in canon:
                if s in c or c in s:
                    return c
            return s

        pred_labels = [canonize(x) for x in pred_labels]
        gold_labels = [canonize(x) for x in gold_labels]

        correct = sum(1 for p, g in zip(pred_labels, gold_labels) if p == g)
        acc = correct / max(1, len(gold_labels))
        return {"label_accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save
    print("Saving model to", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Small sanity check on a couple samples
    model.eval()
    inputs = tokenizer([f"{args.input_prefix}{train_examples[0].text}"], return_tensors='pt', truncation=True, padding=True).to(model.device)
    gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    out = tokenizer.decode(gen[0], skip_special_tokens=True)
    print("Sample generation:", out)


if __name__ == '__main__':
    main()
