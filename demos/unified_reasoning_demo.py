#!/usr/bin/env python3
"""
Inference demo for the unified classification + reasoning model.
Loads a fine-tuned FLAN-T5 model and generates outputs of the form:
  "label: <CATEGORY> | reason: <short explanation>"

We parse the generation to extract label and reason. For a light confidence proxy,
we compute average logprobs for generated tokens if model returns scores.

Example:
  python demos/unified_reasoning_demo.py \
    --model_dir models/unified-flan-t5-small \
    --text "Congratulations! You won a gift card..."
"""
import argparse
import json
import re
from typing import Dict, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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

LABEL_PATTERN = re.compile(r"label\s*:\s*([a-zA-Z0-9_\- ]+)")
REASON_PATTERN = re.compile(r"reason\s*:\s*(.+)$")


def parse_output(text: str) -> Tuple[str, str]:
    # Normalize whitespace
    t = " ".join(text.strip().split())
    label = None
    reason = None
    # Split on bar if present
    if "|" in t:
        parts = [p.strip() for p in t.split("|")]
        for p in parts:
            m1 = LABEL_PATTERN.search(p)
            m2 = REASON_PATTERN.search(p)
            if m1:
                label = m1.group(1).strip().lower()
            if m2:
                reason = m2.group(1).strip()
    else:
        # Try standalone patterns
        m1 = LABEL_PATTERN.search(t)
        if m1:
            label = m1.group(1).strip().lower()
        m2 = REASON_PATTERN.search(t)
        if m2:
            reason = m2.group(1).strip()
    return (label or "unknown"), (reason or "")


INSTRUCTION_PREFIX = (
    "Classify the message into one of these categories and explain briefly:\n"
    "Categories: job_scam, legitimate, phishing, popup_scam, refund_scam, reward_scam, sms_spam, ssn_scam, tech_support_scam\n"
    "Message: "
)


def generate(model, tokenizer, text: str, max_new_tokens: int = 64) -> Dict:
    device = model.device
    inp = INSTRUCTION_PREFIX + text
    inputs = tokenizer([inp], return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    seq = outputs.sequences[0]
    decoded = tokenizer.decode(seq, skip_special_tokens=True)

    # Confidence proxy: average of max softmax over vocab per step
    # Note: this is not a calibrated probability of the label; just a rough signal.
    avg_token_conf = None
    if outputs.scores:
        # scores is a list of logits per generated token step
        probs = [torch.softmax(s, dim=-1).max().item() for s in outputs.scores]
        avg_token_conf = sum(probs) / len(probs)

    label, reason = parse_output(decoded)
    # sanitize label to known set
    if label not in DEFAULT_LABELS:
        # try closest match by simple heuristic
        lower_map = {l.lower(): l for l in DEFAULT_LABELS}
        label = lower_map.get(label.lower(), label)
        if label not in DEFAULT_LABELS:
            # fallback: classify by prompting again with forced format
            pass

    return {
        'input_text': text,
        'raw_output': decoded,
        'predicted_label': label,
        'reason': reason,
        'avg_generation_confidence': avg_token_conf,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', type=str, required=True, help='Path to fine-tuned model directory')
    ap.add_argument('--text', type=str, default="")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

    # Prefer GPU/MPS if available
    if torch.cuda.is_available():
        model.to('cuda')
    elif torch.backends.mps.is_available():
        model.to('mps')
    else:
        model.to('cpu')

    if not args.text:
        samples = [
            "We detected unusual activity on your account. Please login at http://phish.example to verify.",
            "Your interview is scheduled for tomorrow. Please see the attached calendar invite.",
            "Congratulations! You have won a $1000 gift card. Claim now.",
        ]
    else:
        samples = [args.text]

    for i, t in enumerate(samples, 1):
        result = generate(model, tokenizer, t)
        print("\n" + "="*80)
        print(f"Sample {i}")
        print("-"*80)
        print("Text:", t)
        print("Prediction:", result['predicted_label'])
        print("Reason:", result['reason'])
        conf = result['avg_generation_confidence']
        if conf is not None:
            print(f"Confidence (token avg): {conf:.3f}")
        # print raw for debugging
        print("Raw:", result['raw_output'])


if __name__ == '__main__':
    main()
