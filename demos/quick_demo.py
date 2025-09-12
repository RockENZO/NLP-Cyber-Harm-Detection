#!/usr/bin/env python3
"""
Quick Start Demo Script
======================

A simple script to quickly test your fraud detection model.
Run this script to get started immediately with fraud detection.

Usage:
    python quick_demo.py
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

def quick_demo():
    """Quick demonstration of fraud detection"""
    
    print("🚀 Quick Fraud Detection Demo")
    print("=" * 40)
    
    # Class labels
    class_labels = [
        'legitimate', 'job_scam', 'phishing', 'popup_scam', 
        'refund_scam', 'reward_scam', 'sms_spam', 'ssn_scam', 'tech_support_scam'
    ]
    
    try:
        # Load model
        print("📥 Loading model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = BertTokenizer.from_pretrained('../models/bert_tokenizer')
        model = BertForSequenceClassification.from_pretrained('../models/bert_model')
        model.to(device)
        model.eval()
        print("✅ Model loaded!")
        
        # Test examples
        test_messages = [
            ("Hi mom, arrived safely at the airport", "legitimate"),
            ("URGENT: Your account is compromised! Click here now", "phishing"),
            ("Congratulations! You won $10,000! Send fee to claim", "reward_scam"),
            ("Your computer has viruses. Call us immediately", "tech_support_scam"),
            ("Work from home! Make $5000/week! Send $99 registration", "job_scam")
        ]
        
        print(f"\n🧪 Testing {len(test_messages)} examples:")
        print("-" * 60)
        
        correct = 0
        for text, expected in test_messages:
            # Predict
            inputs = tokenizer(text, max_length=128, padding='max_length', 
                             truncation=True, return_tensors='pt')
            
            with torch.no_grad():
                outputs = model(input_ids=inputs['input_ids'].to(device),
                              attention_mask=inputs['attention_mask'].to(device))
                probabilities = torch.softmax(outputs.logits, dim=1)
                predicted_id = torch.argmax(outputs.logits, dim=1).item()
                confidence = probabilities[0][predicted_id].item()
            
            predicted = class_labels[predicted_id]
            is_correct = predicted == expected
            if is_correct:
                correct += 1
            
            # Display result
            status = "✅" if is_correct else "❌"
            fraud_flag = "🚨" if predicted != 'legitimate' else "✅"
            
            print(f"{status} {fraud_flag} {predicted:15s} ({confidence:.3f}) | {text[:45]}...")
            if not is_correct:
                print(f"     Expected: {expected}")
        
        print(f"\n📊 Accuracy: {correct}/{len(test_messages)} ({correct/len(test_messages)*100:.1f}%)")
        
        print(f"\n💡 For more detailed testing, use:")
        print(f"   - fraud_detection_demo.py (command line)")
        print(f"   - fraud_detection_demo.ipynb (Jupyter notebook)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\n💡 Make sure:")
        print(f"   1. Model files are in the '../models/' directory")
        print(f"   2. Check that you downloaded the model from Kaggle correctly")
        print(f"   3. Required packages are installed: pip install torch transformers")

if __name__ == "__main__":
    quick_demo()