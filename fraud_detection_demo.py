#!/usr/bin/env python3
"""
Fraud Detection Model Demo
=========================

This script loads your trained BERT model from Kaggle and provides
interactive fraud detection predictions.

Usage:
    python fraud_detection_demo.py

Features:
- Load saved BERT model and tokenizer
- Interactive text input for fraud detection
- Batch prediction capabilities
- Confidence scores and class probabilities
- Sample test cases for different fraud types
"""

import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionDemo:
    """
    Demo class for fraud detection using trained BERT model
    """
    
    def __init__(self, model_path='models/bert_model', tokenizer_path='models/bert_tokenizer'):
        """
        Initialize the demo with trained model and tokenizer
        
        Args:
            model_path: Path to saved BERT model directory
            tokenizer_path: Path to saved tokenizer directory
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Class labels based on your multiclass training
        # Update these if your actual classes are different
        self.class_labels = [
            'legitimate',
            'job_scam', 
            'phishing',
            'popup_scam',
            'refund_scam',
            'reward_scam',
            'sms_spam',
            'ssn_scam',
            'tech_support_scam'
        ]
        
        print(f"🚀 Initializing Fraud Detection Demo...")
        print(f"Device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
        
    def _load_model(self):
        """Load the trained BERT model and tokenizer"""
        try:
            print("📥 Loading tokenizer...")
            self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)
            
            print("📥 Loading BERT model...")
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ Model loaded successfully!")
            print(f"📊 Model supports {len(self.class_labels)} classes")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict_single(self, text, max_length=128):
        """
        Predict fraud type for a single text message
        
        Args:
            text: Input text to classify
            max_length: Maximum sequence length for tokenization
            
        Returns:
            dict: Prediction results with class, confidence, and probabilities
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class_id = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_class_id].item()
        
        # Format results
        predicted_class = self.class_labels[predicted_class_id]
        all_probabilities = {
            self.class_labels[i]: prob.item() 
            for i, prob in enumerate(probabilities[0])
        }
        
        return {
            'text': text,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'is_fraud': predicted_class != 'legitimate',
            'all_probabilities': all_probabilities
        }
    
    def predict_batch(self, texts, max_length=128, batch_size=16):
        """
        Predict fraud types for multiple texts
        
        Args:
            texts: List of texts to classify
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predicted_classes = torch.argmax(logits, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
            
            # Format batch results
            for j, text in enumerate(batch_texts):
                predicted_class_id = predicted_classes[j].item()
                predicted_class = self.class_labels[predicted_class_id]
                confidence = confidences[j].item()
                
                all_probabilities = {
                    self.class_labels[k]: probabilities[j][k].item() 
                    for k in range(len(self.class_labels))
                }
                
                results.append({
                    'text': text,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'is_fraud': predicted_class != 'legitimate',
                    'all_probabilities': all_probabilities
                })
        
        return results
    
    def print_prediction(self, result):
        """Pretty print a single prediction result"""
        print("\n" + "="*60)
        print("🔍 FRAUD DETECTION RESULT")
        print("="*60)
        print(f"📝 Text: {result['text'][:100]}{'...' if len(result['text']) > 100 else ''}")
        print(f"🎯 Prediction: {result['predicted_class'].upper()}")
        print(f"📊 Confidence: {result['confidence']:.4f}")
        print(f"🚨 Is Fraud: {'YES' if result['is_fraud'] else 'NO'}")
        
        print(f"\n📋 All Class Probabilities:")
        sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
        for class_name, prob in sorted_probs:
            bar = "█" * int(prob * 20)  # Visual bar
            print(f"  {class_name:20s} {prob:.4f} {bar}")
    
    def get_sample_test_cases(self):
        """Get sample test cases for different fraud types"""
        return {
            'legitimate': [
                "Hi mom, just wanted to let you know I arrived safely at the airport.",
                "Meeting rescheduled to 3 PM tomorrow. Please confirm your attendance.",
                "Thank you for your purchase. Your order will be shipped within 2 business days."
            ],
            'phishing': [
                "URGENT: Your bank account has been compromised. Click here to secure it immediately: http://suspicious-bank-link.com",
                "Your PayPal account will be suspended. Verify your information now to avoid closure.",
                "Security Alert: Someone tried to access your account. Confirm your identity here."
            ],
            'tech_support_scam': [
                "ALERT: Your computer is infected with 5 viruses! Call 1-800-SCAM-NOW for immediate assistance.",
                "Microsoft Security: We detected malware on your PC. Call us now to fix it.",
                "Your Windows license expired. Call this number to renew: +1-555-FAKE"
            ],
            'reward_scam': [
                "Congratulations! You've won $10,000 in our lottery. Send processing fee to claim.",
                "WINNER! You've been selected for a $5000 gift card. Click to claim your prize.",
                "Amazing news! You won a free iPhone. Just pay shipping costs to receive it."
            ],
            'job_scam': [
                "Work from home opportunity! Earn $5000/week with no experience required. Send $200 for starter kit.",
                "Urgent: We need someone to process payments. Easy $3000/week. Send bank details.",
                "Be your own boss! Make money online. Just pay $99 registration fee to start."
            ],
            'sms_spam': [
                "FREE! Claim your free trial now! Text STOP to opt out. Msg&data rates apply.",
                "Hot singles in your area want to meet you! Click here now!",
                "Congratulations! You qualify for a debt consolidation loan. Apply now!"
            ]
        }
    
    def run_sample_tests(self):
        """Run predictions on sample test cases"""
        print("\n🧪 RUNNING SAMPLE TEST CASES")
        print("="*60)
        
        test_cases = self.get_sample_test_cases()
        
        for fraud_type, examples in test_cases.items():
            print(f"\n📂 Testing {fraud_type.upper()} examples:")
            print("-" * 40)
            
            for example in examples:
                result = self.predict_single(example)
                
                # Check if prediction matches expected type
                is_correct = (result['predicted_class'] == fraud_type)
                status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                
                print(f"\n{status}")
                print(f"Text: {example[:80]}{'...' if len(example) > 80 else ''}")
                print(f"Expected: {fraud_type} | Predicted: {result['predicted_class']} | Confidence: {result['confidence']:.4f}")
    
    def interactive_demo(self):
        """Run interactive demo where user can input text for prediction"""
        print("\n🎮 INTERACTIVE FRAUD DETECTION DEMO")
        print("="*60)
        print("Enter text messages to check for fraud (type 'quit' to exit)")
        print("Type 'samples' to run sample test cases")
        print("-" * 60)
        
        while True:
            user_input = input("\n📝 Enter text to analyze: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Thanks for using the fraud detection demo!")
                break
                
            elif user_input.lower() == 'samples':
                self.run_sample_tests()
                continue
                
            elif not user_input:
                print("⚠️ Please enter some text to analyze")
                continue
            
            # Get prediction
            result = self.predict_single(user_input)
            self.print_prediction(result)


def main():
    """Main function to run the demo"""
    print("🚀 Fraud Detection Model Demo")
    print("=" * 40)
    
    try:
        # Initialize demo
        demo = FraudDetectionDemo()
        
        # Show menu
        print("\n📋 Available Options:")
        print("1. Interactive demo (type text for prediction)")
        print("2. Run sample test cases")
        print("3. Both")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            demo.interactive_demo()
        elif choice == '2':
            demo.run_sample_tests()
        elif choice == '3':
            demo.run_sample_tests()
            demo.interactive_demo()
        else:
            print("Invalid choice. Running interactive demo...")
            demo.interactive_demo()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your model files are in the 'models/' directory")
        print("2. Check that you have the required packages installed:")
        print("   pip install torch transformers")
        print("3. Verify the model was saved correctly from Kaggle")


if __name__ == "__main__":
    main()